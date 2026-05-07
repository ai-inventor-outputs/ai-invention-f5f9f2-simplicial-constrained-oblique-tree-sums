#!/usr/bin/env python3
"""
XGBoost Simplicial-Constraint Ablation: 5 Modes × 10 Datasets × 5-Fold CV

Ablation experiment comparing 5 XGBoost interaction constraint sources:
  A) Simplicial/TDA-derived constraints
  B) Random-matched constraints (5 seeds, averaged)
  C) Correlation-clustering constraints
  D) Unconstrained baseline
  E) Mutual-information-derived constraints

Output: method_out.json (exp_gen_sol_out schema)
"""

import json
import os
import resource
import sys
import time
from itertools import combinations
from pathlib import Path

import dcor
import gudhi
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from loguru import logger
from scipy.stats import wilcoxon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# ── Resource limits ──────────────────────────────────────────────────
# Cap at 30GB RAM, 3600s CPU
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Workspace paths ──────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/home/adrian/projects/temp/ai-inventor-old3/aii_pipeline/runs/"
    "run__20260228_133939/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
DATA_FILE = DATA_DIR / "full_data_out.json"
OUTPUT_FILE = WORKSPACE / "method_out.json"
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Logging setup ────────────────────────────────────────────────────
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add(LOG_DIR / "experiment.log", rotation="30 MB", level="DEBUG")

# ── Env overrides (for testing) ───────────────────────────────────────
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = no limit
N_ESTIMATORS_OVERRIDE = int(os.environ.get("N_ESTIMATORS", "0"))  # 0 = use default 300
DCOR_SUBSAMPLE = 5000

# ── XGBoost fixed hyperparameters ────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": N_ESTIMATORS_OVERRIDE if N_ESTIMATORS_OVERRIDE > 0 else 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": 2,  # Limited to avoid contention with other agents
    "verbosity": 0,
    "tree_method": "hist",
}

# ── Dataset names (all 10) ───────────────────────────────────────────
DATASETS = [
    "friedman1", "friedman3", "synth_3way", "synth_4way",
    "diabetes", "breast_w", "california_housing", "wine_quality",
    "adult", "spambase",
]

SYNTHETIC_DATASETS = ["friedman1", "friedman3", "synth_3way", "synth_4way"]



# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

@logger.catch
def load_all_datasets() -> dict:
    """Load full_data_out.json and return structured dict of datasets."""
    logger.info(f"Loading data from {DATA_FILE}")
    raw = json.loads(DATA_FILE.read_text())
    logger.info(f"Loaded {len(raw['datasets'])} datasets")

    result = {}
    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        if ds_name not in DATASETS:
            continue

        examples = ds_entry["examples"]
        if MAX_EXAMPLES > 0:
            examples = examples[:MAX_EXAMPLES]

        n_features = examples[0]["metadata_n_features"]
        task_type = examples[0]["metadata_task_type"]
        feature_names = examples[0]["metadata_feature_names"]

        # Parse known_interactions
        known_interactions = None
        if "metadata_known_interactions" in examples[0]:
            ki_str = examples[0]["metadata_known_interactions"]
            known_interactions = json.loads(ki_str) if isinstance(ki_str, str) else ki_str

        # Build X, y, folds
        X = np.array([json.loads(ex["input"]) for ex in examples], dtype=np.float64)
        y_raw = [ex["output"] for ex in examples]
        folds = np.array([ex["metadata_fold"] for ex in examples], dtype=int)

        # Encode y
        if task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y_raw).astype(np.float64)
        else:
            y = np.array([float(v) for v in y_raw], dtype=np.float64)

        result[ds_name] = {
            "X": X,
            "y": y,
            "folds": folds,
            "task_type": task_type,
            "feature_names": feature_names,
            "known_interactions": known_interactions,
            "n_features": n_features,
        }
        logger.info(
            f"  {ds_name}: X={X.shape}, task={task_type}, "
            f"n_features={n_features}, n_folds={len(np.unique(folds))}"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════
# MODE A: SIMPLICIAL (TDA) INTERACTION CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def compute_dcor_matrix(
    X: np.ndarray,
    y: np.ndarray,
    subsample: int = DCOR_SUBSAMPLE,
) -> np.ndarray:
    """Compute pairwise distance correlation dissimilarity matrix.

    Uses sqrt(1 - dCor) as dissimilarity (satisfies triangle inequality).
    """
    n_samples, n_features = X.shape

    # Subsample for speed
    if n_samples > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, size=subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dist_matrix = np.zeros((n_features, n_features), dtype=np.float64)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            try:
                dc = dcor.distance_correlation(X_sub[:, i], X_sub[:, j])
                d = np.sqrt(max(0.0, 1.0 - dc))
            except Exception:
                logger.exception(f"dCor failed for features ({i}, {j})")
                d = 1.0  # max dissimilarity fallback
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def select_threshold(simplex_tree: gudhi.SimplexTree) -> float:
    """Select persistence threshold using largest-gap heuristic on dim-0 deaths."""
    simplex_tree.compute_persistence()
    persistence = simplex_tree.persistence()

    # Extract death values for dimension 0 (connected components)
    deaths = sorted([
        death for dim, (birth, death) in persistence
        if dim == 0 and death != float("inf")
    ])

    if len(deaths) < 2:
        # Fallback: use median of all finite filtration values
        all_vals = [filt for _, filt in simplex_tree.get_filtration() if filt < float("inf")]
        if all_vals:
            return float(np.median(all_vals))
        return 0.5

    # Largest-gap heuristic
    gaps = [deaths[i + 1] - deaths[i] for i in range(len(deaths) - 1)]
    max_gap_idx = int(np.argmax(gaps))
    threshold = (deaths[max_gap_idx] + deaths[max_gap_idx + 1]) / 2.0

    # Cap threshold: if it's > 0.9, the complex is nearly fully connected
    # Fall back to 25th percentile of death values
    if threshold > 0.9 and len(deaths) > 2:
        logger.warning(
            f"Threshold {threshold:.4f} too high (nearly fully connected). "
            f"Falling back to 25th percentile of death values."
        )
        threshold = float(np.percentile(deaths, 25))
        # Ensure it's at least the minimum death value
        threshold = max(threshold, deaths[0] + 1e-6)

    return threshold


def extract_maximal_simplices(
    simplex_tree: gudhi.SimplexTree,
    threshold: float,
    max_simplices: int = 5000,
) -> list[list[int]]:
    """Extract maximal simplices at given filtration threshold.

    If number of simplices exceeds max_simplices, only keep higher-dimensional ones.
    """
    # Collect all simplices with filtration <= threshold, grouped by dimension
    by_dim: dict[int, list[frozenset]] = {}
    total_count = 0
    for simplex, filt in simplex_tree.get_filtration():
        if filt <= threshold:
            fs = frozenset(simplex)
            dim = len(fs)
            by_dim.setdefault(dim, []).append(fs)
            total_count += 1

    if total_count == 0:
        return []

    if total_count > max_simplices:
        logger.warning(
            f"Too many simplices ({total_count}) at threshold {threshold:.4f}. "
            f"Keeping only highest-dimension simplices."
        )
        # Only keep the top dimensions
        dims_desc = sorted(by_dim.keys(), reverse=True)
        all_simplices = []
        for d in dims_desc:
            all_simplices.extend(by_dim[d])
            if len(all_simplices) >= max_simplices:
                break
    else:
        all_simplices = []
        for dim_list in by_dim.values():
            all_simplices.extend(dim_list)

    # Sort by dimension descending for efficient maximal check
    all_simplices.sort(key=len, reverse=True)

    maximal = []
    for s in all_simplices:
        if len(s) == 0:
            continue
        is_maximal = True
        for m in maximal:
            if s.issubset(m):
                is_maximal = False
                break
        if is_maximal:
            maximal.append(s)

    return [sorted(list(s)) for s in maximal]


def build_simplicial_constraints(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[list[list[int]], dict]:
    """Build TDA-based interaction constraints.

    Returns:
        (constraint_groups, diagnostics_dict)
    """
    n_features = X_train.shape[1]

    # Step 1: Compute distance matrix
    dist_matrix = compute_dcor_matrix(X_train, y_train)

    # Step 2: Build Rips complex
    rips = gudhi.RipsComplex(
        distance_matrix=dist_matrix,
        max_edge_length=1.0,
    )
    st = rips.create_simplex_tree(max_dimension=3)

    # Step 3: Select threshold
    threshold = select_threshold(st)

    # Step 4: Extract maximal simplices
    maximal_simplices = extract_maximal_simplices(st, threshold)

    # Step 5: Compute Betti numbers
    st.compute_persistence()
    betti = st.betti_numbers()

    # Step 6: Handle isolated features — add singletons
    covered = set()
    for group in maximal_simplices:
        covered.update(group)

    groups = list(maximal_simplices)
    for i in range(n_features):
        if i not in covered:
            groups.append([i])

    # Count total simplices at threshold
    n_simplices_total = sum(
        1 for _, filt in st.get_filtration() if filt <= threshold
    )
    max_dim = max((len(s) - 1 for s in maximal_simplices), default=0)

    diagnostics = {
        "n_features": n_features,
        "persistence_threshold": round(float(threshold), 6),
        "n_simplices_total": n_simplices_total,
        "n_maximal_simplices": len(maximal_simplices),
        "max_simplex_dimension": max_dim,
        "betti_numbers": betti[:3] if len(betti) >= 3 else betti + [0] * (3 - len(betti)),
    }

    return groups, diagnostics


# ═══════════════════════════════════════════════════════════════════════
# MODE B: RANDOM-MATCHED CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def build_random_constraints(
    simplicial_groups: list[list[int]],
    n_features: int,
    seed: int,
) -> list[list[int]]:
    """Build random constraints matching the cardinality distribution of simplicial groups."""
    rng = np.random.RandomState(seed)
    group_sizes = [len(g) for g in simplicial_groups]

    # Shuffle features
    feature_indices = list(range(n_features))
    rng.shuffle(feature_indices)

    groups = []
    idx = 0
    for size in group_sizes:
        if idx + size <= n_features:
            groups.append(sorted(feature_indices[idx:idx + size]))
            idx += size
        else:
            # Remaining features as one group
            remaining = sorted(feature_indices[idx:])
            if remaining:
                groups.append(remaining)
            idx = n_features
            break

    # Any leftover features → singletons
    if idx < n_features:
        for i in feature_indices[idx:]:
            groups.append([i])

    return groups


# ═══════════════════════════════════════════════════════════════════════
# MODE C: CORRELATION-CLUSTERING CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def build_corr_clustering_constraints(
    X_train: np.ndarray,
    n_groups: int,
) -> list[list[int]]:
    """Build correlation-based hierarchical clustering constraints."""
    n_features = X_train.shape[1]

    if n_groups <= 0:
        n_groups = 1
    if n_groups >= n_features:
        return [[i] for i in range(n_features)]

    # Compute absolute Pearson correlation
    corr = np.abs(np.corrcoef(X_train.T))
    # Handle NaN (constant features)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    dist = 1.0 - corr
    # Ensure non-negative
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    condensed = ssd.squareform(dist, checks=False)
    # Clamp negative values from floating point errors
    condensed = np.clip(condensed, 0.0, None)

    Z = sch.linkage(condensed, method="ward")
    labels = sch.fcluster(Z, t=n_groups, criterion="maxclust")

    groups = {}
    for feat_idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(feat_idx)

    return [sorted(v) for v in groups.values()]


# ═══════════════════════════════════════════════════════════════════════
# MODE E: MI-DERIVED CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════

def build_mi_constraints(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_groups: int,
    task_type: str,
) -> list[list[int]]:
    """Build MI-derived constraints using round-robin assignment."""
    n_features = X_train.shape[1]

    if n_groups <= 0:
        n_groups = 1
    if n_groups >= n_features:
        return [[i] for i in range(n_features)]

    # Compute MI
    if task_type == "classification":
        mi = mutual_info_classif(X_train, y_train, random_state=42)
    else:
        mi = mutual_info_regression(X_train, y_train, random_state=42)

    # Rank features by MI descending
    ranked = np.argsort(-mi)

    # Round-robin assignment
    groups = [[] for _ in range(n_groups)]
    for rank, feat_idx in enumerate(ranked):
        groups[rank % n_groups].append(int(feat_idx))

    return [sorted(g) for g in groups if g]


# ═══════════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    interaction_constraints: list[list[int]] | None = None,
) -> dict:
    """Train XGBoost and evaluate metrics."""
    params = dict(XGBOOST_PARAMS)

    if task_type == "classification":
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            params["eval_metric"] = "logloss"
            params["objective"] = "binary:logistic"
        else:
            params["eval_metric"] = "mlogloss"
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes

        model = XGBClassifier(**params)
    else:
        params["eval_metric"] = "rmse"
        model = XGBRegressor(**params)

    if interaction_constraints is not None:
        # XGBoost 3.x requires string format for interaction_constraints
        model.set_params(interaction_constraints=str(interaction_constraints))

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    metrics = {"train_time_sec": round(train_time, 4)}

    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        metrics["accuracy"] = round(float(acc), 6)
        metrics["r2"] = None
        metrics["rmse"] = None

        # AUROC — skip if only one class in test set
        try:
            unique_test = np.unique(y_test)
            if len(unique_test) < 2:
                logger.warning("Single class in test set, AUROC undefined")
                metrics["auroc"] = None
            else:
                n_classes = len(np.unique(y_train))
                if n_classes == 2:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auroc = roc_auc_score(y_test, y_proba)
                else:
                    y_proba = model.predict_proba(X_test)
                    auroc = roc_auc_score(
                        y_test, y_proba, multi_class="ovr", average="macro"
                    )
                if np.isnan(auroc):
                    metrics["auroc"] = None
                else:
                    metrics["auroc"] = round(float(auroc), 6)
        except (ValueError, TypeError):
            logger.warning("AUROC computation failed")
            metrics["auroc"] = None
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["r2"] = round(float(r2), 6)
        metrics["rmse"] = round(float(rmse), 6)
        metrics["accuracy"] = None
        metrics["auroc"] = None

    return metrics


def average_metrics(seed_metrics: list[dict]) -> dict:
    """Average metrics across random seeds."""
    result = {}
    for key in seed_metrics[0]:
        vals = [
            m[key] for m in seed_metrics
            if m[key] is not None and not (isinstance(m[key], float) and np.isnan(m[key]))
        ]
        if vals and isinstance(vals[0], (int, float)):
            mean_val = float(np.mean(vals))
            result[key] = round(mean_val, 6) if np.isfinite(mean_val) else None
        else:
            result[key] = seed_metrics[0][key]
    return result


# ═══════════════════════════════════════════════════════════════════════
# INTERACTION RECOVERY (SYNTHETIC ONLY)
# ═══════════════════════════════════════════════════════════════════════

def compute_interaction_recovery(
    simplicial_groups: list[list[int]],
    known_interactions: dict,
) -> dict:
    """Compute P/R/F1 for interaction recovery using subset matching."""
    # Build known interaction groups (all interaction tuples)
    known_groups = []
    for key in ["2-way", "3-way", "4-way"]:
        if key in known_interactions:
            for group in known_interactions[key]:
                known_groups.append(frozenset(group))

    if not known_groups:
        return {"precision": None, "recall": None, "f1": None, "known_interactions": [], "discovered_simplices": []}

    # Predicted: all non-singleton simplices
    predicted = [frozenset(g) for g in simplicial_groups if len(g) > 1]

    if not predicted:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "known_interactions": [sorted(list(g)) for g in known_groups],
            "discovered_simplices": [],
        }

    # Precision: fraction of predicted that are subsets of some known group
    correct_predictions = sum(
        1 for p in predicted
        if any(p.issubset(k) for k in known_groups)
    )
    precision = correct_predictions / len(predicted)

    # Recall: fraction of known groups that have at least one predicted subset
    recalled = sum(
        1 for k in known_groups
        if any(p.issubset(k) or k.issubset(p) for p in predicted)
    )
    recall = recalled / len(known_groups)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "known_interactions": [sorted(list(g)) for g in known_groups],
        "discovered_simplices": [sorted(list(g)) for g in predicted],
    }


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════

def compute_statistical_tests(aggregate_results: list[dict]) -> dict:
    """Compute pairwise Wilcoxon signed-rank tests with Holm-Bonferroni."""
    modes = ["A_simplicial", "B_random_avg", "C_corr_cluster", "D_unconstrained", "E_mi_derived"]

    # Build per-dataset mean primary metric for each mode
    # Primary metric: auroc for classification, r2 for regression
    mode_scores = {m: {} for m in modes}
    for rec in aggregate_results:
        ds = rec["dataset"]
        mode = rec["mode"]
        metric_name = rec["metric_name"]
        # Use primary metric
        if metric_name in ("auroc", "r2"):
            mode_scores[mode][ds] = rec["mean_metric"]

    # For datasets that have both auroc and r2 in aggregation, prefer the right one
    # Actually our aggregate_results should have one primary metric per dataset/mode

    pairwise_results = []
    mode_pairs = list(combinations(modes, 2))

    for m1, m2 in mode_pairs:
        # Get common datasets
        common_ds = sorted(set(mode_scores[m1].keys()) & set(mode_scores[m2].keys()))
        if len(common_ds) < 6:
            logger.warning(f"Not enough datasets for Wilcoxon: {m1} vs {m2} ({len(common_ds)})")
            pairwise_results.append({
                "mode_1": m1,
                "mode_2": m2,
                "statistic": None,
                "p_value_raw": None,
                "p_value_holm": None,
                "cohens_d": None,
                "effect_size_label": "insufficient_data",
                "n_datasets": len(common_ds),
                "a_wins": 0,
                "b_wins": 0,
            })
            continue

        scores1 = np.array([mode_scores[m1][ds] for ds in common_ds])
        scores2 = np.array([mode_scores[m2][ds] for ds in common_ds])

        # Check if all differences are zero
        diffs = scores1 - scores2
        if np.all(diffs == 0):
            pairwise_results.append({
                "mode_1": m1,
                "mode_2": m2,
                "statistic": 0.0,
                "p_value_raw": 1.0,
                "p_value_holm": 1.0,
                "cohens_d": 0.0,
                "effect_size_label": "negligible",
                "n_datasets": len(common_ds),
                "a_wins": 0,
                "b_wins": 0,
            })
            continue

        try:
            stat, p_val = wilcoxon(scores1, scores2, alternative="two-sided")
        except ValueError:
            logger.warning(f"Wilcoxon failed for {m1} vs {m2}")
            stat, p_val = 0.0, 1.0

        # Cohen's d
        mean_diff = np.mean(scores1) - np.mean(scores2)
        pooled_std = np.sqrt((np.std(scores1, ddof=1)**2 + np.std(scores2, ddof=1)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            label = "negligible"
        elif abs_d < 0.5:
            label = "small"
        elif abs_d < 0.8:
            label = "medium"
        else:
            label = "large"

        a_wins = int(np.sum(scores1 > scores2))
        b_wins = int(np.sum(scores2 > scores1))

        pairwise_results.append({
            "mode_1": m1,
            "mode_2": m2,
            "statistic": round(float(stat), 6),
            "p_value_raw": round(float(p_val), 6),
            "p_value_holm": None,  # filled below
            "cohens_d": round(float(cohens_d), 6),
            "effect_size_label": label,
            "n_datasets": len(common_ds),
            "a_wins": a_wins,
            "b_wins": b_wins,
        })

    # Holm-Bonferroni correction
    valid = [(i, r) for i, r in enumerate(pairwise_results) if r["p_value_raw"] is not None]
    if valid:
        valid.sort(key=lambda x: x[1]["p_value_raw"])
        n_tests = len(valid)
        for rank, (idx, _) in enumerate(valid):
            raw_p = pairwise_results[idx]["p_value_raw"]
            adjusted = min(1.0, raw_p * (n_tests - rank))
            pairwise_results[idx]["p_value_holm"] = round(adjusted, 6)

    return {"pairwise_wilcoxon": pairwise_results}


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("XGBoost Simplicial-Constraint Ablation Experiment")
    logger.info("=" * 60)

    # Load all datasets
    all_data = load_all_datasets()

    per_fold_results = []
    interaction_recovery_results = []
    tda_diagnostics = []

    # Cache for simplicial groups per dataset/fold (needed for interaction recovery)
    simplicial_cache = {}

    for ds_name in DATASETS:
        if ds_name not in all_data:
            logger.warning(f"Dataset {ds_name} not found, skipping")
            continue

        ds = all_data[ds_name]
        X, y = ds["X"], ds["y"]
        folds, task_type = ds["folds"], ds["task_type"]
        n_features = ds["n_features"]
        known_interactions = ds["known_interactions"]

        logger.info(f"\n{'='*40}")
        logger.info(f"Dataset: {ds_name} | task={task_type} | n_features={n_features} | n_samples={len(y)}")

        for fold_idx in range(5):
            logger.info(f"  Fold {fold_idx}/4")

            train_mask = folds != fold_idx
            test_mask = folds == fold_idx
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # ── Mode A: Simplicial constraints ──
            try:
                t0 = time.time()
                simplicial_groups, diagnostics = build_simplicial_constraints(X_train, y_train)
                tda_build_time = time.time() - t0
                logger.info(
                    f"    Mode A: {len(simplicial_groups)} groups, "
                    f"threshold={diagnostics['persistence_threshold']:.4f}, "
                    f"build_time={tda_build_time:.2f}s"
                )

                diagnostics["dataset"] = ds_name
                diagnostics["fold"] = fold_idx
                tda_diagnostics.append(diagnostics)

                # Cache for interaction recovery
                simplicial_cache[(ds_name, fold_idx)] = simplicial_groups

                n_total_groups = len(simplicial_groups)

            except Exception:
                logger.exception(f"    Mode A FAILED for {ds_name} fold {fold_idx}")
                simplicial_groups = [[i] for i in range(n_features)]
                tda_build_time = 0.0
                n_total_groups = n_features
                simplicial_cache[(ds_name, fold_idx)] = simplicial_groups

            # ── Mode C: Correlation-clustering (match n_groups) ──
            try:
                corr_groups = build_corr_clustering_constraints(X_train, n_groups=n_total_groups)
            except Exception:
                logger.exception(f"    Mode C FAILED for {ds_name} fold {fold_idx}")
                corr_groups = [[i] for i in range(n_features)]

            # ── Mode E: MI-derived (match n_groups) ──
            try:
                mi_groups = build_mi_constraints(X_train, y_train, n_groups=n_total_groups, task_type=task_type)
            except Exception:
                logger.exception(f"    Mode E FAILED for {ds_name} fold {fold_idx}")
                mi_groups = [[i] for i in range(n_features)]

            # ── Evaluate all 5 modes ──
            modes_config = {
                "A_simplicial": simplicial_groups,
                "B_random_avg": "random",  # special handling
                "C_corr_cluster": corr_groups,
                "D_unconstrained": None,
                "E_mi_derived": mi_groups,
            }

            for mode_name, constraints in modes_config.items():
                try:
                    if mode_name == "B_random_avg":
                        # Average over 5 random seeds
                        seed_metrics = []
                        for seed in [42, 43, 44, 45, 46]:
                            random_groups = build_random_constraints(simplicial_groups, n_features, seed)
                            m = train_and_evaluate(
                                X_train, y_train, X_test, y_test,
                                task_type=task_type,
                                interaction_constraints=random_groups,
                            )
                            seed_metrics.append(m)
                        metrics = average_metrics(seed_metrics)
                    elif mode_name == "D_unconstrained":
                        metrics = train_and_evaluate(
                            X_train, y_train, X_test, y_test,
                            task_type=task_type,
                            interaction_constraints=None,
                        )
                    else:
                        metrics = train_and_evaluate(
                            X_train, y_train, X_test, y_test,
                            task_type=task_type,
                            interaction_constraints=constraints,
                        )

                    record = {
                        "dataset": ds_name,
                        "fold": fold_idx,
                        "mode": mode_name,
                        "task_type": task_type,
                        **metrics,
                        "tda_build_time_sec": round(tda_build_time, 4) if mode_name == "A_simplicial" else None,
                        "n_constraint_groups": n_total_groups,
                        "constraint_group_sizes": [len(g) for g in simplicial_groups],
                    }
                    per_fold_results.append(record)

                    # Log primary metric
                    if task_type == "classification":
                        logger.info(f"    {mode_name}: acc={metrics.get('accuracy')}, auroc={metrics.get('auroc')}, t={metrics.get('train_time_sec')}s")
                    else:
                        logger.info(f"    {mode_name}: R²={metrics.get('r2')}, rmse={metrics.get('rmse')}, t={metrics.get('train_time_sec')}s")

                except Exception:
                    logger.exception(f"    {mode_name} FAILED for {ds_name} fold {fold_idx}")
                    # Record NaN
                    per_fold_results.append({
                        "dataset": ds_name,
                        "fold": fold_idx,
                        "mode": mode_name,
                        "task_type": task_type,
                        "accuracy": None, "auroc": None, "r2": None, "rmse": None,
                        "train_time_sec": None,
                        "tda_build_time_sec": None,
                        "n_constraint_groups": n_total_groups,
                        "constraint_group_sizes": [len(g) for g in simplicial_groups],
                    })

    # ── Interaction recovery (synthetic datasets) ──
    logger.info("\n" + "=" * 40)
    logger.info("Interaction Recovery (synthetic datasets)")
    for ds_name in SYNTHETIC_DATASETS:
        if ds_name not in all_data:
            continue
        ki = all_data[ds_name]["known_interactions"]
        if ki is None:
            continue

        for fold_idx in range(5):
            s_groups = simplicial_cache.get((ds_name, fold_idx))
            if s_groups is None:
                continue

            recovery = compute_interaction_recovery(s_groups, ki)
            recovery["dataset"] = ds_name
            recovery["fold"] = fold_idx
            interaction_recovery_results.append(recovery)
            logger.info(
                f"  {ds_name} fold {fold_idx}: P={recovery['precision']}, "
                f"R={recovery['recall']}, F1={recovery['f1']}"
            )

    # ── Aggregate results ──
    logger.info("\n" + "=" * 40)
    logger.info("Computing aggregate results")

    aggregate_results = []
    for ds_name in DATASETS:
        ds_records = [r for r in per_fold_results if r["dataset"] == ds_name]
        if not ds_records:
            continue
        task_type = ds_records[0]["task_type"]
        primary_metric = "auroc" if task_type == "classification" else "r2"

        for mode in ["A_simplicial", "B_random_avg", "C_corr_cluster", "D_unconstrained", "E_mi_derived"]:
            mode_records = [r for r in ds_records if r["mode"] == mode]
            metric_vals = [r[primary_metric] for r in mode_records if r[primary_metric] is not None]
            time_vals = [r["train_time_sec"] for r in mode_records if r.get("train_time_sec") is not None]

            if metric_vals:
                mean_m = round(float(np.mean(metric_vals)), 6)
                std_m = round(float(np.std(metric_vals)), 6)
            else:
                mean_m, std_m = None, None

            mean_time = round(float(np.mean(time_vals)), 4) if time_vals else None

            aggregate_results.append({
                "dataset": ds_name,
                "mode": mode,
                "mean_metric": mean_m,
                "std_metric": std_m,
                "metric_name": primary_metric,
                "mean_train_time": mean_time,
            })

    # ── Statistical tests ──
    logger.info("Computing statistical tests")
    stat_tests = compute_statistical_tests(aggregate_results)

    # ── Build output ──
    output = {
        "experiment_id": "experiment_iter4_dir1",
        "hypothesis": "Simplicial-constrained XGBoost vs baselines ablation",
        "config": {
            "xgboost_params": XGBOOST_PARAMS,
            "modes": ["A_simplicial", "B_random_avg", "C_corr_cluster", "D_unconstrained", "E_mi_derived"],
            "n_folds": 5,
            "random_seeds_mode_B": [42, 43, 44, 45, 46],
            "tda_params": {
                "dissimilarity": "sqrt(1 - dCor)",
                "max_simplex_dim": 3,
                "threshold_method": "largest_gap",
            },
        },
        "per_fold_results": per_fold_results,
        "interaction_recovery": interaction_recovery_results,
        "aggregate_results": aggregate_results,
        "statistical_tests": stat_tests,
        "tda_diagnostics": tda_diagnostics,
    }

    # ── Write in exp_gen_sol_out.json schema ──
    # Convert to schema format: datasets → [{dataset, examples: [{input, output, metadata_*}]}]
    schema_output = build_schema_output(output)
    OUTPUT_FILE.write_text(json.dumps(schema_output, indent=2))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Written: {OUTPUT_FILE.name} ({size_mb:.2f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal experiment time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Mean primary metric by dataset × mode")
    logger.info(f"{'Dataset':<25} {'A_simpl':>8} {'B_rand':>8} {'C_corr':>8} {'D_uncon':>8} {'E_mi':>8}")
    for ds_name in DATASETS:
        row = [ds_name]
        for mode in ["A_simplicial", "B_random_avg", "C_corr_cluster", "D_unconstrained", "E_mi_derived"]:
            rec = [r for r in aggregate_results if r["dataset"] == ds_name and r["mode"] == mode]
            if rec and rec[0]["mean_metric"] is not None:
                row.append(f"{rec[0]['mean_metric']:.4f}")
            else:
                row.append("N/A")
        logger.info(f"{row[0]:<25} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8}")


def build_schema_output(raw_output: dict) -> dict:
    """Convert raw experiment output to exp_gen_sol_out.json schema format.

    Schema requires: {metadata?, datasets: [{dataset, examples: [{input, output, ...metadata_*, predict_*}]}]}
    """
    metadata = {
        "experiment_id": raw_output["experiment_id"],
        "hypothesis": raw_output["hypothesis"],
        "config": raw_output["config"],
        "statistical_tests": raw_output["statistical_tests"],
        "tda_diagnostics": raw_output["tda_diagnostics"],
    }

    # Group per_fold_results by dataset
    ds_groups = {}
    for rec in raw_output["per_fold_results"]:
        ds_name = rec["dataset"]
        ds_groups.setdefault(ds_name, []).append(rec)

    # Also group interaction_recovery by dataset
    ir_groups = {}
    for rec in raw_output["interaction_recovery"]:
        ds_name = rec["dataset"]
        ir_groups.setdefault(ds_name, []).append(rec)

    # Also group aggregate_results by dataset
    agg_groups = {}
    for rec in raw_output["aggregate_results"]:
        ds_name = rec["dataset"]
        agg_groups.setdefault(ds_name, []).append(rec)

    datasets = []
    for ds_name in DATASETS:
        if ds_name not in ds_groups:
            continue

        examples = []

        # Per-fold results as examples
        fold_records = ds_groups[ds_name]
        for rec in fold_records:
            # Input: description of the experiment configuration
            input_str = json.dumps({
                "dataset": rec["dataset"],
                "fold": rec["fold"],
                "mode": rec["mode"],
                "task_type": rec["task_type"],
            })

            # Output: primary metric value
            if rec["task_type"] == "classification":
                primary = rec.get("auroc") or rec.get("accuracy") or "N/A"
            else:
                primary = rec.get("r2") or "N/A"
            output_str = str(primary)

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_fold": rec["fold"],
                "metadata_mode": rec["mode"],
                "metadata_task_type": rec["task_type"],
                "metadata_accuracy": str(rec.get("accuracy", "")),
                "metadata_auroc": str(rec.get("auroc", "")),
                "metadata_r2": str(rec.get("r2", "")),
                "metadata_rmse": str(rec.get("rmse", "")),
                "metadata_train_time_sec": str(rec.get("train_time_sec", "")),
                "metadata_tda_build_time_sec": str(rec.get("tda_build_time_sec", "")),
                "metadata_n_constraint_groups": str(rec.get("n_constraint_groups", "")),
            }

            # Add predict fields for each mode
            mode = rec["mode"]
            if rec["task_type"] == "classification":
                example[f"predict_{mode}"] = str(rec.get("auroc", ""))
            else:
                example[f"predict_{mode}"] = str(rec.get("r2", ""))

            examples.append(example)

        # Add interaction recovery as additional examples
        if ds_name in ir_groups:
            for ir_rec in ir_groups[ds_name]:
                input_str = json.dumps({
                    "dataset": ir_rec["dataset"],
                    "fold": ir_rec["fold"],
                    "type": "interaction_recovery",
                })
                output_str = str(ir_rec.get("f1", "N/A"))
                example = {
                    "input": input_str,
                    "output": output_str,
                    "metadata_fold": ir_rec["fold"],
                    "metadata_type": "interaction_recovery",
                    "metadata_precision": str(ir_rec.get("precision", "")),
                    "metadata_recall": str(ir_rec.get("recall", "")),
                    "metadata_f1": str(ir_rec.get("f1", "")),
                    "metadata_known_interactions": json.dumps(ir_rec.get("known_interactions", [])),
                    "metadata_discovered_simplices": json.dumps(ir_rec.get("discovered_simplices", [])),
                    "predict_interaction_recovery": str(ir_rec.get("f1", "")),
                }
                examples.append(example)

        # Add aggregate results as additional examples
        if ds_name in agg_groups:
            for agg_rec in agg_groups[ds_name]:
                input_str = json.dumps({
                    "dataset": agg_rec["dataset"],
                    "mode": agg_rec["mode"],
                    "type": "aggregate",
                })
                output_str = str(agg_rec.get("mean_metric", "N/A"))
                example = {
                    "input": input_str,
                    "output": output_str,
                    "metadata_type": "aggregate",
                    "metadata_mode": agg_rec["mode"],
                    "metadata_mean_metric": str(agg_rec.get("mean_metric", "")),
                    "metadata_std_metric": str(agg_rec.get("std_metric", "")),
                    "metadata_metric_name": str(agg_rec.get("metric_name", "")),
                    "metadata_mean_train_time": str(agg_rec.get("mean_train_time", "")),
                    "predict_aggregate": str(agg_rec.get("mean_metric", "")),
                }
                examples.append(example)

        datasets.append({"dataset": ds_name, "examples": examples})

    return {"metadata": metadata, "datasets": datasets}


if __name__ == "__main__":
    main()
