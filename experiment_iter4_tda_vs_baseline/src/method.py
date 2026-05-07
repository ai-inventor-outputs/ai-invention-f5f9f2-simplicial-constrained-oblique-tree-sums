#!/usr/bin/env python3
"""
Benchmark 5 interaction detection methods across 10 tabular datasets.
Methods: TDA (persistent homology), ANOVA, MI-screening, Tree-based, Correlation.
Output: method_out.json (exp_gen_sol_out schema).

Key optimization: Expensive per-fold computations are cached, then thresholds
are swept cheaply against those cached results.
"""

import json
import os
import resource
import signal
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import dcor
import gudhi
import networkx as nx
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Constants ────────────────────────────────────────────────────────────────
WS = Path(__file__).resolve().parent
DATA_DEP_DIR = Path(
    "/home/adrian/projects/temp/ai-inventor-old3/aii_pipeline/runs/"
    "run__20260228_133939/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
MAX_SUBSAMPLE = 2000
CLIQUE_TIMEOUT = 60
TOP_FEATURES_CAP = 15

# Threshold sweeps
TDA_THRESHOLDS = np.linspace(0.05, 0.95, 20).tolist()
ANOVA_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
MI_THRESHOLDS = [1.01, 1.05, 1.1, 1.2, 1.5, 2.0]
TREE_THRESHOLDS = np.linspace(0.01, 0.5, 20).tolist()
CORR_THRESHOLDS = np.linspace(0.1, 0.9, 20).tolist()

RUN_MODE = os.environ.get("RUN_MODE", "full")


def _to_py(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, frozenset):
        return frozenset(int(v) for v in obj)
    return obj


def _fs_to_pyint(fs: FrozenSet) -> FrozenSet[int]:
    """Convert a frozenset of numpy ints to Python ints."""
    return frozenset(int(v) for v in fs)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load and parse the benchmark data file."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets_raw = raw["datasets"]
    logger.info(f"Found {len(datasets_raw)} datasets")
    return datasets_raw


def parse_dataset(ds_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single dataset into X, y, folds, ground truth, etc."""
    name = ds_raw["dataset"]
    examples = ds_raw["examples"]

    if len(examples) == 0:
        raise ValueError(f"Dataset {name} has 0 examples")

    ex0 = examples[0]
    task_type = ex0.get("metadata_task_type", "regression")
    feature_names = ex0.get("metadata_feature_names", [])

    X_list, y_list, folds = [], [], []
    for ex in examples:
        inp = json.loads(ex["input"])
        X_list.append([float(v) for v in inp])
        y_list.append(float(ex["output"]))
        folds.append(int(ex["metadata_fold"]))

    X = np.array(X_list)
    y = np.array(y_list)
    fold_arr = np.array(folds)

    # Parse ground truth interactions
    gt_interactions: Optional[Set[FrozenSet[int]]] = None
    ki_str = ex0.get("metadata_known_interactions", None)
    if ki_str:
        ki = json.loads(ki_str) if isinstance(ki_str, str) else ki_str
        gt_set: Set[FrozenSet[int]] = set()
        for key in ["2-way", "3-way", "4-way"]:
            if key in ki:
                for group in ki[key]:
                    gt_set.add(frozenset(group))
        if gt_set:
            gt_interactions = gt_set

    logger.info(
        f"  Dataset '{name}': X={X.shape}, task={task_type}, "
        f"gt={gt_interactions}"
    )

    return {
        "name": name,
        "X": X,
        "y": y,
        "folds": fold_arr,
        "task_type": task_type,
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "gt_interactions": gt_interactions,
    }


# ── Helper: top features by F-test ──────────────────────────────────────────

def _top_features_by_f(
    X: np.ndarray, y: np.ndarray, task_type: str, top_k: int = TOP_FEATURES_CAP
) -> List[int]:
    from sklearn.feature_selection import f_classif, f_regression
    if task_type == "classification":
        f_vals, _ = f_classif(X, y)
    else:
        f_vals, _ = f_regression(X, y)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    top_idx = np.argsort(f_vals)[::-1][:top_k]
    return sorted(top_idx.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# Method 1: TDA  –  distance correlation → Vietoris-Rips persistent homology
# Expensive part: compute dcor matrix ONCE per fold
# Cheap part: sweep thresholds against the simplex tree
# ══════════════════════════════════════════════════════════════════════════════

TDA_MAX_FEATURES = 20  # cap features for TDA dcor computation


def tda_precompute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "regression",
    max_dim: int = 3,
) -> Tuple[gudhi.SimplexTree, Optional[List[int]]]:
    """Build GUDHI simplex tree from interaction-aware distance correlation matrix.

    Uses two complementary signals:
    1. Pairwise dcor between features (captures feature similarity)
    2. Product-target dcor: dcor(xi*xj, y) for each pair (captures interaction strength)

    Returns (simplex_tree, feature_index_map) where feature_index_map is None
    if no feature subsetting was done, or a list mapping local indices back to
    original feature indices.
    """
    n, p_orig = X_train.shape

    # Cap features for high-dimensional datasets
    feat_map: Optional[List[int]] = None
    if p_orig > TDA_MAX_FEATURES:
        feat_idx = _top_features_by_f(X_train, y_train, task_type, TDA_MAX_FEATURES)
        X_use = X_train[:, feat_idx]
        feat_map = feat_idx
        logger.debug(f"    TDA: capped to top {TDA_MAX_FEATURES} features")
    else:
        X_use = X_train

    n, p = X_use.shape

    # Subsample for dcor
    if n > MAX_SUBSAMPLE:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, MAX_SUBSAMPLE, replace=False)
        X_sub, y_sub = X_use[idx], y_train[idx]
    else:
        X_sub, y_sub = X_use, y_train

    # Feature-target dcor
    dcor_target = np.zeros(p)
    for i in range(p):
        try:
            dcor_target[i] = dcor.distance_correlation(X_sub[:, i], y_sub)
        except Exception:
            dcor_target[i] = 0.0

    # Pairwise dcor + product-target interaction dcor
    D_pairwise = np.eye(p)
    D_interact = np.zeros((p, p))

    for i in range(p):
        for j in range(i + 1, p):
            try:
                val = dcor.distance_correlation(X_sub[:, i], X_sub[:, j])
            except Exception:
                val = 0.0
            D_pairwise[i, j] = val
            D_pairwise[j, i] = val

            # Interaction signal: how much does the product xi*xj correlate with y?
            try:
                product = X_sub[:, i] * X_sub[:, j]
                interact_val = dcor.distance_correlation(product, y_sub)
            except Exception:
                interact_val = 0.0
            D_interact[i, j] = interact_val
            D_interact[j, i] = interact_val

    # Combined metric: max of (pairwise dcor weighted by target relevance)
    # and (product-target interaction dcor)
    D_enhanced = np.zeros((p, p))
    for i in range(p):
        D_enhanced[i, i] = 1.0
        for j in range(i + 1, p):
            val_orig = D_pairwise[i, j] * min(dcor_target[i], dcor_target[j])
            val_interact = D_interact[i, j]
            combined = max(val_orig, val_interact)
            D_enhanced[i, j] = combined
            D_enhanced[j, i] = combined

    dist_matrix = 1.0 - D_enhanced
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.maximum(dist_matrix, 0.0)

    rips = gudhi.RipsComplex(distance_matrix=dist_matrix.tolist(), max_edge_length=1.0)
    st = rips.create_simplex_tree(max_dimension=1)
    expand_dim = min(max_dim, p - 1)
    if expand_dim > 1:
        st.expansion(expand_dim)
    st.compute_persistence(min_persistence=-1.0)

    return st, feat_map


def tda_at_threshold(
    st: gudhi.SimplexTree,
    threshold: float,
    feat_map: Optional[List[int]] = None,
) -> Tuple[List[FrozenSet[int]], Dict[str, Any]]:
    """Extract interactions from precomputed simplex tree at a given threshold.

    If feat_map is provided, maps local indices back to original feature indices.
    """
    betti = st.betti_numbers()

    # Extract simplices at threshold (convert numpy ints to Python ints)
    detected = []
    for simplex, filt in st.get_filtration():
        if filt <= threshold and len(simplex) >= 2:
            if feat_map is not None:
                mapped = frozenset(feat_map[int(v)] for v in simplex)
            else:
                mapped = frozenset(int(v) for v in simplex)
            detected.append(mapped)

    # Keep only maximal simplices
    maximal = []
    for s in sorted(detected, key=len, reverse=True):
        if not any(s < m for m in maximal):
            maximal.append(s)

    topo = {
        "betti_0": int(betti[0]) if len(betti) > 0 else 0,
        "betti_1": int(betti[1]) if len(betti) > 1 else 0,
    }
    return maximal, topo


# ══════════════════════════════════════════════════════════════════════════════
# Method 2: ANOVA  –  F-test for interaction terms
# Expensive part: compute p-values for all pairs/triples ONCE per fold
# Cheap part: sweep p-value thresholds
# ══════════════════════════════════════════════════════════════════════════════

def anova_precompute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "regression",
) -> Dict[FrozenSet[int], float]:
    """Compute p-values for all 2-way (and feasible 3-way) interaction terms."""
    n, p = X_train.shape
    pval_map: Dict[FrozenSet[int], float] = {}

    if p > 30:
        feat_idx = _top_features_by_f(X_train, y_train, task_type, TOP_FEATURES_CAP)
    else:
        feat_idx = list(range(p))

    do_3way = len(feat_idx) <= 20
    n_2way = len(list(combinations(feat_idx, 2)))
    n_3way = len(list(combinations(feat_idx, 3))) if do_3way else 0
    n_tests = max(n_2way + n_3way, 1)

    # 2-way
    for i, j in combinations(feat_idx, 2):
        try:
            xi, xj = X_train[:, i], X_train[:, j]
            design = np.column_stack([np.ones(n), xi, xj, xi * xj])
            model = sm.OLS(y_train, design).fit()
            if len(model.pvalues) >= 4:
                raw_p = model.pvalues[3]
                pval_map[frozenset({i, j})] = raw_p * n_tests  # Bonferroni
        except Exception:
            continue

    # 3-way
    if do_3way:
        for i, j, k in combinations(feat_idx, 3):
            try:
                xi, xj, xk = X_train[:, i], X_train[:, j], X_train[:, k]
                design = np.column_stack([
                    np.ones(n), xi, xj, xk,
                    xi * xj, xi * xk, xj * xk,
                    xi * xj * xk,
                ])
                model = sm.OLS(y_train, design).fit()
                if len(model.pvalues) >= 8:
                    raw_p = model.pvalues[7]
                    pval_map[frozenset({i, j, k})] = raw_p * n_tests
            except Exception:
                continue

    return pval_map


def anova_at_threshold(
    pval_map: Dict[FrozenSet[int], float],
    p_threshold: float,
) -> List[FrozenSet[int]]:
    """Filter interactions by p-value threshold."""
    return [s for s, p in pval_map.items() if p < p_threshold]


# ══════════════════════════════════════════════════════════════════════════════
# Method 3: MI-screening  –  mutual information synergy detection
# Expensive part: compute MI values ONCE per fold
# Cheap part: sweep synergy ratios
# ══════════════════════════════════════════════════════════════════════════════

def mi_precompute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "regression",
) -> Dict[FrozenSet[int], float]:
    """Compute synergy ratios for all pairs/triples. Returns {set: ratio}."""
    n, p = X_train.shape
    synergy_map: Dict[FrozenSet[int], float] = {}

    mi_func = mutual_info_regression if task_type == "regression" else mutual_info_classif

    # Individual MI
    mi_individual = np.zeros(p)
    for i in range(p):
        try:
            mi_individual[i] = mi_func(
                X_train[:, [i]], y_train, n_neighbors=5, random_state=42
            )[0]
        except Exception:
            mi_individual[i] = 0.0

    if p > 30:
        feat_idx = _top_features_by_f(X_train, y_train, task_type, TOP_FEATURES_CAP)
    else:
        feat_idx = list(range(p))

    do_3way = len(feat_idx) <= 20

    # Subsample for speed: MI with n_neighbors=5 needs at least ~20 samples
    # but is slow on large n. Subsample to 1000 for speed.
    if n > 1000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 1000, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    # 2-way
    for i, j in combinations(feat_idx, 2):
        try:
            mi_joint = mi_func(
                X_sub[:, [i, j]], y_sub, n_neighbors=5, random_state=42
            )[0]
            sum_marginal = mi_individual[i] + mi_individual[j]
            if sum_marginal > 1e-10:
                synergy_map[frozenset({i, j})] = mi_joint / sum_marginal
            else:
                synergy_map[frozenset({i, j})] = 0.0
        except Exception:
            continue

    # 3-way
    if do_3way:
        for i, j, k in combinations(feat_idx, 3):
            try:
                mi_joint = mi_func(
                    X_sub[:, [i, j, k]], y_sub, n_neighbors=5, random_state=42
                )[0]
                sum_marginal = mi_individual[i] + mi_individual[j] + mi_individual[k]
                if sum_marginal > 1e-10:
                    synergy_map[frozenset({i, j, k})] = mi_joint / sum_marginal
                else:
                    synergy_map[frozenset({i, j, k})] = 0.0
            except Exception:
                continue

    return synergy_map


def mi_at_threshold(
    synergy_map: Dict[FrozenSet[int], float],
    synergy_ratio: float,
) -> List[FrozenSet[int]]:
    """Filter interactions by synergy ratio threshold."""
    return [s for s, r in synergy_map.items() if r > synergy_ratio]


# ══════════════════════════════════════════════════════════════════════════════
# Method 4: Tree-based  –  RF split co-occurrence
# Expensive part: fit RF and count co-occurrences ONCE per fold
# Cheap part: sweep frequency thresholds
# ══════════════════════════════════════════════════════════════════════════════

def tree_precompute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "regression",
) -> Dict[FrozenSet[int], float]:
    """Fit RF, extract co-occurrence frequencies. Returns {set: frequency}."""
    from sklearn.tree import _tree
    TREE_LEAF = _tree.TREE_LEAF

    n, p = X_train.shape
    n_estimators = min(500, max(100, n // 5))

    if task_type == "classification":
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=8, random_state=42, n_jobs=1
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=8, random_state=42, n_jobs=1
        )

    rf.fit(X_train, y_train)

    pair_counts: Dict[FrozenSet[int], int] = defaultdict(int)
    triple_counts: Dict[FrozenSet[int], int] = defaultdict(int)
    total_paths = 0
    do_3way = p <= 20

    for estimator in rf.estimators_:
        tree = estimator.tree_

        # Iterative path extraction (avoid deep recursion)
        stack = [(0, [])]
        while stack:
            node_id, current_path = stack.pop()
            if tree.children_left[node_id] == TREE_LEAF:
                total_paths += 1
                unique_feats = set(current_path)
                for pair in combinations(unique_feats, 2):
                    pair_counts[frozenset(pair)] += 1
                if do_3way:
                    for triple in combinations(unique_feats, 3):
                        triple_counts[frozenset(triple)] += 1
                continue
            feat = tree.feature[node_id]
            new_path = current_path + [feat]
            stack.append((tree.children_left[node_id], new_path))
            stack.append((tree.children_right[node_id], new_path))

    if total_paths == 0:
        return {}

    freq_map: Dict[FrozenSet[int], float] = {}
    for s, count in pair_counts.items():
        freq_map[s] = count / total_paths
    for s, count in triple_counts.items():
        freq_map[s] = count / total_paths

    return freq_map


def tree_at_threshold(
    freq_map: Dict[FrozenSet[int], float],
    freq_threshold: float,
) -> List[FrozenSet[int]]:
    """Filter interactions by frequency threshold."""
    return [s for s, f in freq_map.items() if f >= freq_threshold]


# ══════════════════════════════════════════════════════════════════════════════
# Method 5: Correlation thresholding  –  Pearson → maximal cliques
# Expensive part: compute correlation matrix ONCE per fold
# Cheap part: sweep thresholds, find cliques
# ══════════════════════════════════════════════════════════════════════════════

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Clique enumeration timed out")


def corr_precompute(X_train: np.ndarray) -> np.ndarray:
    """Compute absolute Pearson correlation matrix."""
    C = np.abs(np.corrcoef(X_train.T))
    np.fill_diagonal(C, 0.0)
    return C


def corr_at_threshold(
    C: np.ndarray,
    corr_threshold: float,
) -> List[FrozenSet[int]]:
    """Find maximal cliques at a given correlation threshold."""
    p = C.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(i + 1, p):
            if C[i, j] >= corr_threshold:
                G.add_edge(i, j)

    detected = []
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(CLIQUE_TIMEOUT)
        try:
            count = 0
            for clique in nx.find_cliques(G):
                if len(clique) >= 2:
                    detected.append(frozenset(clique))
                count += 1
                if count >= 1000:
                    logger.warning("Clique enumeration capped at 1000")
                    break
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except _TimeoutError:
        logger.warning(f"Clique enumeration timed out at threshold={corr_threshold}")

    return detected


# ── Evaluation Metrics ───────────────────────────────────────────────────────

def compute_metrics(
    detected: Set[FrozenSet[int]],
    gt: Set[FrozenSet[int]],
) -> Dict[str, float]:
    """Compute exact and superset match P/R/F1."""
    tp = len(gt & detected)
    prec = tp / len(detected) if detected else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    tp_sup = sum(1 for s in detected if any(g <= s for g in gt))
    prec_sup = tp_sup / len(detected) if detected else 0.0
    rec_sup_n = sum(1 for g in gt if any(g <= s for s in detected))
    rec_sup = rec_sup_n / len(gt) if gt else 0.0
    f1_sup = 2 * prec_sup * rec_sup / (prec_sup + rec_sup) if (prec_sup + rec_sup) > 0 else 0.0

    return {
        "precision_exact": round(prec, 6),
        "recall_exact": round(rec, 6),
        "f1_exact": round(f1, 6),
        "precision_superset": round(prec_sup, 6),
        "recall_superset": round(rec_sup, 6),
        "f1_superset": round(f1_sup, 6),
    }


def compute_jaccard(fold_det: Dict[int, Set[FrozenSet[int]]]) -> float:
    """Mean pairwise Jaccard stability across folds."""
    keys = sorted(fold_det.keys())
    if len(keys) < 2:
        return 1.0
    scores = []
    for f1, f2 in combinations(keys, 2):
        s1, s2 = fold_det[f1], fold_det[f2]
        u = s1 | s2
        scores.append(len(s1 & s2) / len(u) if u else 1.0)
    return float(np.mean(scores))


# ── Main Benchmark Loop ─────────────────────────────────────────────────────

METHOD_THRESHOLDS = {
    "TDA": TDA_THRESHOLDS,
    "ANOVA": ANOVA_THRESHOLDS,
    "MI_screening": MI_THRESHOLDS,
    "Tree_based": TREE_THRESHOLDS,
    "Correlation": CORR_THRESHOLDS,
}


@logger.catch
def main():
    t_global_start = time.perf_counter()

    # ── Load data ────────────────────────────────────────────────────────
    if RUN_MODE == "mini":
        data_path = WS / "mini_data_out.json"
    else:
        data_path = DATA_DEP_DIR / "full_data_out.json"

    datasets_raw = load_data(data_path)

    if RUN_MODE == "single":
        datasets_raw = [d for d in datasets_raw if d["dataset"] == "friedman1"]
    elif RUN_MODE == "synthetic":
        synth = {"friedman1", "friedman3", "synth_3way", "synth_4way"}
        datasets_raw = [d for d in datasets_raw if d["dataset"] in synth]

    datasets = []
    for ds_raw in datasets_raw:
        try:
            datasets.append(parse_dataset(ds_raw))
        except Exception:
            logger.exception(f"Failed to parse dataset {ds_raw['dataset']}")

    logger.info(f"Parsed {len(datasets)} datasets")

    # ── Results accumulator ──────────────────────────────────────────────
    all_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ds in datasets:
        name = ds["name"]
        X, y = ds["X"], ds["y"]
        fold_arr = ds["folds"]
        task_type = ds["task_type"]
        gt = ds["gt_interactions"]
        p = ds["n_features"]
        unique_folds = sorted(set(fold_arr.tolist()))

        logger.info(
            f"=== {name} ({X.shape[0]} samples, {p} features, "
            f"{len(unique_folds)} folds) ==="
        )

        for method_name in METHOD_THRESHOLDS:
            thresholds = METHOD_THRESHOLDS[method_name]
            logger.info(
                f"  {method_name}: {len(thresholds)} thresholds × {len(unique_folds)} folds"
            )

            fold_detected_per_thresh: Dict[float, Dict[int, Set[FrozenSet[int]]]] = defaultdict(dict)

            for fold in unique_folds:
                mask_train = fold_arr != fold
                X_train = X[mask_train]
                y_train = y[mask_train]

                if len(X_train) < 10:
                    logger.warning(f"    Fold {fold}: only {len(X_train)} training samples, skipping")
                    continue

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)

                # ── PRECOMPUTE (expensive, once per fold) ────────────
                t_pre = time.perf_counter()
                try:
                    if method_name == "TDA":
                        cache = tda_precompute(X_scaled, y_train, task_type=task_type, max_dim=3)
                    elif method_name == "ANOVA":
                        cache = anova_precompute(X_scaled, y_train, task_type=task_type)
                    elif method_name == "MI_screening":
                        cache = mi_precompute(X_scaled, y_train, task_type=task_type)
                    elif method_name == "Tree_based":
                        cache = tree_precompute(X_scaled, y_train, task_type=task_type)
                    elif method_name == "Correlation":
                        cache = corr_precompute(X_scaled)
                    else:
                        raise ValueError(f"Unknown method: {method_name}")
                except Exception:
                    logger.exception(f"    Precompute failed: {method_name} fold={fold}")
                    continue

                precompute_time = time.perf_counter() - t_pre
                logger.debug(f"    {method_name} fold={fold} precompute: {precompute_time:.2f}s")

                # ── THRESHOLD SWEEP (cheap) ──────────────────────────
                for threshold in thresholds:
                    t0 = time.perf_counter()
                    try:
                        if method_name == "TDA":
                            st, feat_map = cache
                            detected, topo = tda_at_threshold(st, threshold, feat_map)
                        elif method_name == "ANOVA":
                            detected = anova_at_threshold(cache, threshold)
                            topo = {}
                        elif method_name == "MI_screening":
                            detected = mi_at_threshold(cache, threshold)
                            topo = {}
                        elif method_name == "Tree_based":
                            detected = tree_at_threshold(cache, threshold)
                            topo = {}
                        elif method_name == "Correlation":
                            detected = corr_at_threshold(cache, threshold)
                            topo = {}
                        else:
                            detected, topo = [], {}
                        wall_time = time.perf_counter() - t0
                    except Exception:
                        logger.exception(
                            f"    Failed: {method_name} {name} fold={fold} thresh={threshold:.4f}"
                        )
                        detected, topo, wall_time = [], {}, time.perf_counter() - t0

                    # Include precompute time amortized over thresholds
                    total_time = wall_time + precompute_time / len(thresholds)

                    # Ensure all frozenset elements are native Python ints
                    detected = [_fs_to_pyint(s) for s in detected]

                    result: Dict[str, Any] = {
                        "method": method_name,
                        "threshold": round(float(threshold), 6),
                        "fold": int(fold),
                        "detected_interactions": [sorted(list(s)) for s in detected],
                        "num_detected": len(detected),
                        "wall_clock_seconds": round(float(total_time), 4),
                    }

                    if gt is not None:
                        metrics = compute_metrics(set(detected), gt)
                        result.update(metrics)

                    if method_name == "TDA" and topo:
                        result["betti_0"] = topo.get("betti_0", 0)
                        result["betti_1"] = topo.get("betti_1", 0)

                    all_results[name].append(result)
                    fold_detected_per_thresh[threshold][fold] = set(detected)

            # Compute Jaccard per threshold
            for threshold in thresholds:
                fd = fold_detected_per_thresh.get(threshold, {})
                if fd:
                    jac = compute_jaccard(fd)
                    for r in all_results[name]:
                        if r["method"] == method_name and abs(r["threshold"] - threshold) < 1e-9:
                            r["jaccard_stability"] = round(jac, 6)

        elapsed_ds = time.perf_counter() - t_global_start
        logger.info(f"  Cumulative time: {elapsed_ds:.1f}s")

    # ── Summary Metrics ──────────────────────────────────────────────────
    logger.info("Computing summary metrics...")
    synth_names = {"friedman1", "friedman3", "synth_3way", "synth_4way"}
    best_thresholds: Dict[str, float] = {}
    best_f1: Dict[str, float] = {}
    mean_times: Dict[str, float] = {}
    mean_jaccards: Dict[str, float] = {}

    for method_name in METHOD_THRESHOLDS:
        thresh_f1: Dict[float, List[float]] = defaultdict(list)
        all_times: List[float] = []
        all_jac: List[float] = []

        for ds_name, results in all_results.items():
            for r in results:
                if r["method"] == method_name:
                    all_times.append(r["wall_clock_seconds"])
                    if "jaccard_stability" in r:
                        all_jac.append(r["jaccard_stability"])
                    if ds_name in synth_names and "f1_superset" in r:
                        thresh_f1[r["threshold"]].append(r["f1_superset"])

        if thresh_f1:
            bt = max(thresh_f1, key=lambda t: np.mean(thresh_f1[t]))
            best_thresholds[method_name] = round(bt, 6)
            best_f1[method_name] = round(float(np.mean(thresh_f1[bt])), 6)
        else:
            best_thresholds[method_name] = 0.0
            best_f1[method_name] = 0.0

        mean_times[method_name] = round(float(np.mean(all_times)) if all_times else 0.0, 4)
        mean_jaccards[method_name] = round(float(np.mean(all_jac)) if all_jac else 0.0, 6)

    logger.info(f"Best F1: {best_f1}")
    logger.info(f"Best thresholds: {best_thresholds}")
    logger.info(f"Mean times: {mean_times}")

    # ── Build Output JSON ────────────────────────────────────────────────
    output_datasets = []
    total_examples = 0

    for ds_raw in datasets_raw:
        ds_name = ds_raw["dataset"]
        results = all_results.get(ds_name, [])

        examples = []
        for r in results:
            inp = json.dumps({"method": r["method"], "threshold": r["threshold"], "fold": r["fold"]})

            out_d: Dict[str, Any] = {
                "detected": r["detected_interactions"],
                "num_detected": r["num_detected"],
                "wall_clock_seconds": r["wall_clock_seconds"],
            }
            for k in ["f1_exact", "f1_superset", "precision_exact", "recall_exact",
                       "precision_superset", "recall_superset", "jaccard_stability",
                       "betti_0", "betti_1"]:
                if k in r:
                    out_d[k] = r[k]

            # predict_ field key from method name (lowercase, schema pattern)
            predict_key = f"predict_{r['method'].lower()}"
            examples.append({
                "input": inp,
                "output": json.dumps(_to_py(out_d)),
                predict_key: json.dumps(_to_py(out_d)),
                "metadata_method": r["method"],
                "metadata_fold": int(r["fold"]),
                "metadata_threshold": float(r["threshold"]),
            })

        output_datasets.append({"dataset": ds_name, "examples": examples})
        total_examples += len(examples)

    output = {
        "metadata": {
            "experiment": "interaction_detection_benchmark",
            "methods": ["TDA", "ANOVA", "MI_screening", "Tree_based", "Correlation"],
            "threshold_ranges": {
                "TDA": {"min": 0.05, "max": 0.95, "n": len(TDA_THRESHOLDS)},
                "ANOVA": {"values": ANOVA_THRESHOLDS},
                "MI_screening": {"values": MI_THRESHOLDS},
                "Tree_based": {"min": 0.01, "max": 0.5, "n": len(TREE_THRESHOLDS)},
                "Correlation": {"min": 0.1, "max": 0.9, "n": len(CORR_THRESHOLDS)},
            },
            "best_thresholds": best_thresholds,
            "summary_metrics": {
                **{f"{m}_best_f1": best_f1[m] for m in METHOD_THRESHOLDS},
                **{f"{m}_mean_time": mean_times[m] for m in METHOD_THRESHOLDS},
                **{f"{m}_mean_jaccard": mean_jaccards[m] for m in METHOD_THRESHOLDS},
            },
        },
        "datasets": output_datasets,
    }

    out_path = WS / "method_out.json"
    out_path.write_text(json.dumps(_to_py(output), indent=2))

    elapsed = time.perf_counter() - t_global_start
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Output: {out_path}")


if __name__ == "__main__":
    main()
