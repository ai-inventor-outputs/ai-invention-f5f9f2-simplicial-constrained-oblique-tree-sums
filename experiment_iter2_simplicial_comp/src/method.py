#!/usr/bin/env python3
"""
SC-OTS Experiment: Validate persistent-homology-based feature interaction discovery.

Implements the full simplicial complex construction pipeline:
  Step 1: Pairwise distance correlation matrices (all 10 datasets)
  Step 2: Triangle inequality verification (1-dCor vs sqrt(1-dCor))
  Step 3: Rips filtration construction via GUDHI
  Step 4: Precision/Recall on synthetic datasets
  Step 5: Clique inflation analysis
  Step 6: HSIC comparison (friedman1)
  Step 7: Real dataset characterization
  Step 8: Visualization & output assembly

Baseline: Pearson correlation-based dissimilarity for comparison.
"""

import base64
import io
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import dcor
import gudhi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import squareform

# ── Logging setup ──────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (35 * 1024**3, 35 * 1024**3))  # 35GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU

# ── Constants ──────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DEP = Path(
    "/home/adrian/projects/temp/ai-inventor-old3/aii_pipeline/runs/"
    "run__20260228_133939/3_invention_loop/iter_1/gen_art/data_id3_it1__opus"
)
DATA_FILE = DATA_DEP / "full_data_out.json"
OUTPUT_FILE = WORKSPACE / "method_out.json"

SUBSAMPLE_LIMIT = 2000  # dCor is O(n^2), subsample large datasets
RANDOM_STATE = 42
N_THRESHOLDS = 20
MAX_SIMPLEX_DIM = 4  # up to 4-simplices (5-way interactions)

DATASETS_WITH_GROUND_TRUTH = ["friedman1", "friedman3", "synth_3way", "synth_4way"]
REAL_DATASETS = [
    "diabetes", "breast_w", "california_housing", "wine_quality", "adult", "spambase"
]

# Ground truth interactions from the data formulas
GROUND_TRUTH = {
    "friedman1": {
        # y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4
        "simplices": [frozenset({0, 1})],  # 1-simplex: x0*x1
        "active_features": {0, 1, 2, 3, 4},
        "noise_features": set(range(5, 10)),
    },
    "friedman3": {
        # y = arctan((x1*x2 - 1/(x1*x3)) / x0) → all 4 interact
        "simplices": [
            frozenset({0, 1, 2, 3}),  # 3-simplex
            frozenset({1, 2}),  # sub-interaction
            frozenset({1, 3}),  # sub-interaction
        ],
        "active_features": {0, 1, 2, 3},
    },
    "synth_3way": {
        # y = 5*x0*x1*x2 + 3*sin(pi*x3*x4) + 2*x0 + noise
        "simplices": [
            frozenset({0, 1, 2}),  # 3-way interaction
            frozenset({3, 4}),  # 2-way interaction
        ],
        "active_features": {0, 1, 2, 3, 4},
        "noise_features": set(range(5, 15)),
    },
    "synth_4way": {
        # y = 4*x0*x1*x2*x3 + 3*x4*x5 + 2*x0 + 1.5*x4 + noise
        "simplices": [
            frozenset({0, 1, 2, 3}),  # 4-way interaction
            frozenset({4, 5}),  # 2-way interaction
        ],
        "active_features": {0, 1, 2, 3, 4, 5},
        "noise_features": set(range(6, 20)),
    },
}

DOMAIN_NOTES = {
    "diabetes": "Pima Indians diabetes: expected BMI×skin_thickness, glucose×insulin interactions",
    "breast_w": "Wisconsin Breast Cancer: expected cell_size×cell_shape, bare_nuclei×bland_chromatin groupings",
    "california_housing": "California Housing: expected income×location, rooms×occupancy interactions",
    "wine_quality": "Wine Quality: expected acidity×pH, alcohol×residual_sugar interactions",
    "adult": "Adult Census: expected education×occupation, age×hours_per_week interactions",
    "spambase": "Spambase: expected word_freq×char_freq×capital_run_length groupings",
}


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

@logger.catch
def load_all_datasets(data_path: Path) -> dict:
    """Load full_data_out.json and parse into per-dataset structures."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    logger.info(f"Loaded {len(raw['datasets'])} datasets, {raw['metadata']['total_examples']} total examples")

    datasets = {}
    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        n_examples = len(examples)

        # Parse feature matrix and targets
        X = np.array([json.loads(ex["input"]) for ex in examples], dtype=np.float64)
        y_raw = [ex["output"] for ex in examples]

        task_type = examples[0]["metadata_task_type"]
        if task_type == "regression":
            y = np.array([float(v) for v in y_raw], dtype=np.float64)
        else:
            y = np.array([int(v) for v in y_raw], dtype=np.int64)

        feature_names = examples[0].get("metadata_feature_names", [f"f{i}" for i in range(X.shape[1])])
        known_interactions = None
        if "metadata_known_interactions" in examples[0]:
            known_interactions = json.loads(examples[0]["metadata_known_interactions"])

        datasets[ds_name] = {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "task_type": task_type,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "known_interactions": known_interactions,
            "category": examples[0].get("metadata_category", ""),
        }
        logger.info(f"  {ds_name}: X={X.shape}, task={task_type}, features={len(feature_names)}")

    return datasets


# ══════════════════════════════════════════════════════════════════════
# STEP 1: PAIRWISE DISTANCE CORRELATION MATRICES
# ══════════════════════════════════════════════════════════════════════

def compute_dcor_matrix(
    X: np.ndarray,
    subsample_limit: int = SUBSAMPLE_LIMIT,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Compute pairwise distance correlation matrix for all features."""
    n_samples, n_feat = X.shape

    # Subsample if too many samples (dCor is O(n^2) per pair)
    if n_samples > subsample_limit:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=subsample_limit, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dcor_matrix = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            try:
                val = dcor.distance_correlation(X_sub[:, i], X_sub[:, j])
                dcor_matrix[i, j] = val
                dcor_matrix[j, i] = val
            except Exception:
                logger.exception(f"dCor failed for features ({i}, {j})")
                dcor_matrix[i, j] = 0.0
                dcor_matrix[j, i] = 0.0
    np.fill_diagonal(dcor_matrix, 1.0)
    return dcor_matrix


def compute_pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson |r| matrix (baseline)."""
    corr = np.corrcoef(X, rowvar=False)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 1.0)
    return corr


@logger.catch
def step1_dcor_matrices(datasets: dict) -> dict:
    """Step 1: Compute pairwise dCor and Pearson matrices for all datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Pairwise Distance Correlation Matrices")
    logger.info("=" * 60)

    dcor_matrices = {}
    pearson_matrices = {}

    for ds_name, ds in datasets.items():
        t0 = time.time()
        logger.info(f"Computing dCor matrix for {ds_name} (n={ds['n_samples']}, p={ds['n_features']})")

        dcor_mat = compute_dcor_matrix(ds["X"])
        elapsed = time.time() - t0
        logger.info(f"  dCor matrix computed in {elapsed:.1f}s")
        dcor_matrices[ds_name] = dcor_mat

        # Baseline: Pearson
        pearson_mat = compute_pearson_matrix(ds["X"])
        pearson_matrices[ds_name] = pearson_mat

    return {"dcor": dcor_matrices, "pearson": pearson_matrices}


# ══════════════════════════════════════════════════════════════════════
# STEP 1b: FEATURE-TARGET INTERACTION dCor (ENHANCED METHOD)
# ══════════════════════════════════════════════════════════════════════

def compute_interaction_dcor_matrix(
    X: np.ndarray,
    y: np.ndarray,
    subsample_limit: int = SUBSAMPLE_LIMIT,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Compute feature-target interaction dCor matrix.

    For each pair (i, j), compute:
        interaction_dcor[i,j] = dCor((X_i, X_j), y) - max(dCor(X_i, y), dCor(X_j, y))
    This captures the synergistic interaction contribution of the pair
    beyond what individual features provide.
    """
    n_samples, n_feat = X.shape

    if n_samples > subsample_limit:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, size=subsample_limit, replace=False)
        X_sub = X[idx]
        y_sub = y[idx].astype(np.float64)
    else:
        X_sub = X
        y_sub = y.astype(np.float64)

    # First compute individual feature-target dCor
    individual_dcor = np.zeros(n_feat)
    for i in range(n_feat):
        try:
            individual_dcor[i] = dcor.distance_correlation(X_sub[:, i], y_sub)
        except Exception:
            logger.exception(f"Individual dCor failed for feature {i}")
            individual_dcor[i] = 0.0

    # Compute pairwise joint feature-target dCor
    interaction_matrix = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            try:
                joint_dcor = dcor.distance_correlation(
                    X_sub[:, [i, j]], y_sub
                )
                # Interaction = joint effect minus best individual effect
                interaction = joint_dcor - max(individual_dcor[i], individual_dcor[j])
                interaction_matrix[i, j] = max(interaction, 0.0)  # clamp to non-negative
                interaction_matrix[j, i] = interaction_matrix[i, j]
            except Exception:
                logger.exception(f"Joint dCor failed for features ({i}, {j})")

    # Diagonal = individual feature-target dCor (self-association)
    np.fill_diagonal(interaction_matrix, np.max(interaction_matrix) if np.max(interaction_matrix) > 0 else 1.0)

    return interaction_matrix, individual_dcor


@logger.catch
def step1b_interaction_dcor(datasets: dict) -> dict:
    """Step 1b: Compute feature-target interaction dCor matrices for synthetic datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1b: Feature-Target Interaction dCor (Enhanced)")
    logger.info("=" * 60)

    interaction_matrices = {}
    individual_dcors = {}

    for ds_name in DATASETS_WITH_GROUND_TRUTH:
        if ds_name not in datasets:
            continue

        ds = datasets[ds_name]
        t0 = time.time()
        logger.info(f"Computing interaction dCor for {ds_name} (n={ds['n_samples']}, p={ds['n_features']})")

        int_mat, ind_dcor = compute_interaction_dcor_matrix(ds["X"], ds["y"])
        elapsed = time.time() - t0
        logger.info(f"  Interaction dCor matrix computed in {elapsed:.1f}s")

        # Log top interactions
        n_feat = int_mat.shape[0]
        upper = np.triu_indices(n_feat, k=1)
        pairs = list(zip(upper[0], upper[1], int_mat[upper]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"  Top 5 interactions:")
        for i, j, v in pairs[:5]:
            logger.info(f"    ({i},{j}): {v:.4f}")

        # Log individual feature-target dCor
        logger.info(f"  Individual dCor(feature, y):")
        for i, v in enumerate(ind_dcor):
            logger.info(f"    feature {i}: {v:.4f}")

        interaction_matrices[ds_name] = int_mat
        individual_dcors[ds_name] = ind_dcor

    return {"interaction_matrices": interaction_matrices, "individual_dcors": individual_dcors}


def step1b_enhanced_rips(
    interaction_matrices: dict,
    individual_dcors: dict,
) -> dict:
    """Build Rips filtration from interaction dCor and evaluate P/R/F1."""
    logger.info("=" * 60)
    logger.info("STEP 1b RIPS: Enhanced Interaction-Based Rips")
    logger.info("=" * 60)

    enhanced_results = {}

    for ds_name in DATASETS_WITH_GROUND_TRUTH:
        if ds_name not in interaction_matrices:
            continue

        int_mat = interaction_matrices[ds_name]
        max_val = int_mat.max()
        if max_val <= 0:
            logger.warning(f"{ds_name}: No positive interaction values, skipping")
            enhanced_results[ds_name] = {
                "f1_at_optimal": 0.0,
                "f1_at_gap": 0.0,
                "note": "No positive interaction values detected",
            }
            continue

        # Normalize to [0,1] and convert to dissimilarity
        int_norm = int_mat / max_val
        D = np.sqrt(np.clip(1.0 - int_norm, 0.0, None))
        np.fill_diagonal(D, 0.0)

        logger.info(f"Building enhanced Rips for {ds_name}")
        rips = build_rips_filtration(D)

        gt_info = GROUND_TRUTH[ds_name]
        gt_expanded = expand_simplicial_closure(gt_info["simplices"])

        # Evaluate at all thresholds
        best_f1 = 0.0
        best_t = 0.0
        metrics_by_t = {}
        for t in rips["thresholds"]:
            predicted = set(rips["simplices_at_threshold"][t])
            metrics = compute_pr_f1(predicted=predicted, gt_expanded=gt_expanded)
            metrics_by_t[str(round(t, 6))] = metrics
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_t = t

        gap_key = str(round(rips["gap_threshold"], 6))
        gap_metrics = metrics_by_t.get(gap_key, {"f1": 0.0})

        enhanced_results[ds_name] = {
            "f1_at_optimal": round(best_f1, 4),
            "optimal_threshold": round(best_t, 6),
            "f1_at_gap": round(gap_metrics["f1"], 4),
            "gap_threshold": round(rips["gap_threshold"], 6),
            "betti_numbers": rips["betti_numbers"],
            "metrics_by_threshold": metrics_by_t,
        }
        logger.info(f"  {ds_name}: Enhanced F1@optimal={best_f1:.3f}, F1@gap={gap_metrics['f1']:.3f}")

    return enhanced_results


# ══════════════════════════════════════════════════════════════════════
# STEP 2: TRIANGLE INEQUALITY VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def check_triangle_inequality(D: np.ndarray) -> dict:
    """Check triangle inequality violations for a dissimilarity matrix."""
    n = D.shape[0]
    violations = []
    total_checks = 0

    for i, j, k in itertools.combinations(range(n), 3):
        for a, b, c in [(i, j, k), (j, k, i), (i, k, j)]:
            total_checks += 1
            lhs = D[a, c]
            rhs = D[a, b] + D[b, c]
            if lhs > rhs + 1e-12:
                mag = lhs - rhs
                violations.append({
                    "triple": (a, b, c),
                    "violation_magnitude": float(mag),
                    "relative_violation": float(mag / lhs) if lhs > 0 else 0.0,
                })

    result = {
        "total_checks": total_checks,
        "n_violations": len(violations),
        "violation_rate": len(violations) / total_checks if total_checks > 0 else 0.0,
    }
    if violations:
        result["max_violation_magnitude"] = max(v["violation_magnitude"] for v in violations)
        result["mean_violation_magnitude"] = float(np.mean([v["violation_magnitude"] for v in violations]))
        result["max_relative_violation"] = max(v["relative_violation"] for v in violations)
    else:
        result["max_violation_magnitude"] = 0.0
        result["mean_violation_magnitude"] = 0.0
        result["max_relative_violation"] = 0.0

    return result


@logger.catch
def step2_triangle_inequality(dcor_matrices: dict) -> dict:
    """Step 2: Check triangle inequality for 1-dCor and sqrt(1-dCor) transforms."""
    logger.info("=" * 60)
    logger.info("STEP 2: Triangle Inequality Verification")
    logger.info("=" * 60)

    results = {}
    for ds_name, dcor_mat in dcor_matrices.items():
        logger.info(f"Checking triangle inequality for {ds_name} (p={dcor_mat.shape[0]})")

        # D_raw = 1 - dCor
        D_raw = 1.0 - dcor_mat
        np.fill_diagonal(D_raw, 0.0)

        # D_sqrt = sqrt(1 - dCor)
        D_sqrt = np.sqrt(np.clip(1.0 - dcor_mat, 0.0, None))
        np.fill_diagonal(D_sqrt, 0.0)

        raw_result = check_triangle_inequality(D_raw)
        raw_result["dissimilarity_type"] = "1-dCor"
        logger.info(f"  1-dCor: {raw_result['n_violations']}/{raw_result['total_checks']} violations "
                     f"(rate={raw_result['violation_rate']:.6f})")

        sqrt_result = check_triangle_inequality(D_sqrt)
        sqrt_result["dissimilarity_type"] = "sqrt(1-dCor)"
        logger.info(f"  sqrt(1-dCor): {sqrt_result['n_violations']}/{sqrt_result['total_checks']} violations "
                     f"(rate={sqrt_result['violation_rate']:.6f})")

        results[ds_name] = {"raw": raw_result, "sqrt": sqrt_result}

    return results


# ══════════════════════════════════════════════════════════════════════
# STEP 3: RIPS FILTRATION CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def build_rips_filtration(
    D: np.ndarray,
    max_dimension: int = MAX_SIMPLEX_DIM,
    n_thresholds: int = N_THRESHOLDS,
) -> dict:
    """Build Rips filtration using GUDHI and extract persistence/simplices."""
    n_feat = D.shape[0]

    # Build lower-triangular distance matrix for GUDHI
    lower_tri = []
    for i in range(n_feat):
        lower_tri.append(list(D[i, :i].astype(float)))

    # Threshold range from pairwise distances
    upper_tri_vals = D[np.triu_indices(n_feat, k=1)]
    d_min = float(upper_tri_vals.min())
    d_max = float(upper_tri_vals.max())

    # Cap max_edge_length to d_max for memory safety
    # For datasets with many features, limit max_dimension to avoid combinatorial explosion
    effective_max_dim = min(max_dimension, 3) if n_feat > 30 else max_dimension

    try:
        rips = gudhi.RipsComplex(distance_matrix=lower_tri, max_edge_length=d_max)
        st = rips.create_simplex_tree(max_dimension=effective_max_dim)
    except MemoryError:
        logger.warning(f"MemoryError with max_dim={effective_max_dim}, reducing to 2")
        rips = gudhi.RipsComplex(distance_matrix=lower_tri, max_edge_length=d_max)
        st = rips.create_simplex_tree(max_dimension=2)

    # Compute persistence
    st.compute_persistence()
    persistence_pairs = st.persistence()
    betti = st.betti_numbers()

    # Persistence diagrams per dimension
    persistence_diagrams = {}
    max_dim_found = max((dim for dim, _ in persistence_pairs), default=0)
    for dim in range(min(max_dim_found + 1, 5)):
        intervals = st.persistence_intervals_in_dimension(dim)
        if len(intervals) > 0:
            persistence_diagrams[dim] = intervals.tolist()
        else:
            persistence_diagrams[dim] = []

    # Largest gap heuristic for threshold selection
    all_deaths = sorted([d for _, (b, d) in persistence_pairs if d != float("inf") and np.isfinite(d)])
    if len(all_deaths) >= 2:
        gaps = [all_deaths[i + 1] - all_deaths[i] for i in range(len(all_deaths) - 1)]
        largest_gap_idx = int(np.argmax(gaps))
        gap_threshold = (all_deaths[largest_gap_idx] + all_deaths[largest_gap_idx + 1]) / 2
    else:
        gap_threshold = (d_min + d_max) / 2  # fallback: midpoint

    # Evaluation thresholds
    thresholds = np.linspace(d_min, d_max, n_thresholds).tolist()
    eval_thresholds = sorted(set(thresholds + [gap_threshold]))

    # Extract simplices at each threshold
    all_filtration = list(st.get_filtration())
    simplices_at_threshold = {}
    num_simplices_by_dim = {}

    for t in eval_thresholds:
        simplices_t = [
            frozenset(simplex)
            for simplex, filt in all_filtration
            if filt <= t and len(simplex) >= 2
        ]
        simplices_at_threshold[t] = simplices_t

        dim_counts = {}
        for s in simplices_t:
            d = len(s) - 1  # dimension of simplex
            dim_counts[d] = dim_counts.get(d, 0) + 1
        num_simplices_by_dim[t] = dim_counts

    return {
        "persistence_diagrams": persistence_diagrams,
        "betti_numbers": betti,
        "gap_threshold": gap_threshold,
        "d_min": d_min,
        "d_max": d_max,
        "thresholds": eval_thresholds,
        "simplices_at_threshold": simplices_at_threshold,
        "num_simplices_by_dim_and_threshold": {
            str(t): v for t, v in num_simplices_by_dim.items()
        },
    }


@logger.catch
def step3_rips_filtrations(
    dcor_matrices: dict,
    triangle_results: dict,
) -> dict:
    """Step 3: Build Rips filtration for all datasets."""
    logger.info("=" * 60)
    logger.info("STEP 3: Rips Filtration Construction")
    logger.info("=" * 60)

    rips_results = {}
    dissim_choice = {}

    for ds_name, dcor_mat in dcor_matrices.items():
        t0 = time.time()
        logger.info(f"Building Rips filtration for {ds_name}")

        # Choose dissimilarity transform based on triangle inequality results
        tri = triangle_results.get(ds_name, {})
        raw_violations = tri.get("raw", {}).get("n_violations", 0)
        sqrt_violations = tri.get("sqrt", {}).get("n_violations", 0)

        # Use sqrt if it has fewer violations, otherwise raw
        if sqrt_violations < raw_violations:
            D = np.sqrt(np.clip(1.0 - dcor_mat, 0.0, None))
            chosen = "sqrt(1-dCor)"
        else:
            D = 1.0 - dcor_mat
            chosen = "1-dCor"
        np.fill_diagonal(D, 0.0)
        dissim_choice[ds_name] = chosen
        logger.info(f"  Using {chosen} (raw violations={raw_violations}, sqrt violations={sqrt_violations})")

        result = build_rips_filtration(D)
        result["dissimilarity_used"] = chosen
        rips_results[ds_name] = result

        elapsed = time.time() - t0
        logger.info(f"  Built in {elapsed:.1f}s, Betti={result['betti_numbers']}, "
                     f"gap_threshold={result['gap_threshold']:.4f}")

    return rips_results


# ══════════════════════════════════════════════════════════════════════
# STEP 4: PRECISION/RECALL ON SYNTHETIC DATASETS
# ══════════════════════════════════════════════════════════════════════

def expand_simplicial_closure(simplices: list) -> set:
    """Expand a list of simplices to include all sub-simplices of dim >= 1."""
    expanded = set()
    for s in simplices:
        s_list = sorted(s)
        # Add the simplex itself if dim >= 1
        if len(s_list) >= 2:
            expanded.add(frozenset(s_list))
        # Add all sub-simplices of dim >= 1
        for r in range(2, len(s_list) + 1):
            for combo in itertools.combinations(s_list, r):
                expanded.add(frozenset(combo))
    return expanded


def compute_pr_f1(predicted: set, gt_expanded: set) -> dict:
    """Compute precision, recall, F1 for predicted vs ground-truth simplices."""
    tp = predicted & gt_expanded
    fp = predicted - gt_expanded
    fn = gt_expanded - predicted

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
    }


@logger.catch
def step4_interaction_recovery(rips_results: dict) -> dict:
    """Step 4: Precision/Recall/F1 on synthetic datasets."""
    logger.info("=" * 60)
    logger.info("STEP 4: Interaction Recovery (Precision/Recall)")
    logger.info("=" * 60)

    interaction_recovery = {}

    for ds_name in DATASETS_WITH_GROUND_TRUTH:
        if ds_name not in rips_results:
            logger.warning(f"Skipping {ds_name}: no Rips results")
            continue

        gt_info = GROUND_TRUTH[ds_name]
        gt_simplices = gt_info["simplices"]
        gt_expanded = expand_simplicial_closure(gt_simplices)
        logger.info(f"{ds_name}: {len(gt_simplices)} ground-truth simplices, "
                     f"{len(gt_expanded)} after closure expansion")

        rips = rips_results[ds_name]
        eval_thresholds = rips["thresholds"]
        simplices_at_t = rips["simplices_at_threshold"]
        gap_threshold = rips["gap_threshold"]

        metrics_by_threshold = {}
        for t in eval_thresholds:
            predicted = set(simplices_at_t[t])
            metrics = compute_pr_f1(predicted=predicted, gt_expanded=gt_expanded)
            metrics_by_threshold[str(round(t, 6))] = metrics

        # Find optimal threshold (max F1)
        best_t = max(eval_thresholds, key=lambda t: metrics_by_threshold[str(round(t, 6))]["f1"])
        best_metrics = metrics_by_threshold[str(round(best_t, 6))]

        # Gap threshold metrics
        gap_t_key = str(round(gap_threshold, 6))
        gap_metrics = metrics_by_threshold.get(gap_t_key, {"f1": 0.0})

        d_range = rips["d_max"] - rips["d_min"]
        discrepancy = abs(best_t - gap_threshold) / d_range if d_range > 0 else 0.0

        interaction_recovery[ds_name] = {
            "f1_at_optimal": best_metrics["f1"],
            "optimal_threshold": round(best_t, 6),
            "precision_at_optimal": best_metrics["precision"],
            "recall_at_optimal": best_metrics["recall"],
            "f1_at_gap": gap_metrics["f1"],
            "gap_threshold": round(gap_threshold, 6),
            "threshold_discrepancy_normalized": round(discrepancy, 4),
            "gt_simplices_count": len(gt_expanded),
            "metrics_by_threshold": metrics_by_threshold,
        }
        logger.info(f"  {ds_name}: F1@optimal={best_metrics['f1']:.3f} (t={best_t:.4f}), "
                     f"F1@gap={gap_metrics['f1']:.3f} (t={gap_threshold:.4f})")

    return interaction_recovery


# ══════════════════════════════════════════════════════════════════════
# STEP 5: CLIQUE INFLATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

@logger.catch
def step5_clique_inflation(rips_results: dict) -> dict:
    """Step 5: Analyze clique inflation artifacts in Rips complex."""
    logger.info("=" * 60)
    logger.info("STEP 5: Clique Inflation Analysis")
    logger.info("=" * 60)

    clique_inflation = {}

    for ds_name in DATASETS_WITH_GROUND_TRUTH:
        if ds_name not in rips_results:
            continue

        gt_info = GROUND_TRUTH[ds_name]
        gt_expanded = expand_simplicial_closure(gt_info["simplices"])

        rips = rips_results[ds_name]
        gap_threshold = rips["gap_threshold"]
        rips_simplices = rips["simplices_at_threshold"][gap_threshold]
        rips_set = set(rips_simplices)

        logger.info(f"Analyzing clique inflation for {ds_name} at gap_threshold={gap_threshold:.4f}")

        dim_analysis = {}
        for dim in [1, 2, 3]:
            simplices_of_dim = [s for s in rips_simplices if len(s) == dim + 1]
            genuine = [s for s in simplices_of_dim if s in gt_expanded]
            spurious = [s for s in simplices_of_dim if s not in gt_expanded]

            # Check if spurious simplices are due to clique inflation
            inflation_confirmed = []
            for s in spurious:
                # All faces of dimension (dim-1) should be in the complex
                faces = [frozenset(c) for c in itertools.combinations(s, dim)]
                if all(f in rips_set for f in faces):
                    inflation_confirmed.append(sorted(list(s)))

            inflation_rate = len(inflation_confirmed) / len(simplices_of_dim) if simplices_of_dim else 0.0

            dim_analysis[str(dim)] = {
                "total_simplices": len(simplices_of_dim),
                "genuine": len(genuine),
                "spurious_total": len(spurious),
                "spurious_inflation": len(inflation_confirmed),
                "inflation_rate": round(inflation_rate, 4),
            }
            if simplices_of_dim:
                logger.info(f"  dim={dim}: {len(simplices_of_dim)} total, {len(genuine)} genuine, "
                             f"{len(inflation_confirmed)} inflation (rate={inflation_rate:.3f})")

        clique_inflation[ds_name] = dim_analysis

    return clique_inflation


# ══════════════════════════════════════════════════════════════════════
# STEP 6: HSIC COMPARISON
# ══════════════════════════════════════════════════════════════════════

def compute_hsic_matrix(X: np.ndarray, subsample_limit: int = SUBSAMPLE_LIMIT) -> np.ndarray:
    """Compute HSIC-based association matrix."""
    from hyppo.independence import Hsic

    n_samples, n_feat = X.shape
    if n_samples > subsample_limit:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n_samples, size=subsample_limit, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    hsic_matrix = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            try:
                stat, _ = Hsic().test(X_sub[:, i:i + 1], X_sub[:, j:j + 1], auto=True)
                hsic_matrix[i, j] = stat
                hsic_matrix[j, i] = stat
            except Exception:
                logger.exception(f"HSIC failed for features ({i}, {j})")
                hsic_matrix[i, j] = 0.0
                hsic_matrix[j, i] = 0.0

    # Self-association = max
    max_val = hsic_matrix.max()
    if max_val > 0:
        np.fill_diagonal(hsic_matrix, max_val)
    else:
        np.fill_diagonal(hsic_matrix, 1.0)

    return hsic_matrix


@logger.catch
def step6_hsic_comparison(
    datasets: dict,
    dcor_matrices: dict,
    rips_results: dict,
    interaction_recovery: dict,
) -> dict:
    """Step 6: Compare HSIC vs dCor on friedman1."""
    logger.info("=" * 60)
    logger.info("STEP 6: HSIC Comparison (friedman1)")
    logger.info("=" * 60)

    ds_name = "friedman1"
    if ds_name not in datasets:
        logger.warning("friedman1 not found, skipping HSIC comparison")
        return {"skipped": True, "reason": "friedman1 not found"}

    X = datasets[ds_name]["X"]
    t0 = time.time()
    logger.info(f"Computing HSIC matrix for {ds_name} (n={X.shape[0]}, p={X.shape[1]})")

    try:
        hsic_mat = compute_hsic_matrix(X)
        elapsed = time.time() - t0
        logger.info(f"HSIC matrix computed in {elapsed:.1f}s")
    except Exception:
        logger.exception("HSIC computation failed")
        return {"skipped": True, "reason": "HSIC computation failed"}

    # Normalize HSIC to [0,1]
    hsic_max = hsic_mat.max()
    if hsic_max > 0:
        hsic_norm = hsic_mat / hsic_max
    else:
        hsic_norm = hsic_mat

    # Dissimilarity: sqrt(1 - hsic_norm)
    D_hsic = np.sqrt(np.clip(1.0 - hsic_norm, 0.0, None))
    np.fill_diagonal(D_hsic, 0.0)

    # Build Rips filtration
    hsic_rips = build_rips_filtration(D_hsic)

    # Evaluate P/R/F1 on friedman1
    gt_info = GROUND_TRUTH[ds_name]
    gt_expanded = expand_simplicial_closure(gt_info["simplices"])

    hsic_metrics_by_t = {}
    for t in hsic_rips["thresholds"]:
        predicted = set(hsic_rips["simplices_at_threshold"][t])
        metrics = compute_pr_f1(predicted=predicted, gt_expanded=gt_expanded)
        hsic_metrics_by_t[str(round(t, 6))] = metrics

    # Best HSIC F1
    best_hsic_t = max(
        hsic_rips["thresholds"],
        key=lambda t: hsic_metrics_by_t[str(round(t, 6))]["f1"],
    )
    hsic_f1_optimal = hsic_metrics_by_t[str(round(best_hsic_t, 6))]["f1"]
    hsic_f1_gap = hsic_metrics_by_t.get(
        str(round(hsic_rips["gap_threshold"], 6)),
        {"f1": 0.0},
    )["f1"]

    # dCor results for comparison
    dcor_f1_optimal = interaction_recovery.get(ds_name, {}).get("f1_at_optimal", 0.0)
    dcor_f1_gap = interaction_recovery.get(ds_name, {}).get("f1_at_gap", 0.0)

    # Verdict
    if hsic_f1_optimal > dcor_f1_optimal + 0.05:
        verdict = "HSIC_better"
    elif dcor_f1_optimal > hsic_f1_optimal + 0.05:
        verdict = "dCor_better"
    else:
        verdict = "comparable"

    result = {
        "skipped": False,
        "dcor_f1_at_gap": dcor_f1_gap,
        "hsic_f1_at_gap": hsic_f1_gap,
        "dcor_optimal_f1": dcor_f1_optimal,
        "hsic_optimal_f1": hsic_f1_optimal,
        "hsic_gap_threshold": round(hsic_rips["gap_threshold"], 6),
        "hsic_optimal_threshold": round(best_hsic_t, 6),
        "hsic_matrix": hsic_mat.tolist(),
        "hsic_metrics_by_threshold": hsic_metrics_by_t,
        "conclusion": verdict,
    }
    logger.info(f"  dCor optimal F1={dcor_f1_optimal:.3f}, HSIC optimal F1={hsic_f1_optimal:.3f} -> {verdict}")
    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 7: REAL DATASET CHARACTERIZATION
# ══════════════════════════════════════════════════════════════════════

@logger.catch
def step7_real_datasets(
    datasets: dict,
    rips_results: dict,
    dcor_matrices: dict,
) -> dict:
    """Step 7: Characterize real dataset interaction topologies."""
    logger.info("=" * 60)
    logger.info("STEP 7: Real Dataset Characterization")
    logger.info("=" * 60)

    analysis = {}

    for ds_name in REAL_DATASETS:
        if ds_name not in rips_results:
            logger.warning(f"Skipping {ds_name}: no Rips results")
            continue

        rips = rips_results[ds_name]
        gap_threshold = rips["gap_threshold"]
        simplices = rips["simplices_at_threshold"][gap_threshold]

        # Count simplices by dimension
        dim_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in simplices:
            d = len(s) - 1
            if d in dim_counts:
                dim_counts[d] += 1

        # Features participating in any simplex of dim >= 1
        participating_features = set()
        for s in simplices:
            if len(s) >= 2:
                participating_features.update(s)

        n_feat = datasets[ds_name]["n_features"]
        feature_names = datasets[ds_name]["feature_names"]
        participation_rate = len(participating_features) / n_feat if n_feat > 0 else 0.0

        # Top persistent features (from persistence diagram dim 0)
        pd0 = rips["persistence_diagrams"].get(0, [])
        top_persistent = []
        if pd0:
            # Sort by persistence (death - birth)
            pd0_sorted = sorted(pd0, key=lambda x: (x[1] - x[0]) if np.isfinite(x[1]) else float("inf"), reverse=True)
            for birth, death in pd0_sorted[:5]:
                top_persistent.append({
                    "birth": round(birth, 4),
                    "death": round(death, 4) if np.isfinite(death) else "inf",
                    "persistence": round(death - birth, 4) if np.isfinite(death) else "inf",
                })

        # Top edges (highest dCor pairs)
        dcor_mat = dcor_matrices[ds_name]
        upper_tri = np.triu_indices(n_feat, k=1)
        pair_vals = list(zip(upper_tri[0], upper_tri[1], dcor_mat[upper_tri]))
        pair_vals.sort(key=lambda x: x[2], reverse=True)
        top_edges = []
        for i, j, val in pair_vals[:10]:
            fname_i = feature_names[i] if i < len(feature_names) else f"f{i}"
            fname_j = feature_names[j] if j < len(feature_names) else f"f{j}"
            top_edges.append({
                "features": [fname_i, fname_j],
                "dcor": round(float(val), 4),
            })

        analysis[ds_name] = {
            "simplex_counts_by_dim": {str(k): v for k, v in dim_counts.items()},
            "betti_numbers": rips["betti_numbers"],
            "top_persistent_intervals": top_persistent,
            "top_edges": top_edges,
            "feature_participation_rate": round(participation_rate, 4),
            "participating_features": len(participating_features),
            "total_features": n_feat,
            "gap_threshold": round(gap_threshold, 4),
            "domain_notes": DOMAIN_NOTES.get(ds_name, ""),
        }
        logger.info(f"  {ds_name}: edges={dim_counts[1]}, triangles={dim_counts[2]}, "
                     f"participation={participation_rate:.2f}")

    return analysis


# ══════════════════════════════════════════════════════════════════════
# BASELINE: Pearson correlation-based Rips filtration
# ══════════════════════════════════════════════════════════════════════

@logger.catch
def baseline_pearson_pipeline(
    datasets: dict,
    pearson_matrices: dict,
) -> dict:
    """Baseline: Use Pearson |r| dissimilarity for the same Rips pipeline."""
    logger.info("=" * 60)
    logger.info("BASELINE: Pearson Correlation Rips Pipeline")
    logger.info("=" * 60)

    baseline_results = {}

    for ds_name in DATASETS_WITH_GROUND_TRUTH:
        if ds_name not in pearson_matrices:
            continue

        pearson_mat = pearson_matrices[ds_name]
        D = np.sqrt(np.clip(1.0 - pearson_mat, 0.0, None))
        np.fill_diagonal(D, 0.0)

        logger.info(f"Building Pearson-based Rips for {ds_name}")
        rips = build_rips_filtration(D)

        gt_info = GROUND_TRUTH[ds_name]
        gt_expanded = expand_simplicial_closure(gt_info["simplices"])

        # Evaluate at all thresholds
        best_f1 = 0.0
        best_t = 0.0
        for t in rips["thresholds"]:
            predicted = set(rips["simplices_at_threshold"][t])
            metrics = compute_pr_f1(predicted=predicted, gt_expanded=gt_expanded)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_t = t

        # Gap threshold
        gap_predicted = set(rips["simplices_at_threshold"][rips["gap_threshold"]])
        gap_metrics = compute_pr_f1(predicted=gap_predicted, gt_expanded=gt_expanded)

        baseline_results[ds_name] = {
            "f1_at_optimal": round(best_f1, 4),
            "optimal_threshold": round(best_t, 6),
            "f1_at_gap": round(gap_metrics["f1"], 4),
            "gap_threshold": round(rips["gap_threshold"], 6),
            "betti_numbers": rips["betti_numbers"],
        }
        logger.info(f"  {ds_name}: Pearson F1@optimal={best_f1:.3f}, F1@gap={gap_metrics['f1']:.3f}")

    return baseline_results


# ══════════════════════════════════════════════════════════════════════
# STEP 8: VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def figure_f1_vs_threshold(interaction_recovery: dict) -> str:
    """Figure 1: F1 vs Threshold for synthetic datasets."""
    datasets = [d for d in DATASETS_WITH_GROUND_TRUTH if d in interaction_recovery]
    n = len(datasets)
    if n == 0:
        return ""

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for idx, ds_name in enumerate(datasets):
        ax = axes[0, idx]
        rec = interaction_recovery[ds_name]
        metrics = rec["metrics_by_threshold"]

        ts = sorted([float(k) for k in metrics.keys()])
        f1s = [metrics[str(round(t, 6))]["f1"] for t in ts]

        ax.plot(ts, f1s, "b-o", markersize=3, label="F1")
        ax.axvline(rec["gap_threshold"], color="r", linestyle="--", label=f"Gap t={rec['gap_threshold']:.3f}")
        ax.axvline(rec["optimal_threshold"], color="g", linestyle=":", label=f"Opt t={rec['optimal_threshold']:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title(ds_name)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("F1 vs Threshold (dCor-based Rips)", fontsize=14)
    fig.tight_layout()
    return fig_to_base64(fig)


def figure_persistence_diagrams(rips_results: dict) -> str:
    """Figure 2: Persistence diagrams for synthetic datasets."""
    datasets = [d for d in DATASETS_WITH_GROUND_TRUTH if d in rips_results]
    n = len(datasets)
    if n == 0:
        return ""

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    colors = {0: "blue", 1: "orange", 2: "green", 3: "red"}

    for idx, ds_name in enumerate(datasets):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        pd_data = rips_results[ds_name]["persistence_diagrams"]

        max_death = 0
        for dim, intervals in pd_data.items():
            dim = int(dim)
            if not intervals:
                continue
            births = [iv[0] for iv in intervals]
            deaths = [iv[1] for iv in intervals if np.isfinite(iv[1])]
            finite_mask = [np.isfinite(iv[1]) for iv in intervals]

            b_finite = [births[i] for i in range(len(births)) if finite_mask[i]]
            d_finite = [deaths[i] for i, fm in enumerate(finite_mask) if fm]

            if d_finite:
                max_death = max(max_death, max(d_finite))
            if b_finite:
                ax.scatter(b_finite, d_finite, c=colors.get(dim, "black"),
                           label=f"H{dim}", s=20, alpha=0.7)

        if max_death > 0:
            ax.plot([0, max_death * 1.1], [0, max_death * 1.1], "k--", alpha=0.3)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"{ds_name}")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Persistence Diagrams", fontsize=14)
    fig.tight_layout()
    return fig_to_base64(fig)


def figure_triangle_violations(triangle_results: dict) -> str:
    """Figure 3: Triangle inequality violations across datasets."""
    ds_names = sorted(triangle_results.keys())
    if not ds_names:
        return ""

    raw_rates = [triangle_results[d]["raw"]["violation_rate"] for d in ds_names]
    sqrt_rates = [triangle_results[d]["sqrt"]["violation_rate"] for d in ds_names]

    x = np.arange(len(ds_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, raw_rates, width, label="1-dCor", color="salmon")
    ax.bar(x + width / 2, sqrt_rates, width, label="sqrt(1-dCor)", color="skyblue")
    ax.set_ylabel("Violation Rate")
    ax.set_xlabel("Dataset")
    ax.set_title("Triangle Inequality Violation Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)


def figure_clique_inflation(clique_inflation: dict) -> str:
    """Figure 4: Clique inflation rates for synthetic datasets."""
    ds_names = [d for d in DATASETS_WITH_GROUND_TRUTH if d in clique_inflation]
    if not ds_names:
        return ""

    dims = ["1", "2", "3"]
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(ds_names))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, dim in enumerate(dims):
        genuine = []
        spurious = []
        for d in ds_names:
            info = clique_inflation[d].get(dim, {"genuine": 0, "spurious_inflation": 0})
            genuine.append(info["genuine"])
            spurious.append(info["spurious_inflation"])

        ax.bar(x + i * width, genuine, width, label=f"dim={dim} genuine", color=colors[i], alpha=0.7)
        ax.bar(x + i * width, spurious, width, bottom=genuine,
               label=f"dim={dim} spurious", color=colors[i], alpha=0.3, hatch="//")

    ax.set_ylabel("Simplex Count")
    ax.set_xlabel("Dataset")
    ax.set_title("Genuine vs Spurious (Clique-Inflated) Simplices")
    ax.set_xticks(x + width)
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig_to_base64(fig)


def figure_dcor_vs_hsic(
    interaction_recovery: dict,
    hsic_comparison: dict,
) -> str:
    """Figure 5: dCor vs HSIC F1 curves on friedman1."""
    if hsic_comparison.get("skipped", True):
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))

    # dCor F1 curve
    dcor_metrics = interaction_recovery.get("friedman1", {}).get("metrics_by_threshold", {})
    if dcor_metrics:
        ts = sorted([float(k) for k in dcor_metrics.keys()])
        f1s = [dcor_metrics[str(round(t, 6))]["f1"] for t in ts]
        ax.plot(ts, f1s, "b-o", markersize=3, label="dCor")

    # HSIC F1 curve
    hsic_metrics = hsic_comparison.get("hsic_metrics_by_threshold", {})
    if hsic_metrics:
        ts = sorted([float(k) for k in hsic_metrics.keys()])
        f1s = [hsic_metrics[str(round(t, 6))]["f1"] for t in ts]
        ax.plot(ts, f1s, "r-s", markersize=3, label="HSIC")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("dCor vs HSIC on friedman1")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return fig_to_base64(fig)


def figure_interaction_graphs(
    datasets: dict,
    dcor_matrices: dict,
    rips_results: dict,
) -> str:
    """Figure 6: Interaction graphs for small real datasets."""
    # Select datasets with few features for visualization
    viz_datasets = [d for d in ["diabetes", "breast_w", "wine_quality"]
                    if d in rips_results and datasets[d]["n_features"] <= 15]
    if not viz_datasets:
        return ""

    n = len(viz_datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)

    for idx, ds_name in enumerate(viz_datasets):
        ax = axes[0, idx]
        dcor_mat = dcor_matrices[ds_name]
        rips = rips_results[ds_name]
        gap_threshold = rips["gap_threshold"]
        simplices = rips["simplices_at_threshold"][gap_threshold]
        feature_names = datasets[ds_name]["feature_names"]
        n_feat = dcor_mat.shape[0]

        # Build networkx graph
        G = nx.Graph()
        for i in range(n_feat):
            short_name = feature_names[i][:12] if i < len(feature_names) else f"f{i}"
            G.add_node(i, label=short_name)

        # Add edges from 1-simplices
        edges = [s for s in simplices if len(s) == 2]
        for e in edges:
            u, v = sorted(e)
            weight = float(dcor_mat[u, v])
            G.add_edge(u, v, weight=weight)

        # Identify triangles (2-simplices)
        triangles = [s for s in simplices if len(s) == 3]

        pos = nx.spring_layout(G, seed=42)
        labels = {i: G.nodes[i].get("label", str(i)) for i in G.nodes}

        # Draw
        if G.edges:
            edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges]
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, alpha=0.5)

        # Highlight triangles
        for tri in triangles:
            tri_list = sorted(tri)
            if all(n in pos for n in tri_list):
                triangle_coords = np.array([pos[n] for n in tri_list])
                triangle_patch = plt.Polygon(triangle_coords, alpha=0.15, color="orange")
                ax.add_patch(triangle_patch)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)
        ax.set_title(f"{ds_name}\n({len(edges)} edges, {len(triangles)} triangles)")

    fig.suptitle("Feature Interaction Graphs (1-skeleton at gap threshold)", fontsize=14)
    fig.tight_layout()
    return fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════

def build_method_out(
    datasets: dict,
    dcor_matrices: dict,
    triangle_results: dict,
    rips_results: dict,
    interaction_recovery: dict,
    clique_inflation: dict,
    hsic_comparison: dict,
    real_dataset_analysis: dict,
    baseline_results: dict,
    enhanced_results: dict,
    figures: dict,
) -> dict:
    """Assemble the exp_gen_sol_out.json-format output."""
    # Determine key findings
    # Triangle inequality recommendation
    total_raw_violations = sum(
        triangle_results[d]["raw"]["n_violations"] for d in triangle_results
    )
    total_sqrt_violations = sum(
        triangle_results[d]["sqrt"]["n_violations"] for d in triangle_results
    )
    tri_rec = "sqrt(1-dCor)" if total_sqrt_violations < total_raw_violations else "1-dCor"

    # Gap heuristic reliability
    gap_f1s = [
        interaction_recovery[d]["f1_at_gap"]
        for d in DATASETS_WITH_GROUND_TRUTH if d in interaction_recovery
    ]
    mean_f1_gap = float(np.mean(gap_f1s)) if gap_f1s else 0.0
    gap_reliability = "reliable" if mean_f1_gap >= 0.3 else "unreliable"

    # Clique inflation severity
    inflation_rates = []
    for d in DATASETS_WITH_GROUND_TRUTH:
        if d in clique_inflation:
            for dim_info in clique_inflation[d].values():
                if dim_info["total_simplices"] > 0:
                    inflation_rates.append(dim_info["inflation_rate"])
    mean_inflation = float(np.mean(inflation_rates)) if inflation_rates else 0.0
    if mean_inflation < 0.1:
        inflation_severity = "low"
    elif mean_inflation < 0.3:
        inflation_severity = "moderate"
    else:
        inflation_severity = "high"

    # HSIC verdict
    hsic_verdict = hsic_comparison.get("conclusion", "not_evaluated")

    # Build method metadata
    metadata = {
        "experiment_id": "experiment_iter2_dir3",
        "method_name": "SC-OTS Simplicial Complex Construction Pipeline",
        "description": (
            "Validates persistent-homology-based feature interaction discovery using "
            "distance correlation dissimilarity, Rips filtration, and persistence threshold selection. "
            "Compares against Pearson correlation baseline and HSIC alternative."
        ),
        "summary": {
            "total_datasets": len(datasets),
            "synthetic_with_gt": len(DATASETS_WITH_GROUND_TRUTH),
            "real_datasets": len(REAL_DATASETS),
        },
        "step2_triangle_inequality": {
            ds: {
                "raw_violations": triangle_results[ds]["raw"]["n_violations"],
                "raw_violation_rate": triangle_results[ds]["raw"]["violation_rate"],
                "sqrt_violations": triangle_results[ds]["sqrt"]["n_violations"],
                "sqrt_violation_rate": triangle_results[ds]["sqrt"]["violation_rate"],
            }
            for ds in triangle_results
        },
        "step3_rips_filtration_summary": {
            ds: {
                "betti_numbers": rips_results[ds]["betti_numbers"],
                "gap_threshold": rips_results[ds]["gap_threshold"],
                "dissimilarity_used": rips_results[ds]["dissimilarity_used"],
            }
            for ds in rips_results
        },
        "step4_interaction_recovery": {
            ds: {
                "f1_at_optimal": interaction_recovery[ds]["f1_at_optimal"],
                "f1_at_gap": interaction_recovery[ds]["f1_at_gap"],
                "optimal_threshold": interaction_recovery[ds]["optimal_threshold"],
                "gap_threshold": interaction_recovery[ds]["gap_threshold"],
            }
            for ds in interaction_recovery
        },
        "step5_clique_inflation": clique_inflation,
        "step6_hsic_comparison": {
            k: v for k, v in hsic_comparison.items()
            if k not in ("hsic_matrix", "hsic_metrics_by_threshold")
        },
        "step7_real_datasets": {
            ds: {
                "simplex_counts": real_dataset_analysis[ds]["simplex_counts_by_dim"],
                "betti_numbers": real_dataset_analysis[ds]["betti_numbers"],
                "participation_rate": real_dataset_analysis[ds]["feature_participation_rate"],
                "top_edges": real_dataset_analysis[ds]["top_edges"][:5],
            }
            for ds in real_dataset_analysis
        },
        "baseline_pearson": baseline_results,
        "enhanced_interaction_dcor": enhanced_results,
        "figures": figures,
        "key_findings": {
            "triangle_inequality_recommendation": tri_rec,
            "gap_heuristic_reliability": gap_reliability,
            "mean_f1_at_gap_synthetic": round(mean_f1_gap, 4),
            "clique_inflation_severity": inflation_severity,
            "hsic_vs_dcor_verdict": hsic_verdict,
            "total_raw_triangle_violations": total_raw_violations,
            "total_sqrt_triangle_violations": total_sqrt_violations,
            "enhanced_interaction_dcor_f1s": {
                ds: enhanced_results[ds].get("f1_at_optimal", 0.0)
                for ds in enhanced_results
            },
        },
    }

    # Build per-dataset examples in exp_gen_sol_out.json format
    output_datasets = []
    all_dataset_names = sorted(datasets.keys())

    for ds_name in all_dataset_names:
        ds = datasets[ds_name]
        n_examples = ds["n_samples"]

        examples = []
        for i in range(n_examples):
            # Input: feature values as JSON string
            input_vals = [round(float(v), 6) for v in ds["X"][i]]
            input_str = json.dumps(input_vals)

            # Output: target value
            if ds["task_type"] == "regression":
                output_str = str(round(float(ds["y"][i]), 6))
            else:
                output_str = str(int(ds["y"][i]))

            example = {
                "input": input_str,
                "output": output_str,
            }

            # predict_our_method: dCor-based Rips interaction summary
            if ds_name in rips_results:
                rips = rips_results[ds_name]
                gap_t = rips["gap_threshold"]
                simplices = rips["simplices_at_threshold"][gap_t]
                # Summarize interactions as string
                edges = sorted([sorted(list(s)) for s in simplices if len(s) == 2])
                triangles = sorted([sorted(list(s)) for s in simplices if len(s) == 3])
                predict_str = json.dumps({
                    "method": "dCor_Rips",
                    "gap_threshold": round(gap_t, 4),
                    "edges": edges[:20],  # cap for size
                    "triangles": triangles[:10],
                    "n_edges": len(edges),
                    "n_triangles": len(triangles),
                })
                example["predict_our_method"] = predict_str

            # predict_baseline: Pearson-based summary
            if ds_name in DATASETS_WITH_GROUND_TRUTH and ds_name in baseline_results:
                baseline = baseline_results[ds_name]
                predict_baseline_str = json.dumps({
                    "method": "Pearson_Rips",
                    "f1_at_gap": baseline["f1_at_gap"],
                    "f1_at_optimal": baseline["f1_at_optimal"],
                })
                example["predict_baseline"] = predict_baseline_str

            # predict_enhanced: Feature-target interaction dCor-based Rips
            if ds_name in DATASETS_WITH_GROUND_TRUTH and ds_name in enhanced_results:
                enh = enhanced_results[ds_name]
                predict_enhanced_str = json.dumps({
                    "method": "Interaction_dCor_Rips",
                    "f1_at_gap": enh.get("f1_at_gap", 0.0),
                    "f1_at_optimal": enh.get("f1_at_optimal", 0.0),
                })
                example["predict_enhanced"] = predict_enhanced_str

            # metadata fields
            example["metadata_dataset"] = ds_name
            example["metadata_task_type"] = ds["task_type"]
            example["metadata_n_features"] = ds["n_features"]

            examples.append(example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    return {
        "metadata": metadata,
        "datasets": output_datasets,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    t_start = time.time()
    logger.info("SC-OTS Simplicial Complex Construction Pipeline Validation")
    logger.info("=" * 60)

    # ── Step 0: Load data ──
    datasets = load_all_datasets(DATA_FILE)

    # ── Step 1: dCor matrices ──
    matrices = step1_dcor_matrices(datasets)
    dcor_matrices = matrices["dcor"]
    pearson_matrices = matrices["pearson"]

    # ── Step 1b: Feature-target interaction dCor (enhanced) ──
    interaction_data = step1b_interaction_dcor(datasets)
    enhanced_results = step1b_enhanced_rips(
        interaction_matrices=interaction_data["interaction_matrices"],
        individual_dcors=interaction_data["individual_dcors"],
    )

    # ── Step 2: Triangle inequality ──
    triangle_results = step2_triangle_inequality(dcor_matrices)

    # ── Step 3: Rips filtrations ──
    rips_results = step3_rips_filtrations(dcor_matrices, triangle_results)

    # ── Step 4: Interaction recovery ──
    interaction_recovery = step4_interaction_recovery(rips_results)

    # ── Step 5: Clique inflation ──
    clique_inflation = step5_clique_inflation(rips_results)

    # ── Step 6: HSIC comparison ──
    hsic_comparison = step6_hsic_comparison(
        datasets=datasets,
        dcor_matrices=dcor_matrices,
        rips_results=rips_results,
        interaction_recovery=interaction_recovery,
    )

    # ── Step 7: Real dataset characterization ──
    real_dataset_analysis = step7_real_datasets(
        datasets=datasets,
        rips_results=rips_results,
        dcor_matrices=dcor_matrices,
    )

    # ── Baseline: Pearson pipeline ──
    baseline_results = baseline_pearson_pipeline(
        datasets=datasets,
        pearson_matrices=pearson_matrices,
    )

    # ── Step 8: Visualization ──
    logger.info("=" * 60)
    logger.info("STEP 8: Generating Figures")
    logger.info("=" * 60)

    figures = {}
    try:
        figures["f1_vs_threshold"] = figure_f1_vs_threshold(interaction_recovery)
        logger.info("  Figure 1 (F1 vs Threshold): OK")
    except Exception:
        logger.exception("Figure 1 failed")
        figures["f1_vs_threshold"] = ""

    try:
        figures["persistence_diagrams"] = figure_persistence_diagrams(rips_results)
        logger.info("  Figure 2 (Persistence Diagrams): OK")
    except Exception:
        logger.exception("Figure 2 failed")
        figures["persistence_diagrams"] = ""

    try:
        figures["triangle_violations"] = figure_triangle_violations(triangle_results)
        logger.info("  Figure 3 (Triangle Violations): OK")
    except Exception:
        logger.exception("Figure 3 failed")
        figures["triangle_violations"] = ""

    try:
        figures["clique_inflation"] = figure_clique_inflation(clique_inflation)
        logger.info("  Figure 4 (Clique Inflation): OK")
    except Exception:
        logger.exception("Figure 4 failed")
        figures["clique_inflation"] = ""

    try:
        figures["dcor_vs_hsic"] = figure_dcor_vs_hsic(interaction_recovery, hsic_comparison)
        logger.info("  Figure 5 (dCor vs HSIC): OK")
    except Exception:
        logger.exception("Figure 5 failed")
        figures["dcor_vs_hsic"] = ""

    try:
        figures["interaction_graphs"] = figure_interaction_graphs(datasets, dcor_matrices, rips_results)
        logger.info("  Figure 6 (Interaction Graphs): OK")
    except Exception:
        logger.exception("Figure 6 failed")
        figures["interaction_graphs"] = ""

    # ── Assemble output ──
    logger.info("=" * 60)
    logger.info("Assembling method_out.json")
    logger.info("=" * 60)

    output = build_method_out(
        datasets=datasets,
        dcor_matrices=dcor_matrices,
        triangle_results=triangle_results,
        rips_results=rips_results,
        interaction_recovery=interaction_recovery,
        clique_inflation=clique_inflation,
        hsic_comparison=hsic_comparison,
        real_dataset_analysis=real_dataset_analysis,
        baseline_results=baseline_results,
        enhanced_results=enhanced_results,
        figures=figures,
    )

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Written: {OUTPUT_FILE} ({size_mb:.1f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
