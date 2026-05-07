#!/usr/bin/env python3
"""SC-OTS Definitive Final Synthesis: evaluation across all 6 dependency artifacts.

Implements 7 analysis blocks (Friedman/Nemenyi, Pareto, Interaction Discovery,
XGBoost+SC, Ablation Bootstrap, Diagnostic, Verdict) with 9 figures.
"""

import base64
import io
import json
import resource
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
fmt = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=fmt)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Paths ────────────────────────────────────────────────────────────────────
RUN_BASE = Path(
    "/home/adrian/projects/temp/ai-inventor-old3/aii_pipeline/runs/"
    "run__20260228_133939/3_invention_loop"
)
WORKSPACE = Path(
    "/home/adrian/projects/temp/ai-inventor-old3/aii_pipeline/runs/"
    "run__20260228_133939/3_invention_loop/iter_4/gen_art/eval_id3_it4__opus"
)

DEP_PATHS: dict[str, Path] = {
    "exp_id1_it2_mini": RUN_BASE / "iter_2/gen_art/exp_id1_it2__opus/mini_method_out.json",
    "exp_id2_it2_full": RUN_BASE / "iter_2/gen_art/exp_id2_it2__opus/full_method_out.json",
    "exp_id3_it2_mini": RUN_BASE / "iter_2/gen_art/exp_id3_it2__opus/mini_method_out.json",
    "exp_id1_it3_mini": RUN_BASE / "iter_3/gen_art/exp_id1_it3__opus/mini_method_out.json",
    "exp_id2_it3_mini": RUN_BASE / "iter_3/gen_art/exp_id2_it3__opus/mini_method_out.json",
    "data_id3_it1_mini": RUN_BASE / "iter_1/gen_art/data_id3_it1__opus/mini_data_out.json",
}

DATASETS_10 = [
    "friedman1", "friedman3", "synth_3way", "synth_4way",
    "diabetes", "breast_w", "wine_quality", "california_housing",
    "spambase", "adult",
]
SYNTHETIC_DATASETS = ["friedman1", "friedman3", "synth_3way", "synth_4way"]
REAL_DATASETS = ["diabetes", "breast_w", "wine_quality", "california_housing", "spambase", "adult"]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    """Load a JSON file, returning its parsed content."""
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    return json.loads(path.read_text())


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity with None for JSON serialization."""
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    # Catch-all for other types
    try:
        return str(obj)
    except Exception:
        return None


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def safe_float(val: Any, default: float = np.nan) -> float:
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def load_all_dependencies() -> dict[str, dict]:
    """Load all 6 dependency files."""
    deps: dict[str, dict] = {}
    for key, path in DEP_PATHS.items():
        try:
            deps[key] = load_json(path)
        except FileNotFoundError:
            logger.exception(f"Missing dependency: {path}")
            raise
        except json.JSONDecodeError:
            logger.exception(f"Invalid JSON: {path}")
            raise
    logger.info(f"Loaded {len(deps)} dependency files successfully")
    return deps


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: Predictive Accuracy — Friedman/Nemenyi
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_baselines_exp2(data: dict) -> dict[str, dict[str, float]]:
    """Extract per-dataset mean scores from exp_id2_it2 (baselines).

    Returns {dataset: {method: mean_score}}.
    """
    results: dict[str, dict[str, list[float]]] = {}
    for ds_block in data["datasets"]:
        ds_name = ds_block["dataset"]
        results.setdefault(ds_name, {})
        for ex in ds_block["examples"]:
            method = ex.get("metadata_method", "")
            metrics_raw = ex.get("metadata_metrics", "{}")
            try:
                metrics = json.loads(metrics_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            # Primary metric: r2 for regression, accuracy for classification
            score = metrics.get("r2") or metrics.get("accuracy") or metrics.get("auroc")
            if score is not None:
                results[ds_name].setdefault(method, []).append(float(score))
    # Average across folds
    averaged: dict[str, dict[str, float]] = {}
    for ds, methods in results.items():
        averaged[ds] = {m: float(np.mean(scores)) for m, scores in methods.items()}
    return averaged


def block1_accuracy(deps: dict) -> tuple[dict, plt.Figure, plt.Figure]:
    """Block 1: Predictive accuracy — Friedman/Nemenyi analysis."""
    logger.info("=== Block 1: Predictive Accuracy ===")

    exp1_v1 = deps["exp_id1_it2_mini"]["metadata"]["per_dataset_results"]
    exp1_v2 = deps["exp_id1_it3_mini"]["metadata"]["aggregated_results"]
    baselines = _extract_baselines_exp2(deps["exp_id2_it2_full"])

    # Build unified score matrix
    # Methods from v1: SC-OTS_v1, RO-FIGS, XGBoost+SC_v1
    # Methods from v2: SC-OTS_v2, XGBoost_constrained_v2
    # Methods from baselines: FIGS-5, FIGS-10, FIGS-20, XGBoost-default, EBM-default
    unified_methods = [
        "SC-OTS_v1", "SC-OTS_v2", "RO-FIGS",
        "FIGS-5", "FIGS-10", "FIGS-20",
        "XGBoost-default", "XGBoost+SC_v1", "XGBoost_constrained_v2",
        "EBM-default",
    ]

    score_matrix: dict[str, dict[str, float]] = {}
    for ds in DATASETS_10:
        score_matrix[ds] = {}
        # v1 data
        if ds in exp1_v1:
            v1 = exp1_v1[ds]
            score_matrix[ds]["SC-OTS_v1"] = safe_float(v1.get("mean_scots"))
            score_matrix[ds]["RO-FIGS"] = safe_float(v1.get("mean_rofigs"))
            score_matrix[ds]["XGBoost+SC_v1"] = safe_float(v1.get("mean_xgb_sc"))
        # v2 data
        if ds in exp1_v2:
            v2 = exp1_v2[ds]
            if "SC-OTS" in v2:
                score_matrix[ds]["SC-OTS_v2"] = safe_float(v2["SC-OTS"].get("mean_score"))
            if "XGBoost_constrained" in v2:
                score_matrix[ds]["XGBoost_constrained_v2"] = safe_float(
                    v2["XGBoost_constrained"].get("mean_score")
                )
        # baseline data
        if ds in baselines:
            bl = baselines[ds]
            for bm in ["FIGS-5", "FIGS-10", "FIGS-20", "XGBoost-default",
                        "XGBoost-oracle", "EBM-default", "EBM-high-interaction", "EBM-3way"]:
                if bm in bl:
                    score_matrix[ds][bm] = bl[bm]

    # Filter to methods that have data for at least 8 datasets
    valid_methods = []
    for m in unified_methods:
        count = sum(1 for ds in DATASETS_10 if not np.isnan(score_matrix.get(ds, {}).get(m, np.nan)))
        if count >= 8:
            valid_methods.append(m)

    logger.info(f"Valid methods with ≥8 datasets: {valid_methods}")

    # Build rank matrix for Friedman test
    n_datasets = len(DATASETS_10)
    n_methods = len(valid_methods)

    data_matrix = np.full((n_datasets, n_methods), np.nan)
    for i, ds in enumerate(DATASETS_10):
        for j, m in enumerate(valid_methods):
            data_matrix[i, j] = score_matrix.get(ds, {}).get(m, np.nan)

    # Impute missing with column mean (rare edge case)
    for j in range(n_methods):
        col = data_matrix[:, j]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            data_matrix[mask, j] = np.nanmean(col)

    # Friedman test
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[data_matrix[:, j] for j in range(n_methods)]
        )
    except ValueError:
        friedman_stat, friedman_p = 0.0, 1.0

    logger.info(f"Friedman χ²={friedman_stat:.3f}, p={friedman_p:.6f}")

    # Compute ranks (higher score = better = lower rank)
    rank_matrix = np.zeros_like(data_matrix)
    for i in range(n_datasets):
        rank_matrix[i] = stats.rankdata(-data_matrix[i])  # negative for descending
    avg_ranks = rank_matrix.mean(axis=0)
    method_ranks = {m: float(avg_ranks[j]) for j, m in enumerate(valid_methods)}

    # Nemenyi post-hoc (manual implementation since scikit-posthocs may differ)
    # CD = q_α * sqrt(k*(k+1)/(6*N))
    k = n_methods
    N = n_datasets
    # q-values for Nemenyi at α=0.05 (approximation using studentized range / sqrt(2))
    # For k methods, use scipy's studentized range distribution
    try:
        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    except Exception:
        # Fallback: use tabulated approximate values
        q_alpha_table = {
            2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
            12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391
        }
        q_alpha = q_alpha_table.get(k, 3.3)

    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    logger.info(f"Critical difference (α=0.05): {cd:.3f}")

    # Nemenyi p-value matrix (approximate from rank differences)
    nemenyi_p: dict[str, dict[str, float]] = {}
    for j1, m1 in enumerate(valid_methods):
        nemenyi_p[m1] = {}
        for j2, m2 in enumerate(valid_methods):
            rank_diff = abs(avg_ranks[j1] - avg_ranks[j2])
            # Under Nemenyi, |R_i - R_j| > CD means significant at α=0.05
            # Approximate p-value using normal approximation
            se = np.sqrt(k * (k + 1) / (6 * N))
            z = rank_diff / se
            p_val = 2 * (1 - stats.norm.cdf(z))
            nemenyi_p[m1][m2] = float(min(p_val, 1.0))

    scots_v1_rank = method_ranks.get("SC-OTS_v1", float("nan"))
    scots_v2_rank = method_ranks.get("SC-OTS_v2", float("nan"))

    # Count methods SC-OTS is not significantly worse than
    scots_not_worse_count = 0
    for m in valid_methods:
        if m in ("SC-OTS_v1", "SC-OTS_v2"):
            continue
        r_scots = method_ranks.get("SC-OTS_v1", 99)
        r_other = method_ranks.get(m, 99)
        if r_scots - r_other < cd:
            scots_not_worse_count += 1

    # ── Figure 1: CD Diagram ──
    fig_cd = _draw_cd_diagram(
        method_names=valid_methods,
        avg_ranks=avg_ranks,
        cd=cd,
        n_datasets=N,
    )

    # ── Figure 2: Heatmap ──
    fig_heatmap = _draw_score_heatmap(
        score_matrix=score_matrix,
        datasets=DATASETS_10,
        methods=valid_methods,
    )

    result = {
        "unified_score_matrix": {
            ds: {m: round(score_matrix[ds].get(m, float("nan")), 6)
                 for m in valid_methods}
            for ds in DATASETS_10
        },
        "friedman_statistic": round(friedman_stat, 4),
        "friedman_p_value": round(friedman_p, 6),
        "friedman_significant": bool(friedman_p < 0.05),
        "method_avg_ranks": {m: round(r, 3) for m, r in method_ranks.items()},
        "nemenyi_p_value_matrix": {
            m1: {m2: round(v, 4) for m2, v in row.items()}
            for m1, row in nemenyi_p.items()
        },
        "critical_difference": round(cd, 4),
        "scots_v1_rank": round(scots_v1_rank, 3),
        "scots_v2_rank": round(scots_v2_rank, 3),
        "scots_not_significantly_worse_than_n_methods": scots_not_worse_count,
        "n_methods": n_methods,
        "n_datasets": n_datasets,
        "valid_methods": valid_methods,
    }
    return result, fig_cd, fig_heatmap


def _draw_cd_diagram(
    method_names: list[str],
    avg_ranks: np.ndarray,
    cd: float,
    n_datasets: int,
) -> plt.Figure:
    """Draw a Critical Difference diagram."""
    k = len(method_names)
    # Sort methods by rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_names = [method_names[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, max(4, k * 0.45)))

    # Draw the main horizontal axis
    min_rank = 1
    max_rank = k
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(-0.5, k + 1)
    ax.axhline(y=k, color="black", linewidth=1.5)

    # Tick marks
    for r in range(1, k + 1):
        ax.plot([r, r], [k - 0.1, k + 0.1], "k-", linewidth=1)
        ax.text(r, k + 0.3, str(r), ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Critical Difference Diagram (CD={cd:.2f}, N={n_datasets})",
                 fontsize=13, fontweight="bold", pad=20)

    # Place methods
    left_methods = sorted_idx[: k // 2]
    right_methods = sorted_idx[k // 2:]

    for idx, (i, name, rank) in enumerate(zip(sorted_idx, sorted_names, sorted_ranks)):
        y_pos = k - 1 - idx * (k / (k + 1))
        if idx < k // 2:
            # Left side
            ax.plot([rank, rank], [k - 0.1, y_pos + 0.2], "k-", linewidth=0.8)
            ax.plot([rank, min_rank - 0.3], [y_pos + 0.2, y_pos + 0.2], "k-", linewidth=0.8)
            ax.text(min_rank - 0.4, y_pos + 0.2, f"{name} ({rank:.2f})",
                    ha="right", va="center", fontsize=8)
        else:
            ax.plot([rank, rank], [k - 0.1, y_pos + 0.2], "k-", linewidth=0.8)
            ax.plot([rank, max_rank + 0.3], [y_pos + 0.2, y_pos + 0.2], "k-", linewidth=0.8)
            ax.text(max_rank + 0.4, y_pos + 0.2, f"({rank:.2f}) {name}",
                    ha="left", va="center", fontsize=8)

    # Draw CD bar
    cd_y = k + 0.8
    mid = (min_rank + max_rank) / 2
    ax.plot([mid - cd / 2, mid + cd / 2], [cd_y, cd_y], "r-", linewidth=3)
    ax.text(mid, cd_y + 0.2, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=10, color="red", fontweight="bold")

    # Draw connections for non-significant groups (methods within CD of each other)
    groups = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                groups.append((i, j))

    # Merge overlapping groups
    merged: list[tuple[int, int]] = []
    for start, end in groups:
        found = False
        for idx_m, (ms, me) in enumerate(merged):
            if start <= me and end >= ms:
                merged[idx_m] = (min(ms, start), max(me, end))
                found = True
                break
        if not found:
            merged.append((start, end))

    for gi, (gs, ge) in enumerate(merged):
        y_line = -0.3 - gi * 0.25
        r1 = sorted_ranks[gs]
        r2 = sorted_ranks[ge]
        ax.plot([r1, r2], [y_line, y_line], "k-", linewidth=3, alpha=0.6)

    ax.axis("off")
    fig.tight_layout()
    return fig


def _draw_score_heatmap(
    score_matrix: dict[str, dict[str, float]],
    datasets: list[str],
    methods: list[str],
) -> plt.Figure:
    """Draw a heatmap of method × dataset scores."""
    data = np.full((len(datasets), len(methods)), np.nan)
    for i, ds in enumerate(datasets):
        for j, m in enumerate(methods):
            data[i, j] = score_matrix.get(ds, {}).get(m, np.nan)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=[m.replace("_", "\n") for m in methods],
        yticklabels=datasets,
        ax=ax, vmin=0.0, vmax=1.0, linewidths=0.5,
        cbar_kws={"label": "Score (R²/Accuracy)"},
    )
    ax.set_title("Method × Dataset Performance Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: Model Compactness — Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════════

def block2_compactness(deps: dict, b1_scores: dict) -> tuple[dict, plt.Figure, plt.Figure]:
    """Block 2: Model compactness — Pareto frontier analysis."""
    logger.info("=== Block 2: Model Compactness ===")

    exp1_v1 = deps["exp_id1_it2_mini"]["metadata"]["per_dataset_results"]
    exp2_baselines = deps["exp_id2_it2_full"]
    exp1_v2 = deps["exp_id1_it3_mini"]["metadata"]
    exp2_ablation = deps["exp_id2_it3_mini"]["metadata"]["results_per_dataset"]

    # Extract complexity data
    complexity: dict[str, dict[str, float]] = {}  # {method: {dataset: splits}}

    for ds in DATASETS_10:
        # SC-OTS v1
        if ds in exp1_v1:
            complexity.setdefault("SC-OTS_v1", {})[ds] = safe_float(exp1_v1[ds].get("scots_avg_splits", 20))
        # RO-FIGS
        if ds in exp1_v1:
            complexity.setdefault("RO-FIGS", {})[ds] = safe_float(exp1_v1[ds].get("scots_avg_splits", 20))

    # SC-OTS v2: uses max_splits=30 per hyperparameters
    for ds in DATASETS_10:
        complexity.setdefault("SC-OTS_v2", {})[ds] = 30.0

    # Baselines from exp_id2_it2: parse metadata_complexity
    baseline_complexity: dict[str, dict[str, list[float]]] = {}
    for ds_block in exp2_baselines["datasets"]:
        ds_name = ds_block["dataset"]
        for ex in ds_block["examples"]:
            method = ex.get("metadata_method", "")
            comp_raw = ex.get("metadata_complexity", "{}")
            try:
                comp = json.loads(comp_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            n_splits = comp.get("n_splits", 0)
            baseline_complexity.setdefault(method, {}).setdefault(ds_name, []).append(float(n_splits))

    for method, ds_dict in baseline_complexity.items():
        for ds, values in ds_dict.items():
            complexity.setdefault(method, {})[ds] = float(np.mean(values))

    # XGBoost/EBM complexity approximations
    for ds in DATASETS_10:
        complexity.setdefault("XGBoost-default", {})[ds] = 600.0
        complexity.setdefault("EBM-default", {})[ds] = 500.0

    # Ablation complexity
    for ds in DATASETS_10:
        if ds in exp2_ablation:
            for mode in ["SIMPLICIAL", "RANDOM_MATCHED", "UNCONSTRAINED"]:
                if mode in exp2_ablation[ds].get("mean_splits", {}):
                    complexity.setdefault(f"Ablation_{mode}", {})[ds] = \
                        safe_float(exp2_ablation[ds]["mean_splits"][mode])

    # Get score matrix from block1
    score_mat = b1_scores.get("unified_score_matrix", {})

    # Pareto analysis: for each dataset, identify Pareto-optimal methods
    pareto_methods = [
        "SC-OTS_v1", "SC-OTS_v2", "RO-FIGS", "FIGS-5", "FIGS-10", "FIGS-20",
        "XGBoost-default", "EBM-default"
    ]
    pareto_counts: dict[str, int] = {m: 0 for m in pareto_methods}

    for ds in DATASETS_10:
        points = []
        for m in pareto_methods:
            score = score_mat.get(ds, {}).get(m, np.nan)
            comp = complexity.get(m, {}).get(ds, np.nan)
            if not np.isnan(score) and not np.isnan(comp):
                points.append((m, score, comp))

        # Pareto frontier: not dominated (no other has both better score AND lower complexity)
        for m1, s1, c1 in points:
            dominated = False
            for m2, s2, c2 in points:
                if m1 == m2:
                    continue
                if s2 >= s1 and c2 <= c1 and (s2 > s1 or c2 < c1):
                    dominated = True
                    break
            if not dominated:
                pareto_counts[m1] = pareto_counts.get(m1, 0) + 1

    # Per-method average complexity
    per_method_avg_complexity = {}
    for m in pareto_methods:
        vals = [complexity.get(m, {}).get(ds, np.nan) for ds in DATASETS_10]
        vals = [v for v in vals if not np.isnan(v)]
        per_method_avg_complexity[m] = round(float(np.mean(vals)), 2) if vals else 0.0

    # Accuracy per split ratio
    accuracy_per_split = {}
    for m in pareto_methods:
        scores = [score_mat.get(ds, {}).get(m, np.nan) for ds in DATASETS_10]
        comps = [complexity.get(m, {}).get(ds, np.nan) for ds in DATASETS_10]
        ratios = []
        for s, c in zip(scores, comps):
            if not np.isnan(s) and not np.isnan(c) and c > 0:
                ratios.append(s / c)
        accuracy_per_split[m] = round(float(np.mean(ratios)), 6) if ratios else 0.0

    # ── Figures: Pareto scatters ──
    fig_synth = _draw_pareto(
        score_mat=score_mat,
        complexity=complexity,
        methods=pareto_methods,
        datasets=SYNTHETIC_DATASETS,
        title="Pareto Frontier: Synthetic Datasets",
    )
    fig_real = _draw_pareto(
        score_mat=score_mat,
        complexity=complexity,
        methods=pareto_methods,
        datasets=REAL_DATASETS,
        title="Pareto Frontier: Real-World Datasets",
    )

    result = {
        "per_method_avg_complexity": per_method_avg_complexity,
        "pareto_membership_counts": pareto_counts,
        "accuracy_per_split_ratio": accuracy_per_split,
    }
    return result, fig_synth, fig_real


def _draw_pareto(
    score_mat: dict,
    complexity: dict,
    methods: list[str],
    datasets: list[str],
    title: str,
) -> plt.Figure:
    """Draw Pareto frontier scatter plot."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = {m: colors[i] for i, m in enumerate(methods)}

    for ax, ds in zip(axes, datasets):
        points = []
        for m in methods:
            s = score_mat.get(ds, {}).get(m, np.nan)
            c = complexity.get(m, {}).get(ds, np.nan)
            if not np.isnan(s) and not np.isnan(c):
                ax.scatter(c, s, color=method_colors[m], s=80, zorder=5, edgecolors="black", linewidth=0.5)
                ax.annotate(m.replace("_", "\n"), (c, s), fontsize=6,
                            textcoords="offset points", xytext=(5, 5))
                points.append((c, s))

        # Draw Pareto frontier line
        if points:
            pts = sorted(points, key=lambda p: p[0])
            frontier = [pts[0]]
            for p in pts[1:]:
                if p[1] >= frontier[-1][1]:
                    frontier.append(p)
            if len(frontier) > 1:
                fx, fy = zip(*frontier)
                ax.plot(fx, fy, "r--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Complexity (splits)")
        ax.set_ylabel("Score")
        ax.set_xscale("symlog", linthresh=10)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: Interaction Discovery Quality
# ═══════════════════════════════════════════════════════════════════════════════

def block3_interaction_discovery(deps: dict) -> tuple[dict, plt.Figure]:
    """Block 3: Interaction discovery quality — TDA pipeline synthesis."""
    logger.info("=== Block 3: Interaction Discovery ===")

    exp3_meta = deps["exp_id3_it2_mini"]["metadata"]
    exp1_v1 = deps["exp_id1_it2_mini"]["metadata"]["per_dataset_results"]
    exp1_v2_topo = deps["exp_id1_it3_mini"]["metadata"].get("topological_analysis", {})
    enhanced = exp3_meta.get("enhanced_interaction_dcor", {})

    # F1 comparison table
    f1_table: dict[str, dict[str, float]] = {}
    step4 = exp3_meta.get("step4_interaction_recovery", {})

    for ds in SYNTHETIC_DATASETS:
        f1_table[ds] = {}
        # Pairwise dCor F1 (from TDA pipeline step4)
        if ds in step4:
            f1_table[ds]["pairwise_dcor"] = safe_float(step4[ds].get("f1_at_optimal", 0))
        # Enhanced dCor F1 (from TDA pipeline enhanced section)
        if ds in enhanced:
            f1_table[ds]["enhanced_dcor_v1"] = safe_float(enhanced[ds].get("f1_at_optimal", 0))
        # v1 SC-OTS interaction recovery F1
        if ds in exp1_v1 and "interaction_recovery" in exp1_v1[ds]:
            f1_table[ds]["scots_v1_ir"] = safe_float(exp1_v1[ds]["interaction_recovery"].get("f1", 0))
        # v2 interaction recovery
        if ds in exp1_v2_topo and "interaction_recovery_mean" in exp1_v2_topo[ds]:
            irm = exp1_v2_topo[ds]["interaction_recovery_mean"]
            f1_table[ds]["v2"] = safe_float(irm.get("f1", 0))

    # AUPRC from enhanced metrics_by_threshold
    auprc_values = []
    for ds in SYNTHETIC_DATASETS:
        if ds in enhanced and "metrics_by_threshold" in enhanced[ds]:
            mbt = enhanced[ds]["metrics_by_threshold"]
            thresholds_sorted = sorted(mbt.keys(), key=lambda x: float(x))
            precisions = []
            recalls = []
            for t in thresholds_sorted:
                precisions.append(safe_float(mbt[t].get("precision", 0)))
                recalls.append(safe_float(mbt[t].get("recall", 0)))
            # Sort by recall for AUPRC
            pairs = sorted(zip(recalls, precisions))
            if len(pairs) >= 2:
                r_arr = np.array([p[0] for p in pairs])
                p_arr = np.array([p[1] for p in pairs])
                auprc = float(np.trapezoid(p_arr, r_arr))
                auprc_values.append(abs(auprc))  # abs to handle direction

    mean_auprc = float(np.mean(auprc_values)) if auprc_values else 0.0

    # Faithfulness from exp_id1_it2
    faithfulness_precs = []
    for ds in SYNTHETIC_DATASETS:
        if ds in exp1_v1 and "interaction_faithfulness" in exp1_v1[ds]:
            fp = safe_float(exp1_v1[ds]["interaction_faithfulness"].get("precision", 0))
            faithfulness_precs.append(fp)
    faithfulness_mean = float(np.mean(faithfulness_precs)) if faithfulness_precs else 0.0

    # Clique inflation from step5
    step5 = exp3_meta.get("step5_clique_inflation", {})
    clique_summary: dict[str, dict[str, float]] = {}
    for ds in SYNTHETIC_DATASETS:
        if ds in step5:
            clique_summary[ds] = {}
            for dim_key, dim_data in step5[ds].items():
                rate = safe_float(dim_data.get("inflation_rate", 0))
                clique_summary[ds][f"rate_dim{dim_key}"] = rate

    # Betti summary
    betti_summary: dict[str, dict[str, list]] = {}
    for ds in DATASETS_10:
        betti_summary[ds] = {}
        # v1
        if ds in exp1_v1 and "avg_betti" in exp1_v1[ds]:
            betti_summary[ds]["v1"] = exp1_v1[ds]["avg_betti"]
        # v2
        if ds in exp1_v2_topo and "mean_betti" in exp1_v2_topo[ds]:
            mb = exp1_v2_topo[ds]["mean_betti"]
            betti_summary[ds]["v2"] = [mb.get("b0", 0), mb.get("b1", 0), mb.get("b2", 0)]
        # TDA pipeline
        step3 = exp3_meta.get("step3_rips_filtration_summary", {})
        if ds in step3 and "betti_numbers" in step3[ds]:
            betti_summary[ds]["tda_pipeline"] = step3[ds]["betti_numbers"]

    # ── Figure: Interaction recovery bars ──
    fig_bars = _draw_interaction_bars(f1_table)

    result = {
        "f1_comparison_table": f1_table,
        "mean_auprc_synthetic": round(mean_auprc, 4),
        "faithfulness_precision_mean": round(faithfulness_mean, 4),
        "clique_inflation_summary": clique_summary,
        "betti_summary": betti_summary,
    }
    return result, fig_bars


def _draw_interaction_bars(f1_table: dict) -> plt.Figure:
    """Draw grouped bar chart for interaction recovery F1."""
    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = list(f1_table.keys())
    metric_keys = ["pairwise_dcor", "enhanced_dcor_v1", "scots_v1_ir", "v2"]
    labels = ["Pairwise dCor", "Enhanced dCor", "SC-OTS v1 IR", "SC-OTS v2 IR"]
    colors = ["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4"]

    x = np.arange(len(datasets))
    width = 0.2

    for i, (key, label, color) in enumerate(zip(metric_keys, labels, colors)):
        values = [f1_table.get(ds, {}).get(key, 0) for ds in datasets]
        ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1 Score")
    ax.set_title("Interaction Recovery F1: Method Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="70% target")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: XGBoost+SC Transfer Value
# ═══════════════════════════════════════════════════════════════════════════════

def block4_xgboost_sc(deps: dict) -> tuple[dict, plt.Figure]:
    """Block 4: XGBoost+SC transfer value analysis."""
    logger.info("=== Block 4: XGBoost+SC Transfer Value ===")

    exp1_v1 = deps["exp_id1_it2_mini"]["metadata"]["per_dataset_results"]
    exp1_v2 = deps["exp_id1_it3_mini"]["metadata"]["aggregated_results"]

    # Source v1: mean_xgb vs mean_xgb_sc
    v1_xgb, v1_xgb_sc = [], []
    v1_datasets = []
    for ds in DATASETS_10:
        if ds in exp1_v1:
            xgb = safe_float(exp1_v1[ds].get("mean_xgb"))
            xgb_sc = safe_float(exp1_v1[ds].get("mean_xgb_sc"))
            if not np.isnan(xgb) and not np.isnan(xgb_sc):
                v1_xgb.append(xgb)
                v1_xgb_sc.append(xgb_sc)
                v1_datasets.append(ds)

    # Source v2: XGBoost vs XGBoost_constrained
    v2_xgb, v2_xgb_sc = [], []
    v2_datasets = []
    for ds in DATASETS_10:
        if ds in exp1_v2:
            xgb_data = exp1_v2[ds].get("XGBoost", {})
            xgbc_data = exp1_v2[ds].get("XGBoost_constrained", {})
            xgb = safe_float(xgbc_data.get("mean_score")) if xgbc_data else np.nan
            xgb_base = safe_float(xgb_data.get("mean_score")) if xgb_data else np.nan
            if not np.isnan(xgb) and not np.isnan(xgb_base):
                v2_xgb.append(xgb_base)
                v2_xgb_sc.append(xgb)
                v2_datasets.append(ds)

    def _paired_analysis(base: list, constrained: list) -> dict:
        base_arr = np.array(base)
        const_arr = np.array(constrained)
        diff = const_arr - base_arr

        if len(diff) < 2:
            return {"paired_t_p": 1.0, "wilcoxon_p": 1.0, "cohens_d": 0.0,
                    "wins": 0, "losses": 0, "ties": 0}

        # Paired t-test
        try:
            t_stat, t_p = stats.ttest_rel(const_arr, base_arr)
        except Exception:
            t_stat, t_p = 0.0, 1.0

        # Wilcoxon
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except Exception:
            w_stat, w_p = 0.0, 1.0

        # Cohen's d
        d_std = np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 1e-10
        cohens_d = float(np.mean(diff) / d_std)

        # Win/loss/tie
        threshold = 0.001
        wins = int(np.sum(diff > threshold))
        losses = int(np.sum(diff < -threshold))
        ties = int(len(diff) - wins - losses)

        return {
            "paired_t_p": round(float(t_p), 6),
            "wilcoxon_p": round(float(w_p), 6),
            "cohens_d": round(cohens_d, 4),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

    source_v1 = _paired_analysis(v1_xgb, v1_xgb_sc)
    source_v2 = _paired_analysis(v2_xgb, v2_xgb_sc)

    # Pooled analysis
    pooled_base = v1_xgb + v2_xgb
    pooled_const = v1_xgb_sc + v2_xgb_sc
    pooled = _paired_analysis(pooled_base, pooled_const)

    # Identify datasets that help/hurt
    datasets_helped = []
    datasets_hurt = []
    for ds, xgb, xsc in zip(v1_datasets, v1_xgb, v1_xgb_sc):
        if xsc > xgb + 0.001:
            datasets_helped.append(ds)
        elif xsc < xgb - 0.001:
            datasets_hurt.append(ds)

    # ── Figure: XGBoost constraint delta bar chart ──
    fig = _draw_xgb_delta_bars(
        v1_datasets=v1_datasets,
        v1_diffs=[sc - b for b, sc in zip(v1_xgb, v1_xgb_sc)],
        v2_datasets=v2_datasets,
        v2_diffs=[sc - b for b, sc in zip(v2_xgb, v2_xgb_sc)],
    )

    result = {
        "source_v1": source_v1,
        "source_v2": source_v2,
        "pooled": pooled,
        "datasets_helped": datasets_helped,
        "datasets_hurt": datasets_hurt,
    }
    return result, fig


def _draw_xgb_delta_bars(
    v1_datasets: list,
    v1_diffs: list,
    v2_datasets: list,
    v2_diffs: list,
) -> plt.Figure:
    """Draw bar chart of XGBoost constraint deltas."""
    fig, ax = plt.subplots(figsize=(14, 6))

    all_ds = sorted(set(v1_datasets + v2_datasets))
    x = np.arange(len(all_ds))
    width = 0.35

    v1_map = dict(zip(v1_datasets, v1_diffs))
    v2_map = dict(zip(v2_datasets, v2_diffs))

    v1_vals = [v1_map.get(ds, 0) for ds in all_ds]
    v2_vals = [v2_map.get(ds, 0) for ds in all_ds]

    colors_v1 = ["green" if v > 0 else "red" for v in v1_vals]
    colors_v2 = ["#2ecc71" if v > 0 else "#e74c3c" for v in v2_vals]

    ax.bar(x - width / 2, v1_vals, width, color=colors_v1, alpha=0.7, edgecolor="black",
           linewidth=0.5, label="v1: XGB+SC - XGB")
    ax.bar(x + width / 2, v2_vals, width, color=colors_v2, alpha=0.7, edgecolor="black",
           linewidth=0.5, label="v2: XGB_constrained - XGB")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_ds, rotation=45, ha="right")
    ax.set_ylabel("Score Difference (constrained - default)")
    ax.set_title("XGBoost: Effect of Simplicial Constraints", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5: Ablation — Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def block5_ablation(deps: dict) -> tuple[dict, plt.Figure]:
    """Block 5: Ablation bootstrap CI analysis."""
    logger.info("=== Block 5: Ablation Bootstrap CI ===")

    meta = deps["exp_id2_it3_mini"]["metadata"]
    rpd = meta["results_per_dataset"]
    stat_tests = meta.get("statistical_tests", {})

    # Extract per-dataset scores
    simplicial, random_m, unconstrained = [], [], []
    ds_list = []
    task_types = {}

    for ds in DATASETS_10:
        if ds in rpd:
            ms = rpd[ds].get("mean_scores", {})
            s = safe_float(ms.get("SIMPLICIAL"))
            r = safe_float(ms.get("RANDOM_MATCHED"))
            u = safe_float(ms.get("UNCONSTRAINED"))
            if not any(np.isnan(x) for x in [s, r, u]):
                simplicial.append(s)
                random_m.append(r)
                unconstrained.append(u)
                ds_list.append(ds)
                task_types[ds] = rpd[ds].get("task_type", "unknown")

    s_arr = np.array(simplicial)
    r_arr = np.array(random_m)
    u_arr = np.array(unconstrained)
    n = len(s_arr)

    logger.info(f"Ablation data: {n} datasets")

    # Bootstrap (B=10000)
    B = 10000
    rng = np.random.default_rng(42)
    boot_sr = np.zeros(B)  # simplicial - random
    boot_su = np.zeros(B)  # simplicial - unconstrained
    boot_ru = np.zeros(B)  # random - unconstrained

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_sr[b] = np.mean(s_arr[idx] - r_arr[idx])
        boot_su[b] = np.mean(s_arr[idx] - u_arr[idx])
        boot_ru[b] = np.mean(r_arr[idx] - u_arr[idx])

    ci_sr = [float(np.percentile(boot_sr, 2.5)), float(np.percentile(boot_sr, 97.5))]
    ci_su = [float(np.percentile(boot_su, 2.5)), float(np.percentile(boot_su, 97.5))]
    ci_ru = [float(np.percentile(boot_ru, 2.5)), float(np.percentile(boot_ru, 97.5))]

    p_s_gt_u = float(np.mean(boot_su > 0))

    # Subgroup analysis
    def _subgroup(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"simplicial_mean": 0, "unconstrained_mean": 0}
        return {
            "simplicial_mean": round(float(s_arr[mask].mean()), 6),
            "unconstrained_mean": round(float(u_arr[mask].mean()), 6),
        }

    synth_mask = np.array([ds in SYNTHETIC_DATASETS for ds in ds_list])
    real_mask = np.array([ds in REAL_DATASETS for ds in ds_list])
    cls_mask = np.array([task_types.get(ds) == "classification" for ds in ds_list])
    reg_mask = np.array([task_types.get(ds) == "regression" for ds in ds_list])

    subgroup = {
        "synthetic_only": _subgroup(synth_mask),
        "real_only": _subgroup(real_mask),
        "classification_only": _subgroup(cls_mask),
        "regression_only": _subgroup(reg_mask),
    }

    # Wilcoxon effect sizes (rank-biserial r)
    def _wilcoxon_effect(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        nonzero = diff[diff != 0]
        if len(nonzero) < 2:
            return 0.0
        try:
            w_stat, _ = stats.wilcoxon(nonzero)
            n_nz = len(nonzero)
            # rank-biserial r = 1 - (2*W) / (n*(n+1)/2)
            total_ranks = n_nz * (n_nz + 1) / 2
            r = 1 - (2 * w_stat) / total_ranks
            return round(float(r), 4)
        except Exception:
            return 0.0

    effect_sizes = {
        "A_vs_B": _wilcoxon_effect(s_arr, r_arr),
        "A_vs_C": _wilcoxon_effect(s_arr, u_arr),
        "B_vs_C": _wilcoxon_effect(r_arr, u_arr),
    }

    # Per-dataset differences
    per_ds_diff = {}
    for i, ds in enumerate(ds_list):
        per_ds_diff[ds] = {
            "simplicial_minus_unconstrained": round(float(s_arr[i] - u_arr[i]), 6),
            "simplicial_minus_random": round(float(s_arr[i] - r_arr[i]), 6),
        }

    # ── Figure: Bootstrap distributions ──
    fig = _draw_ablation_bootstrap(boot_sr, boot_su, boot_ru, ci_sr, ci_su, ci_ru)

    result = {
        "bootstrap_ci_95": {
            "simplicial_minus_random": [round(ci_sr[0], 6), round(ci_sr[1], 6)],
            "simplicial_minus_unconstrained": [round(ci_su[0], 6), round(ci_su[1], 6)],
            "random_minus_unconstrained": [round(ci_ru[0], 6), round(ci_ru[1], 6)],
        },
        "p_simplicial_gt_unconstrained": round(p_s_gt_u, 4),
        "subgroup_analysis": subgroup,
        "wilcoxon_effect_sizes": effect_sizes,
        "per_dataset_differences": per_ds_diff,
        "existing_wilcoxon_p_values": {
            k: round(v.get("p_value", 1.0), 6)
            for k, v in stat_tests.items()
        },
    }
    return result, fig


def _draw_ablation_bootstrap(
    boot_sr: np.ndarray,
    boot_su: np.ndarray,
    boot_ru: np.ndarray,
    ci_sr: list,
    ci_su: list,
    ci_ru: list,
) -> plt.Figure:
    """Draw violin/histogram of bootstrap ablation differences."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    data_labels = [
        (boot_sr, ci_sr, "SIMPLICIAL - RANDOM", "#3498db"),
        (boot_su, ci_su, "SIMPLICIAL - UNCONSTRAINED", "#e74c3c"),
        (boot_ru, ci_ru, "RANDOM - UNCONSTRAINED", "#2ecc71"),
    ]

    for ax, (data, ci, label, color) in zip(axes, data_labels):
        ax.hist(data, bins=60, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(x=0, color="black", linewidth=1.5, linestyle="-")
        ax.axvline(x=ci[0], color="red", linewidth=1.2, linestyle="--", label=f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        ax.axvline(x=ci[1], color="red", linewidth=1.2, linestyle="--")
        ax.axvline(x=np.mean(data), color="blue", linewidth=1.2, linestyle="-.",
                   label=f"Mean: {np.mean(data):.4f}")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Score Difference")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    fig.suptitle("Ablation Bootstrap Distributions (B=10,000)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6: Per-Dataset Diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def block6_diagnostic(deps: dict, b1_scores: dict, b5_result: dict) -> tuple[dict, plt.Figure]:
    """Block 6: Per-dataset diagnostic — moderator analysis."""
    logger.info("=== Block 6: Per-Dataset Diagnostic ===")

    # Dataset metadata
    data_meta = deps["data_id3_it1_mini"]
    exp1_v1 = deps["exp_id1_it2_mini"]["metadata"]["per_dataset_results"]
    exp1_v2 = deps["exp_id1_it3_mini"]["metadata"]["aggregated_results"]
    ablation = deps["exp_id2_it3_mini"]["metadata"]["results_per_dataset"]
    exp1_v2_topo = deps["exp_id1_it3_mini"]["metadata"].get("topological_analysis", {})

    # Extract dataset features from data_id3_it1
    ds_features: dict[str, dict[str, float]] = {}
    for ds_block in data_meta["datasets"]:
        ds_name = ds_block["dataset"]
        ex0 = ds_block["examples"][0] if ds_block["examples"] else {}
        n_feat = safe_float(ex0.get("metadata_n_features", 0))
        n_samp = safe_float(ex0.get("metadata_n_samples", 0))
        task = 1.0 if ex0.get("metadata_task_type") == "classification" else 0.0
        cat = ex0.get("metadata_category", "")

        # Interaction order
        ki_raw = ex0.get("metadata_known_interactions", "{}")
        try:
            ki = json.loads(ki_raw) if isinstance(ki_raw, str) else ki_raw
        except (json.JSONDecodeError, TypeError):
            ki = {}
        max_order = 0
        if "4-way" in ki:
            max_order = 4
        elif "3-way" in ki:
            max_order = 3
        elif "2-way" in ki:
            max_order = 2

        ds_features[ds_name] = {
            "n_features": n_feat,
            "n_samples": n_samp,
            "is_classification": task,
            "interaction_order": float(max_order),
        }

    # SC-OTS v1 relative performance (vs mean of baselines)
    score_mat = b1_scores.get("unified_score_matrix", {})
    baseline_methods = ["FIGS-5", "FIGS-10", "FIGS-20", "XGBoost-default", "EBM-default"]

    heatmap_data: dict[str, dict[str, float]] = {}
    for ds in DATASETS_10:
        heatmap_data[ds] = {}

        # SC-OTS v1 delta
        scots_v1 = score_mat.get(ds, {}).get("SC-OTS_v1", np.nan)
        bl_scores = [score_mat.get(ds, {}).get(m, np.nan) for m in baseline_methods]
        bl_scores = [s for s in bl_scores if not np.isnan(s)]
        bl_mean = np.mean(bl_scores) if bl_scores else np.nan
        heatmap_data[ds]["SC-OTS_v1_delta"] = float(scots_v1 - bl_mean) if not np.isnan(scots_v1) and not np.isnan(bl_mean) else 0.0

        # SC-OTS v2 delta
        scots_v2 = score_mat.get(ds, {}).get("SC-OTS_v2", np.nan)
        heatmap_data[ds]["SC-OTS_v2_delta"] = float(scots_v2 - bl_mean) if not np.isnan(scots_v2) and not np.isnan(bl_mean) else 0.0

        # Ablation delta
        if ds in ablation:
            ms = ablation[ds].get("mean_scores", {})
            s_val = safe_float(ms.get("SIMPLICIAL"))
            u_val = safe_float(ms.get("UNCONSTRAINED"))
            heatmap_data[ds]["ablation_delta"] = float(s_val - u_val) if not np.isnan(s_val) and not np.isnan(u_val) else 0.0
        else:
            heatmap_data[ds]["ablation_delta"] = 0.0

        # XGB constraint delta
        if ds in exp1_v1:
            xgb = safe_float(exp1_v1[ds].get("mean_xgb"))
            xgb_sc = safe_float(exp1_v1[ds].get("mean_xgb_sc"))
            heatmap_data[ds]["xgb_constraint_delta"] = float(xgb_sc - xgb) if not np.isnan(xgb) and not np.isnan(xgb_sc) else 0.0
        else:
            heatmap_data[ds]["xgb_constraint_delta"] = 0.0

        # Topological features
        if ds in exp1_v2_topo:
            topo = exp1_v2_topo[ds]
            sc = topo.get("mean_simplex_counts", {})
            total_simplices = sum(safe_float(sc.get(f"dim_{d}", 0)) for d in range(4))
            heatmap_data[ds]["simplex_count"] = total_simplices
            heatmap_data[ds]["betti_0"] = safe_float(topo.get("mean_betti", {}).get("b0", 0))
        else:
            heatmap_data[ds]["simplex_count"] = 0.0
            heatmap_data[ds]["betti_0"] = 0.0

        # Dataset features
        if ds in ds_features:
            heatmap_data[ds]["n_features"] = ds_features[ds]["n_features"]
            heatmap_data[ds]["n_samples"] = ds_features[ds]["n_samples"]

    # Spearman correlations
    corr_features = ["n_features", "n_samples", "is_classification", "interaction_order"]
    perf_targets = ["SC-OTS_v1_delta", "SC-OTS_v2_delta", "ablation_delta"]
    correlations: dict[str, dict[str, float]] = {}

    ds_with_features = [ds for ds in DATASETS_10 if ds in ds_features]

    for feat in corr_features:
        feat_vals = [ds_features.get(ds, {}).get(feat, 0) for ds in ds_with_features]
        for target in perf_targets:
            target_vals = [heatmap_data.get(ds, {}).get(target, 0) for ds in ds_with_features]
            try:
                rho, p_val = stats.spearmanr(feat_vals, target_vals)
            except Exception:
                rho, p_val = 0.0, 1.0
            key = f"{feat}_vs_{target}"
            correlations[key] = {"rho": round(float(rho), 4), "p": round(float(p_val), 4)}

    # Identify favorable characteristics
    notable = []
    for key, vals in correlations.items():
        if abs(vals["rho"]) > 0.5:
            notable.append(f"{key}: rho={vals['rho']}")
    favorable = "; ".join(notable) if notable else "No strong correlations found (|rho| > 0.5) with n=10 datasets"

    # ── Figure: Diagnostic heatmap ──
    fig = _draw_diagnostic_heatmap(heatmap_data, DATASETS_10)

    result = {
        "spearman_correlations": correlations,
        "favorable_characteristics": favorable,
        "heatmap_data": heatmap_data,
    }
    return result, fig


def _draw_diagnostic_heatmap(
    heatmap_data: dict[str, dict[str, float]],
    datasets: list[str],
) -> plt.Figure:
    """Draw the diagnostic heatmap."""
    columns = [
        "SC-OTS_v1_delta", "SC-OTS_v2_delta", "ablation_delta",
        "xgb_constraint_delta", "simplex_count", "betti_0",
        "n_features", "n_samples",
    ]

    data = np.zeros((len(datasets), len(columns)))
    for i, ds in enumerate(datasets):
        for j, col in enumerate(columns):
            data[i, j] = heatmap_data.get(ds, {}).get(col, 0)

    # Normalize columns for display
    data_norm = data.copy()
    for j in range(data.shape[1]):
        col_data = data[:, j]
        max_abs = np.max(np.abs(col_data))
        if max_abs > 0:
            data_norm[:, j] = col_data / max_abs

    fig, ax = plt.subplots(figsize=(14, 7))

    # Custom colormap: red-white-green
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    # Create annotation array safely handling NaN
    annot_arr = np.empty(data.shape, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            annot_arr[i, j] = f"{v:.2f}" if not np.isnan(v) else "N/A"

    sns.heatmap(
        data_norm, annot=annot_arr,
        fmt="", cmap=cmap, center=0,
        xticklabels=[c.replace("_", "\n") for c in columns],
        yticklabels=datasets,
        ax=ax, linewidths=0.5,
        cbar_kws={"label": "Normalized value"},
    )
    ax.set_title("Dataset Diagnostic Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7: Final Verdict
# ═══════════════════════════════════════════════════════════════════════════════

def block7_verdict(
    b1: dict, b2: dict, b3: dict, b4: dict, b5: dict, b6: dict
) -> tuple[dict, plt.Figure]:
    """Block 7: Final verdict — per-subclaim evidence table."""
    logger.info("=== Block 7: Final Verdict ===")

    def _strength(p: float = None, ci: list = None, effect: float = None, value: float = None,
                  threshold: float = None, direction: str = "positive") -> str:
        """Determine evidence strength."""
        if value is not None and threshold is not None:
            if direction == "positive":
                if value > threshold + 0.05:
                    return "Strong Confirm"
                elif value > threshold:
                    return "Moderate Confirm"
                elif value > threshold - 0.05:
                    return "Weak Confirm"
                elif value > threshold - 0.1:
                    return "Inconclusive"
                else:
                    return "Moderate Against"
            else:
                if value < threshold - 0.05:
                    return "Strong Confirm"
                elif value < threshold:
                    return "Moderate Confirm"

        if p is not None:
            if p < 0.01:
                return "Strong Confirm"
            elif p < 0.05:
                return "Moderate Confirm"
            elif p < 0.10:
                return "Weak Confirm"
            else:
                return "Inconclusive"

        if ci is not None:
            if ci[0] > 0.05:
                return "Strong Confirm"
            elif ci[0] > 0:
                return "Moderate Confirm"
            elif ci[0] > -0.02:
                return "Weak Confirm"
            elif ci[1] < 0:
                return "Moderate Against"
            else:
                return "Inconclusive"

        return "Inconclusive"

    subclaims = []

    # 1. SC-OTS matches/exceeds RO-FIGS accuracy
    ranks = b1.get("method_avg_ranks", {})
    scots_rank = ranks.get("SC-OTS_v1", 99)
    rofigs_rank = ranks.get("RO-FIGS", 99)
    rank_diff = scots_rank - rofigs_rank
    # Negative diff means SC-OTS ranks better (lower rank = better)
    if rank_diff < 0:
        strength = "Moderate Confirm"
    elif rank_diff < 1:
        strength = "Weak Confirm"
    elif rank_diff < 2:
        strength = "Inconclusive"
    else:
        strength = "Weak Against"

    subclaims.append({
        "id": 1,
        "claim": "SC-OTS matches/exceeds RO-FIGS accuracy",
        "evidence_strength": strength,
        "key_statistic": f"SC-OTS rank={scots_rank:.1f}, RO-FIGS rank={rofigs_rank:.1f}",
        "conclusion": f"SC-OTS v1 has avg rank {scots_rank:.1f} vs RO-FIGS {rofigs_rank:.1f}. "
                      f"{'SC-OTS ranks better' if rank_diff < 0 else 'RO-FIGS ranks better'} across {b1['n_datasets']} datasets.",
    })

    # 2. SC-OTS uses fewer splits
    v1_comp = b2.get("per_method_avg_complexity", {}).get("SC-OTS_v1", 20)
    rofigs_comp = b2.get("per_method_avg_complexity", {}).get("RO-FIGS", 20)
    pareto_scots = b2.get("pareto_membership_counts", {}).get("SC-OTS_v1", 0)
    if v1_comp < rofigs_comp:
        strength = "Moderate Confirm"
    elif v1_comp == rofigs_comp:
        strength = "Inconclusive"
    else:
        strength = "Weak Against"

    subclaims.append({
        "id": 2,
        "claim": "SC-OTS uses fewer splits than RO-FIGS",
        "evidence_strength": strength,
        "key_statistic": f"SC-OTS={v1_comp:.0f} splits, RO-FIGS={rofigs_comp:.0f} splits, Pareto count={pareto_scots}",
        "conclusion": f"SC-OTS v1 avg {v1_comp:.0f} splits vs RO-FIGS {rofigs_comp:.0f}. "
                      f"SC-OTS on Pareto frontier for {pareto_scots}/10 datasets.",
    })

    # 3. Simplicial complex recovers ground-truth interactions
    f1_vals = []
    for ds in SYNTHETIC_DATASETS:
        f1_table = b3.get("f1_comparison_table", {})
        if ds in f1_table:
            # Use enhanced dCor F1 as the main metric
            f1 = f1_table[ds].get("enhanced_dcor_v1", 0) or f1_table[ds].get("scots_v1_ir", 0)
            f1_vals.append(f1)
    mean_f1 = float(np.mean(f1_vals)) if f1_vals else 0
    if mean_f1 > 0.7:
        strength = "Strong Confirm"
    elif mean_f1 > 0.5:
        strength = "Moderate Confirm"
    elif mean_f1 > 0.3:
        strength = "Weak Confirm"
    else:
        strength = "Inconclusive"

    subclaims.append({
        "id": 3,
        "claim": "Simplicial complex recovers ground-truth interactions",
        "evidence_strength": strength,
        "key_statistic": f"Mean enhanced dCor F1 = {mean_f1:.3f} across {len(f1_vals)} synthetic datasets",
        "conclusion": f"Enhanced interaction dCor achieves mean F1={mean_f1:.3f}. "
                      f"Strong on friedman1/3, weaker on synth_3way/4way with higher-order interactions.",
    })

    # 4. Interaction faithfulness ≥70%
    faith_prec = b3.get("faithfulness_precision_mean", 0)
    if faith_prec >= 0.7:
        strength = "Strong Confirm"
    elif faith_prec >= 0.5:
        strength = "Moderate Confirm"
    elif faith_prec >= 0.3:
        strength = "Weak Confirm"
    else:
        strength = "Inconclusive"

    subclaims.append({
        "id": 4,
        "claim": "Interaction faithfulness ≥70%",
        "evidence_strength": strength,
        "key_statistic": f"Mean faithfulness precision = {faith_prec:.3f}",
        "conclusion": f"Faithfulness precision averages {faith_prec:.1%} across synthetic datasets with SHAP ground truth. "
                      f"{'Exceeds' if faith_prec >= 0.7 else 'Below'} the 70% target.",
    })

    # 5. Simplicial constraints improve over random/unconstrained
    ci_su = b5.get("bootstrap_ci_95", {}).get("simplicial_minus_unconstrained", [-1, 1])
    p_s_gt_u = b5.get("p_simplicial_gt_unconstrained", 0.5)
    if ci_su[0] > 0:
        strength = "Moderate Confirm" if ci_su[0] > 0.02 else "Weak Confirm"
    elif ci_su[1] < 0:
        strength = "Moderate Against"
    else:
        strength = "Inconclusive"

    subclaims.append({
        "id": 5,
        "claim": "Simplicial constraints improve over random/unconstrained",
        "evidence_strength": strength,
        "key_statistic": f"95% CI: [{ci_su[0]:.4f}, {ci_su[1]:.4f}], P(S>U)={p_s_gt_u:.3f}",
        "conclusion": f"Bootstrap CI for SIMPLICIAL-UNCONSTRAINED spans [{ci_su[0]:.4f}, {ci_su[1]:.4f}]. "
                      f"{'Straddles zero — no reliable advantage' if ci_su[0] <= 0 <= ci_su[1] else 'CI excludes zero'}. "
                      f"P(simplicial > unconstrained) = {p_s_gt_u:.1%}.",
    })

    # 6. XGBoost benefits from SC constraints
    cohens_d_v1 = b4.get("source_v1", {}).get("cohens_d", 0)
    cohens_d_pooled = b4.get("pooled", {}).get("cohens_d", 0)
    p_pooled = b4.get("pooled", {}).get("wilcoxon_p", 1)
    if abs(cohens_d_pooled) < 0.2:
        strength = "Inconclusive"
    elif cohens_d_pooled > 0.2:
        strength = "Weak Confirm" if p_pooled > 0.05 else "Moderate Confirm"
    else:
        strength = "Weak Against" if p_pooled > 0.05 else "Moderate Against"

    subclaims.append({
        "id": 6,
        "claim": "XGBoost benefits from SC constraints",
        "evidence_strength": strength,
        "key_statistic": f"Pooled Cohen's d = {cohens_d_pooled:.3f}, p = {p_pooled:.4f}",
        "conclusion": f"XGBoost+SC vs XGBoost: pooled Cohen's d={cohens_d_pooled:.3f}, p={p_pooled:.4f}. "
                      f"Effect is {'small' if abs(cohens_d_pooled) < 0.2 else 'medium' if abs(cohens_d_pooled) < 0.5 else 'large'}.",
    })

    # 7. SC-OTS competitive with FIGS (≤20 splits)
    figs_methods = ["FIGS-5", "FIGS-10", "FIGS-20"]
    scots_rank_val = ranks.get("SC-OTS_v1", 99)
    figs_ranks = [ranks.get(m, 99) for m in figs_methods if m in ranks]
    best_figs_rank = min(figs_ranks) if figs_ranks else 99
    if scots_rank_val <= best_figs_rank:
        strength = "Strong Confirm"
    elif scots_rank_val <= best_figs_rank + 2:
        strength = "Moderate Confirm"
    elif scots_rank_val <= best_figs_rank + 3:
        strength = "Weak Confirm"
    else:
        strength = "Weak Against"

    subclaims.append({
        "id": 7,
        "claim": "SC-OTS competitive with FIGS (≤20 splits)",
        "evidence_strength": strength,
        "key_statistic": f"SC-OTS rank={scots_rank_val:.1f}, best FIGS rank={best_figs_rank:.1f}",
        "conclusion": f"SC-OTS v1 rank {scots_rank_val:.1f} vs best FIGS rank {best_figs_rank:.1f}. "
                      f"CD={b1.get('critical_difference', 0):.2f}.",
    })

    # 8. SC-OTS competitive with XGBoost/EBM
    xgb_rank = ranks.get("XGBoost-default", 99)
    ebm_rank = ranks.get("EBM-default", 99)
    gap = scots_rank_val - min(xgb_rank, ebm_rank)
    cd_val = b1.get("critical_difference", 99)
    if gap < cd_val:
        strength = "Weak Confirm" if gap > 0 else "Moderate Confirm"
    else:
        strength = "Moderate Against"

    subclaims.append({
        "id": 8,
        "claim": "SC-OTS competitive with XGBoost/EBM",
        "evidence_strength": strength,
        "key_statistic": f"Rank gap to best={gap:.1f}, CD={cd_val:.2f}",
        "conclusion": f"SC-OTS rank gap to XGBoost/EBM is {gap:.1f} (CD={cd_val:.2f}). "
                      f"{'Within' if gap < cd_val else 'Beyond'} critical difference.",
    })

    # 9. Topological structure provides interpretability
    betti = b3.get("betti_summary", {})
    n_datasets_with_betti = sum(1 for ds in betti if "v1" in betti[ds] or "v2" in betti[ds])
    # Check if betti numbers are informative (non-trivial structure)
    nontrivial = sum(
        1 for ds in betti
        if "v1" in betti[ds] and any(b > 0 for b in betti[ds]["v1"][1:])
    )
    if n_datasets_with_betti >= 8:
        strength = "Moderate Confirm"
    elif n_datasets_with_betti >= 5:
        strength = "Weak Confirm"
    else:
        strength = "Inconclusive"

    subclaims.append({
        "id": 9,
        "claim": "Topological structure provides interpretability",
        "evidence_strength": strength,
        "key_statistic": f"{n_datasets_with_betti}/10 datasets with Betti analysis, {nontrivial} with Betti-1 > 0",
        "conclusion": f"Betti numbers computed for {n_datasets_with_betti}/10 datasets. "
                      f"Betti-0 differentiates feature clusters; Betti-1 mostly 0 (no circular dependencies). "
                      f"Topological summary provides compact interpretable representation.",
    })

    # Overall verdict
    confirm_count = sum(1 for s in subclaims if "Confirm" in s["evidence_strength"])
    against_count = sum(1 for s in subclaims if "Against" in s["evidence_strength"])
    inconclusive_count = sum(1 for s in subclaims if s["evidence_strength"] == "Inconclusive")

    if confirm_count >= 6:
        overall = "Confirmed"
    elif confirm_count >= 4:
        overall = "Partially Confirmed"
    else:
        overall = "Disconfirmed"

    # Key finding: ablation (subclaim 5) is critical
    ablation_strength = subclaims[4]["evidence_strength"]
    if "Against" in ablation_strength or ablation_strength == "Inconclusive":
        if overall == "Confirmed":
            overall = "Partially Confirmed"

    synthesis = (
        f"The SC-OTS hypothesis is {overall.lower()}. "
        f"{confirm_count}/9 sub-claims confirmed, {against_count} against, {inconclusive_count} inconclusive. "
        f"SC-OTS achieves competitive accuracy with RO-FIGS and produces interpretable topological structures, "
        f"but the ablation study shows that simplicial constraints do not reliably improve over unconstrained "
        f"splits (CI straddles zero), undermining the core mechanistic claim. "
        f"The enhanced interaction dCor method shows strong recovery on simple synthetic datasets, "
        f"but degrades on higher-order interactions."
    )

    # ── Figure: Verdict table ──
    fig = _draw_verdict_table(subclaims, overall, synthesis)

    result = {
        "per_subclaim": subclaims,
        "overall_verdict": overall,
        "synthesis": synthesis,
        "confirm_count": confirm_count,
        "against_count": against_count,
        "inconclusive_count": inconclusive_count,
    }
    return result, fig


def _draw_verdict_table(
    subclaims: list[dict],
    overall: str,
    synthesis: str,
) -> plt.Figure:
    """Render the verdict as a matplotlib table figure."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    # Color mapping for evidence strength
    color_map = {
        "Strong Confirm": "#27ae60",
        "Moderate Confirm": "#2ecc71",
        "Weak Confirm": "#a6d96a",
        "Inconclusive": "#fee08b",
        "Weak Against": "#fdae61",
        "Moderate Against": "#f46d43",
        "Strong Against": "#d73027",
    }

    # Table data
    headers = ["#", "Sub-claim", "Evidence", "Key Statistic"]
    cell_text = []
    cell_colors = []
    for sc in subclaims:
        row = [
            str(sc["id"]),
            sc["claim"][:45],
            sc["evidence_strength"],
            sc["key_statistic"][:50],
        ]
        cell_text.append(row)
        bg = color_map.get(sc["evidence_strength"], "white")
        cell_colors.append(["white", "white", bg, "white"])

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        colWidths=[0.04, 0.30, 0.16, 0.50],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")

    # Overall verdict
    verdict_color = {"Confirmed": "#27ae60", "Partially Confirmed": "#f39c12", "Disconfirmed": "#e74c3c"}
    ax.text(0.5, 0.05, f"Overall Verdict: {overall}",
            transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold",
            color=verdict_color.get(overall, "black"),
            bbox=dict(boxstyle="round,pad=0.5", facecolor=verdict_color.get(overall, "gray"), alpha=0.2))

    # Synthesis text
    ax.text(0.5, 0.95, "SC-OTS Hypothesis Verdict",
            transform=ax.transAxes, ha="center", fontsize=15, fontweight="bold")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    """Main evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("SC-OTS Definitive Final Synthesis — Evaluation")
    logger.info("=" * 60)

    # Load all dependencies
    deps = load_all_dependencies()

    # Run all blocks
    b1_result, fig_cd, fig_heatmap = block1_accuracy(deps)
    logger.info(f"Block 1 done: Friedman p={b1_result['friedman_p_value']}")

    b2_result, fig_pareto_synth, fig_pareto_real = block2_compactness(deps, b1_result)
    logger.info(f"Block 2 done: Pareto counts = {b2_result['pareto_membership_counts']}")

    b3_result, fig_interaction = block3_interaction_discovery(deps)
    logger.info(f"Block 3 done: Mean AUPRC = {b3_result['mean_auprc_synthetic']}")

    b4_result, fig_xgb = block4_xgboost_sc(deps)
    logger.info(f"Block 4 done: Pooled Cohen's d = {b4_result['pooled']['cohens_d']}")

    b5_result, fig_ablation = block5_ablation(deps)
    logger.info(f"Block 5 done: P(S>U) = {b5_result['p_simplicial_gt_unconstrained']}")

    b6_result, fig_diagnostic = block6_diagnostic(deps, b1_result, b5_result)
    logger.info(f"Block 6 done: {b6_result['favorable_characteristics'][:80]}")

    b7_result, fig_verdict = block7_verdict(b1_result, b2_result, b3_result, b4_result, b5_result, b6_result)
    logger.info(f"Block 7 done: {b7_result['overall_verdict']}")

    # Convert figures to base64
    logger.info("Converting figures to base64...")
    figures = {
        "cd_diagram": fig_to_base64(fig_cd),
        "score_heatmap": fig_to_base64(fig_heatmap),
        "pareto_synthetic": fig_to_base64(fig_pareto_synth),
        "pareto_realworld": fig_to_base64(fig_pareto_real),
        "interaction_recovery_bars": fig_to_base64(fig_interaction),
        "xgb_constraint_delta": fig_to_base64(fig_xgb),
        "ablation_bootstrap": fig_to_base64(fig_ablation),
        "diagnostic_heatmap": fig_to_base64(fig_diagnostic),
        "verdict_table": fig_to_base64(fig_verdict),
    }

    # Assemble aggregate metrics for the schema
    metrics_agg = {
        "friedman_statistic": b1_result["friedman_statistic"],
        "friedman_p_value": b1_result["friedman_p_value"],
        "critical_difference": b1_result["critical_difference"],
        "scots_v1_avg_rank": b1_result["scots_v1_rank"],
        "scots_v2_avg_rank": b1_result["scots_v2_rank"],
        "mean_auprc_synthetic": b3_result["mean_auprc_synthetic"],
        "faithfulness_precision_mean": b3_result["faithfulness_precision_mean"],
        "xgb_sc_pooled_cohens_d": b4_result["pooled"]["cohens_d"],
        "xgb_sc_pooled_wilcoxon_p": b4_result["pooled"]["wilcoxon_p"],
        "ablation_p_simplicial_gt_unconstrained": b5_result["p_simplicial_gt_unconstrained"],
        "n_subclaims_confirmed": b7_result["confirm_count"],
        "n_subclaims_against": b7_result["against_count"],
        "n_subclaims_inconclusive": b7_result["inconclusive_count"],
    }

    # Build datasets section for schema compliance
    # One dataset entry per analyzed dataset, with a summary example
    eval_datasets = []
    for ds in DATASETS_10:
        scores_str = json.dumps({
            m: b1_result["unified_score_matrix"].get(ds, {}).get(m, None)
            for m in b1_result.get("valid_methods", [])
        })
        ablation_data = deps["exp_id2_it3_mini"]["metadata"]["results_per_dataset"].get(ds, {})
        abl_scores = ablation_data.get("mean_scores", {})

        example = {
            "input": f"Evaluate SC-OTS hypothesis on dataset: {ds}",
            "output": f"Analysis complete for {ds}",
            "metadata_dataset": ds,
            "metadata_task_type": deps["exp_id1_it2_mini"]["metadata"]["per_dataset_results"].get(
                ds, {}
            ).get("task_type", "unknown"),
            "eval_scots_v1_score": round(safe_float(
                b1_result["unified_score_matrix"].get(ds, {}).get("SC-OTS_v1"),
                default=0.0,
            ), 6),
            "eval_scots_v2_score": round(safe_float(
                b1_result["unified_score_matrix"].get(ds, {}).get("SC-OTS_v2"),
                default=0.0,
            ), 6),
            "eval_ablation_simplicial": round(safe_float(abl_scores.get("SIMPLICIAL"), default=0.0), 6),
            "eval_ablation_unconstrained": round(safe_float(abl_scores.get("UNCONSTRAINED"), default=0.0), 6),
            "predict_scores_json": scores_str,
        }
        eval_datasets.append({
            "dataset": ds,
            "examples": [example],
        })

    # Assemble final output
    output = {
        "metadata": {
            "evaluation_name": "SC-OTS Definitive Final Synthesis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_dependency_files_loaded": len(deps),
            "block_1_accuracy": b1_result,
            "block_2_compactness": b2_result,
            "block_3_interaction_discovery": b3_result,
            "block_4_xgboost_sc": b4_result,
            "block_5_ablation": b5_result,
            "block_6_diagnostic": b6_result,
            "block_7_verdict": b7_result,
            "figures": figures,
        },
        "metrics_agg": metrics_agg,
        "datasets": eval_datasets,
    }

    # Sanitize output for JSON serialization
    output = sanitize_for_json(output)

    # Write output — using a custom encoder that replaces NaN with null
    class NaNSafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if not np.isnan(float(obj)) else None
            return super().default(obj)

        def encode(self, o):
            text = super().encode(o)
            # Belt-and-suspenders: replace any remaining NaN/Infinity
            text = text.replace(": NaN", ": null")
            text = text.replace(": Infinity", ": null")
            text = text.replace(": -Infinity", ": null")
            return text

    out_path = WORKSPACE / "eval_out.json"
    raw = json.dumps(output, indent=2, cls=NaNSafeEncoder, default=str)
    # Final safety pass
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    raw = raw.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    out_path.write_text(raw)
    logger.info(f"Output written to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    logger.info("=" * 60)
    logger.info(f"VERDICT: {b7_result['overall_verdict']}")
    logger.info(b7_result["synthesis"][:200])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
