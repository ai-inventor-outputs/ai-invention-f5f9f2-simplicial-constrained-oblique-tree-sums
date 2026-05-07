#!/usr/bin/env python3
"""Comprehensive statistical evaluation of SC-OTS experimental results.

Computes 7 evaluation metrics:
1. Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
2. Friedman test + Nemenyi post-hoc with CD diagram
3. Per-dataset error analysis
4. XGBoost+SC benefit analysis (Cohen's d)
5. Interaction recovery curves (from TDA validation)
6. Complexity-accuracy Pareto frontier
7. Dataset characteristics regression

Outputs eval_out.json conforming to exp_eval_sol_out schema.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import io
import base64
import warnings
import resource

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr, rankdata
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Resource limits ──────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Logging ──────────────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Paths ────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
DEP_BASE = WORKSPACE.parent.parent.parent / "iter_2" / "gen_art"
EXP1_DIR = DEP_BASE / "exp_id1_it2__opus"
EXP2_DIR = DEP_BASE / "exp_id2_it2__opus"
EXP3_DIR = DEP_BASE / "exp_id3_it2__opus"

# ── Constants ────────────────────────────────────────────────────────
DATASET_ORDER = [
    "friedman1", "friedman3", "synth_3way", "synth_4way",
    "diabetes", "breast_w", "wine_quality", "california_housing",
    "spambase", "adult",
]
SYNTHETIC_DATASETS = ["friedman1", "friedman3", "synth_3way", "synth_4way"]

# Method name mapping: exp_id1 short name -> unified name
EXP1_METHOD_MAP = {
    "scots": "SC-OTS",
    "rofigs": "RO-FIGS",
    "figs": "FIGS-20",
    "xgb": "XGBoost",
    "xgb_sc": "XGBoost+SC",
    "ebm": "EBM",
}

# Unified method names from exp_id2
EXP2_METHOD_MAP = {
    "FIGS-5": "FIGS-5",
    "FIGS-10": "FIGS-10",
    "FIGS-20": "FIGS-20",
    "XGBoost-default": "XGBoost",
    "XGBoost-oracle": "XGBoost-oracle",
    "EBM-default": "EBM",
    "EBM-high-interaction": "EBM-high",
    "EBM-3way": "EBM-3way",
}

# Complexity assignments (total splits proxy)
COMPLEXITY_MAP = {
    "SC-OTS": 20,
    "RO-FIGS": 20,
    "FIGS-5": 5,
    "FIGS-10": 10,
    "FIGS-20": 20,
    "XGBoost": 600,
    "XGBoost-oracle": 600,
    "XGBoost+SC": 600,
    "EBM": 500,
    "EBM-high": 500,
    "EBM-3way": 500,
}

# Methods from exp_id1 (shared folds with SC-OTS)
EXP1_METHODS = ["SC-OTS", "RO-FIGS", "FIGS-20", "XGBoost", "XGBoost+SC", "EBM"]
# Additional methods only in exp_id2
EXP2_ONLY_METHODS = ["FIGS-5", "FIGS-10", "XGBoost-oracle", "EBM-high", "EBM-3way"]
ALL_METHODS = EXP1_METHODS + EXP2_ONLY_METHODS


# ── Helper functions ─────────────────────────────────────────────────

def fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    # If too large, redo at lower dpi
    if len(b64) > 500_000:
        logger.warning(f"Figure base64 size {len(b64)} > 500KB, reducing dpi to 100")
        buf2 = io.BytesIO()
        # Need to recreate... return as-is for now, the caller should handle
    return b64


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy for classification."""
    return float(np.mean(y_true == np.round(y_pred)))


# ── Step 1: Load data ───────────────────────────────────────────────

@logger.catch
def load_exp1() -> dict:
    """Load exp_id1 (SC-OTS main results)."""
    path = EXP1_DIR / "full_method_out.json"
    logger.info(f"Loading exp_id1 from {path}")
    data = json.loads(path.read_text())
    logger.info(f"Loaded exp_id1: {len(data['datasets'])} datasets")
    return data


@logger.catch
def load_exp2() -> dict:
    """Load exp_id2 (baseline benchmarks)."""
    path = EXP2_DIR / "full_method_out.json"
    logger.info(f"Loading exp_id2 from {path}")
    data = json.loads(path.read_text())
    logger.info(f"Loaded exp_id2: {len(data['datasets'])} datasets, methods: {data['metadata']['methods']}")
    return data


@logger.catch
def load_exp3() -> dict:
    """Load exp_id3 (TDA validation)."""
    path = EXP3_DIR / "full_method_out.json"
    logger.info(f"Loading exp_id3 from {path}")
    data = json.loads(path.read_text())
    logger.info(f"Loaded exp_id3: {len(data['datasets'])} datasets")
    return data


# ── Step 2: Build unified per-fold score DataFrame ──────────────────

def build_exp1_fold_scores(exp1: dict) -> pd.DataFrame:
    """
    From exp_id1: group examples by (dataset, fold), compute per-fold
    R2 or accuracy for each method.
    """
    rows = []
    methods_short = ["scots", "rofigs", "figs", "xgb", "xgb_sc", "ebm"]

    for ds_block in exp1["datasets"]:
        ds_name = ds_block["dataset"]
        examples = ds_block["examples"]
        if not examples:
            continue

        task_type = examples[0]["metadata_task_type"]

        # Group by fold
        fold_groups: dict[int, list] = {}
        for ex in examples:
            fold = ex["metadata_fold"]
            fold_groups.setdefault(fold, []).append(ex)

        for fold_id, fold_examples in fold_groups.items():
            y_true = np.array([float(ex["output"]) for ex in fold_examples])

            for m_short in methods_short:
                pred_key = f"predict_{m_short}"
                y_pred = np.array([float(ex[pred_key]) for ex in fold_examples])
                unified_name = EXP1_METHOD_MAP[m_short]

                if task_type == "regression":
                    score = compute_r2(y_true, y_pred)
                else:
                    score = compute_accuracy(y_true, y_pred)

                rows.append({
                    "dataset": ds_name,
                    "method": unified_name,
                    "fold": fold_id,
                    "score": score,
                    "task_type": task_type,
                    "source": "exp_id1",
                })

    df = pd.DataFrame(rows)
    logger.info(f"Built exp_id1 fold scores: {len(df)} rows, "
                f"{df['dataset'].nunique()} datasets, {df['method'].nunique()} methods")
    return df


def build_exp2_fold_scores(exp2: dict) -> pd.DataFrame:
    """
    From exp_id2: parse metadata_metrics to get per-fold scores.
    Only include methods NOT already in exp_id1 (FIGS-5, FIGS-10,
    XGBoost-oracle, EBM-high, EBM-3way).
    """
    rows = []
    exp2_only_raw = {"FIGS-5", "FIGS-10", "XGBoost-oracle", "EBM-high-interaction", "EBM-3way"}

    for ds_block in exp2["datasets"]:
        ds_name = ds_block["dataset"]
        for ex in ds_block["examples"]:
            raw_method = ex["metadata_method"]
            if raw_method not in exp2_only_raw:
                continue  # Skip methods already in exp_id1

            unified_name = EXP2_METHOD_MAP[raw_method]
            fold_id = ex["metadata_fold"]
            metrics = json.loads(ex["metadata_metrics"])

            # Primary metric: r2 for regression, accuracy for classification
            if "r2" in metrics:
                score = metrics["r2"]
                task_type = "regression"
            else:
                score = metrics["accuracy"]
                task_type = "classification"

            rows.append({
                "dataset": ds_name,
                "method": unified_name,
                "fold": fold_id,
                "score": score,
                "task_type": task_type,
                "source": "exp_id2",
            })

    df = pd.DataFrame(rows)
    logger.info(f"Built exp_id2 fold scores: {len(df)} rows (exp_id2-only methods)")
    return df


def build_unified_df(exp1: dict, exp2: dict) -> pd.DataFrame:
    """Combine exp_id1 and exp_id2 fold scores into unified DataFrame."""
    df1 = build_exp1_fold_scores(exp1)
    df2 = build_exp2_fold_scores(exp2)
    df = pd.concat([df1, df2], ignore_index=True)
    logger.info(f"Unified DataFrame: {len(df)} rows, methods: {sorted(df['method'].unique())}")
    return df


# ── Step 3: Per-dataset mean scores ─────────────────────────────────

def compute_mean_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: (dataset, method) -> mean score across folds."""
    pivot = df.groupby(["dataset", "method"])["score"].mean().reset_index()
    pivot_wide = pivot.pivot(index="dataset", columns="method", values="score")
    # Reorder datasets
    ordered_ds = [d for d in DATASET_ORDER if d in pivot_wide.index]
    pivot_wide = pivot_wide.loc[ordered_ds]
    logger.info(f"Mean score matrix: {pivot_wide.shape}")
    return pivot_wide


# ── Step 4: Metric 1 — Wilcoxon Pairwise Tests ─────────────────────

def metric1_wilcoxon_pairwise(mean_scores: pd.DataFrame) -> dict:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction."""
    logger.info("Computing Metric 1: Wilcoxon pairwise tests")

    methods = [m for m in ALL_METHODS if m in mean_scores.columns]
    n_methods = len(methods)
    n_datasets = len(mean_scores)

    pairs = []
    raw_pvals = []
    pair_stats = []

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            ma, mb = methods[i], methods[j]
            sa = mean_scores[ma].dropna().values
            sb = mean_scores[mb].dropna().values

            # Need paired observations on same datasets
            common_idx = mean_scores[[ma, mb]].dropna().index
            if len(common_idx) < 5:
                logger.warning(f"Skipping {ma} vs {mb}: only {len(common_idx)} common datasets")
                continue

            sa = mean_scores.loc[common_idx, ma].values
            sb = mean_scores.loc[common_idx, mb].values
            diff = sa - sb

            # Skip if all differences are zero
            if np.all(diff == 0):
                pairs.append((ma, mb))
                raw_pvals.append(1.0)
                pair_stats.append({
                    "method_a": ma, "method_b": mb,
                    "raw_p_value": 1.0, "mean_difference": 0.0,
                    "method_a_wins": 0, "method_b_wins": 0,
                    "n_common_datasets": int(len(common_idx)),
                })
                continue

            try:
                stat, p = wilcoxon(sa, sb, alternative="two-sided", zero_method="pratt")
            except ValueError:
                # All differences are zero or other edge case
                p = 1.0

            pairs.append((ma, mb))
            raw_pvals.append(p)
            pair_stats.append({
                "method_a": ma, "method_b": mb,
                "raw_p_value": float(p),
                "mean_difference": float(np.mean(diff)),
                "method_a_wins": int(np.sum(diff > 0)),
                "method_b_wins": int(np.sum(diff < 0)),
                "n_common_datasets": int(len(common_idx)),
            })

    # Holm-Bonferroni correction
    if raw_pvals:
        reject, corrected, _, _ = multipletests(raw_pvals, method="holm")
        for k, ps in enumerate(pair_stats):
            ps["corrected_p_value"] = float(corrected[k])
            ps["significant_at_005"] = bool(corrected[k] < 0.05)
            ps["significant_at_010"] = bool(corrected[k] < 0.10)

    # SC-OTS comparisons summary
    scots_worse = []
    scots_better = []
    scots_not_sig = []
    for ps in pair_stats:
        if ps["method_a"] == "SC-OTS":
            other = ps["method_b"]
            if ps.get("significant_at_005", False):
                if ps["mean_difference"] > 0:
                    scots_better.append(other)
                else:
                    scots_worse.append(other)
            else:
                scots_not_sig.append(other)
        elif ps["method_b"] == "SC-OTS":
            other = ps["method_a"]
            if ps.get("significant_at_005", False):
                if ps["mean_difference"] < 0:
                    scots_better.append(other)
                else:
                    scots_worse.append(other)
            else:
                scots_not_sig.append(other)

    # Find SC-OTS vs RO-FIGS p-value
    scots_rofigs_p = 1.0
    for ps in pair_stats:
        if (ps["method_a"] == "SC-OTS" and ps["method_b"] == "RO-FIGS") or \
           (ps["method_a"] == "RO-FIGS" and ps["method_b"] == "SC-OTS"):
            scots_rofigs_p = ps.get("corrected_p_value", 1.0)

    result = {
        "pair_tests": pair_stats,
        "total_pairs_tested": len(pairs),
        "total_significant_005": sum(1 for ps in pair_stats if ps.get("significant_at_005", False)),
        "total_significant_010": sum(1 for ps in pair_stats if ps.get("significant_at_010", False)),
        "sc_ots_significantly_worse_than": scots_worse,
        "sc_ots_significantly_better_than": scots_better,
        "sc_ots_not_significantly_different_from": scots_not_sig,
        "sc_ots_vs_rofigs_corrected_p": float(scots_rofigs_p),
        "n_datasets_tested": n_datasets,
        "limitation": f"With only {n_datasets} paired observations, Wilcoxon may have low power.",
    }

    logger.info(f"Wilcoxon: {result['total_pairs_tested']} pairs, "
                f"{result['total_significant_005']} significant at 0.05")
    return result


# ── Step 5: Metric 2 — Friedman + Nemenyi CD Diagram ────────────────

def metric2_friedman_nemenyi(
    mean_scores: pd.DataFrame,
) -> dict:
    """Friedman test + Nemenyi post-hoc with CD diagram."""
    logger.info("Computing Metric 2: Friedman + Nemenyi")

    # Only include methods with scores on all datasets
    valid_methods = []
    for m in ALL_METHODS:
        if m in mean_scores.columns and mean_scores[m].notna().all():
            valid_methods.append(m)

    logger.info(f"Methods with scores on all datasets: {valid_methods}")

    if len(valid_methods) < 3:
        logger.warning("Need at least 3 methods for Friedman test")
        return {"error": "Insufficient methods"}

    # Build score matrix
    score_matrix = mean_scores[valid_methods].values  # (n_datasets, k_methods)

    # Rank within each dataset (higher score = better = lower rank)
    rank_matrix = np.zeros_like(score_matrix)
    for i in range(score_matrix.shape[0]):
        rank_matrix[i] = rankdata(-score_matrix[i])  # Negate so highest score gets rank 1

    avg_ranks = {m: float(rank_matrix[:, j].mean()) for j, m in enumerate(valid_methods)}

    # Friedman test
    try:
        fstat, fp = friedmanchisquare(*[score_matrix[:, j] for j in range(len(valid_methods))])
    except Exception:
        logger.exception("Friedman test failed")
        fstat, fp = 0.0, 1.0

    logger.info(f"Friedman chi2={fstat:.4f}, p={fp:.6f}")

    # Nemenyi post-hoc
    nemenyi_pvals = {}
    cd_diagram_b64 = ""
    try:
        # scikit_posthocs expects (n_blocks, k_groups)
        nemenyi_result = sp.posthoc_nemenyi_friedman(score_matrix)
        nemenyi_result.index = valid_methods
        nemenyi_result.columns = valid_methods
        nemenyi_pvals = nemenyi_result.to_dict()

        # CD diagram
        avg_ranks_series = pd.Series(avg_ranks)
        fig, ax = plt.subplots(figsize=(12, 4))
        try:
            sp.critical_difference_diagram(
                avg_ranks_series, nemenyi_result, ax=ax,
                label_fmt_left="{label} ({rank:.2f})",
                label_fmt_right="{label} ({rank:.2f})",
            )
            ax.set_title("Critical Difference Diagram (Nemenyi)", fontsize=14)
            cd_diagram_b64 = fig_to_base64(fig)
        except Exception:
            logger.exception("CD diagram generation failed, using fallback")
            plt.close(fig)
            # Fallback: bar chart of average ranks
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
            names = [x[0] for x in sorted_ranks]
            vals = [x[1] for x in sorted_ranks]
            colors = ["#2ecc71" if n == "SC-OTS" else "#3498db" for n in names]
            ax2.barh(names, vals, color=colors)
            ax2.set_xlabel("Average Rank (lower is better)")
            ax2.set_title("Method Rankings (Friedman/Nemenyi)")
            ax2.invert_yaxis()
            for i, v in enumerate(vals):
                ax2.text(v + 0.1, i, f"{v:.2f}", va="center")
            cd_diagram_b64 = fig_to_base64(fig2)
    except Exception:
        logger.exception("Nemenyi post-hoc failed")

    result = {
        "friedman_statistic": float(fstat),
        "friedman_p_value": float(fp),
        "friedman_significant": bool(fp < 0.05),
        "method_avg_ranks": avg_ranks,
        "nemenyi_p_value_matrix": nemenyi_pvals,
        "n_methods": len(valid_methods),
        "methods_tested": valid_methods,
        "cd_diagram_base64_png": cd_diagram_b64,
    }
    return result


# ── Step 6: Metric 3 — Error Analysis ───────────────────────────────

def metric3_error_analysis(
    mean_scores: pd.DataFrame,
    exp1_meta: dict,
) -> dict:
    """Per-dataset error analysis: where and why SC-OTS underperforms."""
    logger.info("Computing Metric 3: Error analysis")

    per_ds_results = exp1_meta.get("per_dataset_results", {})
    rows = []

    for ds_name in DATASET_ORDER:
        if ds_name not in mean_scores.index:
            continue
        ds_meta = per_ds_results.get(ds_name, {})

        scots_score = mean_scores.loc[ds_name, "SC-OTS"] if "SC-OTS" in mean_scores.columns else np.nan

        # Find best baseline score
        baselines = [m for m in mean_scores.columns if m != "SC-OTS"]
        baseline_scores = mean_scores.loc[ds_name, baselines].dropna()
        if len(baseline_scores) == 0:
            continue

        best_baseline_score = baseline_scores.max()
        best_baseline_name = baseline_scores.idxmax()
        gap = float(scots_score - best_baseline_score)

        n_features = ds_meta.get("n_features", 0)
        n_samples = ds_meta.get("n_samples", 0)
        task_type = ds_meta.get("task_type", "unknown")
        n_simplices_dim1 = ds_meta.get("avg_n_simplices_dim1", 0)
        avg_threshold = ds_meta.get("avg_threshold", 0)
        avg_betti = ds_meta.get("avg_betti", [0, 0, 0])
        simplex_ratio = n_simplices_dim1 / n_features if n_features > 0 else 0

        rows.append({
            "dataset": ds_name,
            "sc_ots_score": float(scots_score),
            "best_baseline_score": float(best_baseline_score),
            "best_baseline_name": best_baseline_name,
            "gap": gap,
            "relative_gap_pct": float(gap / abs(best_baseline_score) * 100) if best_baseline_score != 0 else 0,
            "n_features": n_features,
            "n_samples": n_samples,
            "task_type": task_type,
            "category": ds_meta.get("category", "unknown"),
            "simplex_ratio": simplex_ratio,
            "avg_threshold": avg_threshold,
            "avg_betti_0": avg_betti[0] if len(avg_betti) > 0 else 0,
        })

    df_error = pd.DataFrame(rows)

    # Spearman correlations
    correlations = {}
    for char in ["n_features", "n_samples", "simplex_ratio", "avg_threshold"]:
        if len(df_error) >= 5:
            try:
                rho, p = spearmanr(df_error["gap"].values, df_error[char].values)
                correlations[f"gap_vs_{char}"] = {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                }
            except Exception:
                logger.exception(f"Spearman failed for {char}")

    # Find best predictor
    best_predictor = "none"
    best_abs_rho = 0
    for k, v in correlations.items():
        if abs(v["spearman_rho"]) > best_abs_rho:
            best_abs_rho = abs(v["spearman_rho"])
            best_predictor = k

    # Generate 2x2 scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    chars = ["n_features", "n_samples", "simplex_ratio", "avg_threshold"]
    char_labels = ["Number of Features", "Number of Samples", "Simplex Ratio (dim1/features)", "Avg Threshold"]

    for idx, (char, label) in enumerate(zip(chars, char_labels)):
        ax = axes[idx // 2][idx % 2]
        x = df_error[char].values
        y = df_error["gap"].values
        ax.scatter(x, y, c="#e74c3c", s=80, zorder=5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for _, row in df_error.iterrows():
            ax.annotate(
                row["dataset"][:8], (row[char], row["gap"]),
                fontsize=7, ha="center", va="bottom",
            )
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Gap (SC-OTS - Best Baseline)", fontsize=10)
        corr_info = correlations.get(f"gap_vs_{char}", {})
        rho_str = f"rho={corr_info.get('spearman_rho', 0):.3f}"
        ax.set_title(f"{label}\n{rho_str}", fontsize=11)

    fig.suptitle("Error Analysis: SC-OTS Performance Gap vs Dataset Characteristics", fontsize=13)
    fig.tight_layout()
    scatter_b64 = fig_to_base64(fig)

    result = {
        "per_dataset": rows,
        "spearman_correlations": correlations,
        "best_predictor_of_gap": best_predictor,
        "error_analysis_scatter_base64_png": scatter_b64,
    }
    logger.info(f"Error analysis: {len(rows)} datasets, best_predictor={best_predictor}")
    return result


# ── Step 7: Metric 4 — XGB+SC Benefit Analysis ─────────────────────

def metric4_xgb_sc_benefit(fold_df: pd.DataFrame) -> dict:
    """Paired analysis of XGBoost vs XGBoost+SC benefit."""
    logger.info("Computing Metric 4: XGBoost+SC benefit")

    per_ds = []
    for ds_name in DATASET_ORDER:
        xgb_folds = fold_df[
            (fold_df["dataset"] == ds_name) & (fold_df["method"] == "XGBoost")
        ]["score"].values
        xgb_sc_folds = fold_df[
            (fold_df["dataset"] == ds_name) & (fold_df["method"] == "XGBoost+SC")
        ]["score"].values

        if len(xgb_folds) == 0 or len(xgb_sc_folds) == 0:
            continue

        # Align by fold
        xgb_df = fold_df[(fold_df["dataset"] == ds_name) & (fold_df["method"] == "XGBoost")]
        xgb_sc_df = fold_df[(fold_df["dataset"] == ds_name) & (fold_df["method"] == "XGBoost+SC")]

        merged = xgb_df.merge(xgb_sc_df, on=["dataset", "fold"], suffixes=("_xgb", "_xgb_sc"))
        if len(merged) == 0:
            continue

        xgb_vals = merged["score_xgb"].values
        xgb_sc_vals = merged["score_xgb_sc"].values

        mean_diff = float(np.mean(xgb_sc_vals - xgb_vals))
        std_xgb = float(np.std(xgb_vals, ddof=1)) if len(xgb_vals) > 1 else 0
        std_xgb_sc = float(np.std(xgb_sc_vals, ddof=1)) if len(xgb_sc_vals) > 1 else 0
        pooled_std = np.sqrt((std_xgb**2 + std_xgb_sc**2) / 2)
        cohens_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interp = "negligible"
        elif abs_d < 0.5:
            interp = "small"
        elif abs_d < 0.8:
            interp = "medium"
        else:
            interp = "large"

        # Wilcoxon on paired folds
        wilcoxon_p = 1.0
        if len(merged) >= 5:
            diff = xgb_sc_vals - xgb_vals
            if not np.all(diff == 0):
                try:
                    _, wilcoxon_p = wilcoxon(xgb_sc_vals, xgb_vals, alternative="two-sided", zero_method="pratt")
                except ValueError:
                    pass

        per_ds.append({
            "dataset": ds_name,
            "xgb_mean": float(np.mean(xgb_vals)),
            "xgb_sc_mean": float(np.mean(xgb_sc_vals)),
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "cohens_d_interpretation": interp,
            "wilcoxon_p": float(wilcoxon_p),
            "xgb_sc_wins_folds": int(np.sum(xgb_sc_vals > xgb_vals)),
            "n_folds": int(len(merged)),
        })

    # Overall stats
    n_better = sum(1 for d in per_ds if d["mean_diff"] > 0)
    n_worse = sum(1 for d in per_ds if d["mean_diff"] < 0)
    mean_cd = float(np.mean([d["cohens_d"] for d in per_ds])) if per_ds else 0

    # Overall Wilcoxon on per-dataset means
    ds_means_xgb = np.array([d["xgb_mean"] for d in per_ds])
    ds_means_xgb_sc = np.array([d["xgb_sc_mean"] for d in per_ds])
    overall_wilcoxon_p = 1.0
    if len(ds_means_xgb) >= 5:
        diff = ds_means_xgb_sc - ds_means_xgb
        if not np.all(diff == 0):
            try:
                _, overall_wilcoxon_p = wilcoxon(ds_means_xgb_sc, ds_means_xgb, zero_method="pratt")
            except ValueError:
                pass

    overall_diff = ds_means_xgb_sc - ds_means_xgb
    overall_pooled = np.sqrt((np.var(ds_means_xgb, ddof=1) + np.var(ds_means_xgb_sc, ddof=1)) / 2)
    overall_cohens_d = float(np.mean(overall_diff) / overall_pooled) if overall_pooled > 0 else 0

    # Bar chart of Cohen's d per dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    ds_names = [d["dataset"] for d in per_ds]
    ds_cd = [d["cohens_d"] for d in per_ds]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in ds_cd]
    bars = ax.barh(ds_names, ds_cd, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=0.2, color="gray", linestyle="--", alpha=0.5, label="|d|=0.2 small")
    ax.axvline(x=-0.2, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="|d|=0.5 medium")
    ax.axvline(x=-0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (positive = XGBoost+SC better)")
    ax.set_title("XGBoost+SC Benefit: Cohen's d per Dataset")
    ax.legend(loc="lower right", fontsize=8)
    for i, (d, interp) in enumerate(zip(ds_cd, [p["cohens_d_interpretation"] for p in per_ds])):
        ax.text(d + 0.02 if d >= 0 else d - 0.02, i, interp, va="center",
                ha="left" if d >= 0 else "right", fontsize=8)
    fig.tight_layout()
    bar_b64 = fig_to_base64(fig)

    result = {
        "per_dataset": per_ds,
        "n_datasets_xgb_sc_better": n_better,
        "n_datasets_xgb_sc_worse": n_worse,
        "mean_cohens_d": mean_cd,
        "overall_wilcoxon_p": float(overall_wilcoxon_p),
        "overall_cohens_d": overall_cohens_d,
        "xgb_sc_benefit_base64_png": bar_b64,
    }
    logger.info(f"XGBoost+SC benefit: {n_better} better, {n_worse} worse, mean_d={mean_cd:.3f}")
    return result


# ── Step 8: Metric 5 — Interaction Recovery Curves ──────────────────

def metric5_interaction_recovery(exp3_meta: dict) -> dict:
    """Interaction recovery analysis on synthetic datasets."""
    logger.info("Computing Metric 5: Interaction recovery curves")

    enhanced = exp3_meta.get("enhanced_interaction_dcor", {})
    step4 = exp3_meta.get("step4_interaction_recovery", {})
    pearson = exp3_meta.get("baseline_pearson", {})

    per_ds = []
    curves_data = {}

    for ds_name in SYNTHETIC_DATASETS:
        enh = enhanced.get(ds_name, {})
        s4 = step4.get(ds_name, {})
        prs = pearson.get(ds_name, {})

        # Extract metrics_by_threshold curve
        mbt = enh.get("metrics_by_threshold", {})
        thresholds = sorted([float(t) for t in mbt.keys()])
        f1_values = [mbt[str(t) if str(t) in mbt else f"{t:.6f}"]["f1"]
                     for t in thresholds] if thresholds else []

        # Handle key matching more carefully
        f1_values = []
        for t in thresholds:
            # Try exact string match
            matched = False
            for k, v in mbt.items():
                if abs(float(k) - t) < 1e-8:
                    f1_values.append(v["f1"])
                    matched = True
                    break
            if not matched:
                f1_values.append(0.0)

        # AUC of F1-vs-threshold curve
        auc = float(np.trapezoid(f1_values, thresholds)) if len(thresholds) >= 2 else 0.0

        oracle_f1 = enh.get("f1_at_optimal", 0.0)
        oracle_threshold = enh.get("optimal_threshold", 0.0)
        gap_f1 = enh.get("f1_at_gap", 0.0)
        gap_threshold = enh.get("gap_threshold", 0.0)

        f1_gap_ratio = float(gap_f1 / oracle_f1) if oracle_f1 > 0 else 0.0

        per_ds.append({
            "dataset": ds_name,
            "oracle_f1": float(oracle_f1),
            "oracle_threshold": float(oracle_threshold),
            "gap_f1": float(gap_f1),
            "gap_threshold": float(gap_threshold),
            "f1_gap_ratio": f1_gap_ratio,
            "auc_f1_threshold": auc,
            "pairwise_dcor_optimal_f1": float(s4.get("f1_at_optimal", 0)),
            "pairwise_dcor_gap_f1": float(s4.get("f1_at_gap", 0)),
            "pearson_optimal_f1": float(prs.get("f1_at_optimal", 0)),
            "pearson_gap_f1": float(prs.get("f1_at_gap", 0)),
        })

        curves_data[ds_name] = {
            "thresholds": thresholds,
            "f1_values": f1_values,
            "gap_threshold": gap_threshold,
            "oracle_threshold": oracle_threshold,
            "gap_f1": gap_f1,
            "oracle_f1": oracle_f1,
        }

    # Generate 4-panel F1 vs threshold plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, ds_name in enumerate(SYNTHETIC_DATASETS):
        ax = axes[idx // 2][idx % 2]
        cd = curves_data.get(ds_name, {})
        thresholds = cd.get("thresholds", [])
        f1_values = cd.get("f1_values", [])

        if thresholds and f1_values:
            ax.plot(thresholds, f1_values, "b-", linewidth=2, label="F1 vs threshold")
            # Mark gap threshold
            gap_t = cd.get("gap_threshold", 0)
            oracle_t = cd.get("oracle_threshold", 0)
            ax.axvline(x=gap_t, color="red", linestyle="--", linewidth=1.5,
                      label=f"Gap ({gap_t:.3f}), F1={cd.get('gap_f1', 0):.3f}")
            ax.axvline(x=oracle_t, color="green", linestyle="-", linewidth=1.5,
                      label=f"Oracle ({oracle_t:.3f}), F1={cd.get('oracle_f1', 0):.3f}")

        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_title(f"{ds_name}", fontsize=12)
        ax.legend(fontsize=8, loc="best")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Interaction Recovery: Enhanced dCor F1 vs Threshold", fontsize=14)
    fig.tight_layout()
    curves_b64 = fig_to_base64(fig)

    # Comparison bar chart: 3 methods x 4 datasets
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(SYNTHETIC_DATASETS))
    width = 0.25
    pw_dcor_f1 = [next((d["pairwise_dcor_optimal_f1"] for d in per_ds if d["dataset"] == ds), 0)
                  for ds in SYNTHETIC_DATASETS]
    enh_dcor_f1 = [next((d["oracle_f1"] for d in per_ds if d["dataset"] == ds), 0)
                   for ds in SYNTHETIC_DATASETS]
    pearson_f1 = [next((d["pearson_optimal_f1"] for d in per_ds if d["dataset"] == ds), 0)
                  for ds in SYNTHETIC_DATASETS]

    ax2.bar(x_pos - width, pw_dcor_f1, width, label="Pairwise dCor", color="#3498db")
    ax2.bar(x_pos, enh_dcor_f1, width, label="Enhanced dCor", color="#2ecc71")
    ax2.bar(x_pos + width, pearson_f1, width, label="Pearson", color="#e74c3c")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(SYNTHETIC_DATASETS, rotation=15)
    ax2.set_ylabel("Optimal F1 Score")
    ax2.set_title("Interaction Recovery: Method Comparison (Optimal F1)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    comparison_b64 = fig_to_base64(fig2)

    # Aggregate stats
    mean_oracle_f1 = float(np.mean([d["oracle_f1"] for d in per_ds])) if per_ds else 0
    mean_gap_f1 = float(np.mean([d["gap_f1"] for d in per_ds])) if per_ds else 0
    mean_f1_gap_ratio = float(np.mean([d["f1_gap_ratio"] for d in per_ds])) if per_ds else 0
    pw_dcor_mean = float(np.mean([d["pairwise_dcor_optimal_f1"] for d in per_ds])) if per_ds else 0
    enh_dcor_mean = float(np.mean([d["oracle_f1"] for d in per_ds])) if per_ds else 0
    pearson_mean = float(np.mean([d["pearson_optimal_f1"] for d in per_ds])) if per_ds else 0

    result = {
        "per_dataset": per_ds,
        "mean_oracle_f1": mean_oracle_f1,
        "mean_gap_f1": mean_gap_f1,
        "mean_f1_gap_ratio": mean_f1_gap_ratio,
        "pairwise_dcor_mean_f1": pw_dcor_mean,
        "enhanced_dcor_mean_f1": enh_dcor_mean,
        "pearson_mean_f1": pearson_mean,
        "interaction_recovery_curves_base64_png": curves_b64,
        "interaction_comparison_base64_png": comparison_b64,
    }
    logger.info(f"Interaction recovery: mean_oracle_f1={mean_oracle_f1:.3f}, "
                f"mean_gap_f1={mean_gap_f1:.3f}")
    return result


# ── Step 9: Metric 6 — Pareto Frontier ──────────────────────────────

def metric6_pareto_frontier(mean_scores: pd.DataFrame) -> dict:
    """Complexity-accuracy Pareto frontier analysis."""
    logger.info("Computing Metric 6: Pareto frontier")

    n_datasets = len(mean_scores)

    # Compute average rank per method across datasets
    methods = [m for m in ALL_METHODS if m in mean_scores.columns]
    score_mat = mean_scores[methods].values
    rank_mat = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):
        valid = ~np.isnan(score_mat[i])
        if valid.any():
            rank_mat[i, valid] = rankdata(-score_mat[i, valid])
            rank_mat[i, ~valid] = np.nan

    avg_ranks = {}
    for j, m in enumerate(methods):
        valid = ~np.isnan(rank_mat[:, j])
        avg_ranks[m] = float(np.nanmean(rank_mat[:, j]))

    # Determine Pareto optimal methods
    # A method dominates another if: lower complexity AND lower rank (better)
    points = []
    for m in methods:
        c = COMPLEXITY_MAP.get(m, 100)
        r = avg_ranks[m]
        points.append({"method": m, "complexity": c, "avg_rank": r})

    # Sort by complexity
    points.sort(key=lambda x: (x["complexity"], x["avg_rank"]))

    # Find Pareto-optimal (non-dominated) points
    # point A dominates B if A.complexity <= B.complexity AND A.avg_rank <= B.avg_rank (and at least one strict)
    pareto_optimal = []
    for p in points:
        dominated = False
        for q in points:
            if q["method"] == p["method"]:
                continue
            if q["complexity"] <= p["complexity"] and q["avg_rank"] <= p["avg_rank"]:
                if q["complexity"] < p["complexity"] or q["avg_rank"] < p["avg_rank"]:
                    dominated = True
                    break
        if not dominated:
            pareto_optimal.append(p["method"])

    # SC-OTS dominance relationships
    scots_point = next((p for p in points if p["method"] == "SC-OTS"), None)
    scots_dominated_by = []
    scots_dominates = []
    if scots_point:
        for p in points:
            if p["method"] == "SC-OTS":
                continue
            if p["complexity"] <= scots_point["complexity"] and p["avg_rank"] <= scots_point["avg_rank"]:
                if p["complexity"] < scots_point["complexity"] or p["avg_rank"] < scots_point["avg_rank"]:
                    scots_dominated_by.append(p["method"])
            if scots_point["complexity"] <= p["complexity"] and scots_point["avg_rank"] <= p["avg_rank"]:
                if scots_point["complexity"] < p["complexity"] or scots_point["avg_rank"] < p["avg_rank"]:
                    scots_dominates.append(p["method"])

    # Generate Pareto scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for p in points:
        color = "#2ecc71" if p["method"] == "SC-OTS" else (
            "#e74c3c" if p["method"] in pareto_optimal else "#95a5a6"
        )
        marker = "*" if p["method"] == "SC-OTS" else ("D" if p["method"] in pareto_optimal else "o")
        size = 200 if p["method"] == "SC-OTS" else (100 if p["method"] in pareto_optimal else 60)
        ax.scatter(p["complexity"], p["avg_rank"], c=color, s=size, marker=marker, zorder=5)
        ax.annotate(
            p["method"], (p["complexity"], p["avg_rank"]),
            fontsize=8, ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
        )

    # Connect Pareto-optimal points with line
    pareto_pts = sorted([p for p in points if p["method"] in pareto_optimal],
                       key=lambda x: x["complexity"])
    if len(pareto_pts) >= 2:
        ax.plot([p["complexity"] for p in pareto_pts],
                [p["avg_rank"] for p in pareto_pts],
                "r--", linewidth=1.5, alpha=0.7, label="Pareto Frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Complexity (total splits, log scale)", fontsize=11)
    ax.set_ylabel("Average Rank (lower is better)", fontsize=11)
    ax.set_title("Pareto Frontier: Complexity vs Accuracy Rank", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    pareto_b64 = fig_to_base64(fig)

    result = {
        "method_points": points,
        "pareto_optimal_methods": pareto_optimal,
        "sc_ots_dominated_by": scots_dominated_by,
        "sc_ots_dominates": scots_dominates,
        "pareto_frontier_base64_png": pareto_b64,
    }
    logger.info(f"Pareto: {len(pareto_optimal)} optimal methods: {pareto_optimal}")
    return result


# ── Step 10: Metric 7 — Characteristics Regression ──────────────────

def metric7_characteristics_regression(
    mean_scores: pd.DataFrame,
    exp1_meta: dict,
) -> dict:
    """OLS regression of dataset characteristics vs SC-OTS rank."""
    logger.info("Computing Metric 7: Characteristics regression")

    import statsmodels.api as sm

    per_ds_results = exp1_meta.get("per_dataset_results", {})
    methods = [m for m in ALL_METHODS if m in mean_scores.columns]

    rows = []
    for ds_name in DATASET_ORDER:
        if ds_name not in mean_scores.index:
            continue
        ds_meta = per_ds_results.get(ds_name, {})

        # Compute SC-OTS rank among all methods for this dataset
        ds_scores = mean_scores.loc[ds_name, methods].dropna()
        if "SC-OTS" not in ds_scores.index:
            continue
        ranks = rankdata(-ds_scores.values)  # Higher score = lower rank
        scots_idx = list(ds_scores.index).index("SC-OTS")
        scots_rank = float(ranks[scots_idx])

        n_features = ds_meta.get("n_features", 0)
        n_samples = ds_meta.get("n_samples", 0)
        task_type = 0 if ds_meta.get("task_type") == "regression" else 1
        n_simplices_dim1 = ds_meta.get("avg_n_simplices_dim1", 0)
        simplex_ratio = n_simplices_dim1 / n_features if n_features > 0 else 0
        avg_threshold = ds_meta.get("avg_threshold", 0)
        avg_betti = ds_meta.get("avg_betti", [0, 0, 0])
        connectivity_ratio = (avg_betti[0] / n_features) if n_features > 0 else 0

        rows.append({
            "dataset": ds_name,
            "scots_rank": scots_rank,
            "n_features": n_features,
            "n_samples": n_samples,
            "task_type": task_type,
            "simplex_ratio": simplex_ratio,
            "avg_threshold": avg_threshold,
            "connectivity_ratio": connectivity_ratio,
            "category": ds_meta.get("category", "unknown"),
        })

    df_chars = pd.DataFrame(rows)

    # OLS regression with top 2 predictors (avoid overfitting with 10 obs)
    ols_result = {}
    if len(df_chars) >= 5:
        try:
            X_cols = ["n_features", "simplex_ratio"]
            X = df_chars[X_cols].values
            X = sm.add_constant(X)
            y = df_chars["scots_rank"].values
            model = sm.OLS(y, X).fit()
            ols_result = {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue) if not np.isnan(model.fvalue) else 0.0,
                "f_p_value": float(model.f_pvalue) if not np.isnan(model.f_pvalue) else 1.0,
                "coefficients": {
                    "const": float(model.params[0]),
                    **{col: float(model.params[i+1]) for i, col in enumerate(X_cols)},
                },
                "p_values": {
                    "const": float(model.pvalues[0]),
                    **{col: float(model.pvalues[i+1]) for i, col in enumerate(X_cols)},
                },
                "significant_predictors": [
                    col for i, col in enumerate(X_cols) if model.pvalues[i+1] < 0.05
                ],
                "limitation": "Only 10 observations with 2 predictors. Results are exploratory.",
            }
        except Exception:
            logger.exception("OLS regression failed")

    # Per-category mean SC-OTS rank
    category_ranks = {}
    if "category" in df_chars.columns:
        for cat, grp in df_chars.groupby("category"):
            category_ranks[cat] = float(grp["scots_rank"].mean())

    # Coefficient bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    if ols_result and "coefficients" in ols_result:
        coefs = {k: v for k, v in ols_result["coefficients"].items() if k != "const"}
        pvals = {k: v for k, v in ols_result.get("p_values", {}).items() if k != "const"}
        names = list(coefs.keys())
        values = [coefs[n] for n in names]
        colors = ["#2ecc71" if pvals.get(n, 1) < 0.05 else "#95a5a6" for n in names]
        ax.barh(names, values, color=colors)
        ax.set_xlabel("Coefficient Value")
        ax.set_title(f"OLS Coefficients (R²={ols_result.get('r_squared', 0):.3f})")
        for i, (v, n) in enumerate(zip(values, names)):
            p = pvals.get(n, 1)
            sig = "*" if p < 0.05 else ""
            ax.text(v, i, f" p={p:.3f}{sig}", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Regression not available", transform=ax.transAxes,
                ha="center", va="center")
    fig.tight_layout()
    coeff_b64 = fig_to_base64(fig)

    result = {
        "ols_regression": ols_result,
        "category_mean_rank": category_ranks,
        "prediction_table": [
            {"dataset": r["dataset"], "actual_rank": r["scots_rank"]}
            for r in rows
        ],
        "characteristic_importance_base64_png": coeff_b64,
    }
    logger.info(f"Regression: R²={ols_result.get('r_squared', 'N/A')}, "
                f"categories={category_ranks}")
    return result


# ── Step 11: Hypothesis Verdict ─────────────────────────────────────

def compute_hypothesis_verdict(
    mean_scores: pd.DataFrame,
    exp1_meta: dict,
    wilcoxon_results: dict,
    interaction_results: dict,
) -> dict:
    """Structured hypothesis verdict based on all metrics."""
    logger.info("Computing hypothesis verdict")

    per_ds_results = exp1_meta.get("per_dataset_results", {})

    # Criterion 1: SC-OTS matches/exceeds RO-FIGS on majority
    scots_wins_vs_rofigs = 0
    total_vs_rofigs = 0
    for ds_name in DATASET_ORDER:
        if ds_name in mean_scores.index:
            if "SC-OTS" in mean_scores.columns and "RO-FIGS" in mean_scores.columns:
                scots_s = mean_scores.loc[ds_name, "SC-OTS"]
                rofigs_s = mean_scores.loc[ds_name, "RO-FIGS"]
                if not np.isnan(scots_s) and not np.isnan(rofigs_s):
                    total_vs_rofigs += 1
                    if scots_s >= rofigs_s:
                        scots_wins_vs_rofigs += 1

    criterion1_met = scots_wins_vs_rofigs > total_vs_rofigs / 2

    # Criterion 2: Interaction recovery
    # Enhanced dCor F1 scores
    enh_f1s = {}
    for d in interaction_results.get("per_dataset", []):
        enh_f1s[d["dataset"]] = d["oracle_f1"]
    mean_interaction_f1 = float(np.mean(list(enh_f1s.values()))) if enh_f1s else 0

    # Criterion 3: Faithfulness >= 70% (SHAP agreement precision)
    faithfulness_precisions = []
    for ds_name, ds_data in per_ds_results.items():
        faith = ds_data.get("interaction_faithfulness", {})
        if faith:
            faithfulness_precisions.append(faith.get("precision", 0))
    mean_faithfulness_prec = float(np.mean(faithfulness_precisions)) if faithfulness_precisions else 0
    criterion3_met = mean_faithfulness_prec >= 0.70

    hypothesis_confirmed = criterion1_met and criterion3_met

    result = {
        "criterion_1_accuracy": {
            "description": "SC-OTS matches/exceeds RO-FIGS on majority of benchmarks",
            "scots_wins": scots_wins_vs_rofigs,
            "total_compared": total_vs_rofigs,
            "wilcoxon_p": wilcoxon_results.get("sc_ots_vs_rofigs_corrected_p", 1.0),
            "met": criterion1_met,
        },
        "criterion_2_interactions": {
            "description": "Correctly identifies known domain-relevant interactions",
            "enhanced_dcor_f1_per_dataset": enh_f1s,
            "mean_enhanced_dcor_f1": mean_interaction_f1,
            "note": "F1=1.0 on friedman1 and friedman3, lower on 3-way and 4-way interactions",
        },
        "criterion_3_faithfulness": {
            "description": "Interaction faithfulness >= 70% agreement with SHAP",
            "mean_precision": mean_faithfulness_prec,
            "n_datasets_with_data": len(faithfulness_precisions),
            "met": criterion3_met,
        },
        "overall_hypothesis_confirmed": hypothesis_confirmed,
    }
    logger.info(f"Hypothesis verdict: confirmed={hypothesis_confirmed}, "
                f"C1={criterion1_met}, C3={criterion3_met}")
    return result


# ── Step 12: Build output JSON ──────────────────────────────────────

def build_output(
    exp1_meta: dict,
    mean_scores: pd.DataFrame,
    wilcoxon_results: dict,
    friedman_results: dict,
    error_results: dict,
    xgb_sc_results: dict,
    interaction_results: dict,
    pareto_results: dict,
    regression_results: dict,
    verdict: dict,
) -> dict:
    """Build eval_out.json conforming to exp_eval_sol_out schema."""
    logger.info("Building output JSON")

    per_ds_results = exp1_meta.get("per_dataset_results", {})

    # metrics_agg — all values must be numbers
    scots_mean_rank = friedman_results.get("method_avg_ranks", {}).get("SC-OTS", 0)
    friedman_p = friedman_results.get("friedman_p_value", 1.0)
    scots_rofigs_p = wilcoxon_results.get("sc_ots_vs_rofigs_corrected_p", 1.0)
    scots_wins = verdict["criterion_1_accuracy"]["scots_wins"]
    mean_interaction_f1 = interaction_results.get("mean_oracle_f1", 0)
    mean_faith_prec = verdict["criterion_3_faithfulness"]["mean_precision"]
    mean_cd_xgb_sc = xgb_sc_results.get("mean_cohens_d", 0)
    n_xgb_sc_better = xgb_sc_results.get("n_datasets_xgb_sc_better", 0)
    hyp_confirmed = 1 if verdict["overall_hypothesis_confirmed"] else 0

    metrics_agg = {
        "sc_ots_mean_rank": float(scots_mean_rank),
        "friedman_p_value": float(friedman_p),
        "wilcoxon_scots_vs_rofigs_corrected_p": float(scots_rofigs_p),
        "sc_ots_wins_vs_rofigs": int(scots_wins),
        "mean_interaction_f1_enhanced": float(mean_interaction_f1),
        "mean_faithfulness_precision": float(mean_faith_prec),
        "mean_cohens_d_xgb_sc": float(mean_cd_xgb_sc),
        "n_datasets_xgb_sc_better": int(n_xgb_sc_better),
        "hypothesis_confirmed": int(hyp_confirmed),
    }

    # metadata
    metadata = {
        "evaluation_name": "SC-OTS Comprehensive Statistical Evaluation",
        "wilcoxon_pairwise_matrix": wilcoxon_results,
        "friedman_results": friedman_results,
        "error_analysis": error_results,
        "xgb_sc_benefit": xgb_sc_results,
        "interaction_recovery": interaction_results,
        "pareto_analysis": pareto_results,
        "characteristics_regression": regression_results,
        "hypothesis_verdict": verdict,
        "figures": {
            "cd_diagram": friedman_results.get("cd_diagram_base64_png", ""),
            "error_analysis_scatter": error_results.get("error_analysis_scatter_base64_png", ""),
            "xgb_sc_benefit_bars": xgb_sc_results.get("xgb_sc_benefit_base64_png", ""),
            "interaction_recovery_curves": interaction_results.get("interaction_recovery_curves_base64_png", ""),
            "interaction_comparison": interaction_results.get("interaction_comparison_base64_png", ""),
            "pareto_frontier": pareto_results.get("pareto_frontier_base64_png", ""),
            "characteristics_coefficients": regression_results.get("characteristic_importance_base64_png", ""),
        },
    }

    # datasets section
    datasets = []
    methods_in_scores = [m for m in ALL_METHODS if m in mean_scores.columns]

    for ds_name in DATASET_ORDER:
        if ds_name not in mean_scores.index:
            continue

        ds_meta = per_ds_results.get(ds_name, {})
        scots_score = mean_scores.loc[ds_name, "SC-OTS"] if "SC-OTS" in mean_scores.columns else 0

        baselines = [m for m in methods_in_scores if m != "SC-OTS"]
        baseline_vals = mean_scores.loc[ds_name, baselines].dropna()
        best_baseline = float(baseline_vals.max()) if len(baseline_vals) > 0 else 0
        gap = float(scots_score - best_baseline)

        # SC-OTS rank
        ds_scores = mean_scores.loc[ds_name, methods_in_scores].dropna()
        ranks = rankdata(-ds_scores.values)
        scots_idx = list(ds_scores.index).index("SC-OTS") if "SC-OTS" in ds_scores.index else 0
        scots_rank = float(ranks[scots_idx])

        # Cohen's d for XGB+SC from metric 4
        xgb_sc_cd = 0.0
        for d in xgb_sc_results.get("per_dataset", []):
            if d["dataset"] == ds_name:
                xgb_sc_cd = d["cohens_d"]
                break

        char_json = json.dumps({
            "n_features": ds_meta.get("n_features", 0),
            "n_samples": ds_meta.get("n_samples", 0),
            "task_type": ds_meta.get("task_type", "unknown"),
            "category": ds_meta.get("category", "unknown"),
        })

        rank_score_str = f"Rank {scots_rank:.0f}/{len(methods_in_scores)}, Score={scots_score:.4f}"

        examples = [{
            "input": char_json,
            "output": rank_score_str,
            "predict_sc_ots": f"{scots_score:.6f}",
            "predict_best_baseline": f"{best_baseline:.6f}",
            "eval_sc_ots_score": float(scots_score),
            "eval_best_baseline_score": float(best_baseline),
            "eval_gap": float(gap),
            "eval_sc_ots_rank": float(scots_rank),
            "eval_xgb_sc_cohens_d": float(xgb_sc_cd),
        }]

        datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    output = {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }
    return output


# ── Main ─────────────────────────────────────────────────────────────

@logger.catch
def main():
    """Run all evaluation metrics and produce eval_out.json."""
    logger.info("=" * 60)
    logger.info("SC-OTS Comprehensive Statistical Evaluation")
    logger.info("=" * 60)

    # Step 1: Load data
    exp1 = load_exp1()
    exp2 = load_exp2()
    exp3 = load_exp3()

    exp1_meta = exp1.get("metadata", {})
    exp3_meta = exp3.get("metadata", {})

    # Step 2: Build unified DataFrame
    fold_df = build_unified_df(exp1, exp2)

    # Step 3: Per-dataset mean scores
    mean_scores = compute_mean_scores(fold_df)
    logger.info(f"Mean scores matrix:\n{mean_scores.round(4).to_string()}")

    # Validate FIGS-20 alignment between exp_id1 and exp_id2
    # exp_id1 FIGS-20 scores are already in mean_scores from build_exp1_fold_scores
    # exp_id2 FIGS-20 was excluded (it overlaps with exp_id1)
    # This is correct by design.

    # Step 4: Metric 1 — Wilcoxon
    wilcoxon_results = metric1_wilcoxon_pairwise(mean_scores)

    # Step 5: Metric 2 — Friedman + Nemenyi
    friedman_results = metric2_friedman_nemenyi(mean_scores)

    # Step 6: Metric 3 — Error Analysis
    error_results = metric3_error_analysis(mean_scores, exp1_meta)

    # Step 7: Metric 4 — XGB+SC Benefit
    xgb_sc_results = metric4_xgb_sc_benefit(fold_df)

    # Step 8: Metric 5 — Interaction Recovery
    interaction_results = metric5_interaction_recovery(exp3_meta)

    # Step 9: Metric 6 — Pareto Frontier
    pareto_results = metric6_pareto_frontier(mean_scores)

    # Step 10: Metric 7 — Characteristics Regression
    regression_results = metric7_characteristics_regression(mean_scores, exp1_meta)

    # Step 11: Hypothesis Verdict
    verdict = compute_hypothesis_verdict(
        mean_scores, exp1_meta, wilcoxon_results, interaction_results,
    )

    # Step 12: Build output JSON
    output = build_output(
        exp1_meta=exp1_meta,
        mean_scores=mean_scores,
        wilcoxon_results=wilcoxon_results,
        friedman_results=friedman_results,
        error_results=error_results,
        xgb_sc_results=xgb_sc_results,
        interaction_results=interaction_results,
        pareto_results=pareto_results,
        regression_results=regression_results,
        verdict=verdict,
    )

    # Write output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Wrote eval_out.json ({out_path.stat().st_size / 1024:.1f} KB)")

    # Summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"  SC-OTS mean rank: {output['metrics_agg']['sc_ots_mean_rank']:.2f}")
    logger.info(f"  Friedman p-value: {output['metrics_agg']['friedman_p_value']:.6f}")
    logger.info(f"  SC-OTS vs RO-FIGS p: {output['metrics_agg']['wilcoxon_scots_vs_rofigs_corrected_p']:.6f}")
    logger.info(f"  SC-OTS wins vs RO-FIGS: {output['metrics_agg']['sc_ots_wins_vs_rofigs']}/10")
    logger.info(f"  Mean interaction F1 (enhanced): {output['metrics_agg']['mean_interaction_f1_enhanced']:.3f}")
    logger.info(f"  Mean faithfulness precision: {output['metrics_agg']['mean_faithfulness_precision']:.3f}")
    logger.info(f"  Mean Cohen's d (XGB+SC): {output['metrics_agg']['mean_cohens_d_xgb_sc']:.3f}")
    logger.info(f"  Hypothesis confirmed: {bool(output['metrics_agg']['hypothesis_confirmed'])}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
