#!/usr/bin/env python3
"""Baseline Benchmarks: FIGS, XGBoost, XGBoost-Oracle, EBM on 10 SC-OTS Datasets (5-Fold CV).

Runs FIGS (3 complexity levels), XGBoost (unconstrained + oracle-constrained),
and EBM (default + high-interaction + 3-way) across all 10 SC-OTS datasets with
5-fold CV, measuring accuracy/R²/AUROC, model complexity, and wall-clock time.
"""

import json
import signal
import sys
import time
import resource
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


class FitTimeout(Exception):
    """Raised when model.fit exceeds time limit."""
    pass


def _timeout_handler(signum, frame):
    raise FitTimeout("Model fit exceeded time limit")

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Logging setup ===
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
LOG_FMT = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"
logger.add(sys.stdout, level="INFO", format=LOG_FMT)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG", format=LOG_FMT)

# === Resource limits (leave headroom for OS & other agents) ===
try:
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except (ValueError, OSError):
    pass

# === Paths ===
WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE.parent.parent.parent / "iter_1" / "gen_art" / "data_id3_it1__opus"
MINI_DATA = DATA_DIR / "mini_data_out.json"
FULL_DATA = DATA_DIR / "full_data_out.json"

# === Config ===
CONFIG = {
    "random_seed": 42,
    "run_mode": "full",
    "oracle_constraints": {
        "friedman1": {
            "groups_idx": [[0, 1], [2], [3], [4]],
            "description": "x0*x1 interact; x2,x3,x4 additive; rest noise",
        },
        "friedman3": {
            "groups_idx": [[0, 1, 2, 3]],
            "description": "all 4 features interact in atan formula",
        },
        "synth_3way": "read_from_metadata",
        "synth_4way": "read_from_metadata",
    },
    # EBM speed tuning for resource-constrained environments (shared CPU).
    # outer_bags=1 and max_rounds=200 keeps runs under timeout while still
    # producing valid EBM models. Full data (63k samples) requires these
    # reductions because EBM scales O(bags * rounds * features * n).
    "ebm_outer_bags": 1,
    "ebm_inner_bags": 0,
    "ebm_max_rounds": 200,
    # EBM-3way is especially expensive — use even fewer rounds
    "ebm_3way_max_rounds": 100,
    # Per-fit timeout in seconds (protects against EBM hangs on large datasets)
    "fit_timeout_sec": 120,
}

SYNTHETIC_DATASETS = {"friedman1", "friedman3", "synth_3way", "synth_4way"}
THREE_WAY_DATASETS = {"synth_3way", "synth_4way"}


# ============================================================================
#  DATA LOADER
# ============================================================================

def load_datasets(json_path: Path) -> dict:
    """Load SC-OTS benchmark data from JSON, return dict keyed by dataset name.

    Each value contains X, y, folds, task_type, feature_names, known_interactions.
    """
    logger.info(f"Loading data from {json_path}")
    raw = json.loads(json_path.read_text())

    # Handle the {metadata, datasets[{dataset, examples}]} schema
    ds_list = raw.get("datasets", raw)
    if isinstance(ds_list, dict):
        ds_list = [ds_list]

    result = {}
    for ds_entry in ds_list:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        if not examples:
            logger.warning(f"Empty dataset: {ds_name}")
            continue

        first = examples[0]
        task_type = first["metadata_task_type"]
        feature_names = first.get("metadata_feature_names", [])
        known_raw = first.get("metadata_known_interactions")

        # Parse known interactions
        known_interactions = None
        if known_raw:
            if isinstance(known_raw, str):
                try:
                    known_interactions = json.loads(known_raw)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse known_interactions for {ds_name}")
            elif isinstance(known_raw, dict):
                known_interactions = known_raw

        rows = []
        targets = []
        folds = []
        for ex in examples:
            features = json.loads(ex["input"]) if isinstance(ex["input"], str) else ex["input"]
            rows.append(features)
            folds.append(ex["metadata_fold"])
            targets.append(ex["output"])

        X = np.array(rows, dtype=np.float64)
        fold_arr = np.array(folds, dtype=int)

        if task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform([str(t) for t in targets])
            y = y.astype(np.int64)
        else:
            y = np.array([float(t) for t in targets], dtype=np.float64)

        result[ds_name] = {
            "X": X,
            "y": y,
            "folds": fold_arr,
            "task_type": task_type,
            "feature_names": feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])],
            "known_interactions": known_interactions,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }

    logger.info(f"Loaded {len(result)} datasets")
    for name, ds in result.items():
        logger.info(
            f"  {name}: {ds['n_samples']} samples, {ds['n_features']} features, "
            f"task={ds['task_type']}, folds={sorted(np.unique(ds['folds']))}"
        )
    return result


def get_fold_splits(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    fold_idx: int,
) -> tuple:
    """Return (X_train, y_train, X_test, y_test) for a given fold."""
    train_mask = folds != fold_idx
    test_mask = folds == fold_idx
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


# ============================================================================
#  METRICS
# ============================================================================

def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> dict:
    """Compute performance metrics for a fitted model on test data."""
    metrics = {}
    preds = model.predict(X_test)

    if task_type == "classification":
        metrics["accuracy"] = float(round(accuracy_score(y_test, preds), 6))
        # AUROC
        try:
            n_classes = len(np.unique(y_test))
            if n_classes < 2:
                metrics["auroc"] = float("nan")
                logger.warning("Only one class in test fold — AUROC undefined")
            elif n_classes == 2:
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    metrics["auroc"] = float(round(roc_auc_score(y_test, proba[:, 1]), 6))
                else:
                    metrics["auroc"] = float(round(roc_auc_score(y_test, proba[:, 0]), 6))
            else:
                proba = model.predict_proba(X_test)
                metrics["auroc"] = float(round(
                    roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"), 6
                ))
        except Exception:
            logger.exception("AUROC computation failed")
            metrics["auroc"] = float("nan")
    else:
        metrics["r2"] = float(round(r2_score(y_test, preds), 6))
        metrics["rmse"] = float(round(np.sqrt(mean_squared_error(y_test, preds)), 6))

    return metrics


def extract_complexity(model, method_name: str) -> dict:
    """Extract complexity metrics from a fitted model."""
    complexity = {}

    if method_name.startswith("FIGS"):
        complexity["n_splits"] = int(getattr(model, "complexity_", 0))
        complexity["n_trees"] = len(getattr(model, "trees_", []))

    elif method_name.startswith("XGBoost"):
        try:
            booster = model.get_booster()
            dump = booster.get_dump()
            n_trees = len(dump)
            n_splits = sum(
                1
                for tree_str in dump
                for line in tree_str.strip().split("\n")
                if "[" in line
            )
            complexity["n_splits"] = n_splits
            complexity["n_trees"] = n_trees
        except Exception:
            logger.exception("XGBoost complexity extraction failed")
            complexity["n_splits"] = -1
            complexity["n_trees"] = -1

    elif method_name.startswith("EBM"):
        terms = getattr(model, "term_features_", [])
        complexity["n_total_terms"] = len(terms)
        complexity["n_main_effects"] = sum(1 for t in terms if len(t) == 1)
        complexity["n_pairwise"] = sum(1 for t in terms if len(t) == 2)
        complexity["n_3way"] = sum(1 for t in terms if len(t) == 3)

    return complexity


def timed_fit(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    timeout_sec: int | None = None,
) -> tuple:
    """Fit model with optional timeout. Returns (fitted_model, wall_clock_seconds)."""
    if timeout_sec is None:
        timeout_sec = CONFIG.get("fit_timeout_sec", 300)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
    finally:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
    return model, elapsed


# ============================================================================
#  MODEL CONFIGURATIONS
# ============================================================================

def _get_oracle_constraints_for_xgb(
    dataset_name: str,
    dataset_meta: dict,
    feature_names: list[str],
) -> list[list[str]] | None:
    """Build interaction_constraints for XGBoost oracle, using feature name strings."""
    cfg = CONFIG["oracle_constraints"].get(dataset_name)
    if cfg is None:
        return None

    n_features = len(feature_names)

    if isinstance(cfg, str) and cfg == "read_from_metadata":
        ki = dataset_meta.get("known_interactions")
        if ki is None:
            logger.warning(f"No known_interactions for {dataset_name}")
            return None
        # Build groups from known_interactions structure
        interacting_features = set()
        groups = []

        # Collect all mentioned interacting index groups
        for key in ["4-way", "3-way", "2-way"]:
            if key in ki:
                for group in ki[key]:
                    groups.append(group)
                    interacting_features.update(group)

        # Add additive features as singleton groups
        for idx in ki.get("additive", []):
            if idx not in interacting_features:
                groups.append([idx])
                interacting_features.add(idx)

        # Add remaining features as singletons (noise features get their own group)
        for idx in range(n_features):
            if idx not in interacting_features:
                groups.append([idx])

        # Convert to feature name strings
        return [[feature_names[i] for i in g] for g in groups]

    elif isinstance(cfg, dict) and "groups_idx" in cfg:
        idx_groups = cfg["groups_idx"]
        covered = set()
        for g in idx_groups:
            covered.update(g)
        # Add uncovered features as singletons
        all_groups = list(idx_groups)
        for i in range(n_features):
            if i not in covered:
                all_groups.append([i])
        return [[feature_names[i] for i in g] for g in all_groups]

    return None


def _get_3way_interactions_for_ebm(
    dataset_name: str,
    dataset_meta: dict,
) -> list[tuple] | None:
    """Build explicit 3-way interaction tuples for EBM from known interactions."""
    ki = dataset_meta.get("known_interactions")
    if ki is None:
        return None

    interactions = []

    # Add 3-way and 4-way groups as explicit tuples
    for key in ["3-way", "4-way"]:
        if key in ki:
            for group in ki[key]:
                if len(group) == 3:
                    interactions.append(tuple(group))
                elif len(group) == 4:
                    # For 4-way, add all 3-way subsets
                    from itertools import combinations
                    for combo in combinations(group, 3):
                        interactions.append(combo)

    # Add 2-way groups
    for key in ["2-way"]:
        if key in ki:
            for group in ki[key]:
                interactions.append(tuple(group))

    return interactions if interactions else None


def build_model_configs(datasets: dict) -> list[dict]:
    """Build all model configurations for benchmarking."""
    from imodels import FIGSClassifier, FIGSRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

    seed = CONFIG["random_seed"]
    ebm_ob = CONFIG["ebm_outer_bags"]
    ebm_ib = CONFIG["ebm_inner_bags"]
    ebm_mr = CONFIG["ebm_max_rounds"]

    configs = []

    # 1. FIGS variants
    for max_rules in [5, 10, 20]:
        def _figs_factory(task_type, dataset_name, dataset_meta, _mr=max_rules):
            if task_type == "classification":
                return FIGSClassifier(max_rules=_mr, random_state=seed)
            return FIGSRegressor(max_rules=_mr, random_state=seed)

        configs.append({
            "name": f"FIGS-{max_rules}",
            "factory": _figs_factory,
            "is_applicable": lambda ds_name, ds: True,
        })

    # 2. XGBoost default (unconstrained)
    def _xgb_default_factory(task_type, dataset_name, dataset_meta):
        if task_type == "classification":
            return XGBClassifier(
                n_estimators=100,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=1,
                verbosity=0,
            )
        return XGBRegressor(
            n_estimators=100,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )

    configs.append({
        "name": "XGBoost-default",
        "factory": _xgb_default_factory,
        "is_applicable": lambda ds_name, ds: True,
    })

    # 3. XGBoost oracle (synthetic only)
    def _xgb_oracle_factory(task_type, dataset_name, dataset_meta):
        feature_names = dataset_meta["feature_names"]
        constraints = _get_oracle_constraints_for_xgb(
            dataset_name, dataset_meta, feature_names
        )
        if constraints is None:
            raise ValueError(f"No oracle constraints for {dataset_name}")

        if task_type == "classification":
            return XGBClassifier(
                n_estimators=100,
                random_state=seed,
                eval_metric="logloss",
                interaction_constraints=constraints,
                n_jobs=1,
                verbosity=0,
            )
        return XGBRegressor(
            n_estimators=100,
            random_state=seed,
            interaction_constraints=constraints,
            n_jobs=1,
            verbosity=0,
        )

    configs.append({
        "name": "XGBoost-oracle",
        "factory": _xgb_oracle_factory,
        "is_applicable": lambda ds_name, ds: ds_name in SYNTHETIC_DATASETS,
    })

    # 4. EBM default (pairwise auto-detected)
    def _ebm_default_factory(task_type, dataset_name, dataset_meta):
        if task_type == "classification":
            return ExplainableBoostingClassifier(
                random_state=seed,
                n_jobs=1,
                outer_bags=ebm_ob,
                inner_bags=ebm_ib,
                max_rounds=ebm_mr,
                max_bins=128,
            )
        return ExplainableBoostingRegressor(
            random_state=seed,
            n_jobs=1,
            outer_bags=ebm_ob,
            inner_bags=ebm_ib,
            max_rounds=ebm_mr,
            max_bins=128,
        )

    configs.append({
        "name": "EBM-default",
        "factory": _ebm_default_factory,
        "is_applicable": lambda ds_name, ds: True,
    })

    # 5. EBM high-interaction
    def _ebm_high_factory(task_type, dataset_name, dataset_meta):
        if task_type == "classification":
            return ExplainableBoostingClassifier(
                interactions=30,
                max_bins=128,
                max_interaction_bins=32,
                random_state=seed,
                n_jobs=1,
                outer_bags=ebm_ob,
                inner_bags=ebm_ib,
                max_rounds=ebm_mr,
            )
        return ExplainableBoostingRegressor(
            interactions=30,
            max_bins=128,
            max_interaction_bins=32,
            random_state=seed,
            n_jobs=1,
            outer_bags=ebm_ob,
            inner_bags=ebm_ib,
            max_rounds=ebm_mr,
        )

    configs.append({
        "name": "EBM-high-interaction",
        "factory": _ebm_high_factory,
        "is_applicable": lambda ds_name, ds: True,
    })

    # 6. EBM 3-way (synth_3way, synth_4way only)
    ebm_3way_mr = CONFIG.get("ebm_3way_max_rounds", 100)

    def _ebm_3way_factory(task_type, dataset_name, dataset_meta, _3mr=ebm_3way_mr):
        interactions_3way = _get_3way_interactions_for_ebm(dataset_name, dataset_meta)
        if interactions_3way is None:
            raise ValueError(f"No 3-way interactions for {dataset_name}")

        if task_type == "classification":
            return ExplainableBoostingClassifier(
                interactions=interactions_3way,
                max_bins=32,
                max_interaction_bins=32,
                random_state=seed,
                n_jobs=1,
                outer_bags=ebm_ob,
                inner_bags=ebm_ib,
                max_rounds=_3mr,
            )
        return ExplainableBoostingRegressor(
            interactions=interactions_3way,
            max_bins=32,
            max_interaction_bins=32,
            random_state=seed,
            n_jobs=1,
            outer_bags=ebm_ob,
            inner_bags=ebm_ib,
            max_rounds=_3mr,
        )

    configs.append({
        "name": "EBM-3way",
        "factory": _ebm_3way_factory,
        "is_applicable": lambda ds_name, ds: ds_name in THREE_WAY_DATASETS,
    })

    return configs


# ============================================================================
#  FOLD SUMMARY
# ============================================================================

def compute_fold_summary(
    fold_metrics: list[dict],
    fold_complexities: list[dict],
    fold_times: list[float],
) -> dict:
    """Aggregate per-fold results into mean±std summary."""
    summary = {}

    if not fold_metrics:
        return summary

    # Metric aggregation
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics if not (isinstance(fm[key], float) and np.isnan(fm[key]))]
        if vals:
            summary[f"{key}_mean"] = round(float(np.mean(vals)), 6)
            summary[f"{key}_std"] = round(float(np.std(vals)), 6)
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")

    # Complexity: take fold 0 (deterministic models give same complexity)
    if fold_complexities:
        summary["complexity"] = fold_complexities[0]

    # Time
    summary["train_time_mean"] = round(float(np.mean(fold_times)), 4)
    summary["train_time_std"] = round(float(np.std(fold_times)), 4)

    return summary


# ============================================================================
#  MAIN BENCHMARK
# ============================================================================

@logger.catch
def run_benchmark(
    data_path: Path,
    run_mode: str = "mini",
    max_examples_per_dataset: int | None = None,
) -> dict:
    """Run the full benchmark pipeline."""
    datasets = load_datasets(data_path)

    # Optionally limit examples per dataset (for gradual scaling)
    if max_examples_per_dataset is not None:
        for ds_name in datasets:
            ds = datasets[ds_name]
            n = min(max_examples_per_dataset, ds["n_samples"])
            if n < ds["n_samples"]:
                datasets[ds_name] = {
                    "X": ds["X"][:n],
                    "y": ds["y"][:n],
                    "folds": ds["folds"][:n],
                    "task_type": ds["task_type"],
                    "feature_names": ds["feature_names"],
                    "known_interactions": ds["known_interactions"],
                    "n_samples": n,
                    "n_features": ds["n_features"],
                }
                logger.info(f"  Limited {ds_name} to {n} examples")

    model_configs = build_model_configs(datasets)
    logger.info(f"Built {len(model_configs)} model configs: {[m['name'] for m in model_configs]}")

    all_results = {}

    for ds_name, ds in datasets.items():
        logger.info(f"=== Dataset: {ds_name} ({ds['task_type']}, {ds['n_samples']} samples) ===")
        all_results[ds_name] = {}

        # For XGBoost oracle, pass data as DataFrame with feature names
        feature_names = ds["feature_names"]

        unique_folds = sorted(np.unique(ds["folds"]))

        for mcfg in model_configs:
            method_name = mcfg["name"]

            if not mcfg["is_applicable"](ds_name, ds):
                logger.info(f"  Skipping {method_name} (not applicable to {ds_name})")
                continue

            logger.info(f"  Running {method_name}...")
            all_results[ds_name][method_name] = {}
            fold_metrics_list = []
            fold_complexities_list = []
            fold_times_list = []

            for fold_idx in unique_folds:
                X_train, y_train, X_test, y_test = get_fold_splits(
                    ds["X"], ds["y"], ds["folds"], fold_idx
                )

                # Skip fold if too few samples or single class
                if len(y_train) < 2 or len(y_test) < 1:
                    logger.warning(f"    Fold {fold_idx} skipped: too few samples")
                    continue

                try:
                    model = mcfg["factory"](
                        task_type=ds["task_type"],
                        dataset_name=ds_name,
                        dataset_meta=ds,
                    )
                except Exception:
                    logger.exception(f"    Fold {fold_idx} factory FAILED for {method_name} on {ds_name}")
                    all_results[ds_name][method_name][str(fold_idx)] = {"error": "factory_failed"}
                    continue

                # For XGBoost with constraints, need DataFrame with feature names
                if method_name.startswith("XGBoost"):
                    X_train_fit = pd.DataFrame(X_train, columns=feature_names)
                    X_test_fit = pd.DataFrame(X_test, columns=feature_names)
                else:
                    X_train_fit = X_train
                    X_test_fit = X_test

                try:
                    model, elapsed = timed_fit(model, X_train_fit, y_train)
                except FitTimeout:
                    logger.warning(
                        f"    Fold {fold_idx} TIMEOUT for {method_name} on {ds_name} "
                        f"(>{CONFIG['fit_timeout_sec']}s)"
                    )
                    all_results[ds_name][method_name][str(fold_idx)] = {"error": "timeout"}
                    continue
                except Exception:
                    logger.exception(f"    Fold {fold_idx} FIT FAILED for {method_name} on {ds_name}")
                    all_results[ds_name][method_name][str(fold_idx)] = {"error": "fit_failed"}
                    continue

                try:
                    metrics = compute_metrics(model, X_test_fit, y_test, ds["task_type"])
                except Exception:
                    logger.exception(f"    Fold {fold_idx} METRICS FAILED for {method_name} on {ds_name}")
                    metrics = {}

                try:
                    complexity = extract_complexity(model, method_name)
                except Exception:
                    logger.exception(f"    Fold {fold_idx} COMPLEXITY FAILED for {method_name}")
                    complexity = {}

                fold_result = {
                    "metrics": metrics,
                    "complexity": complexity,
                    "train_time_sec": round(elapsed, 4),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                }
                all_results[ds_name][method_name][str(fold_idx)] = fold_result
                fold_metrics_list.append(metrics)
                fold_complexities_list.append(complexity)
                fold_times_list.append(elapsed)

                logger.info(
                    f"    Fold {fold_idx}: {metrics} | complexity={complexity} | time={elapsed:.2f}s"
                )

            # Compute summary across folds
            if fold_metrics_list:
                summary = compute_fold_summary(
                    fold_metrics_list, fold_complexities_list, fold_times_list
                )
                all_results[ds_name][method_name]["summary"] = summary
                logger.info(f"  {method_name} summary: {summary}")

    return {
        "experiment": "baseline_benchmarks",
        "run_mode": run_mode,
        "n_datasets": len(datasets),
        "methods": [m["name"] for m in model_configs],
        "results": all_results,
    }


def convert_to_schema_format(benchmark_output: dict) -> dict:
    """Convert benchmark results to the exp_gen_sol_out.json schema format.

    Schema requires: {datasets: [{dataset: str, examples: [{input, output, ...metadata}]}]}
    Each (dataset, method, fold) becomes one example with predictions as metadata.
    """
    datasets_out = []
    results = benchmark_output.get("results", {})

    for ds_name, ds_results in results.items():
        examples = []
        for method_name, method_results in ds_results.items():
            for fold_key, fold_data in method_results.items():
                if fold_key == "summary":
                    continue
                if isinstance(fold_data, dict) and "error" in fold_data:
                    continue

                metrics = fold_data.get("metrics", {})
                complexity = fold_data.get("complexity", {})

                # Build a summary string for input
                input_str = json.dumps({
                    "dataset": ds_name,
                    "method": method_name,
                    "fold": int(fold_key),
                    "n_train": fold_data.get("n_train", 0),
                    "n_test": fold_data.get("n_test", 0),
                })

                # Build output as the primary metric result
                primary_metric = ""
                if "accuracy" in metrics:
                    primary_metric = f"accuracy={metrics['accuracy']}"
                    if "auroc" in metrics and not np.isnan(metrics.get("auroc", float("nan"))):
                        primary_metric += f", auroc={metrics['auroc']}"
                elif "r2" in metrics:
                    primary_metric = f"r2={metrics['r2']}, rmse={metrics['rmse']}"

                example = {
                    "input": input_str,
                    "output": primary_metric,
                    "metadata_method": method_name,
                    "metadata_fold": int(fold_key),
                    "metadata_dataset": ds_name,
                    "metadata_train_time_sec": fold_data.get("train_time_sec", 0),
                    "metadata_complexity": json.dumps(complexity),
                    "metadata_metrics": json.dumps(metrics),
                }

                # Add predict_ fields for each metric
                for mk, mv in metrics.items():
                    example[f"predict_{method_name.replace('-', '_')}_{mk}"] = str(mv)

                examples.append(example)

        if examples:
            datasets_out.append({
                "dataset": ds_name,
                "examples": examples,
            })

    output = {
        "metadata": {
            "experiment": benchmark_output.get("experiment", "baseline_benchmarks"),
            "run_mode": benchmark_output.get("run_mode", "unknown"),
            "n_datasets": benchmark_output.get("n_datasets", 0),
            "methods": benchmark_output.get("methods", []),
        },
        "datasets": datasets_out,
    }
    return output


@logger.catch
def main():
    """Main entry point — run full benchmark directly.

    Validation was already completed in a prior session (all models work
    correctly on 50-sample subsets). Now run the full data benchmark.
    """
    Path("logs").mkdir(exist_ok=True)
    Path("tmp").mkdir(exist_ok=True)

    total_start = time.time()

    # ===== Full data run =====
    logger.info("=" * 60)
    logger.info("FULL DATA BENCHMARK RUN")
    logger.info("=" * 60)

    t0 = time.time()
    full_output = run_benchmark(FULL_DATA, run_mode="full")
    full_time = time.time() - t0
    logger.info(f"Full run completed in {full_time:.1f}s")

    _validate_results(full_output, is_mini=False)

    # ===== Write outputs =====
    raw_json = json.dumps(full_output, indent=2, default=str)
    (WORKSPACE / "tmp" / "full_benchmark_results.json").write_text(raw_json)

    # Convert to schema format and save as method_out.json
    schema_output = convert_to_schema_format(full_output)
    method_out_path = WORKSPACE / "method_out.json"
    method_out_path.write_text(json.dumps(schema_output, indent=2, default=str))
    logger.info(f"Schema-formatted output written to {method_out_path}")

    _print_summary_table(full_output)

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"  Full run: {full_time:.1f}s")
    logger.info(f"  Total wall-clock: {total_time:.1f}s")
    logger.info("=" * 60)


def _validate_results(output: dict, is_mini: bool = False) -> None:
    """Validate benchmark results for completeness and sanity."""
    results = output.get("results", {})
    n_datasets = len(results)

    logger.info(f"Validating results: {n_datasets} datasets")

    issues = []

    for ds_name, ds_results in results.items():
        for method_name, method_results in ds_results.items():
            fold_count = 0
            error_count = 0
            for fold_key, fold_data in method_results.items():
                if fold_key == "summary":
                    continue
                if isinstance(fold_data, dict) and "error" in fold_data:
                    error_count += 1
                    issues.append(f"ERROR: {ds_name}/{method_name}/fold{fold_key}: {fold_data['error']}")
                else:
                    fold_count += 1
                    metrics = fold_data.get("metrics", {})
                    # Sanity checks
                    if "r2" in metrics and metrics["r2"] < -10:
                        issues.append(
                            f"WARNING: {ds_name}/{method_name}/fold{fold_key}: R²={metrics['r2']:.4f} very negative"
                        )
                    if "auroc" in metrics and not np.isnan(metrics.get("auroc", float("nan"))):
                        if metrics["auroc"] < 0.45:
                            issues.append(
                                f"WARNING: {ds_name}/{method_name}/fold{fold_key}: AUROC={metrics['auroc']:.4f} < 0.45"
                            )

            if fold_count == 0 and error_count > 0:
                issues.append(f"CRITICAL: {ds_name}/{method_name}: all folds failed!")

    if issues:
        for issue in issues:
            logger.warning(issue)
    else:
        logger.info("Validation passed — no issues found")


def _print_summary_table(output: dict) -> None:
    """Print a summary table of results."""
    results = output.get("results", {})

    logger.info("\n===== RESULTS SUMMARY =====")
    for ds_name, ds_results in results.items():
        task_type = None
        logger.info(f"\n--- {ds_name} ---")
        for method_name, method_results in ds_results.items():
            summary = method_results.get("summary", {})
            if not summary:
                continue

            parts = [f"  {method_name:25s}"]
            if "accuracy_mean" in summary:
                task_type = "classification"
                parts.append(f"acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}")
                if "auroc_mean" in summary and not np.isnan(summary.get("auroc_mean", float("nan"))):
                    parts.append(f"auroc={summary['auroc_mean']:.4f}±{summary['auroc_std']:.4f}")
            elif "r2_mean" in summary:
                task_type = "regression"
                parts.append(f"R²={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}")
                parts.append(f"RMSE={summary['rmse_mean']:.4f}±{summary['rmse_std']:.4f}")

            if "train_time_mean" in summary:
                parts.append(f"t={summary['train_time_mean']:.2f}s")

            logger.info(" | ".join(parts))


if __name__ == "__main__":
    main()
