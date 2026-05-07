# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
# ]
# ///
"""
SC-OTS Tabular Benchmark Suite: Dataset Assembly Script
Loads the best 10 datasets from tmp/datasets/, preprocesses them, creates 5-fold CV splits,
and outputs in exp_sel_data_out.json schema format.

Selected 10 (from 15 candidates):
  4 synthetic with known ground-truth interactions (friedman1, friedman3, synth_3way, synth_4way)
  4 real-world well-studied (diabetes, breast_w, california_housing, wine_quality)
  2 medium-complexity (spambase, adult)

Dropped:
  friedman2 — functionally similar to friedman1/3, less diversity
  heart_c — only 303 samples, too small for robust 5-fold evaluation
  ames_housing — 297 features after encoding (overwhelmingly dominated by one-hot dummies)
  ozone_level_8hr — V-named features, severely imbalanced (93.7% class 1)
  qsar_biodeg — V-named molecular descriptors, least relevant for testing general interaction detection
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

WORKSPACE = Path(__file__).parent
DATA_DIR = WORKSPACE / "tmp" / "datasets"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"

# ── Dataset definitions ─────────────────────────────────────────────
# Top 15 datasets (dropped: credit_approval=anonymized, sonar=208 samples, musk=wrong version)
DATASET_CONFIGS = [
    # Category A — Synthetic (regression, known ground-truth interactions)
    {
        "name": "friedman1",
        "file": "friedman1.csv",
        "task_type": "regression",
        "category": "A_synthetic",
        "source": "sklearn.datasets.make_friedman1",
        "known_interactions": {
            "2-way": [[0, 1]],
            "additive": [2, 3, 4],
            "noise_features": [5, 6, 7, 8, 9],
            "formula": "y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4",
        },
    },
    {
        "name": "friedman3",
        "file": "friedman3.csv",
        "task_type": "regression",
        "category": "A_synthetic",
        "source": "sklearn.datasets.make_friedman3",
        "known_interactions": {
            "3-way": [[0, 1, 2], [0, 1, 3]],
            "formula": "y = arctan((x1*x2 - 1/(x1*x3)) / x0)",
        },
    },
    {
        "name": "synth_3way",
        "file": "synth_3way.csv",
        "task_type": "regression",
        "category": "A_synthetic",
        "source": "custom_generator",
        "known_interactions": {
            "3-way": [[0, 1, 2]],
            "2-way": [[3, 4]],
            "additive": [0],
            "noise_features": list(range(5, 15)),
            "formula": "y = 5*x0*x1*x2 + 3*sin(pi*x3*x4) + 2*x0 + noise",
        },
    },
    {
        "name": "synth_4way",
        "file": "synth_4way.csv",
        "task_type": "regression",
        "category": "A_synthetic",
        "source": "custom_generator",
        "known_interactions": {
            "4-way": [[0, 1, 2, 3]],
            "2-way": [[4, 5]],
            "additive": [0, 4],
            "noise_features": list(range(6, 20)),
            "formula": "y = 4*x0*x1*x2*x3 + 3*x4*x5 + 2*x0 + 1.5*x4 + noise",
        },
    },
    # Category B — Real-world
    {
        "name": "diabetes",
        "file": "diabetes.csv",
        "task_type": "classification",
        "category": "B_realworld",
        "source": "OpenML ID 37 (Pima Indians)",
        "domain_hint": "BMI x skin_thickness, glucose x insulin",
    },
    {
        "name": "breast_w",
        "file": "breast_w.csv",
        "task_type": "classification",
        "category": "B_realworld",
        "source": "OpenML ID 15 (Wisconsin Breast Cancer)",
        "domain_hint": "cell_size x cell_shape, bare_nuclei x bland_chromatin",
    },
    {
        "name": "adult",
        "file": "adult.csv",
        "task_type": "classification",
        "category": "B_realworld",
        "source": "OpenML ID 1590 (Adult Census Income)",
        "domain_hint": "education x occupation, age x hours_per_week",
        "subsample": 10000,
    },
    {
        "name": "wine_quality",
        "file": "wine_quality.csv",
        "task_type": "regression",
        "category": "B_realworld",
        "source": "OpenML ID 287 (Wine Quality)",
        "domain_hint": "acidity x pH, alcohol x residual_sugar",
    },
    {
        "name": "california_housing",
        "file": "california_housing.csv",
        "task_type": "regression",
        "category": "B_realworld",
        "source": "sklearn.datasets.fetch_california_housing",
        "domain_hint": "income x location(lat,lon), rooms x occupancy",
    },
    # Category C — Medium-complexity
    {
        "name": "spambase",
        "file": "spambase.csv",
        "task_type": "classification",
        "category": "C_medium_complexity",
        "source": "OpenML ID 44 (Spambase)",
        "domain_hint": "word_freq x char_freq x capital_run_length",
    },
]

# ── Target label mappings ────────────────────────────────────────────
TARGET_MAP = {
    "diabetes": {"tested_positive": 1, "tested_negative": 0},
    "breast_w": {"malignant": 1, "benign": 0},
    "heart_c": {"P": 1, "N": 0},  # P=positive/disease, N=negative/healthy
    "adult": {">50K": 1, "<=50K": 0, ">50K.": 1, "<=50K.": 0},
    "spambase": {"1": 1, "0": 0, 1: 1, 0: 0},
    "ozone_level_8hr": {"1": 1, "2": 0, 1: 1, 2: 0},  # 1=ozone day, 2=normal
    "qsar_biodeg": {"1": 1, "2": 0, 1: 1, 2: 0},  # 1=ready biodeg, 2=not ready
}


def preprocess_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    task_type: str,
    subsample: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocess a dataset: encode categoricals, handle missing values, encode target.

    Returns: (X_numeric, y, feature_names)
    """
    # Separate target
    target_col = "target"
    y_raw = df[target_col].copy()
    X_df = df.drop(columns=[target_col])

    # ── Special handling per dataset ──
    # musk: drop molecule_name (categorical identifier, not a feature)
    if "molecule_name" in X_df.columns:
        X_df = X_df.drop(columns=["molecule_name"])

    # ── Subsample if needed (before encoding to save memory) ──
    if subsample is not None and len(X_df) > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_df), size=subsample, replace=False)
        idx.sort()
        X_df = X_df.iloc[idx].reset_index(drop=True)
        y_raw = y_raw.iloc[idx].reset_index(drop=True)

    # ── Handle missing values ──
    # Identify column types
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # For numeric cols: impute with median if <20% missing per column, else drop column
    cols_to_drop = []
    for col in numeric_cols:
        pct_missing = X_df[col].isna().mean()
        if pct_missing > 0.20:
            cols_to_drop.append(col)
        elif pct_missing > 0:
            X_df[col] = X_df[col].fillna(X_df[col].median())

    # For categorical cols: impute with mode if <20% missing, else drop
    for col in cat_cols:
        pct_missing = X_df[col].isna().mean()
        if pct_missing > 0.20:
            cols_to_drop.append(col)
        elif pct_missing > 0:
            mode_val = X_df[col].mode()
            if len(mode_val) > 0:
                X_df[col] = X_df[col].fillna(mode_val.iloc[0])

    if cols_to_drop:
        print(f"  Dropping columns with >20% missing: {cols_to_drop}")
        X_df = X_df.drop(columns=cols_to_drop)
        numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]
        cat_cols = [c for c in cat_cols if c not in cols_to_drop]

    # Drop any remaining rows with NaN (should be very few)
    mask = X_df.isna().any(axis=1) | y_raw.isna()
    if mask.sum() > 0:
        print(f"  Dropping {mask.sum()} rows with remaining NaN")
        X_df = X_df[~mask].reset_index(drop=True)
        y_raw = y_raw[~mask].reset_index(drop=True)

    # ── One-hot encode categoricals (drop_first to avoid collinearity) ──
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=float)
        print(f"  One-hot encoded {len(cat_cols)} categorical columns -> {len(X_df.columns)} total features")

    # ── Encode target ──
    if task_type == "classification":
        mapping = TARGET_MAP.get(dataset_name)
        if mapping:
            y = y_raw.map(mapping)
            unmapped = y.isna().sum()
            if unmapped > 0:
                print(f"  WARNING: {unmapped} unmapped target values in {dataset_name}")
                print(f"  Unique values: {y_raw.unique()}")
                # Try to force-convert
                y = y.fillna(0)
            y = y.astype(int)
        else:
            # Try numeric conversion
            try:
                y = y_raw.astype(float).astype(int)
            except (ValueError, TypeError):
                # Label encode
                unique_labels = sorted(y_raw.unique())
                label_map = {v: i for i, v in enumerate(unique_labels)}
                y = y_raw.map(label_map).astype(int)
        y = y.values
    else:
        y = y_raw.astype(float).values

    # ── Final conversion to numeric array ──
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    # Verify no NaN remains
    assert not np.isnan(X).any(), f"NaN found in X for {dataset_name}"
    assert not np.isnan(y).any(), f"NaN found in y for {dataset_name}"

    return X, y, feature_names


def create_fold_assignments(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Create fold assignments for 5-fold CV."""
    if task_type == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_arg = y
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_arg = None

    fold_assignments = np.zeros(len(X), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(kf.split(X, split_arg)):
        fold_assignments[test_idx] = fold_idx

    return fold_assignments


def build_dataset_entry(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    folds: np.ndarray,
) -> dict:
    """Build a single dataset entry in the output schema."""
    task_type = config["task_type"]
    n_samples, n_features = X.shape

    # Compute n_classes for classification
    if task_type == "classification":
        n_classes = int(len(np.unique(y)))
    else:
        n_classes = 0  # regression

    # Build examples (one per row)
    examples = []
    for i in range(n_samples):
        # Input: JSON string of feature values (rounded to 6 decimal places)
        input_vals = [round(float(v), 6) for v in X[i]]
        input_str = json.dumps(input_vals)

        # Output: target as string
        if task_type == "classification":
            output_str = str(int(y[i]))
        else:
            output_str = str(round(float(y[i]), 6))

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_fold": int(folds[i]),
            "metadata_feature_names": feature_names,
            "metadata_task_type": task_type,
            "metadata_category": config["category"],
            "metadata_row_index": i,
        }

        # Add classification-specific metadata
        if task_type == "classification":
            example["metadata_n_classes"] = n_classes

        # Add interaction metadata for synthetic datasets
        if "known_interactions" in config:
            example["metadata_known_interactions"] = json.dumps(config["known_interactions"])

        if "domain_hint" in config:
            example["metadata_domain_hint"] = config["domain_hint"]

        example["metadata_source"] = config["source"]
        example["metadata_n_features"] = n_features
        example["metadata_n_samples"] = n_samples

        examples.append(example)

    return {"dataset": config["name"], "examples": examples}


def main() -> None:
    print("=" * 60)
    print("SC-OTS Tabular Benchmark Suite — Dataset Assembly")
    print("=" * 60)

    all_datasets = []

    for config in DATASET_CONFIGS:
        name = config["name"]
        filepath = DATA_DIR / config["file"]

        print(f"\n── Processing: {name} ──")
        if not filepath.exists():
            print(f"  ERROR: File not found: {filepath}")
            continue

        # Load CSV
        df = pd.read_csv(filepath)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Preprocess
        X, y, feature_names = preprocess_dataset(
            df=df,
            dataset_name=name,
            task_type=config["task_type"],
            subsample=config.get("subsample"),
        )
        print(f"  After preprocessing: X={X.shape}, y={y.shape}, features={len(feature_names)}")

        # Create fold assignments
        folds = create_fold_assignments(X, y, config["task_type"])
        fold_counts = np.bincount(folds)
        print(f"  Fold sizes: {fold_counts.tolist()}")

        # Build dataset entry
        entry = build_dataset_entry(config, X, y, feature_names, folds)
        all_datasets.append(entry)

        # Summary
        print(f"  ✓ {name}: {len(entry['examples'])} examples, {len(feature_names)} features")

    # ── Assemble final output ──
    output = {
        "metadata": {
            "benchmark_name": "SC-OTS Tabular Benchmark v1",
            "description": "10 tabular datasets (4 synthetic + 4 real-world + 2 medium-complexity) "
            "with standardized schema, 5-fold CV splits, and interaction metadata for SC-OTS evaluation",
            "n_datasets": len(all_datasets),
            "total_examples": sum(len(d["examples"]) for d in all_datasets),
            "categories": {
                "A_synthetic": "Known ground-truth interactions for validation",
                "B_realworld": "Well-studied datasets from FIGS/EBM/RO-FIGS literature",
                "C_medium_complexity": "40-200 features with plausible higher-order interactions",
            },
        },
        "datasets": all_datasets,
    }

    # ── Validation checks ──
    print("\n" + "=" * 60)
    print("Validation Checks")
    print("=" * 60)

    errors = []
    for ds in all_datasets:
        ds_name = ds["dataset"]
        examples = ds["examples"]

        # Check examples exist
        if len(examples) == 0:
            errors.append(f"{ds_name}: No examples")
            continue

        # Check all inputs have same length
        first_len = len(json.loads(examples[0]["input"]))
        for i, ex in enumerate(examples):
            inp = json.loads(ex["input"])
            if len(inp) != first_len:
                errors.append(f"{ds_name}: Example {i} input length {len(inp)} != {first_len}")
                break
            # Check no NaN
            if any(v != v for v in inp if isinstance(v, float)):
                errors.append(f"{ds_name}: NaN in example {i} input")
                break

        # Check folds
        fold_vals = set(ex["metadata_fold"] for ex in examples)
        if fold_vals != {0, 1, 2, 3, 4}:
            errors.append(f"{ds_name}: Expected folds {{0,1,2,3,4}}, got {fold_vals}")

        # Check feature_names length matches input
        n_feat = examples[0]["metadata_n_features"]
        if first_len != n_feat:
            errors.append(f"{ds_name}: Feature count mismatch: input len={first_len}, n_features={n_feat}")

        feat_names_len = len(examples[0]["metadata_feature_names"])
        if feat_names_len != first_len:
            errors.append(f"{ds_name}: feature_names length {feat_names_len} != input length {first_len}")

        # Check target values
        task = examples[0]["metadata_task_type"]
        if task == "classification":
            targets = set(ex["output"] for ex in examples)
            if not targets.issubset({"0", "1"}):
                errors.append(f"{ds_name}: Classification targets not binary: {targets}")
        else:
            for ex in examples[:10]:
                try:
                    v = float(ex["output"])
                    if not np.isfinite(v):
                        errors.append(f"{ds_name}: Non-finite regression target: {v}")
                        break
                except ValueError:
                    errors.append(f"{ds_name}: Non-numeric regression target: {ex['output']}")
                    break

        print(f"  ✓ {ds_name}: {len(examples)} examples, {first_len} features, {len(fold_vals)} folds")

    if errors:
        print(f"\n✗ VALIDATION FAILED with {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"\n✓ All {len(all_datasets)} datasets passed validation!")

    # ── Write output ──
    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ Written: {OUTPUT_FILE.name} ({size_mb:.1f} MB)")
    print(f"  Total datasets: {len(all_datasets)}")
    print(f"  Total examples: {sum(len(d['examples']) for d in all_datasets)}")


if __name__ == "__main__":
    main()
