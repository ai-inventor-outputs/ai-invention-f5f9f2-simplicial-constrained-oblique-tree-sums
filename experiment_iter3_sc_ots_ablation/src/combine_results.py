#!/usr/bin/env python3
"""Combine results from multiple partial runs into final method_out.json."""

import json
import sys
from pathlib import Path

from loguru import logger

GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)

WORKSPACE = Path(__file__).parent
DATA_PATH = (
    WORKSPACE.parents[2]
    / "iter_1"
    / "gen_art"
    / "data_id3_it1__opus"
    / "full_data_out.json"
)
OUTPUT_PATH = WORKSPACE / "method_out.json"


def main() -> None:
    """Combine checkpoint results + latest run results into final output."""
    # Load checkpoint (datasets 0-5)
    checkpoint_path = WORKSPACE / "tmp" / "checkpoint_results.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        results_1 = checkpoint["results"]
        logger.info(f"Loaded checkpoint with {len(results_1)} datasets: {list(results_1.keys())}")
    else:
        results_1 = {}
        logger.warning("No checkpoint found")

    # Load latest run results (from method_out.json metadata)
    latest_out = WORKSPACE / "method_out.json"
    if latest_out.exists():
        latest_data = json.loads(latest_out.read_text())
        results_2 = latest_data.get("metadata", {}).get("results_per_dataset", {})
        logger.info(f"Loaded latest output with {len(results_2)} datasets: {list(results_2.keys())}")
    else:
        results_2 = {}
        logger.warning("No latest output found")

    # Also check for checkpoint2
    checkpoint2_path = WORKSPACE / "tmp" / "checkpoint_results.json"
    # The second run's checkpoint replaces the first one, so we need both
    # Actually, the method_out.json from run 2 has the run 2 results

    # Merge results: checkpoint_1 has full fold data, method_out has summary
    # We need the full fold data from both runs
    # Let's merge all checkpoints
    all_checkpoints = sorted(WORKSPACE.glob("tmp/checkpoint_*.json"))
    all_results = {}
    for cp_path in all_checkpoints:
        cp = json.loads(cp_path.read_text())
        cp_results = cp["results"]
        all_results.update(cp_results)
        logger.info(f"  Loaded {cp_path.name}: {list(cp_results.keys())}")

    # If we only have one checkpoint, we need to load the latest run's full data too
    # Load the raw data to reconstruct predictions
    logger.info(f"Loading full dataset from {DATA_PATH}")
    raw_data = json.loads(DATA_PATH.read_text())
    datasets_raw = {ds["dataset"]: ds for ds in raw_data["datasets"]}

    # Build predictions from the method_out.json (which has predict_* fields)
    predictions = {}
    if latest_out.exists():
        for ds_entry in latest_data.get("datasets", []):
            ds_name = ds_entry["dataset"]
            predictions[ds_name] = {}
            for i, ex in enumerate(ds_entry["examples"]):
                preds = {}
                for key in ["predict_SIMPLICIAL", "predict_RANDOM_MATCHED", "predict_UNCONSTRAINED"]:
                    if key in ex:
                        preds[key.replace("predict_", "")] = float(ex[key])
                if preds:
                    predictions[ds_name][i] = preds

    # Now build the combined results and output
    from scipy import stats
    import numpy as np

    # Compute statistics using whatever results we have
    from method import compute_statistics, build_output, DatasetInfo

    # Reconstruct datasets dict
    datasets = {}
    for ds_entry in raw_data["datasets"]:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        X = np.array([json.loads(ex["input"]) for ex in examples], dtype=np.float64)
        task_type = examples[0]["metadata_task_type"]
        if task_type == "classification":
            y = np.array([int(ex["output"]) for ex in examples], dtype=np.int64)
        else:
            y = np.array([float(ex["output"]) for ex in examples], dtype=np.float64)
        fold_ids = np.array([ex["metadata_fold"] for ex in examples], dtype=np.int64)
        feature_names = examples[0]["metadata_feature_names"]
        ki_str = examples[0].get("metadata_known_interactions")
        known_interactions = json.loads(ki_str) if ki_str else None
        datasets[ds_name] = DatasetInfo(
            name=ds_name, X=X, y=y, fold_ids=fold_ids,
            task_type=task_type, feature_names=feature_names,
            known_interactions=known_interactions,
        )

    logger.info(f"Combined results for: {list(all_results.keys())}")
    statistics = compute_statistics(all_results)
    logger.info(f"Conclusion: {statistics['conclusion']}")
    logger.info(f"Aggregate scores: {statistics['aggregate'].get('mean_score', {})}")

    output = build_output(datasets, all_results, predictions, statistics)

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Output written: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
