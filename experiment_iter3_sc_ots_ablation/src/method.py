#!/usr/bin/env python3
"""SC-OTS Ablation: Simplicial vs Random-Matched vs Unconstrained Oblique Tree Splits.

Three-mode ablation (SIMPLICIAL vs RANDOM_MATCHED vs UNCONSTRAINED oblique tree splits)
across 10 datasets with 5-fold CV, testing whether topological constraints from persistent
homology genuinely improve predictive accuracy and interaction discovery over random or
unconstrained baselines.
"""

from __future__ import annotations

import itertools
import json
import resource
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dcor
import gudhi
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ───────────────────────────────────────────────────────────
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))  # 30GB RAM
# Note: CPU time limit removed to avoid premature kills on multi-core systems

# ── Paths ────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_PATH = (
    WORKSPACE.parents[2]
    / "iter_1"
    / "gen_art"
    / "data_id3_it1__opus"
    / "full_data_out.json"
)
OUTPUT_PATH = WORKSPACE / "method_out.json"

# ── Constants ────────────────────────────────────────────────────────
MAX_TOTAL_SPLITS = 12
MAX_TREES = 4
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 5
DCOR_SUBSAMPLE = 200
PEARSON_FEATURE_THRESHOLD = 30  # Use Pearson instead of dCor above this
MAX_SUBSETS_PER_EVAL = 30
N_RANDOM_SEEDS = 5
N_FOLDS = 5


# ════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════


@dataclass
class DatasetInfo:
    name: str
    X: np.ndarray
    y: np.ndarray
    fold_ids: np.ndarray
    task_type: str
    feature_names: list[str]
    known_interactions: Optional[dict] = None
    n_samples: int = 0
    n_features: int = 0

    def __post_init__(self) -> None:
        self.n_samples, self.n_features = self.X.shape


@dataclass
class ObliqueSplit:
    feature_indices: tuple
    weights: np.ndarray
    bias: float
    scaler: Optional[StandardScaler]
    threshold: float
    impurity_decrease: float
    left_mask: np.ndarray
    right_mask: np.ndarray


@dataclass
class TreeNode:
    """Node in a decision tree."""

    is_leaf: bool = True
    value: float = 0.0
    split: Optional[ObliqueSplit] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    sample_indices: Optional[np.ndarray] = None
    depth: int = 0
    n_samples: int = 0


@dataclass
class DecisionTree:
    root: Optional[TreeNode] = None
    n_splits: int = 0


# ════════════════════════════════════════════════════════════════════
# PHASE 0: DATA LOADING
# ════════════════════════════════════════════════════════════════════


@logger.catch
def load_datasets(data_path: Path) -> dict[str, DatasetInfo]:
    """Load all datasets from full_data_out.json."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    logger.info(f"Loaded JSON with {len(raw['datasets'])} datasets")

    datasets: dict[str, DatasetInfo] = {}
    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        n = len(examples)

        X = np.array([json.loads(ex["input"]) for ex in examples], dtype=np.float64)
        task_type = examples[0]["metadata_task_type"]

        if task_type == "classification":
            y = np.array([int(ex["output"]) for ex in examples], dtype=np.int64)
        else:
            y = np.array([float(ex["output"]) for ex in examples], dtype=np.float64)

        fold_ids = np.array([ex["metadata_fold"] for ex in examples], dtype=np.int64)
        feature_names = examples[0]["metadata_feature_names"]

        # Parse known interactions
        known_interactions = None
        ki_str = examples[0].get("metadata_known_interactions")
        if ki_str:
            known_interactions = json.loads(ki_str)

        ds_info = DatasetInfo(
            name=ds_name,
            X=X,
            y=y,
            fold_ids=fold_ids,
            task_type=task_type,
            feature_names=feature_names,
            known_interactions=known_interactions,
        )
        datasets[ds_name] = ds_info
        logger.info(
            f"  {ds_name}: {n} examples, {X.shape[1]} features, "
            f"task={task_type}, folds={np.unique(fold_ids).tolist()}"
        )

    return datasets


# ════════════════════════════════════════════════════════════════════
# PHASE 1: OBLIQUE SPLIT OPTIMIZER
# ════════════════════════════════════════════════════════════════════


def compute_impurity(y: np.ndarray, task_type: str) -> float:
    """Compute impurity of a node: variance for regression, Gini for classification."""
    if len(y) == 0:
        return 0.0
    if task_type == "regression":
        return float(np.var(y))
    else:
        # Gini impurity
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs**2))


def compute_impurity_decrease(
    y: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    task_type: str,
) -> float:
    """Compute weighted impurity decrease from a split."""
    n = len(y)
    n_left = left_mask.sum()
    n_right = right_mask.sum()
    if n_left == 0 or n_right == 0:
        return 0.0

    parent_impurity = compute_impurity(y, task_type)
    left_impurity = compute_impurity(y[left_mask], task_type)
    right_impurity = compute_impurity(y[right_mask], task_type)

    weighted_child = (n_left / n) * left_impurity + (n_right / n) * right_impurity
    return parent_impurity - weighted_child


def fit_oblique_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_subset: tuple,
    task_type: str,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
) -> Optional[ObliqueSplit]:
    """Fit an oblique split on the given feature subset.

    Returns None if no valid split can be found.
    """
    n_samples = len(y)
    if n_samples < 2 * min_samples_leaf:
        return None

    X_sub = X[:, list(feature_subset)]
    scaler = StandardScaler()

    try:
        X_scaled = scaler.fit_transform(X_sub)
    except ValueError:
        return None

    # Check for constant features
    if np.all(scaler.scale_ < 1e-10):
        return None

    # Fit linear model
    try:
        if task_type == "classification":
            n_classes = len(np.unique(y))
            if n_classes < 2:
                return None
            model = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=0.85,
                max_iter=100,
                tol=1e-4,
                random_state=42,
            )
            model.fit(X_scaled, y)
            w = model.coef_.flatten()
            b = float(np.asarray(model.intercept_).flatten()[0])
        else:
            model = SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                l1_ratio=0.85,
                max_iter=100,
                tol=1e-4,
                random_state=42,
            )
            model.fit(X_scaled, y)
            w = model.coef_.flatten()
            b = float(np.asarray(model.intercept_).flatten()[0])
    except Exception:
        # Fallback: axis-aligned split on the feature with highest abs correlation
        return _axis_aligned_fallback(X_sub, y, feature_subset, task_type, min_samples_leaf, scaler)

    # Check for degenerate weights
    if np.all(np.abs(w) < 1e-10):
        return _axis_aligned_fallback(X_sub, y, feature_subset, task_type, min_samples_leaf, scaler)

    # Compute oblique projection
    z = X_scaled @ w + b

    # Find best threshold via percentile candidates
    try:
        candidates = np.percentile(z, np.arange(10, 100, 10))
    except Exception:
        return None

    best_thresh = None
    best_decrease = -1.0

    for t in candidates:
        left_mask = z <= t
        right_mask = ~left_mask
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue
        decrease = compute_impurity_decrease(y, left_mask, right_mask, task_type)
        if decrease > best_decrease:
            best_decrease = decrease
            best_thresh = t

    if best_thresh is None or best_decrease <= 0:
        return None

    left_mask = z <= best_thresh
    right_mask = ~left_mask

    return ObliqueSplit(
        feature_indices=feature_subset,
        weights=w,
        bias=b,
        scaler=scaler,
        threshold=best_thresh,
        impurity_decrease=best_decrease,
        left_mask=left_mask,
        right_mask=right_mask,
    )


def _axis_aligned_fallback(
    X_sub: np.ndarray,
    y: np.ndarray,
    feature_subset: tuple,
    task_type: str,
    min_samples_leaf: int,
    scaler: StandardScaler,
) -> Optional[ObliqueSplit]:
    """Fallback: find best single-feature threshold split."""
    n_features_sub = X_sub.shape[1]
    best_decrease = -1.0
    best_split = None

    for fi in range(n_features_sub):
        col = X_sub[:, fi]
        unique_vals = np.unique(col)
        if len(unique_vals) < 2:
            continue

        # Use percentile candidates
        try:
            candidates = np.percentile(col, np.arange(10, 100, 10))
        except Exception:
            continue

        for t in candidates:
            left_mask = col <= t
            right_mask = ~left_mask
            if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                continue
            decrease = compute_impurity_decrease(y, left_mask, right_mask, task_type)
            if decrease > best_decrease:
                best_decrease = decrease
                w = np.zeros(n_features_sub)
                w[fi] = 1.0
                best_split = ObliqueSplit(
                    feature_indices=feature_subset,
                    weights=w,
                    bias=0.0,
                    scaler=None,  # No scaling for axis-aligned
                    threshold=t,
                    impurity_decrease=decrease,
                    left_mask=left_mask,
                    right_mask=right_mask,
                )

    return best_split


# ════════════════════════════════════════════════════════════════════
# PHASE 2: FIGS-STYLE GREEDY TREE SUM
# ════════════════════════════════════════════════════════════════════


class FIGSTreeSum:
    """Greedy additive tree sum (FIGS-style) with oblique splits."""

    def __init__(
        self,
        max_total_splits: int = MAX_TOTAL_SPLITS,
        max_trees: int = MAX_TREES,
        max_depth: int = MAX_DEPTH,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
    ) -> None:
        self.max_total_splits = max_total_splits
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees_: list[DecisionTree] = []
        self.task_type: str = "regression"
        self._global_mean: float = 0.0

    def _get_total_splits(self) -> int:
        return sum(t.n_splits for t in self.trees_)

    def _predict_tree(self, tree: DecisionTree, X: np.ndarray) -> np.ndarray:
        """Predict using a single tree for all samples."""
        preds = np.full(X.shape[0], 0.0)
        if tree.root is None:
            return preds
        self._predict_node(tree.root, X, np.arange(X.shape[0]), preds)
        return preds

    def _predict_node(
        self,
        node: TreeNode,
        X: np.ndarray,
        indices: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        """Recursively predict for a node."""
        if len(indices) == 0:
            return
        if node.is_leaf:
            preds[indices] = node.value
            return

        split = node.split
        if split is None:
            preds[indices] = node.value
            return

        feat_idx = list(split.feature_indices)
        X_sub = X[np.ix_(indices, feat_idx)]

        if split.scaler is not None:
            X_scaled = split.scaler.transform(X_sub)
            z = X_scaled @ split.weights + split.bias
        else:
            z = X_sub @ split.weights + split.bias

        left_mask = z <= split.threshold
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        self._predict_node(node.left, X, left_indices, preds)
        self._predict_node(node.right, X, right_indices, preds)

    def _compute_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals: y - sum of all tree predictions."""
        total_pred = np.full(len(y), self._global_mean)
        for tree in self.trees_:
            total_pred += self._predict_tree(tree, X)
        return y - total_pred

    def _get_leaves(self, tree: DecisionTree) -> list[TreeNode]:
        """Get all leaf nodes from a tree."""
        if tree.root is None:
            return []
        leaves = []
        self._collect_leaves(tree.root, leaves)
        return leaves

    def _collect_leaves(self, node: TreeNode, leaves: list[TreeNode]) -> None:
        if node.is_leaf:
            leaves.append(node)
        else:
            if node.left:
                self._collect_leaves(node.left, leaves)
            if node.right:
                self._collect_leaves(node.right, leaves)

    def _route_samples(self, node: TreeNode, X: np.ndarray, indices: np.ndarray) -> dict:
        """Route samples to leaves, returning {leaf_id: sample_indices}."""
        result: dict[int, np.ndarray] = {}
        self._route_recursive(node, X, indices, result)
        return result

    def _route_recursive(
        self,
        node: TreeNode,
        X: np.ndarray,
        indices: np.ndarray,
        result: dict,
    ) -> None:
        if len(indices) == 0:
            return
        if node.is_leaf:
            result[id(node)] = indices
            return

        split = node.split
        feat_idx = list(split.feature_indices)
        X_sub = X[np.ix_(indices, feat_idx)]

        if split.scaler is not None:
            X_scaled = split.scaler.transform(X_sub)
            z = X_scaled @ split.weights + split.bias
        else:
            z = X_sub @ split.weights + split.bias

        left_mask = z <= split.threshold
        right_mask = ~left_mask

        self._route_recursive(node.left, X, indices[left_mask], result)
        self._route_recursive(node.right, X, indices[right_mask], result)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        constraint_source: "ConstraintSource",
        task_type: str,
    ) -> "FIGSTreeSum":
        """Fit the FIGS tree sum model."""
        self.task_type = task_type
        self.trees_ = []

        if task_type == "regression":
            self._global_mean = float(np.mean(y_train))
        else:
            # For classification, use log-odds of positive class
            p = np.mean(y_train)
            p = np.clip(p, 0.01, 0.99)
            self._global_mean = float(np.log(p / (1 - p)))

        n_train = len(y_train)

        # Internal validation split for early stopping
        if n_train > 50:
            try:
                if task_type == "classification":
                    tr_idx, val_idx = train_test_split(
                        np.arange(n_train),
                        test_size=0.1,
                        stratify=y_train,
                        random_state=42,
                    )
                else:
                    tr_idx, val_idx = train_test_split(
                        np.arange(n_train),
                        test_size=0.1,
                        random_state=42,
                    )
            except ValueError:
                # If stratification fails, do random split
                tr_idx, val_idx = train_test_split(
                    np.arange(n_train), test_size=0.1, random_state=42
                )
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

        no_improvement_count = 0
        best_val_score = -np.inf

        allowed_subsets = constraint_source.get_subsets()
        if len(allowed_subsets) == 0:
            logger.warning("No allowed subsets from constraint source, model will be trivial")
            return self

        # Subsample allowed subsets if too many
        if len(allowed_subsets) > MAX_SUBSETS_PER_EVAL:
            rng = np.random.RandomState(42)
            subset_indices = rng.choice(len(allowed_subsets), MAX_SUBSETS_PER_EVAL, replace=False)
            eval_subsets = [allowed_subsets[i] for i in subset_indices]
        else:
            eval_subsets = allowed_subsets

        for _step in range(self.max_total_splits):
            if self._get_total_splits() >= self.max_total_splits:
                break
            if len(self.trees_) >= self.max_trees and all(
                len(self._get_leaves(t)) == 0 or all(
                    leaf.depth >= self.max_depth for leaf in self._get_leaves(t)
                )
                for t in self.trees_
            ):
                break

            residuals = self._compute_residuals(X_tr, y_tr)

            # For classification with log-odds residuals
            if task_type == "classification":
                total_pred = np.full(len(y_tr), self._global_mean)
                for tree in self.trees_:
                    total_pred += self._predict_tree(tree, X_tr)
                probs = 1.0 / (1.0 + np.exp(-total_pred))
                probs = np.clip(probs, 1e-6, 1 - 1e-6)
                # Gradient residuals for log-loss
                residuals = y_tr - probs

            best_overall_split = None
            best_overall_decrease = -1.0
            best_position = None  # (tree_idx, leaf_node) or ('new', None)

            # Evaluate splitting existing leaves
            for tree_idx, tree in enumerate(self.trees_):
                if tree.root is None:
                    continue

                leaf_samples = self._route_samples(tree.root, X_tr, np.arange(len(X_tr)))

                for leaf in self._get_leaves(tree):
                    if leaf.depth >= self.max_depth:
                        continue

                    leaf_id = id(leaf)
                    if leaf_id not in leaf_samples:
                        continue
                    sample_idx = leaf_samples[leaf_id]
                    if len(sample_idx) < 2 * self.min_samples_leaf:
                        continue

                    X_leaf = X_tr[sample_idx]
                    r_leaf = residuals[sample_idx]

                    for subset in eval_subsets:
                        split = fit_oblique_split(
                            X_leaf, r_leaf, subset, "regression", self.min_samples_leaf
                        )
                        if split is not None and split.impurity_decrease > best_overall_decrease:
                            best_overall_decrease = split.impurity_decrease
                            best_overall_split = split
                            best_position = (tree_idx, leaf)

            # Evaluate creating a new tree (if we have room)
            if len(self.trees_) < self.max_trees:
                for subset in eval_subsets:
                    split = fit_oblique_split(
                        X_tr, residuals, subset, "regression", self.min_samples_leaf
                    )
                    if split is not None and split.impurity_decrease > best_overall_decrease:
                        best_overall_decrease = split.impurity_decrease
                        best_overall_split = split
                        best_position = ("new", None)

            if best_overall_split is None or best_overall_decrease <= 1e-8:
                break

            # Install the split
            split = best_overall_split
            if best_position[0] == "new":
                # Create new tree
                root = TreeNode(is_leaf=False, depth=0, n_samples=len(X_tr))
                root.split = split

                left_idx = np.where(split.left_mask)[0]
                right_idx = np.where(split.right_mask)[0]

                root.left = TreeNode(
                    is_leaf=True,
                    value=float(np.mean(residuals[left_idx])),
                    depth=1,
                    n_samples=len(left_idx),
                )
                root.right = TreeNode(
                    is_leaf=True,
                    value=float(np.mean(residuals[right_idx])),
                    depth=1,
                    n_samples=len(right_idx),
                )

                new_tree = DecisionTree(root=root, n_splits=1)
                self.trees_.append(new_tree)
            else:
                # Split an existing leaf
                tree_idx, leaf_node = best_position
                leaf_id = id(leaf_node)
                leaf_samples_map = self._route_samples(
                    self.trees_[tree_idx].root, X_tr, np.arange(len(X_tr))
                )
                sample_idx = leaf_samples_map.get(leaf_id, np.array([], dtype=int))

                leaf_node.is_leaf = False
                leaf_node.split = split

                left_idx = sample_idx[split.left_mask]
                right_idx = sample_idx[split.right_mask]

                leaf_node.left = TreeNode(
                    is_leaf=True,
                    value=float(np.mean(residuals[left_idx])) if len(left_idx) > 0 else 0.0,
                    depth=leaf_node.depth + 1,
                    n_samples=len(left_idx),
                )
                leaf_node.right = TreeNode(
                    is_leaf=True,
                    value=float(np.mean(residuals[right_idx])) if len(right_idx) > 0 else 0.0,
                    depth=leaf_node.depth + 1,
                    n_samples=len(right_idx),
                )

                self.trees_[tree_idx].n_splits += 1

            # Early stopping check
            if X_val is not None:
                val_score = self._evaluate_score(X_val, y_val)
                if val_score > best_val_score:
                    best_val_score = val_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if no_improvement_count >= 3:
                    break

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict raw values (log-odds for classification)."""
        preds = np.full(X.shape[0], self._global_mean)
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels/values."""
        raw = self.predict_raw(X)
        if self.task_type == "classification":
            probs = 1.0 / (1.0 + np.exp(-raw))
            return (probs >= 0.5).astype(int)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for classification."""
        raw = self.predict_raw(X)
        probs = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(probs, 0.0, 1.0)

    def _evaluate_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model score on data."""
        if self.task_type == "classification":
            try:
                probs = self.predict_proba(X)
                return float(roc_auc_score(y, probs))
            except ValueError:
                preds = self.predict(X)
                return float(accuracy_score(y, preds))
        else:
            preds = self.predict_raw(X)
            return float(r2_score(y, preds))


# ════════════════════════════════════════════════════════════════════
# PHASE 3: CONSTRAINT SOURCES
# ════════════════════════════════════════════════════════════════════


class ConstraintSource:
    """Base class for constraint sources."""

    def __init__(self) -> None:
        self.allowed_subsets: list[tuple] = []

    def get_subsets(self) -> list[tuple]:
        return self.allowed_subsets


class SimplicialConstraintSource(ConstraintSource):
    """TDA pipeline: dCor → GUDHI Rips → persistence → simplices → MI filter."""

    def __init__(self) -> None:
        super().__init__()
        self.threshold: float = 0.0
        self.betti_numbers: list[int] = [0, 0, 0]
        self.n_simplices_by_dim: dict[str, int] = {}

    def build(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str,
        feature_names: list[str],
    ) -> "SimplicialConstraintSource":
        """Build simplicial complex from training data."""
        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]

        # ── Step 1: Distance matrix ──
        if n_features > PEARSON_FEATURE_THRESHOLD:
            logger.debug(f"Using Pearson correlation (n_features={n_features} > {PEARSON_FEATURE_THRESHOLD})")
            D = self._compute_pearson_distance(X_train)
        else:
            logger.debug(f"Using dCor distance (n_features={n_features})")
            D = self._compute_dcor_distance(X_train)

        # ── Step 2: Target-aware MI weighting ──
        try:
            if task_type == "classification":
                mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

            # Weight distances: reduce for features that both relate to target
            epsilon = 0.01
            D_weighted = D.copy()
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    weight = np.sqrt(mi_scores[i] * mi_scores[j]) + epsilon
                    D_weighted[i, j] /= weight
                    D_weighted[j, i] /= weight
        except Exception:
            logger.warning("MI weighting failed, using unweighted distances")
            D_weighted = D.copy()
            mi_scores = np.ones(n_features)

        # ── Step 3: Build GUDHI Rips complex ──
        try:
            max_edge = float(np.max(D_weighted[D_weighted < np.inf]))
            if max_edge <= 0 or not np.isfinite(max_edge):
                max_edge = 2.0

            # Ensure distance matrix has zeros on diagonal
            np.fill_diagonal(D_weighted, 0.0)

            rips = gudhi.RipsComplex(
                distance_matrix=D_weighted.tolist(),
                max_edge_length=max_edge,
            )
            max_dim = min(3, n_features - 1)
            st = rips.create_simplex_tree(max_dimension=max_dim)
        except Exception:
            logger.exception("GUDHI Rips construction failed")
            return self._fallback_mi_pairs(mi_scores, n_features)

        # ── Step 4: Persistence and threshold ──
        try:
            persistence = st.persistence()
            threshold = self._select_persistence_threshold(st, persistence)
            self.threshold = threshold
        except Exception:
            logger.exception("Persistence computation failed")
            threshold = max_edge * 0.75
            self.threshold = threshold

        # ── Step 5: Extract persistent simplices ──
        simplices = []
        for simplex, filt_value in st.get_filtration():
            if filt_value <= threshold and len(simplex) >= 2:
                simplices.append(tuple(sorted(simplex)))

        # Remove duplicates
        simplices = list(set(simplices))

        # ── Step 6: MI-based filtering ──
        if len(mi_scores) > 0:
            mi_threshold = np.percentile(mi_scores, 25)
            filtered = [
                s for s in simplices if all(mi_scores[f] >= mi_threshold for f in s if f < len(mi_scores))
            ]
            if len(filtered) >= 3:
                simplices = filtered

        # ── Cap at manageable number ──
        if len(simplices) > 500:
            # Keep top simplices by avg MI
            scored = []
            for s in simplices:
                avg_mi = np.mean([mi_scores[f] for f in s if f < len(mi_scores)])
                scored.append((avg_mi, s))
            scored.sort(reverse=True)
            simplices = [s for _, s in scored[:200]]

        # ── Step 7: Safety net ──
        simplices = self._safety_net(simplices, mi_scores, n_features)

        self.allowed_subsets = simplices
        self._compute_complex_info(simplices, st)

        logger.debug(f"Simplicial complex: {len(simplices)} simplices, dims={self.n_simplices_by_dim}")
        return self

    def _compute_dcor_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance correlation and convert to distance."""
        n_features = X.shape[1]
        n_sub = min(X.shape[0], DCOR_SUBSAMPLE)
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], n_sub, replace=False)
        X_sub = X[idx]

        dcor_matrix = np.ones((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    dc = dcor.distance_correlation(X_sub[:, i], X_sub[:, j])
                    if not np.isfinite(dc):
                        dc = 0.0
                    dcor_matrix[i, j] = dc
                    dcor_matrix[j, i] = dc
                except Exception:
                    dcor_matrix[i, j] = 0.0
                    dcor_matrix[j, i] = 0.0

        # Convert to dissimilarity
        D = np.sqrt(np.clip(1.0 - dcor_matrix, 0.0, 1.0))
        np.fill_diagonal(D, 0.0)
        return D

    def _compute_pearson_distance(self, X: np.ndarray) -> np.ndarray:
        """Fast distance using Pearson correlation."""
        n_sub = min(X.shape[0], 500)
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], n_sub, replace=False)
        X_sub = X[idx]

        # Correlation matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(X_sub.T)

        # Handle NaN
        corr = np.nan_to_num(corr, nan=0.0)
        D = 1.0 - np.abs(corr)
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, 2.0)
        return D

    def _select_persistence_threshold(
        self,
        st: "gudhi.SimplexTree",
        persistence: list,
    ) -> float:
        """Select threshold using largest-gap heuristic."""
        death_values = []
        for dim, (birth, death) in persistence:
            if np.isfinite(death) and death > 0:
                death_values.append(death)

        if len(death_values) < 2:
            # Fallback: use 75th percentile of filtration values
            filt_values = [fv for _, fv in st.get_filtration() if fv > 0]
            if filt_values:
                return float(np.percentile(filt_values, 75))
            return 1.0

        death_values.sort()
        # Largest gap heuristic
        gaps = np.diff(death_values)
        if len(gaps) == 0:
            return float(death_values[-1])

        largest_gap_idx = np.argmax(gaps)
        threshold = (death_values[largest_gap_idx] + death_values[largest_gap_idx + 1]) / 2.0
        return float(threshold)

    def _safety_net(
        self,
        simplices: list[tuple],
        mi_scores: np.ndarray,
        n_features: int,
    ) -> list[tuple]:
        """Ensure enough simplices for the model to work."""
        if len(simplices) >= max(3, n_features // 2):
            return simplices

        # Add top-MI feature pairs
        ranked = np.argsort(mi_scores)[::-1]
        top_features = ranked[: max(n_features // 2, 3)]
        additional = []
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                pair = tuple(sorted([int(top_features[i]), int(top_features[j])]))
                if pair not in simplices and pair not in additional:
                    additional.append(pair)

        simplices = simplices + additional
        return simplices

    def _fallback_mi_pairs(
        self,
        mi_scores: np.ndarray,
        n_features: int,
    ) -> "SimplicialConstraintSource":
        """Fallback: use top-MI feature pairs as simplices."""
        logger.warning("Using MI-pair fallback for simplicial complex")
        ranked = np.argsort(mi_scores)[::-1]
        top_k = max(n_features // 2, 3)
        top_features = ranked[:top_k]

        simplices = []
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                simplices.append(tuple(sorted([int(top_features[i]), int(top_features[j])])))

        self.allowed_subsets = simplices
        self.threshold = 0.0
        self.betti_numbers = [0, 0, 0]
        self.n_simplices_by_dim = {"1": len(simplices)}
        return self

    def _compute_complex_info(self, simplices: list[tuple], st: "gudhi.SimplexTree") -> None:
        """Compute Betti numbers and simplex counts."""
        dim_counts: dict[str, int] = {}
        for s in simplices:
            dim = str(len(s) - 1)
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        self.n_simplices_by_dim = dim_counts

        # Betti numbers from persistence
        try:
            betti = st.betti_numbers()
            self.betti_numbers = list(betti[:3]) if len(betti) >= 3 else list(betti) + [0] * (3 - len(betti))
        except Exception:
            self.betti_numbers = [0, 0, 0]


class RandomMatchedConstraintSource(ConstraintSource):
    """Generate random feature subsets matching Mode A's cardinality distribution."""

    def build(
        self,
        simplicial_source: SimplicialConstraintSource,
        n_features: int,
        seed: int,
    ) -> "RandomMatchedConstraintSource":
        """Build random-matched subsets."""
        cardinality_dist = Counter(len(s) for s in simplicial_source.allowed_subsets)
        rng = np.random.RandomState(seed)

        subsets = []
        for size, count in cardinality_dist.items():
            if size > n_features:
                size = min(size, n_features)
            for _ in range(count):
                subset = tuple(sorted(rng.choice(n_features, size=size, replace=False)))
                subsets.append(subset)

        # Ensure feature coverage
        covered = set(f for s in subsets for f in s)
        uncovered = set(range(n_features)) - covered
        for f in uncovered:
            partner = rng.randint(0, n_features)
            while partner == f:
                partner = rng.randint(0, n_features)
            subsets.append(tuple(sorted([f, partner])))

        # Remove duplicates
        self.allowed_subsets = list(set(subsets))
        return self


class UnconstrainedConstraintSource(ConstraintSource):
    """All feature subsets up to size 4. Cap at MAX_SUBSETS_PER_EVAL random candidates."""

    def build(
        self,
        n_features: int,
        max_size: int = 4,
        max_candidates: int = 200,
    ) -> "UnconstrainedConstraintSource":
        """Build unconstrained subsets."""
        all_subsets = []
        actual_max_size = min(max_size, n_features)
        for k in range(2, actual_max_size + 1):
            all_subsets.extend(itertools.combinations(range(n_features), k))

        if len(all_subsets) > max_candidates:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(all_subsets), max_candidates, replace=False)
            self.allowed_subsets = [tuple(all_subsets[i]) for i in indices]
        else:
            self.allowed_subsets = [tuple(s) for s in all_subsets]

        return self


# ════════════════════════════════════════════════════════════════════
# PHASE 4: EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: FIGSTreeSum,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> float:
    """Evaluate model score."""
    if task_type == "classification":
        try:
            probs = model.predict_proba(X_test)
            return float(roc_auc_score(y_test, probs))
        except ValueError:
            preds = model.predict(X_test)
            return float(accuracy_score(y_test, preds))
    else:
        preds = model.predict_raw(X_test)
        return float(r2_score(y_test, preds))


def get_model_predictions(
    model: FIGSTreeSum,
    X_test: np.ndarray,
    task_type: str,
) -> np.ndarray:
    """Get predictions for output."""
    if task_type == "classification":
        return model.predict_proba(X_test)
    else:
        return model.predict_raw(X_test)


def compute_interaction_recovery(
    allowed_subsets: list[tuple],
    known_interactions: dict,
) -> dict[str, float]:
    """Compute precision/recall/F1 for interaction recovery."""
    # Build set of known interaction tuples
    known_set = set()
    for key in ["2-way", "3-way", "4-way"]:
        if key in known_interactions:
            for interaction in known_interactions[key]:
                known_set.add(tuple(sorted(interaction)))

    if len(known_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Build set of predicted interaction tuples (simplices of matching sizes)
    predicted_set = set()
    for s in allowed_subsets:
        predicted_set.add(tuple(sorted(s)))

    # Compute overlap
    tp = len(predicted_set & known_set)
    precision = tp / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = tp / len(known_set) if len(known_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ════════════════════════════════════════════════════════════════════
# PHASE 5: MAIN EXPERIMENT LOOP
# ════════════════════════════════════════════════════════════════════


@logger.catch
def run_experiment(
    datasets: dict[str, DatasetInfo],
    max_datasets: Optional[int] = None,
    max_folds: Optional[int] = None,
    start_dataset_idx: int = 0,
) -> dict:
    """Run the full ablation experiment."""
    # Load existing checkpoint to merge with (resume support)
    checkpoint_path = WORKSPACE / "tmp" / "checkpoint_results.json"
    results: dict = {}
    predictions: dict[str, dict[int, dict[str, float]]] = {}
    if checkpoint_path.exists():
        try:
            existing = json.loads(checkpoint_path.read_text())
            results = existing.get("results", {})
            # Convert string keys back to int (JSON only supports string keys)
            raw_preds = existing.get("predictions", {})
            for ds_name, ds_preds in raw_preds.items():
                predictions[ds_name] = {int(k): v for k, v in ds_preds.items()}
            logger.info(f"Resuming: loaded {len(results)} completed datasets from checkpoint: {list(results.keys())}")
        except Exception:
            logger.warning("Failed to load existing checkpoint, starting fresh")

    dataset_list = list(datasets.items())
    if max_datasets is not None:
        end_idx = start_dataset_idx + max_datasets
        dataset_list = dataset_list[start_dataset_idx:end_idx]
    else:
        dataset_list = dataset_list[start_dataset_idx:]

    total_start = time.time()
    n_folds_to_run = max_folds if max_folds is not None else N_FOLDS

    for ds_idx, (ds_name, ds) in enumerate(dataset_list):
        # Skip already-completed datasets
        if ds_name in results and len(results[ds_name].get("folds", {})) >= n_folds_to_run:
            logger.info(f"═══ SKIPPING {ds_name} (already completed with {len(results[ds_name]['folds'])} folds) ═══")
            continue
        logger.info(f"═══ Dataset {ds_idx+1}/{len(dataset_list)}: {ds_name} "
                     f"(n={ds.n_samples}, p={ds.n_features}, task={ds.task_type}) ═══")

        results[ds_name] = {
            "task_type": ds.task_type,
            "n_samples": ds.n_samples,
            "n_features": ds.n_features,
            "folds": {},
        }
        predictions[ds_name] = {}

        ds_start = time.time()

        for fold in range(n_folds_to_run):
            fold_start = time.time()
            logger.info(f"  Fold {fold}/{n_folds_to_run-1}")

            train_mask = ds.fold_ids != fold
            test_mask = ds.fold_ids == fold

            X_train, y_train = ds.X[train_mask], ds.y[train_mask]
            X_test, y_test = ds.X[test_mask], ds.y[test_mask]
            test_indices = np.where(test_mask)[0]

            # Standardize features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            fold_results: dict = {}

            # ── MODE A: SIMPLICIAL ──
            try:
                simp_src = SimplicialConstraintSource()
                simp_src.build(
                    X_train=X_train_s,
                    y_train=y_train,
                    task_type=ds.task_type,
                    feature_names=ds.feature_names,
                )

                model_A = FIGSTreeSum(
                    max_total_splits=MAX_TOTAL_SPLITS,
                    max_trees=MAX_TREES,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                )
                t0 = time.time()
                model_A.fit(X_train_s, y_train, simp_src, ds.task_type)
                time_A = time.time() - t0

                score_A = evaluate_model(model_A, X_test_s, y_test, ds.task_type)
                splits_A = model_A._get_total_splits()
                preds_A = get_model_predictions(model_A, X_test_s, ds.task_type)

                fold_results["SIMPLICIAL"] = {
                    "score": round(score_A, 6),
                    "splits": splits_A,
                    "time_s": round(time_A, 3),
                }

                logger.info(f"    Mode A (SIMPLICIAL): score={score_A:.4f}, "
                           f"splits={splits_A}, time={time_A:.1f}s, "
                           f"n_simplices={len(simp_src.allowed_subsets)}")
            except Exception:
                logger.exception(f"    Mode A failed on {ds_name} fold {fold}")
                fold_results["SIMPLICIAL"] = {"score": 0.0, "splits": 0, "time_s": 0.0}
                preds_A = np.zeros(len(y_test))
                simp_src = SimplicialConstraintSource()
                simp_src.allowed_subsets = [(0, 1)]

            # ── MODE B: RANDOM-MATCHED ──
            scores_B = []
            splits_B_list = []
            times_B = []
            preds_B_all = []

            for seed in range(N_RANDOM_SEEDS):
                try:
                    rand_src = RandomMatchedConstraintSource()
                    rand_src.build(simp_src, ds.n_features, seed)

                    model_B = FIGSTreeSum(
                        max_total_splits=MAX_TOTAL_SPLITS,
                        max_trees=MAX_TREES,
                        max_depth=MAX_DEPTH,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                    )
                    t0 = time.time()
                    model_B.fit(X_train_s, y_train, rand_src, ds.task_type)
                    t_B = time.time() - t0

                    s_B = evaluate_model(model_B, X_test_s, y_test, ds.task_type)
                    sp_B = model_B._get_total_splits()
                    p_B = get_model_predictions(model_B, X_test_s, ds.task_type)

                    scores_B.append(s_B)
                    splits_B_list.append(sp_B)
                    times_B.append(t_B)
                    preds_B_all.append(p_B)
                except Exception:
                    logger.exception(f"    Mode B seed {seed} failed on {ds_name} fold {fold}")
                    scores_B.append(0.0)
                    splits_B_list.append(0)
                    times_B.append(0.0)
                    preds_B_all.append(np.zeros(len(y_test)))

            score_B_mean = float(np.mean(scores_B))
            splits_B_mean = float(np.mean(splits_B_list))
            time_B_mean = float(np.mean(times_B))
            preds_B = np.mean(preds_B_all, axis=0) if preds_B_all else np.zeros(len(y_test))

            fold_results["RANDOM_MATCHED"] = {
                "score": round(score_B_mean, 6),
                "splits": round(splits_B_mean, 1),
                "time_s": round(time_B_mean, 3),
                "scores_per_seed": [round(s, 6) for s in scores_B],
            }

            logger.info(f"    Mode B (RANDOM_MATCHED): score={score_B_mean:.4f}, "
                        f"splits={splits_B_mean:.1f}, time={time_B_mean:.1f}s")

            # ── MODE C: UNCONSTRAINED ──
            try:
                uncon_src = UnconstrainedConstraintSource()
                uncon_src.build(ds.n_features)

                model_C = FIGSTreeSum(
                    max_total_splits=MAX_TOTAL_SPLITS,
                    max_trees=MAX_TREES,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                )
                t0 = time.time()
                model_C.fit(X_train_s, y_train, uncon_src, ds.task_type)
                time_C = time.time() - t0

                score_C = evaluate_model(model_C, X_test_s, y_test, ds.task_type)
                splits_C = model_C._get_total_splits()
                preds_C = get_model_predictions(model_C, X_test_s, ds.task_type)

                fold_results["UNCONSTRAINED"] = {
                    "score": round(score_C, 6),
                    "splits": splits_C,
                    "time_s": round(time_C, 3),
                }

                logger.info(f"    Mode C (UNCONSTRAINED): score={score_C:.4f}, "
                           f"splits={splits_C}, time={time_C:.1f}s")
            except Exception:
                logger.exception(f"    Mode C failed on {ds_name} fold {fold}")
                fold_results["UNCONSTRAINED"] = {"score": 0.0, "splits": 0, "time_s": 0.0}
                preds_C = np.zeros(len(y_test))

            # ── Interaction recovery (synthetic only) ──
            if ds.known_interactions is not None:
                try:
                    interaction_metrics = compute_interaction_recovery(
                        simp_src.allowed_subsets, ds.known_interactions
                    )
                    fold_results["interaction_recovery"] = {
                        k: round(v, 4) for k, v in interaction_metrics.items()
                    }
                except Exception:
                    logger.exception("Interaction recovery failed")
                    fold_results["interaction_recovery"] = None
            else:
                fold_results["interaction_recovery"] = None

            # Simplicial complex info
            fold_results["simplicial_complex_info"] = {
                "n_simplices_by_dim": simp_src.n_simplices_by_dim,
                "betti_numbers": simp_src.betti_numbers,
                "persistence_threshold": round(simp_src.threshold, 6),
            }

            results[ds_name]["folds"][str(fold)] = fold_results

            # Store per-example predictions
            for local_i, global_i in enumerate(test_indices):
                predictions[ds_name][int(global_i)] = {
                    "SIMPLICIAL": float(preds_A[local_i]),
                    "RANDOM_MATCHED": float(preds_B[local_i]),
                    "UNCONSTRAINED": float(preds_C[local_i]),
                }

            fold_time = time.time() - fold_start
            logger.info(f"  Fold {fold} completed in {fold_time:.1f}s")

        # Per-dataset aggregation
        _aggregate_dataset(results[ds_name], n_folds_to_run)

        ds_time = time.time() - ds_start
        logger.info(f"  Dataset {ds_name} completed in {ds_time:.1f}s")

        # Checkpoint: save intermediate results (including predictions)
        checkpoint_path = WORKSPACE / "tmp" / "checkpoint_results.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Convert prediction keys to strings for JSON
            preds_json = {}
            for pds_name, pds_data in predictions.items():
                preds_json[pds_name] = {str(k): v for k, v in pds_data.items()}
            checkpoint_path.write_text(json.dumps({
                "completed_datasets": list(results.keys()),
                "results": results,
                "predictions": preds_json,
            }, default=str))
        except Exception:
            logger.warning("Failed to save checkpoint")

    total_time = time.time() - total_start
    logger.info(f"Total experiment time: {total_time:.1f}s")

    return results, predictions


def _aggregate_dataset(ds_results: dict, n_folds: int) -> None:
    """Compute per-dataset mean scores, splits, times."""
    modes = ["SIMPLICIAL", "RANDOM_MATCHED", "UNCONSTRAINED"]
    mean_scores: dict[str, float] = {}
    mean_splits: dict[str, float] = {}
    mean_times: dict[str, float] = {}

    for mode in modes:
        scores = []
        splits = []
        times = []
        for fold_key in ds_results["folds"]:
            fold_data = ds_results["folds"][fold_key]
            if mode in fold_data:
                scores.append(fold_data[mode]["score"])
                sp = fold_data[mode]["splits"]
                splits.append(float(sp) if isinstance(sp, (int, float)) else 0.0)
                times.append(fold_data[mode]["time_s"])

        mean_scores[mode] = round(float(np.mean(scores)) if scores else 0.0, 6)
        mean_splits[mode] = round(float(np.mean(splits)) if splits else 0.0, 1)
        mean_times[mode] = round(float(np.mean(times)) if times else 0.0, 3)

    ds_results["mean_scores"] = mean_scores
    ds_results["mean_splits"] = mean_splits
    ds_results["mean_time_s"] = mean_times


# ════════════════════════════════════════════════════════════════════
# PHASE 6: STATISTICAL TESTS & OUTPUT
# ════════════════════════════════════════════════════════════════════


def compute_statistics(results: dict) -> dict:
    """Compute Wilcoxon tests, win/tie/loss, aggregates."""
    modes = ["SIMPLICIAL", "RANDOM_MATCHED", "UNCONSTRAINED"]
    dataset_names = list(results.keys())

    # Per-dataset mean scores
    per_ds_scores: dict[str, dict[str, float]] = {}
    for ds_name in dataset_names:
        per_ds_scores[ds_name] = results[ds_name].get("mean_scores", {})

    # Arrays for statistical tests
    scores_A = [per_ds_scores[ds].get("SIMPLICIAL", 0.0) for ds in dataset_names]
    scores_B = [per_ds_scores[ds].get("RANDOM_MATCHED", 0.0) for ds in dataset_names]
    scores_C = [per_ds_scores[ds].get("UNCONSTRAINED", 0.0) for ds in dataset_names]

    # Wilcoxon signed-rank tests
    stat_tests = {}
    for label, s1, s2 in [
        ("A_vs_B", scores_A, scores_B),
        ("A_vs_C", scores_A, scores_C),
        ("B_vs_C", scores_B, scores_C),
    ]:
        try:
            diffs = np.array(s1) - np.array(s2)
            # Remove zero differences for Wilcoxon
            nonzero = np.abs(diffs) > 1e-10
            if nonzero.sum() >= 6:
                stat_val, p_val = stats.wilcoxon(
                    np.array(s1)[nonzero],
                    np.array(s2)[nonzero],
                    alternative="two-sided",
                )
                interpretation = (
                    "significant (p<0.05)" if p_val < 0.05 else "not significant (p>=0.05)"
                )
            else:
                # Too few non-tied pairs; use permutation test
                stat_val, p_val = _permutation_test(s1, s2)
                interpretation = f"permutation test (n_nonzero<6): {'sig' if p_val < 0.05 else 'not sig'}"

            stat_tests[label] = {
                "statistic": round(float(stat_val), 4),
                "p_value": round(float(p_val), 6),
                "interpretation": interpretation,
            }
        except Exception as e:
            stat_tests[label] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "interpretation": f"test failed: {str(e)[:50]}",
            }

    # Win/tie/loss
    win_tie_loss = {}
    for label, s1, s2, name1, name2 in [
        ("A_vs_B", scores_A, scores_B, "A", "B"),
        ("A_vs_C", scores_A, scores_C, "A", "C"),
        ("B_vs_C", scores_B, scores_C, "B", "C"),
    ]:
        wins_1 = sum(1 for a, b in zip(s1, s2) if a - b > 0.005)
        wins_2 = sum(1 for a, b in zip(s1, s2) if b - a > 0.005)
        ties = len(s1) - wins_1 - wins_2
        win_tie_loss[label] = {
            f"{name1}_wins": wins_1,
            "ties": ties,
            f"{name2}_wins": wins_2,
        }

    # Aggregate across datasets
    aggregate = {}
    for stat_name, score_list, label in [
        ("mean_score", [scores_A, scores_B, scores_C], modes),
        ("std_score", [scores_A, scores_B, scores_C], modes),
    ]:
        if "mean" in stat_name:
            aggregate[stat_name] = {m: round(float(np.mean(s)), 6) for m, s in zip(label, score_list)}
        else:
            aggregate[stat_name] = {m: round(float(np.std(s)), 6) for m, s in zip(label, score_list)}

    # Mean splits and time
    for metric in ["mean_splits", "mean_time_s"]:
        vals = {m: [] for m in modes}
        for ds_name in dataset_names:
            ds_data = results[ds_name]
            for mode in modes:
                v = ds_data.get(metric, {}).get(mode, 0.0)
                vals[mode].append(v)
        aggregate[metric] = {m: round(float(np.mean(v)), 3) for m, v in vals.items()}

    # Conclusion
    mean_A = aggregate["mean_score"]["SIMPLICIAL"]
    mean_B = aggregate["mean_score"]["RANDOM_MATCHED"]
    mean_C = aggregate["mean_score"]["UNCONSTRAINED"]

    if mean_A > mean_B + 0.005 and mean_A > mean_C + 0.005:
        conclusion = "A > B ≈ C (topology helps)"
    elif mean_A > mean_C + 0.005 and abs(mean_A - mean_B) <= 0.005:
        conclusion = "A ≈ B > C (any constraint helps)"
    elif abs(mean_A - mean_B) <= 0.005 and abs(mean_A - mean_C) <= 0.005:
        conclusion = "A ≈ B ≈ C (constraints irrelevant)"
    elif mean_B > mean_A + 0.005:
        conclusion = "B > A (random constraints beat topology)"
    elif mean_C > mean_A + 0.005:
        conclusion = "C > A (unconstrained best)"
    else:
        conclusion = "Mixed results — no clear winner"

    return {
        "statistical_tests": stat_tests,
        "win_tie_loss": win_tie_loss,
        "aggregate": aggregate,
        "conclusion": conclusion,
    }


def _permutation_test(
    s1: list[float],
    s2: list[float],
    n_perms: int = 10000,
) -> tuple[float, float]:
    """Permutation test for paired differences."""
    diffs = np.array(s1) - np.array(s2)
    observed = np.mean(diffs)
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = np.mean(diffs * signs)
        if abs(perm_mean) >= abs(observed):
            count += 1
    p_value = count / n_perms
    return abs(observed), p_value


# ════════════════════════════════════════════════════════════════════
# PHASE 7: OUTPUT FORMATTING (exp_gen_sol_out.json schema)
# ════════════════════════════════════════════════════════════════════


def build_output(
    datasets: dict[str, DatasetInfo],
    results: dict,
    predictions: dict[str, dict[int, dict[str, float]]],
    statistics: dict,
) -> dict:
    """Build output in exp_gen_sol_out.json schema format."""
    output_datasets = []

    for ds_name, ds in datasets.items():
        examples = []
        for i in range(ds.n_samples):
            # Input: JSON array of features
            input_vals = [round(float(v), 6) for v in ds.X[i]]
            input_str = json.dumps(input_vals)

            # Output: target value
            if ds.task_type == "classification":
                output_str = str(int(ds.y[i]))
            else:
                output_str = str(round(float(ds.y[i]), 6))

            example: dict = {
                "input": input_str,
                "output": output_str,
            }

            # Add predictions if available
            preds_for_i = predictions.get(ds_name, {}).get(i, {})
            if preds_for_i:
                example["predict_SIMPLICIAL"] = str(round(preds_for_i.get("SIMPLICIAL", 0.0), 6))
                example["predict_RANDOM_MATCHED"] = str(round(preds_for_i.get("RANDOM_MATCHED", 0.0), 6))
                example["predict_UNCONSTRAINED"] = str(round(preds_for_i.get("UNCONSTRAINED", 0.0), 6))

            # Metadata
            example["metadata_fold"] = int(ds.fold_ids[i])
            example["metadata_task_type"] = ds.task_type
            example["metadata_dataset_name"] = ds_name

            examples.append(example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    # Top-level metadata with experiment details
    metadata = {
        "experiment": "sc_ots_ablation_v1",
        "modes": ["SIMPLICIAL", "RANDOM_MATCHED", "UNCONSTRAINED"],
        "description": "Three-mode ablation testing whether topological constraints from "
                       "persistent homology improve predictive accuracy over random/unconstrained baselines",
        "results_per_dataset": {
            ds_name: {
                "task_type": ds_data.get("task_type", ""),
                "n_samples": ds_data.get("n_samples", 0),
                "n_features": ds_data.get("n_features", 0),
                "mean_scores": ds_data.get("mean_scores", {}),
                "mean_splits": ds_data.get("mean_splits", {}),
                "mean_time_s": ds_data.get("mean_time_s", {}),
            }
            for ds_name, ds_data in results.items()
        },
        **statistics,
    }

    return {
        "metadata": metadata,
        "datasets": output_datasets,
    }


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════


@logger.catch
def main(
    max_datasets: Optional[int] = None,
    max_folds: Optional[int] = None,
    start_dataset_idx: int = 0,
) -> None:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("SC-OTS Ablation Experiment")
    logger.info("=" * 60)

    # Load data
    datasets = load_datasets(DATA_PATH)
    logger.info(f"Loaded {len(datasets)} datasets")

    # Run experiment
    results, predictions = run_experiment(
        datasets=datasets,
        max_datasets=max_datasets,
        max_folds=max_folds,
        start_dataset_idx=start_dataset_idx,
    )

    # Compute statistics
    logger.info("Computing statistics...")
    statistics = compute_statistics(results)

    logger.info(f"Conclusion: {statistics['conclusion']}")
    logger.info(f"Aggregate mean scores: {statistics['aggregate'].get('mean_score', {})}")

    # Build output
    logger.info("Building output...")
    output = build_output(datasets, results, predictions, statistics)

    # Write output
    logger.info(f"Writing output to {OUTPUT_PATH}")
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Output written: {size_mb:.1f} MB")


if __name__ == "__main__":
    # Parse CLI: python method.py [max_datasets] [max_folds] [start_dataset_idx]
    max_ds = int(sys.argv[1]) if len(sys.argv) > 1 else None
    max_fl = int(sys.argv[2]) if len(sys.argv) > 2 else None
    start_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(max_datasets=max_ds, max_folds=max_fl, start_dataset_idx=start_idx)
