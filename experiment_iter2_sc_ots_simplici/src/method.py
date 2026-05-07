#!/usr/bin/env python3
"""SC-OTS: Simplicial-Constrained Oblique Tree Sums Experiment.

Implements the complete SC-OTS pipeline:
1. Compute pairwise target-aware interaction distance matrices
2. Build Vietoris-Rips simplicial complexes via GUDHI
3. Extract persistent simplices as feature interaction groups
4. Use them to constrain oblique splits in a custom FIGS-style greedy tree sum
5. Evaluate against 5 baselines (FIGS, RO-FIGS, XGBoost, XGBoost+SC, EBM)
   on all 10 benchmark datasets with 5-fold CV
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gudhi
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ── Logging setup ──────────────────────────────────────────────────────
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
LOG_FMT = (
    f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}"
    f"|{CYAN}{{function}}{END}| {{message}}"
)

logger.remove()
logger.add(sys.stdout, level="INFO", format=LOG_FMT)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG", format=LOG_FMT)

# ── Constants ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path(
    "/home/adrian/projects/temp/ai-inventor-old3/aii_pipeline/runs/"
    "run__20260228_133939/3_invention_loop/iter_1/gen_art/"
    "data_id3_it1__opus"
)
MAX_DCOR_SAMPLES = 2000
MAX_RULES = 20
N_FOLDS = 5
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════
# STEP 0: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DatasetInfo:
    """Container for a single dataset."""

    name: str
    X: np.ndarray
    y: np.ndarray
    folds: np.ndarray
    task_type: str
    feature_names: list
    known_interactions: dict
    n_features: int
    n_samples: int
    category: str


def load_datasets(data_path: Path) -> List[DatasetInfo]:
    """Load all datasets from full_data_out.json."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets: List[DatasetInfo] = []

    for ds_block in raw["datasets"]:
        name = ds_block["dataset"]
        examples = ds_block["examples"]
        ex0 = examples[0]

        X = np.array(
            [json.loads(ex["input"]) for ex in examples],
            dtype=np.float64,
        )
        task_type = ex0["metadata_task_type"]

        if task_type == "regression":
            y = np.array(
                [float(ex["output"]) for ex in examples],
                dtype=np.float64,
            )
        else:
            y = np.array(
                [int(float(ex["output"])) for ex in examples],
                dtype=np.int64,
            )

        folds = np.array([int(ex["metadata_fold"]) for ex in examples])
        feature_names = ex0.get(
            "metadata_feature_names",
            [f"f{i}" for i in range(X.shape[1])],
        )
        ki_raw = ex0.get("metadata_known_interactions", "{}")
        known_interactions = (
            json.loads(ki_raw) if isinstance(ki_raw, str) else ki_raw
        )

        info = DatasetInfo(
            name=name,
            X=X,
            y=y,
            folds=folds,
            task_type=task_type,
            feature_names=feature_names,
            known_interactions=known_interactions,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            category=ex0.get("metadata_category", ""),
        )
        datasets.append(info)
        logger.info(
            f"  {name}: {info.n_samples} samples, "
            f"{info.n_features} features, task={task_type}"
        )

    logger.info(f"Loaded {len(datasets)} datasets total")
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: INTERACTION DISTANCE MATRIX (Target-Aware)
# ═══════════════════════════════════════════════════════════════════════


def compute_interaction_distance_matrix(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = MAX_DCOR_SAMPLES,
) -> np.ndarray:
    """Compute target-aware interaction distance using decision trees.

    For each pair (i,j):
      1. Fit a tree on x_i alone -> R²_i
      2. Fit a tree on x_j alone -> R²_j
      3. Fit a tree on (x_i, x_j) together -> R²_ij
      4. Interaction strength = max(0, R²_ij - max(R²_i, R²_j))
      5. Distance = 1 - interaction_strength

    This captures NONLINEAR interactions (unlike Pearson correlation)
    and is TARGET-AWARE (unlike feature-to-feature dCor).

    For friedman1: x0, x1 together predict y much better than either
    alone because the tree can approximate sin(pi*x0*x1).
    """
    n, p = X.shape
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # For very high-dimensional data, pre-filter features
    feature_subset = list(range(p))
    if p > 40:
        # Keep top-30 features by univariate tree R²
        univariate_r2 = []
        for j in range(p):
            try:
                tree = DecisionTreeRegressor(
                    max_depth=4, random_state=RANDOM_STATE
                )
                tree.fit(X[:, j:j + 1], y)
                r2 = max(0, tree.score(X[:, j:j + 1], y))
                univariate_r2.append((r2, j))
            except Exception:
                univariate_r2.append((0.0, j))
        univariate_r2.sort(reverse=True)
        feature_subset = [j for _, j in univariate_r2[:30]]

    # Compute univariate R² for each feature
    r2_single = {}
    for j in feature_subset:
        try:
            tree = DecisionTreeRegressor(
                max_depth=4, random_state=RANDOM_STATE
            )
            tree.fit(X[:, j:j + 1], y)
            r2_single[j] = max(0.0, tree.score(X[:, j:j + 1], y))
        except Exception:
            r2_single[j] = 0.0

    # Compute pairwise interaction strength
    D = np.ones((p, p))
    np.fill_diagonal(D, 0.0)

    for i_idx in range(len(feature_subset)):
        for j_idx in range(i_idx + 1, len(feature_subset)):
            i = feature_subset[i_idx]
            j = feature_subset[j_idx]
            try:
                tree_pair = DecisionTreeRegressor(
                    max_depth=4, random_state=RANDOM_STATE
                )
                tree_pair.fit(X[:, [i, j]], y)
                r2_pair = max(0.0, tree_pair.score(X[:, [i, j]], y))

                # Interaction = how much the pair improves over best single
                interaction_strength = max(
                    0.0,
                    r2_pair - max(r2_single.get(i, 0), r2_single.get(j, 0)),
                )
                D[i, j] = D[j, i] = 1.0 - interaction_strength
            except Exception:
                D[i, j] = D[j, i] = 1.0

    return D


def compute_dcor_distance_matrix(
    X: np.ndarray,
    max_samples: int = MAX_DCOR_SAMPLES,
) -> np.ndarray:
    """Compute D[i,j] = 1 - dcor(X[:,i], X[:,j]).

    Used as a supplementary metric for feature correlation structure.
    For datasets with >30 features, falls back to Pearson.
    """
    import dcor

    n, p = X.shape
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        X = X[idx]

    if p > 30:
        corr = np.abs(np.corrcoef(X.T))
        corr = np.nan_to_num(corr, nan=0.0)
        D = np.sqrt(np.clip(1.0 - corr, 0.0, 1.0))
        np.fill_diagonal(D, 0.0)
        return D

    D = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            try:
                c = dcor.distance_correlation(X[:, i], X[:, j])
                D[i, j] = D[j, i] = 1.0 - c
            except Exception:
                D[i, j] = D[j, i] = 1.0
    return D


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: VIETORIS-RIPS FILTRATION & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════


def build_simplicial_complex(
    D: np.ndarray,
    max_dim: int = 3,
) -> Tuple[Any, Dict[int, list], float, list, list]:
    """Build Rips complex, compute persistence, select threshold."""
    rips = gudhi.RipsComplex(
        distance_matrix=D.tolist(), max_edge_length=1.0
    )
    st = rips.create_simplex_tree(max_dimension=max_dim)
    persistence = st.persistence()

    # Largest-gap threshold selection on dimension 0
    lifetimes = sorted(
        [
            death - birth
            for dim, (birth, death) in persistence
            if dim == 0 and death != float("inf") and death > birth
        ]
    )

    if len(lifetimes) >= 2:
        gaps = [
            lifetimes[i + 1] - lifetimes[i]
            for i in range(len(lifetimes) - 1)
        ]
        gap_idx = int(np.argmax(gaps))
        threshold = (lifetimes[gap_idx] + lifetimes[gap_idx + 1]) / 2
    else:
        off_diag = D[D > 0]
        threshold = (
            float(np.median(off_diag)) if len(off_diag) > 0 else 0.5
        )

    # Ensure threshold is in a useful range
    threshold = max(0.05, min(threshold, 0.95))

    # Check if we have enough edges; if not, raise threshold
    st_test = gudhi.RipsComplex(
        distance_matrix=D.tolist(), max_edge_length=1.0
    ).create_simplex_tree(max_dimension=max_dim)
    st_test.prune_above_filtration(threshold)
    n_edges = sum(
        1
        for s, _ in st_test.get_filtration()
        if len(s) == 2
    )
    p = D.shape[0]
    min_edges = max(3, p // 3)
    if n_edges < min_edges:
        # Raise threshold to include more edges
        off_diag = []
        for i in range(p):
            for j in range(i + 1, p):
                if D[i, j] < 1.0:
                    off_diag.append(D[i, j])
        if off_diag:
            off_diag.sort()
            # Take threshold to include at least min_edges pairs
            idx = min(min_edges - 1, len(off_diag) - 1)
            threshold = off_diag[idx] + 0.001
            threshold = min(threshold, 0.99)

    st.prune_above_filtration(threshold)

    simplices_by_dim: Dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    for simplex, filt in st.get_filtration():
        dim = len(simplex) - 1
        if dim in simplices_by_dim:
            simplices_by_dim[dim].append(list(simplex))

    try:
        betti = list(st.persistent_betti_numbers(0, threshold))
    except Exception:
        betti = [0, 0, 0]

    return st, simplices_by_dim, threshold, betti, persistence


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: SC-OTS — SIMPLICIAL-CONSTRAINED OBLIQUE TREE SUM
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SCOTSNode:
    """Node in the SC-OTS tree."""

    idxs: np.ndarray
    value: float
    impurity: float
    tree_num: int
    is_root: bool = False
    depth: int = 0
    simplex: Optional[list] = None
    weights: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    impurity_reduction: Optional[float] = None
    left: Optional["SCOTSNode"] = None
    right: Optional["SCOTSNode"] = None
    left_temp: Optional["SCOTSNode"] = None
    right_temp: Optional["SCOTSNode"] = None


def construct_oblique_split(
    X: np.ndarray,
    y: np.ndarray,
    idxs: np.ndarray,
    simplex_features: list,
) -> Optional[Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]]:
    """Create best oblique split using features from one simplex.

    Uses Ridge regression to find weights, then finds optimal
    threshold on the linear combination via quantile search.
    """
    n_at_node = int(idxs.sum())
    if n_at_node < 10:
        return None

    X_sub = X[idxs][:, simplex_features]
    y_sub = y[idxs]

    if np.std(y_sub) < 1e-10:
        return None

    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_sub, y_sub)
        weights = ridge.coef_.copy()
    except Exception:
        return None

    if np.all(np.abs(weights) < 1e-10):
        return None

    # Compute linear combination for ALL samples
    linear_combo = X[:, simplex_features] @ weights
    combo_at_node = linear_combo[idxs]

    sorted_vals = np.sort(combo_at_node)
    n_vals = len(sorted_vals)
    if n_vals < 2:
        return None

    # ~50 candidate thresholds at quantile positions
    n_candidates = min(50, n_vals - 1)
    indices = np.linspace(0, n_vals - 2, n_candidates, dtype=int)
    midpoints = (sorted_vals[indices] + sorted_vals[indices + 1]) / 2.0

    best_impurity_red = -np.inf
    best_threshold = None
    parent_impurity = np.var(y_sub) * n_at_node
    min_leaf = max(5, n_at_node // 20)

    for t in midpoints:
        left_mask = combo_at_node <= t
        n_left = int(left_mask.sum())
        n_right = n_at_node - n_left
        if n_left < min_leaf or n_right < min_leaf:
            continue

        right_mask = ~left_mask
        imp_left = np.var(y_sub[left_mask]) * n_left
        imp_right = np.var(y_sub[right_mask]) * n_right
        imp_red = parent_impurity - imp_left - imp_right

        if imp_red > best_impurity_red:
            best_impurity_red = imp_red
            best_threshold = t

    if best_threshold is None or best_impurity_red <= 0:
        return None

    idxs_left = (linear_combo <= best_threshold) & idxs
    idxs_right = (linear_combo > best_threshold) & idxs

    if idxs_left.sum() < 1 or idxs_right.sum() < 1:
        return None

    return weights, best_threshold, best_impurity_red, idxs_left, idxs_right


class SCOTS:
    """Simplicial-Constrained Oblique Tree Sum (SC-OTS)."""

    def __init__(
        self,
        simplices: list,
        max_rules: int = MAX_RULES,
        max_trees: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        task_type: str = "regression",
    ):
        self.simplices = simplices
        self.max_rules = max_rules
        self.max_trees = max_trees
        self.min_impurity_decrease = min_impurity_decrease
        self.task_type = task_type
        self.trees_: List[SCOTSNode] = []
        self.complexity_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SCOTS":
        """Fit the SC-OTS model using greedy tree sum (FIGS-style)."""
        n, p = X.shape
        # Always work with float64 for residual computation
        y = y.astype(np.float64)
        self.trees_ = []
        self.complexity_ = 0

        y_predictions_per_tree: Dict[int, np.ndarray] = {}
        y_residuals_per_tree: Dict[int, np.ndarray] = {}

        idxs = np.ones(n, dtype=bool)
        node_init = self._best_split_over_simplices(X, y, idxs, -1)
        node_init.is_root = True
        potential_splits = [node_init]

        max_iters = self.max_rules * 3
        iter_count = 0

        while potential_splits and self.complexity_ < self.max_rules:
            iter_count += 1
            if iter_count > max_iters:
                break

            potential_splits.sort(
                key=lambda nd: (
                    nd.impurity_reduction
                    if nd.impurity_reduction is not None
                    else -np.inf
                )
            )
            split_node = potential_splits.pop()

            if (
                split_node.impurity_reduction is None
                or split_node.impurity_reduction <= self.min_impurity_decrease
            ):
                break

            self.complexity_ += 1

            if split_node.is_root:
                self.trees_.append(split_node)
                for node_ in [
                    split_node,
                    split_node.left_temp,
                    split_node.right_temp,
                ]:
                    if node_ is not None:
                        node_.tree_num = len(self.trees_) - 1

                new_root = SCOTSNode(
                    idxs=np.ones(n, dtype=bool),
                    value=float(np.mean(y)),
                    impurity=float(np.var(y) * n),
                    tree_num=-1,
                    is_root=True,
                )
                potential_splits.append(new_root)

            split_node.left = split_node.left_temp
            split_node.right = split_node.right_temp
            split_node.left_temp = None
            split_node.right_temp = None

            if split_node.left is not None:
                potential_splits.append(split_node.left)
            if split_node.right is not None:
                potential_splits.append(split_node.right)

            # Update residuals
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self._predict_tree(
                    self.trees_[tree_num_], X
                )
            y_predictions_per_tree[-1] = np.zeros(n)

            for tree_num_ in list(range(len(self.trees_))) + [-1]:
                y_residuals_per_tree[tree_num_] = y.copy()
                for other in range(len(self.trees_)):
                    if other != tree_num_:
                        y_residuals_per_tree[tree_num_] -= (
                            y_predictions_per_tree[other]
                        )

            new_potentials = []
            for ps in potential_splits:
                if ps.idxs.sum() < 10:
                    continue
                y_target = y_residuals_per_tree.get(ps.tree_num, y)
                updated = self._best_split_over_simplices(
                    X, y_target, ps.idxs, ps.tree_num
                )
                updated.is_root = ps.is_root
                updated.tree_num = ps.tree_num
                updated.depth = ps.depth
                new_potentials.append(updated)
            potential_splits = new_potentials

            if (
                self.max_trees is not None
                and len(self.trees_) >= self.max_trees
            ):
                potential_splits = [
                    p for p in potential_splits if not p.is_root
                ]

        return self

    def _best_split_over_simplices(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idxs: np.ndarray,
        tree_num: int,
    ) -> SCOTSNode:
        """Try oblique splits for each simplex, return best Node."""
        n_at_node = int(idxs.sum())
        if n_at_node == 0:
            return SCOTSNode(
                idxs=idxs,
                value=0.0,
                impurity=0.0,
                tree_num=tree_num,
                impurity_reduction=None,
            )

        y_at_node = y[idxs]
        best_node = SCOTSNode(
            idxs=idxs,
            value=float(np.mean(y_at_node)),
            impurity=float(np.var(y_at_node) * n_at_node),
            tree_num=tree_num,
            impurity_reduction=None,
        )

        # Oblique splits from simplices (dim >= 1)
        candidates = [s for s in self.simplices if len(s) >= 2]
        if len(candidates) > 30:
            candidates = random.sample(candidates, 30)

        for simplex in candidates:
            if any(f >= X.shape[1] for f in simplex):
                continue
            result = construct_oblique_split(X, y, idxs, simplex)
            if result is None:
                continue
            weights, threshold, imp_red, idxs_left, idxs_right = result
            if imp_red > (best_node.impurity_reduction or 0):
                y_l = y[idxs_left]
                y_r = y[idxs_right]
                best_node = SCOTSNode(
                    idxs=idxs,
                    value=float(np.mean(y_at_node)),
                    impurity=float(np.var(y_at_node) * n_at_node),
                    tree_num=tree_num,
                    simplex=list(simplex),
                    weights=weights,
                    threshold=threshold,
                    impurity_reduction=imp_red,
                )
                best_node.left_temp = SCOTSNode(
                    idxs=idxs_left,
                    value=float(np.mean(y_l)),
                    impurity=float(np.var(y_l) * idxs_left.sum()),
                    tree_num=tree_num,
                )
                best_node.right_temp = SCOTSNode(
                    idxs=idxs_right,
                    value=float(np.mean(y_r)),
                    impurity=float(np.var(y_r) * idxs_right.sum()),
                    tree_num=tree_num,
                )

        # Axis-aligned splits (0-simplices) via sklearn stump
        if n_at_node >= 10:
            try:
                stump = DecisionTreeRegressor(
                    max_depth=1, random_state=RANDOM_STATE
                )
                stump.fit(X[idxs], y_at_node)
                tree = stump.tree_
                if (
                    tree.n_node_samples[0] > 0
                    and len(tree.feature) > 1
                    and tree.feature[0] >= 0
                ):
                    feat = tree.feature[0]
                    thresh = tree.threshold[0]
                    imp = tree.impurity
                    n_ns = tree.n_node_samples
                    if n_ns[0] > 0:
                        imp_red_s = (
                            imp[0]
                            - imp[1] * n_ns[1] / n_ns[0]
                            - imp[2] * n_ns[2] / n_ns[0]
                        ) * n_ns[0]
                        if imp_red_s > (
                            best_node.impurity_reduction or 0
                        ):
                            idxs_l = (X[:, feat] <= thresh) & idxs
                            idxs_r = (X[:, feat] > thresh) & idxs
                            if idxs_l.sum() > 0 and idxs_r.sum() > 0:
                                y_l = y[idxs_l]
                                y_r = y[idxs_r]
                                best_node = SCOTSNode(
                                    idxs=idxs,
                                    value=float(np.mean(y_at_node)),
                                    impurity=float(
                                        np.var(y_at_node) * n_at_node
                                    ),
                                    tree_num=tree_num,
                                    simplex=[int(feat)],
                                    weights=np.array([1.0]),
                                    threshold=float(thresh),
                                    impurity_reduction=float(imp_red_s),
                                )
                                best_node.left_temp = SCOTSNode(
                                    idxs=idxs_l,
                                    value=float(np.mean(y_l)),
                                    impurity=float(
                                        np.var(y_l) * idxs_l.sum()
                                    ),
                                    tree_num=tree_num,
                                )
                                best_node.right_temp = SCOTSNode(
                                    idxs=idxs_r,
                                    value=float(np.mean(y_r)),
                                    impurity=float(
                                        np.var(y_r) * idxs_r.sum()
                                    ),
                                    tree_num=tree_num,
                                )
            except Exception:
                pass

        return best_node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        return preds

    def _predict_tree(
        self, node: SCOTSNode, X: np.ndarray
    ) -> np.ndarray:
        preds = np.full(X.shape[0], np.nan)
        mask = np.ones(X.shape[0], dtype=bool)
        self._predict_recursive(node, X, preds, mask)
        return np.nan_to_num(preds, nan=0.0)

    def _predict_recursive(
        self,
        node: Optional[SCOTSNode],
        X: np.ndarray,
        preds: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if node is None or mask.sum() == 0:
            return
        if node.left is None and node.right is None:
            preds[mask] = node.value
            return
        if (
            node.simplex is None
            or node.weights is None
            or node.threshold is None
        ):
            preds[mask] = node.value
            return

        combo = X[:, node.simplex] @ node.weights
        left_mask = (combo <= node.threshold) & mask
        right_mask = (combo > node.threshold) & mask
        self._predict_recursive(node.left, X, preds, left_mask)
        self._predict_recursive(node.right, X, preds, right_mask)

    def get_used_simplices(self) -> List[list]:
        """Collect all simplices used in splits."""
        used: List[list] = []
        for tree in self.trees_:
            self._collect_simplices(tree, used)
        return used

    def _collect_simplices(
        self, node: Optional[SCOTSNode], acc: list
    ) -> None:
        if node is None:
            return
        if node.simplex is not None:
            acc.append(node.simplex)
        self._collect_simplices(node.left, acc)
        self._collect_simplices(node.right, acc)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: BASELINES
# ═══════════════════════════════════════════════════════════════════════


def generate_random_subsets(
    n_features: int,
    n_subsets: int,
    max_size: int = 4,
) -> list:
    """Generate random feature subsets for RO-FIGS ablation."""
    subsets = []
    for _ in range(n_subsets):
        size = random.randint(2, min(max_size, n_features))
        subset = sorted(random.sample(range(n_features), size))
        subsets.append(subset)
    for i in range(n_features):
        subsets.append([i])
    return subsets


def train_figs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    max_rules: int = MAX_RULES,
):
    """Train FIGS baseline."""
    from imodels import FIGSClassifier, FIGSRegressor

    if task_type == "regression":
        model = FIGSRegressor(max_rules=max_rules)
    else:
        model = FIGSClassifier(max_rules=max_rules)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
):
    """Train XGBoost baseline."""
    from xgboost import XGBClassifier, XGBRegressor

    if task_type == "regression":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE,
            verbosity=0,
            eval_metric="logloss",
        )
    model.fit(X_train, y_train)
    return model


def train_xgboost_sc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    valid_simplices: list,
):
    """Train XGBoost with simplicial interaction constraints."""
    from xgboost import XGBClassifier, XGBRegressor

    n_features = X_train.shape[1]
    col_names = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X_train, columns=col_names)

    constraints: List[List[str]] = []
    for s in valid_simplices:
        if len(s) >= 2:
            c = [col_names[f] for f in s if f < n_features]
            if len(c) >= 2:
                constraints.append(c)

    all_feats: set = set()
    for c in constraints:
        all_feats.update(c)
    for f_name in col_names:
        if f_name not in all_feats:
            constraints.append([f_name])

    if task_type == "regression":
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            interaction_constraints=constraints,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            interaction_constraints=constraints,
            random_state=RANDOM_STATE,
            verbosity=0,
            eval_metric="logloss",
        )
    model.fit(X_df, y_train)
    return model


def train_ebm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
):
    """Train EBM baseline."""
    from interpret.glassbox import (
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
    )

    if task_type == "regression":
        model = ExplainableBoostingRegressor(
            random_state=RANDOM_STATE,
            max_rounds=500,
            interactions=5,
        )
    else:
        model = ExplainableBoostingClassifier(
            random_state=RANDOM_STATE,
            max_rounds=500,
            interactions=5,
        )
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: METRICS
# ═══════════════════════════════════════════════════════════════════════


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> float:
    """R² for regression, accuracy for classification."""
    if task_type == "regression":
        return float(r2_score(y_true, y_pred))
    else:
        y_pred_labels = np.round(y_pred).astype(int)
        y_pred_labels = np.clip(y_pred_labels, 0, 1)
        return float(accuracy_score(y_true, y_pred_labels))


def safe_predict_proba(model, X, task_type: str):
    """Safely get probability predictions."""
    if task_type != "classification":
        return None
    try:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: INTERACTION RECOVERY & FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════


def compute_interaction_recovery(
    valid_simplices: list,
    known_interactions: dict,
) -> dict:
    """Compare simplices against ground truth interactions."""
    if not known_interactions:
        return {}

    gt_sets: set = set()
    for key in ["2-way", "3-way", "4-way"]:
        if key in known_interactions:
            for interaction in known_interactions[key]:
                gt_sets.add(tuple(sorted(interaction)))

    if not gt_sets:
        return {}

    # Expand to pairwise level
    gt_pairwise: set = set()
    for gt in gt_sets:
        if len(gt) == 2:
            gt_pairwise.add(gt)
        elif len(gt) > 2:
            for pair in combinations(gt, 2):
                gt_pairwise.add(tuple(sorted(pair)))

    pred_pairwise: set = set()
    for s in valid_simplices:
        if len(s) == 2:
            pred_pairwise.add(tuple(sorted(s)))
        elif len(s) > 2:
            for pair in combinations(sorted(s), 2):
                pred_pairwise.add(pair)

    if not gt_pairwise:
        return {}

    confirmed = pred_pairwise & gt_pairwise
    precision = (
        len(confirmed) / len(pred_pairwise) if pred_pairwise else 0.0
    )
    recall = (
        len(confirmed) / len(gt_pairwise) if gt_pairwise else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_interaction_faithfulness(
    valid_simplices: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str = "regression",
    shap_threshold: float = 0.01,
) -> dict:
    """Compute SHAP faithfulness of simplicial complex edges."""
    try:
        import shap
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        # Use sklearn GBM instead of XGBoost for SHAP compatibility
        if task_type == "regression":
            bb = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        else:
            bb = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        bb.fit(X_train, y_train)

        n_shap = min(50, X_test.shape[0])
        X_shap = X_test[:n_shap].copy()

        explainer = shap.TreeExplainer(bb)
        shap_interaction = explainer.shap_interaction_values(X_shap)

        if isinstance(shap_interaction, list):
            shap_interaction = shap_interaction[0]

        mean_interaction = np.abs(shap_interaction).mean(axis=0)

        significant_shap_pairs: set = set()
        for i in range(mean_interaction.shape[0]):
            for j in range(i + 1, mean_interaction.shape[1]):
                if mean_interaction[i, j] > shap_threshold:
                    significant_shap_pairs.add((i, j))

        complex_edges: set = set()
        for simplex in valid_simplices:
            if len(simplex) == 2:
                complex_edges.add(tuple(sorted(simplex)))

        if len(complex_edges) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        confirmed = complex_edges & significant_shap_pairs
        precision = (
            len(confirmed) / len(complex_edges) if complex_edges else 0.0
        )
        recall = (
            len(confirmed) / len(significant_shap_pairs)
            if significant_shap_pairs
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    except Exception as e:
        logger.warning(f"SHAP faithfulness failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ═══════════════════════════════════════════════════════════════════════
# STEP 7: EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════


def evaluate_single_fold(
    dataset: DatasetInfo,
    fold_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Run all models on a single fold."""
    task_type = dataset.task_type
    n_features = dataset.n_features
    fold_result: Dict[str, Any] = {"fold": fold_id}
    fold_preds: Dict[str, np.ndarray] = {}

    # ── A: Build simplicial complex ──
    logger.info(f"    Building complex for {dataset.name} fold {fold_id}")
    t0 = time.time()
    D = compute_interaction_distance_matrix(X_train, y_train)
    dcor_time = time.time() - t0

    st, simplices_by_dim, threshold, betti, persistence = (
        build_simplicial_complex(D, max_dim=min(3, n_features - 1))
    )

    valid_simplices: List[list] = []
    for dim in [1, 2, 3]:
        valid_simplices.extend(simplices_by_dim.get(dim, []))
    for i in range(n_features):
        valid_simplices.append([i])

    fold_result["n_simplices_dim0"] = len(simplices_by_dim.get(0, []))
    fold_result["n_simplices_dim1"] = len(simplices_by_dim.get(1, []))
    fold_result["n_simplices_dim2"] = len(simplices_by_dim.get(2, []))
    fold_result["n_simplices_dim3"] = len(simplices_by_dim.get(3, []))
    fold_result["betti"] = betti
    fold_result["threshold"] = threshold
    fold_result["dcor_time"] = dcor_time

    n_oblique = len([s for s in valid_simplices if len(s) >= 2])
    logger.info(
        f"    Simplices: dim1={fold_result['n_simplices_dim1']}, "
        f"dim2={fold_result['n_simplices_dim2']}, "
        f"dim3={fold_result['n_simplices_dim3']}, "
        f"thr={threshold:.3f}, oblique={n_oblique}"
    )

    # ── B1: SC-OTS ──
    t0 = time.time()
    scots = SCOTS(
        simplices=valid_simplices,
        max_rules=MAX_RULES,
        task_type=task_type,
    )
    scots.fit(X_train, y_train)
    scots_time = time.time() - t0
    scots_pred = scots.predict(X_test)
    fold_preds["scots"] = scots_pred

    fold_result["scots_metric"] = compute_metric(
        y_test, scots_pred, task_type
    )
    fold_result["scots_splits"] = scots.complexity_
    fold_result["scots_trees"] = len(scots.trees_)
    fold_result["scots_time"] = scots_time
    logger.info(
        f"    SC-OTS: {fold_result['scots_metric']:.4f}, "
        f"splits={scots.complexity_}, time={scots_time:.1f}s"
    )

    # ── B2: RO-FIGS (ablation) ──
    t0 = time.time()
    rand_simplices = generate_random_subsets(
        n_features=n_features,
        n_subsets=15,
        max_size=3,
    )
    rofigs = SCOTS(
        simplices=rand_simplices,
        max_rules=MAX_RULES,
        task_type=task_type,
    )
    rofigs.fit(X_train, y_train)
    rofigs_time = time.time() - t0
    rofigs_pred = rofigs.predict(X_test)
    fold_preds["rofigs"] = rofigs_pred
    fold_result["rofigs_metric"] = compute_metric(
        y_test, rofigs_pred, task_type
    )
    fold_result["rofigs_splits"] = rofigs.complexity_
    fold_result["rofigs_time"] = rofigs_time
    logger.info(f"    RO-FIGS: {fold_result['rofigs_metric']:.4f}")

    # ── B3: FIGS ──
    t0 = time.time()
    figs_model = None
    try:
        figs_model = train_figs(X_train, y_train, task_type)
        figs_pred = figs_model.predict(X_test)
        fold_preds["figs"] = figs_pred
        fold_result["figs_metric"] = compute_metric(
            y_test, figs_pred, task_type
        )
        fold_result["figs_time"] = time.time() - t0
    except Exception as e:
        logger.warning(f"    FIGS failed: {e}")
        fold_result["figs_metric"] = float("nan")
        fold_result["figs_time"] = 0.0
    logger.info(f"    FIGS: {fold_result['figs_metric']:.4f}")

    # ── B4: XGBoost ──
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train, task_type)
    xgb_pred = xgb_model.predict(X_test)
    fold_preds["xgb"] = xgb_pred
    fold_result["xgb_metric"] = compute_metric(
        y_test, xgb_pred, task_type
    )
    fold_result["xgb_time"] = time.time() - t0
    logger.info(f"    XGBoost: {fold_result['xgb_metric']:.4f}")

    # ── B5: XGBoost + SC ──
    t0 = time.time()
    xgb_sc_model = None
    try:
        xgb_sc_model = train_xgboost_sc(
            X_train, y_train, task_type, valid_simplices
        )
        xgb_sc_pred = xgb_sc_model.predict(X_test)
        fold_preds["xgb_sc"] = xgb_sc_pred
        fold_result["xgb_sc_metric"] = compute_metric(
            y_test, xgb_sc_pred, task_type
        )
        fold_result["xgb_sc_time"] = time.time() - t0
    except Exception as e:
        logger.warning(f"    XGBoost+SC failed: {e}")
        fold_result["xgb_sc_metric"] = float("nan")
        fold_result["xgb_sc_time"] = 0.0
    logger.info(
        f"    XGBoost+SC: {fold_result.get('xgb_sc_metric', 'nan')}"
    )

    # ── B6: EBM ──
    t0 = time.time()
    ebm_model = None
    try:
        ebm_model = train_ebm(X_train, y_train, task_type)
        ebm_pred = ebm_model.predict(X_test)
        fold_preds["ebm"] = ebm_pred
        fold_result["ebm_metric"] = compute_metric(
            y_test, ebm_pred, task_type
        )
        fold_result["ebm_time"] = time.time() - t0
    except Exception as e:
        logger.warning(f"    EBM failed: {e}")
        fold_result["ebm_metric"] = float("nan")
        fold_result["ebm_time"] = 0.0
    logger.info(f"    EBM: {fold_result.get('ebm_metric', 'nan')}")

    # ── AUROC for classification ──
    if task_type == "classification":
        for mn, mobj in [
            ("xgb", xgb_model),
            ("figs", figs_model),
            ("ebm", ebm_model),
            ("xgb_sc", xgb_sc_model),
        ]:
            if mobj is not None:
                proba = safe_predict_proba(mobj, X_test, task_type)
                if proba is not None:
                    try:
                        fold_result[f"{mn}_auroc"] = float(
                            roc_auc_score(y_test, proba)
                        )
                    except Exception:
                        pass

    # ── Interaction recovery (synthetic only) ──
    if dataset.known_interactions:
        recovery = compute_interaction_recovery(
            valid_simplices, dataset.known_interactions
        )
        fold_result["interaction_recovery"] = recovery
        if recovery:
            logger.info(
                f"    Interaction recovery: "
                f"P={recovery.get('precision', 0):.3f}, "
                f"R={recovery.get('recall', 0):.3f}"
            )

    # ── Interaction faithfulness (SHAP) — only synthetic/small ──
    if X_test.shape[0] >= 20 and n_features <= 15 and dataset.category == "A_synthetic":
        faithfulness = compute_interaction_faithfulness(
            valid_simplices=valid_simplices,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            task_type=task_type,
        )
        fold_result["interaction_faithfulness"] = faithfulness
        logger.info(
            f"    Faithfulness: P={faithfulness['precision']:.3f}, "
            f"R={faithfulness['recall']:.3f}"
        )

    gc.collect()
    return fold_result, fold_preds


def run_dataset(
    dataset: DatasetInfo,
) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Run full 5-fold CV evaluation on a single dataset."""
    logger.info(
        f"  === {dataset.name} ({dataset.n_samples}×"
        f"{dataset.n_features}, {dataset.task_type}) ==="
    )
    fold_results: List[dict] = []

    model_names = ["scots", "rofigs", "figs", "xgb", "xgb_sc", "ebm"]
    ds_preds: Dict[str, np.ndarray] = {
        mn: np.full(dataset.n_samples, np.nan) for mn in model_names
    }

    for fold_id in range(N_FOLDS):
        logger.info(f"  -- Fold {fold_id} for {dataset.name} --")
        train_mask = dataset.folds != fold_id
        test_mask = dataset.folds == fold_id
        test_indices = np.where(test_mask)[0]

        X_train = dataset.X[train_mask]
        y_train = dataset.y[train_mask]
        X_test = dataset.X[test_mask]
        y_test = dataset.y[test_mask]

        try:
            fold_result, fold_preds = evaluate_single_fold(
                dataset=dataset,
                fold_id=fold_id,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_indices=test_indices,
            )
            fold_results.append(fold_result)

            for mn, preds in fold_preds.items():
                if mn in ds_preds:
                    ds_preds[mn][test_indices] = preds

        except Exception:
            logger.exception(f"  Fold {fold_id} failed for {dataset.name}")
            continue

    if not fold_results:
        logger.error(f"  No successful folds for {dataset.name}")
        return {}, ds_preds

    # Aggregate across folds
    metric_name = (
        "R²" if dataset.task_type == "regression" else "Accuracy"
    )

    aggregated: Dict[str, Any] = {
        "dataset": dataset.name,
        "task_type": dataset.task_type,
        "n_features": dataset.n_features,
        "n_samples": dataset.n_samples,
        "category": dataset.category,
        "metric_name": metric_name,
        "n_folds_completed": len(fold_results),
    }

    for mn in model_names:
        vals = [
            fr[f"{mn}_metric"]
            for fr in fold_results
            if not np.isnan(fr.get(f"{mn}_metric", float("nan")))
        ]
        if vals:
            aggregated[f"mean_{mn}"] = float(np.mean(vals))
            aggregated[f"std_{mn}"] = float(np.std(vals))
        else:
            aggregated[f"mean_{mn}"] = None
            aggregated[f"std_{mn}"] = None

    aggregated["scots_avg_splits"] = float(
        np.mean([fr["scots_splits"] for fr in fold_results])
    )
    aggregated["scots_avg_trees"] = float(
        np.mean([fr["scots_trees"] for fr in fold_results])
    )
    aggregated["scots_avg_time"] = float(
        np.mean([fr["scots_time"] for fr in fold_results])
    )

    for dim in range(4):
        key = f"n_simplices_dim{dim}"
        aggregated[f"avg_{key}"] = float(
            np.mean([fr[key] for fr in fold_results])
        )
    aggregated["avg_threshold"] = float(
        np.mean([fr["threshold"] for fr in fold_results])
    )

    betti_lists = [
        fr["betti"] for fr in fold_results if fr["betti"]
    ]
    if betti_lists:
        max_len = max(len(b) for b in betti_lists)
        padded = [b + [0] * (max_len - len(b)) for b in betti_lists]
        aggregated["avg_betti"] = [
            float(np.mean([p[i] for p in padded]))
            for i in range(max_len)
        ]
    else:
        aggregated["avg_betti"] = [0, 0, 0]

    recoveries = [
        fr["interaction_recovery"]
        for fr in fold_results
        if "interaction_recovery" in fr and fr["interaction_recovery"]
    ]
    if recoveries:
        aggregated["interaction_recovery"] = {
            "precision": float(
                np.mean([r["precision"] for r in recoveries])
            ),
            "recall": float(
                np.mean([r["recall"] for r in recoveries])
            ),
            "f1": float(np.mean([r["f1"] for r in recoveries])),
        }

    faith_list = [
        fr["interaction_faithfulness"]
        for fr in fold_results
        if "interaction_faithfulness" in fr
    ]
    if faith_list:
        aggregated["interaction_faithfulness"] = {
            "precision": float(
                np.mean([f["precision"] for f in faith_list])
            ),
            "recall": float(
                np.mean([f["recall"] for f in faith_list])
            ),
            "f1": float(np.mean([f["f1"] for f in faith_list])),
        }

    logger.info(f"  === {dataset.name} Summary ===")
    for mn in model_names:
        val = aggregated.get(f"mean_{mn}")
        std = aggregated.get(f"std_{mn}")
        if val is not None:
            logger.info(f"    {mn}: {val:.4f} ± {std:.4f}")
        else:
            logger.info(f"    {mn}: FAILED")

    return aggregated, ds_preds


# ═══════════════════════════════════════════════════════════════════════
# STEP 8: OUTPUT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════


def build_summary_table(all_results: list) -> str:
    """Build a markdown summary table."""
    lines = [
        "| Dataset | Task | SC-OTS | RO-FIGS | FIGS"
        " | XGB | XGB+SC | EBM |",
        "|---------|------|--------|---------|------"
        "|-----|--------|-----|",
    ]

    for r in all_results:
        if not r:
            continue
        name = r["dataset"][:12]
        task = r["task_type"][:4]
        vals = []
        for mn in ["scots", "rofigs", "figs", "xgb", "xgb_sc", "ebm"]:
            v = r.get(f"mean_{mn}")
            if v is not None:
                vals.append(f"{v:.3f}")
            else:
                vals.append("N/A")
        lines.append(
            f"| {name} | {task} | {' | '.join(vals)} |"
        )

    return "\n".join(lines)


def build_output_json(
    all_results: list,
    datasets: List[DatasetInfo],
    all_predictions: dict,
    summary_table: str,
) -> dict:
    """Build method_out.json in exp_gen_sol_out.json schema format."""
    output: Dict[str, Any] = {
        "metadata": {
            "method_name": "SC-OTS",
            "description": (
                "Simplicial-Constrained Oblique Tree Sums: "
                "uses persistent homology on target-aware feature "
                "interaction distances to discover multi-way "
                "feature interactions, constraining oblique tree "
                "splits to these interactions."
            ),
            "summary_table": summary_table,
            "per_dataset_results": {},
        },
        "datasets": [],
    }

    for r in all_results:
        if not r:
            continue
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                clean_r[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                clean_r[k] = float(v)
            else:
                clean_r[k] = v
        output["metadata"]["per_dataset_results"][r["dataset"]] = (
            clean_r
        )

    for ds in datasets:
        ds_block: Dict[str, Any] = {
            "dataset": ds.name,
            "examples": [],
        }

        ds_preds = all_predictions.get(ds.name, {})

        for idx in range(ds.n_samples):
            example: Dict[str, Any] = {
                "input": json.dumps(ds.X[idx].tolist()),
                "output": str(ds.y[idx]),
                "metadata_fold": int(ds.folds[idx]),
                "metadata_task_type": ds.task_type,
            }

            for mn in [
                "scots",
                "rofigs",
                "figs",
                "xgb",
                "xgb_sc",
                "ebm",
            ]:
                preds = ds_preds.get(mn)
                if preds is not None and idx < len(preds):
                    val = preds[idx]
                    if not np.isnan(val):
                        example[f"predict_{mn}"] = str(float(val))

            ds_block["examples"].append(example)

        output["datasets"].append(ds_block)

    return output


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


@logger.catch
def main():
    """Main entry point for SC-OTS experiment."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SC-OTS EXPERIMENT START")
    logger.info("=" * 60)

    data_path = DATA_DIR / "full_data_out.json"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    datasets = load_datasets(data_path)

    all_results: List[dict] = []
    all_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    # Synthetic first (fast debug), then real-world
    dataset_order = [
        "friedman1",
        "friedman3",
        "synth_3way",
        "synth_4way",
        "diabetes",
        "breast_w",
        "wine_quality",
        "california_housing",
        "spambase",
        "adult",
    ]

    ds_by_name = {ds.name: ds for ds in datasets}
    ordered_datasets = [
        ds_by_name[n] for n in dataset_order if n in ds_by_name
    ]
    for ds in datasets:
        if ds.name not in dataset_order:
            ordered_datasets.append(ds)

    def _save_checkpoint(label: str = "checkpoint"):
        """Save partial results as checkpoint."""
        if not all_results:
            return
        summary_table = build_summary_table(all_results)
        output = build_output_json(
            all_results=all_results,
            datasets=[
                ds_by_name[r["dataset"]]
                for r in all_results
                if r and r.get("dataset") in ds_by_name
            ],
            all_predictions=all_predictions,
            summary_table=summary_table,
        )
        scots_wins = 0
        scots_total = 0
        for r in all_results:
            if (
                not r
                or r.get("mean_scots") is None
                or r.get("mean_figs") is None
            ):
                continue
            scots_total += 1
            if r["mean_scots"] >= r["mean_figs"] - 0.01:
                scots_wins += 1
        hypothesis_confirmed = (
            scots_wins >= scots_total * 0.6
            if scots_total > 0
            else False
        )
        output["metadata"]["hypothesis_confirmed"] = (
            hypothesis_confirmed
        )
        output["metadata"]["success_criteria"] = {
            "scots_competitive_with_figs": (
                f"{scots_wins}/{scots_total} datasets"
            ),
            "hypothesis_confirmed": hypothesis_confirmed,
        }
        output["metadata"]["checkpoint"] = label
        output["metadata"]["datasets_completed"] = len(all_results)
        out_path = WORKSPACE / "method_out.json"
        out_path.write_text(
            json.dumps(output, indent=2, default=str)
        )
        logger.info(
            f"  Checkpoint [{label}] saved: "
            f"{out_path.stat().st_size / 1024 / 1024:.1f} MB"
        )

    for ds in ordered_datasets:
        ds_start = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {ds.name}")
        logger.info(f"{'='*50}")

        result, ds_preds = run_dataset(ds)
        all_results.append(result)
        all_predictions[ds.name] = ds_preds

        ds_time = time.time() - ds_start
        elapsed = time.time() - start_time
        logger.info(
            f"  {ds.name}: {ds_time:.1f}s "
            f"(total: {elapsed:.0f}s)"
        )

        # Incremental checkpoint after each dataset
        _save_checkpoint(f"after_{ds.name}")

        if elapsed > 2700:
            logger.warning(
                f"Time budget nearly exhausted ({elapsed:.0f}s). "
                "Stopping early."
            )
            break

    # Final save
    summary_table = build_summary_table(all_results)
    logger.info(f"\n{summary_table}")

    output = build_output_json(
        all_results=all_results,
        datasets=[
            ds_by_name[r["dataset"]]
            for r in all_results
            if r and r.get("dataset") in ds_by_name
        ],
        all_predictions=all_predictions,
        summary_table=summary_table,
    )

    scots_wins = 0
    scots_total = 0
    for r in all_results:
        if (
            not r
            or r.get("mean_scots") is None
            or r.get("mean_figs") is None
        ):
            continue
        scots_total += 1
        if r["mean_scots"] >= r["mean_figs"] - 0.01:
            scots_wins += 1

    hypothesis_confirmed = (
        scots_wins >= scots_total * 0.6 if scots_total > 0 else False
    )
    output["metadata"]["hypothesis_confirmed"] = hypothesis_confirmed
    output["metadata"]["success_criteria"] = {
        "scots_competitive_with_figs": (
            f"{scots_wins}/{scots_total} datasets"
        ),
        "hypothesis_confirmed": hypothesis_confirmed,
    }
    output["metadata"]["checkpoint"] = "final"
    output["metadata"]["datasets_completed"] = len(all_results)

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output saved to {out_path}")
    logger.info(
        f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB"
    )

    total_time = time.time() - start_time
    logger.info(
        f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)"
    )
    logger.info("SC-OTS EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
