#!/usr/bin/env python3
"""SC-OTS v2: Simplicial-Constrained Oblique Tree Sums with Elastic Net Splits.

Pipeline:
  1. Load datasets from full_data_out.json
  2. For each dataset × fold:
     a. Compute pairwise distance correlation matrix
     b. Build Vietoris-Rips filtration via GUDHI
     c. Select persistence threshold via internal CV
     d. Filter simplices via MI-based criterion
     e. Fit SC-OTS (oblique tree sums constrained to simplex feature groups)
     f. Fit baselines: FIGS, XGBoost, XGBoost+constraints, EBM
  3. Evaluate: accuracy/R², complexity, Betti numbers, interaction recovery
  4. Output method_out.json in exp_gen_sol_out.json schema format
"""

import json
import resource
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import dcor
import gudhi
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ── Logging setup ──────────────────────────────────────────────────
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
LOG_FMT = f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}"

logger.remove()
logger.add(sys.stdout, level="INFO", format=LOG_FMT)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG", format=LOG_FMT)

# ── Resource limits ────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (18 * 1024**3, 18 * 1024**3))  # 18GB RAM
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU

WORKSPACE = Path(__file__).parent
DATA_FILE = WORKSPACE / "full_data_out.json"
MINI_DATA_FILE = WORKSPACE / "mini_data_out.json"
OUTPUT_FILE = WORKSPACE / "method_out.json"

# ── Configuration ──────────────────────────────────────────────────
MAX_SAMPLES_DCOR = 2000       # subsample for dCor computation
MAX_SPLITS_SCOTS = 30         # max splits for SC-OTS
VAL_PATIENCE = 3              # early stopping patience
L1_RATIO = 0.85               # elastic net L1 ratio
ALPHA = 0.001                 # elastic net alpha
MI_THRESHOLD_RATIO = 0.10     # MI excess threshold for simplex filtering
FIGS_MAX_RULES_CANDIDATES = [5, 10, 15, 20, 30]
N_FOLDS = 5
MAX_DIM_RIPS = 3              # max simplicial dimension
MAX_SCOTS_TIME_PER_FOLD = 120  # seconds — hard time limit for SC-OTS per fold
MAX_SCOTS_TRAIN_SAMPLES = 5000  # subsample training for SC-OTS on large datasets


# ====================================================================
# DATA LOADING
# ====================================================================
@logger.catch
def load_data(data_path: Path) -> dict:
    """Load and parse the benchmark dataset JSON."""
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets = {}

    for ds_entry in raw["datasets"]:
        ds_name = ds_entry["dataset"]
        examples = ds_entry["examples"]
        if len(examples) == 0:
            logger.warning(f"Skipping empty dataset: {ds_name}")
            continue

        ex0 = examples[0]
        task_type = ex0["metadata_task_type"]
        n_features = ex0["metadata_n_features"]

        # Parse feature names
        feature_names = ex0.get("metadata_feature_names", [f"f{i}" for i in range(n_features)])
        if len(feature_names) < n_features:
            feature_names = [f"f{i}" for i in range(n_features)]

        # Parse known interactions if available
        known_interactions_str = ex0.get("metadata_known_interactions", None)
        known_interactions = None
        if known_interactions_str:
            try:
                known_interactions = json.loads(known_interactions_str)
            except (json.JSONDecodeError, TypeError):
                known_interactions = None

        # Build arrays
        X_list = []
        y_list = []
        folds_list = []
        for ex in examples:
            inp = json.loads(ex["input"])
            X_list.append(inp)
            if task_type == "classification":
                y_list.append(int(ex["output"]))
            else:
                y_list.append(float(ex["output"]))
            folds_list.append(int(ex["metadata_fold"]))

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        folds = np.array(folds_list, dtype=int)

        datasets[ds_name] = {
            "X": X,
            "y": y,
            "folds": folds,
            "task_type": task_type,
            "n_features": n_features,
            "feature_names": feature_names,
            "known_interactions": known_interactions,
            "n_samples": len(examples),
            "category": ex0.get("metadata_category", "unknown"),
        }
        logger.info(f"  {ds_name}: n={X.shape[0]}, p={X.shape[1]}, task={task_type}")

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


# ====================================================================
# STEP 1: DISTANCE CORRELATION MATRIX
# ====================================================================
def compute_dcor_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise distance correlation matrix.

    Subsample if dataset is too large for performance.
    For very wide datasets (>40 features), use abs(Pearson) as a fast fallback.
    """
    n_samples, n_features = X.shape

    # For very wide datasets, use Pearson correlation as fast fallback
    if n_features > 40:
        logger.info(f"    Using Pearson correlation fallback for {n_features} features")
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        return np.abs(corr)

    # Subsample for speed
    if n_samples > MAX_SAMPLES_DCOR:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, size=MAX_SAMPLES_DCOR, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dcor_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            try:
                dc = dcor.distance_correlation(X_sub[:, i], X_sub[:, j])
                if not np.isfinite(dc):
                    dc = 0.0
            except Exception:
                dc = 0.0
            dcor_matrix[i, j] = dc
            dcor_matrix[j, i] = dc

    return dcor_matrix


def dcor_to_dissimilarity(dcor_matrix: np.ndarray) -> np.ndarray:
    """Convert distance correlation to dissimilarity: sqrt(1 - dCor)."""
    clipped = np.clip(dcor_matrix, 0.0, 1.0)
    return np.sqrt(1.0 - clipped)


# ====================================================================
# STEP 2: VIETORIS-RIPS FILTRATION & PERSISTENT HOMOLOGY
# ====================================================================
def build_rips_persistence(
    dissimilarity_matrix: np.ndarray,
    max_dim: int = 3,
) -> tuple:
    """Build Rips complex from dissimilarity matrix and compute persistence."""
    n = dissimilarity_matrix.shape[0]

    # Convert to lower-triangular list format for GUDHI
    lower_tri = []
    for i in range(n):
        row = []
        for j in range(i):
            row.append(float(dissimilarity_matrix[i, j]))
        lower_tri.append(row)

    rips = gudhi.RipsComplex(
        distance_matrix=lower_tri,
        max_edge_length=1.0,
    )
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    persistence = simplex_tree.persistence()

    return simplex_tree, persistence


# ====================================================================
# STEP 3: PERSISTENCE THRESHOLD SELECTION
# ====================================================================
def extract_persistent_simplices(
    simplex_tree: gudhi.SimplexTree,
    persistence: list,
    min_persistence: float,
) -> dict:
    """Extract simplices from the complex at a filtration level
    determined by persistent features above the threshold."""
    # Collect birth values of persistent features
    birth_values = []
    for dim, (birth, death) in persistence:
        if dim >= 1 and np.isfinite(death) and (death - birth) >= min_persistence:
            birth_values.append(death)

    if not birth_values:
        # Fallback: use median filtration across all 1-simplices
        all_filts = [filt for simplex, filt in simplex_tree.get_skeleton(1)
                     if len(simplex) == 2]
        if all_filts:
            eps = float(np.median(all_filts))
        else:
            eps = 0.5
    else:
        eps = max(birth_values)

    # Prune complex at this threshold
    st_copy = gudhi.SimplexTree()
    for simplex, filt in simplex_tree.get_simplices():
        st_copy.insert(simplex, filtration=filt)
    st_copy.prune_above_filtration(eps)

    # Extract simplices by dimension
    simplices_by_dim = {0: [], 1: [], 2: [], 3: []}
    for simplex, filt in st_copy.get_simplices():
        dim = len(simplex) - 1
        if dim in simplices_by_dim:
            simplices_by_dim[dim].append(tuple(simplex))

    return simplices_by_dim


def select_persistence_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    simplex_tree: gudhi.SimplexTree,
    persistence: list,
    task_type: str,
    n_internal_folds: int = 2,
) -> float:
    """Select persistence threshold via internal CV with fast proxy model."""
    # Collect all persistence values (death - birth) for dim >= 1
    all_persistence_values = []
    for dim, (birth, death) in persistence:
        if dim >= 1 and np.isfinite(death):
            pers_val = death - birth
            if pers_val > 0:
                all_persistence_values.append(pers_val)

    if len(all_persistence_values) == 0:
        return 0.0

    all_persistence_values = np.array(all_persistence_values)

    # 5 candidate thresholds
    candidates = [
        float(np.percentile(all_persistence_values, 20)),
        float(np.percentile(all_persistence_values, 40)),
        float(np.percentile(all_persistence_values, 60)),
        float(np.percentile(all_persistence_values, 80)),
    ]

    # Add gap-selected threshold
    sorted_pers = np.sort(all_persistence_values)
    if len(sorted_pers) >= 2:
        gaps = np.diff(sorted_pers)
        gap_idx = int(np.argmax(gaps))
        gap_threshold = float((sorted_pers[gap_idx] + sorted_pers[gap_idx + 1]) / 2)
        candidates.append(gap_threshold)

    candidates = sorted(set(candidates))

    # Quick 2-fold internal CV using a decision tree proxy
    best_score = -np.inf
    best_threshold = candidates[len(candidates) // 2]

    # Subsample for speed in threshold selection
    n_sub = min(X_train.shape[0], 2000)
    if X_train.shape[0] > n_sub:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train.shape[0], size=n_sub, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
    else:
        X_sub = X_train
        y_sub = y_train

    kf = KFold(n_splits=n_internal_folds, shuffle=True, random_state=42)

    for threshold in candidates:
        scores = []
        for train_idx, val_idx in kf.split(X_sub):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]

            simplices = extract_persistent_simplices(
                simplex_tree, persistence, min_persistence=threshold
            )

            # Collect all features in simplices of dim >= 1
            all_features = set()
            for d in [1, 2, 3]:
                for s in simplices.get(d, []):
                    all_features.update(s)
            if not all_features:
                all_features = set(range(X_train.shape[1]))
            feat_list = sorted(all_features)

            # Fast proxy: decision tree on the selected features
            if task_type == "regression":
                proxy = DecisionTreeRegressor(max_depth=5, random_state=42)
            else:
                proxy = DecisionTreeClassifier(max_depth=5, random_state=42)
            proxy.fit(X_tr[:, feat_list], y_tr)
            score = proxy.score(X_val[:, feat_list], y_val)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_threshold = threshold

    return best_threshold


# ====================================================================
# STEP 4: MI-BASED SIMPLEX FILTERING
# ====================================================================
def filter_simplices_by_mi(
    simplices_by_dim: dict,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    mi_threshold_ratio: float = MI_THRESHOLD_RATIO,
) -> dict:
    """Filter higher-order simplices using decision tree score proxy for MI."""
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )

    mi_func = mutual_info_classif if task_type == "classification" else mutual_info_regression

    # Subsample for speed
    n = min(X.shape[0], 2000)
    rng = np.random.RandomState(42)
    if X.shape[0] > n:
        idx = rng.choice(X.shape[0], size=n, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    # Compute individual MI for each feature
    n_neighbors_mi = min(5, max(1, X_sub.shape[0] - 2))
    if n_neighbors_mi < 1 or X_sub.shape[0] < 5:
        return simplices_by_dim
    try:
        individual_mi = mi_func(X_sub, y_sub, random_state=42, n_neighbors=n_neighbors_mi)
    except ValueError:
        return simplices_by_dim

    filtered = {0: simplices_by_dim[0], 1: simplices_by_dim[1]}

    for dim in [2, 3]:
        filtered[dim] = []
        for simplex in simplices_by_dim.get(dim, []):
            feature_indices = list(simplex)

            if task_type == "regression":
                TreeModel = DecisionTreeRegressor
            else:
                TreeModel = DecisionTreeClassifier

            # Joint score
            joint_tree = TreeModel(max_depth=3, random_state=42)
            joint_tree.fit(X_sub[:, feature_indices], y_sub)
            joint_score = joint_tree.score(X_sub[:, feature_indices], y_sub)

            # Sum of individual MIs
            sum_individual = sum(individual_mi[fi] for fi in feature_indices if fi < len(individual_mi))

            if sum_individual > 0 and (joint_score / sum_individual) >= (1 + mi_threshold_ratio):
                filtered[dim].append(simplex)
            elif sum_individual == 0 and joint_score > 0:
                filtered[dim].append(simplex)

    # Log filtering results
    for dim in [2, 3]:
        before = len(simplices_by_dim.get(dim, []))
        after = len(filtered[dim])
        if before > 0:
            logger.debug(f"    MI filter dim {dim}: {before} -> {after} simplices "
                         f"({100 * (before - after) / before:.0f}% removed)")

    return filtered


# ====================================================================
# STEP 5: SC-OTS MODEL
# ====================================================================
class ObliqueSplitNode:
    """A node in an oblique decision tree."""
    __slots__ = [
        "coefficients", "feature_indices", "threshold", "value",
        "left", "right", "impurity_reduction", "n_samples",
        "is_leaf", "tree_num", "depth",
    ]

    def __init__(self):
        self.coefficients = None
        self.feature_indices = None
        self.threshold = None
        self.value = 0.0
        self.left = None
        self.right = None
        self.impurity_reduction = 0.0
        self.n_samples = 0
        self.is_leaf = True
        self.tree_num = 0
        self.depth = 0

    def predict_single(self, x: np.ndarray) -> float:
        if self.is_leaf:
            return self.value
        projection = np.dot(x[self.feature_indices], self.coefficients)
        if projection <= self.threshold:
            return self.left.predict_single(x)
        else:
            return self.right.predict_single(x)


class SCOTSModel:
    """Simplicial-Constrained Oblique Tree Sum."""

    def __init__(
        self,
        simplices_by_dim: dict,
        task_type: str,
        max_splits: int = MAX_SPLITS_SCOTS,
        val_patience: int = VAL_PATIENCE,
        l1_ratio: float = L1_RATIO,
        alpha: float = ALPHA,
        time_limit: float = MAX_SCOTS_TIME_PER_FOLD,
    ):
        self.simplices_by_dim = simplices_by_dim
        self.task_type = task_type
        self.max_splits = max_splits
        self.val_patience = val_patience
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.time_limit = time_limit
        self.trees_ = []
        self.n_splits_ = 0

    def _get_candidate_simplices(self) -> list:
        """Return all simplices as candidate feature groups."""
        candidates = []
        for dim in [1, 2, 3]:
            for simplex in self.simplices_by_dim.get(dim, []):
                candidates.append(list(simplex))
        for simplex in self.simplices_by_dim.get(0, []):
            candidates.append(list(simplex))
        return candidates

    def _fit_oblique_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_mask: np.ndarray,
        feature_indices: list,
    ):
        """Fit an oblique split on the given feature subset."""
        X_sub = X[sample_mask][:, feature_indices]
        y_sub = y[sample_mask]
        n = X_sub.shape[0]

        if n < 10:
            return None

        if len(feature_indices) == 1:
            # Single feature: axis-aligned split via CART stump
            stump = DecisionTreeRegressor(max_depth=1)
            stump.fit(X_sub, y_sub)
            if stump.tree_.feature[0] < 0:
                return None
            threshold = stump.tree_.threshold[0]
            coefficients = np.array([1.0])
            left_mask_local = X_sub[:, 0] <= threshold
        else:
            # Multi-feature: oblique split via elastic net SGDRegressor
            best_coefs = None
            for try_alpha in [self.alpha, self.alpha * 0.1, self.alpha * 0.01]:
                sgd = SGDRegressor(
                    penalty="elasticnet",
                    l1_ratio=self.l1_ratio,
                    alpha=try_alpha,
                    max_iter=1000,
                    tol=1e-4,
                    random_state=42,
                )
                try:
                    sgd.fit(X_sub, y_sub)
                    if not np.allclose(sgd.coef_, 0):
                        best_coefs = sgd.coef_.copy()
                        break
                except Exception:
                    continue

            if best_coefs is None:
                # Fallback: try Ridge (no L1)
                try:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_sub, y_sub)
                    best_coefs = ridge.coef_.copy()
                except Exception:
                    return None
                if np.allclose(best_coefs, 0):
                    return None

            coefficients = best_coefs

            norm = np.linalg.norm(coefficients)
            if norm > 0:
                coefficients = coefficients / norm

            projections = X_sub @ coefficients
            candidate_thresholds = np.percentile(projections, [25, 50, 75])

            best_impurity_reduction = -np.inf
            best_threshold = candidate_thresholds[1]
            total_var = np.var(y_sub) * n
            left_mask_local = projections <= best_threshold

            for t in candidate_thresholds:
                left_local = projections <= t
                right_local = ~left_local
                n_left = int(left_local.sum())
                n_right = int(right_local.sum())
                if n_left < 5 or n_right < 5:
                    continue
                impurity_left = np.var(y_sub[left_local]) * n_left
                impurity_right = np.var(y_sub[right_local]) * n_right
                reduction = total_var - impurity_left - impurity_right
                if reduction > best_impurity_reduction:
                    best_impurity_reduction = reduction
                    best_threshold = t
                    left_mask_local = left_local

            threshold = best_threshold

        # Map local mask back to global
        sample_indices = np.where(sample_mask)[0]
        global_left_mask = np.zeros(len(X), dtype=bool)
        global_right_mask = np.zeros(len(X), dtype=bool)
        global_left_mask[sample_indices[left_mask_local]] = True
        global_right_mask[sample_indices[~left_mask_local]] = True

        # Compute impurity reduction
        y_parent = y[sample_mask]
        y_left = y[global_left_mask]
        y_right = y[global_right_mask]
        if len(y_left) == 0 or len(y_right) == 0:
            return None
        impurity_parent = np.var(y_parent) * len(y_parent)
        impurity_left = np.var(y_left) * len(y_left)
        impurity_right = np.var(y_right) * len(y_right)
        impurity_reduction = impurity_parent - impurity_left - impurity_right

        return (coefficients, threshold, impurity_reduction,
                global_left_mask, global_right_mask, feature_indices)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """Greedy FIGS-style tree sum with oblique splits and validation stopping."""
        fit_start = time.time()
        n, p = X.shape
        candidate_simplices = self._get_candidate_simplices()

        if len(candidate_simplices) == 0:
            candidate_simplices = [[i] for i in range(p)]

        # Limit candidate simplices to avoid blowup
        if len(candidate_simplices) > 50:
            by_dim = sorted(candidate_simplices, key=lambda s: -len(s))
            candidate_simplices = by_dim[:50]

        # Subsample training data for large datasets
        if n > MAX_SCOTS_TRAIN_SAMPLES:
            rng = np.random.RandomState(42)
            sub_idx = rng.choice(n, size=MAX_SCOTS_TRAIN_SAMPLES, replace=False)
            X_fit = X[sub_idx]
            y_fit = y[sub_idx].copy()
            n_fit = MAX_SCOTS_TRAIN_SAMPLES
            logger.debug(f"    SC-OTS: subsampled {n} -> {n_fit} for fitting")
        else:
            X_fit = X
            y_fit = y.copy()
            n_fit = n

        self.trees_ = []
        residuals = y_fit.copy().astype(float)
        n_splits = 0
        best_val_score = -np.inf
        patience_counter = 0
        n_trees_to_grow = max(1, self.max_splits // 5)

        for tree_idx in range(n_trees_to_grow):
            # Check time limit
            elapsed = time.time() - fit_start
            if elapsed > self.time_limit:
                logger.debug(f"    SC-OTS: time limit ({self.time_limit}s) reached at tree {tree_idx}")
                break

            root = ObliqueSplitNode()
            root.value = float(np.mean(residuals))
            root.is_leaf = True
            root.n_samples = n_fit
            root.tree_num = tree_idx

            potential_leaves = [(root, np.ones(n_fit, dtype=bool))]
            splits_this_tree = 0
            max_splits_per_tree = min(5, self.max_splits - n_splits)

            for _ in range(max_splits_per_tree):
                # Time check within inner loop
                if time.time() - fit_start > self.time_limit:
                    break

                best_split = None
                best_reduction = -np.inf
                best_leaf_idx = -1

                for leaf_idx, (leaf_node, mask) in enumerate(potential_leaves):
                    if mask.sum() < 10:
                        continue
                    for simplex_features in candidate_simplices:
                        if max(simplex_features) >= p:
                            continue
                        result = self._fit_oblique_split(X_fit, residuals, mask, simplex_features)
                        if result is None:
                            continue
                        coefs, thresh, reduction, left_m, right_m, feats = result
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_split = result
                            best_leaf_idx = leaf_idx

                if best_split is None or best_reduction <= 0:
                    break

                coefs, thresh, reduction, left_m, right_m, feats = best_split
                leaf_node, mask = potential_leaves[best_leaf_idx]

                leaf_node.is_leaf = False
                leaf_node.coefficients = coefs
                leaf_node.threshold = thresh
                leaf_node.feature_indices = feats

                left_child = ObliqueSplitNode()
                left_child.value = float(np.mean(residuals[left_m])) if left_m.sum() > 0 else 0.0
                left_child.n_samples = int(left_m.sum())
                left_child.tree_num = tree_idx
                left_child.depth = leaf_node.depth + 1

                right_child = ObliqueSplitNode()
                right_child.value = float(np.mean(residuals[right_m])) if right_m.sum() > 0 else 0.0
                right_child.n_samples = int(right_m.sum())
                right_child.tree_num = tree_idx
                right_child.depth = leaf_node.depth + 1

                leaf_node.left = left_child
                leaf_node.right = right_child

                potential_leaves.pop(best_leaf_idx)
                potential_leaves.append((left_child, left_m))
                potential_leaves.append((right_child, right_m))

                n_splits += 1
                splits_this_tree += 1

            self.trees_.append(root)

            # Update residuals after this tree
            predictions = self._predict_array(X_fit)
            residuals = y_fit.astype(float) - predictions

            # Validation-based early stopping
            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                if self.task_type == "regression":
                    var_y = np.var(y_val)
                    val_score = 1 - np.mean((y_val - val_preds) ** 2) / var_y if var_y > 0 else 0.0
                else:
                    try:
                        val_score = roc_auc_score(y_val, val_preds)
                    except ValueError:
                        val_score = accuracy_score(y_val, (val_preds > 0.5).astype(int))

                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.val_patience:
                    logger.debug(f"    Early stopping at tree {tree_idx+1}, "
                                 f"n_splits={n_splits}, val_score={val_score:.4f}")
                    break

            if splits_this_tree == 0:
                break

        self.n_splits_ = n_splits

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        """Prediction using per-sample tree traversal."""
        predictions = np.zeros(X.shape[0])
        for tree_root in self.trees_:
            for i in range(X.shape[0]):
                predictions[i] += tree_root.predict_single(X[i])
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_array(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        if self.task_type == "regression":
            var_y = np.var(y)
            return 1 - np.mean((y - preds) ** 2) / var_y if var_y > 0 else 0.0
        else:
            try:
                return roc_auc_score(y, preds)
            except ValueError:
                return accuracy_score(y, (preds > 0.5).astype(int))


# ====================================================================
# STEP 6: BASELINES
# ====================================================================
def fit_figs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
) -> dict:
    """Fit FIGS baseline with max_rules selection."""
    from imodels import FIGSClassifier, FIGSRegressor

    best_score = -np.inf
    best_model = None
    best_mr = 10

    for mr in FIGS_MAX_RULES_CANDIDATES:
        try:
            if task_type == "classification":
                model = FIGSClassifier(max_rules=mr)
            else:
                model = FIGSRegressor(max_rules=mr)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            if task_type == "regression":
                var_y = np.var(y_val)
                sc = 1 - np.mean((y_val - preds_val) ** 2) / var_y if var_y > 0 else 0.0
            else:
                try:
                    sc = roc_auc_score(y_val, preds_val)
                except ValueError:
                    sc = accuracy_score(y_val, preds_val)
            if sc > best_score:
                best_score = sc
                best_model = model
                best_mr = mr
        except Exception:
            logger.debug(f"FIGS max_rules={mr} failed")
            continue

    if best_model is None:
        return {"score": 0.0, "total_splits": 0, "training_time_s": 0.0}

    if task_type == "classification" and hasattr(best_model, "predict_proba"):
        test_preds = best_model.predict_proba(X_test)[:, 1]
    else:
        test_preds = best_model.predict(X_test).astype(float)
    return {
        "predictions": test_preds,
        "model": best_model,
        "best_max_rules": best_mr,
    }


def fit_xgboost_default(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
) -> dict:
    """Fit XGBoost baseline."""
    from xgboost import XGBClassifier, XGBRegressor

    if task_type == "classification":
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
            use_label_encoder=False, verbosity=0,
        )
    else:
        model = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric="rmse", random_state=42, verbosity=0,
        )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    if task_type == "classification" and hasattr(model, "predict_proba"):
        test_preds = model.predict_proba(X_test)[:, 1]
    else:
        test_preds = model.predict(X_test).astype(float)
    return {"predictions": test_preds, "model": model}


def fit_xgboost_constrained(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
    simplices_by_dim: dict,
    n_features: int,
) -> dict:
    """Fit XGBoost with simplicial interaction constraints."""
    from xgboost import XGBClassifier, XGBRegressor

    feat_names = [f"f{i}" for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    X_val_df = pd.DataFrame(X_val, columns=feat_names)
    X_test_df = pd.DataFrame(X_test, columns=feat_names)

    # Convert simplicial complex to XGBoost constraint groups using feature names
    groups = []
    seen_features = set()
    for dim in [1, 2, 3]:
        for simplex in simplices_by_dim.get(dim, []):
            grp = [f"f{f}" for f in simplex if f < n_features]
            if grp:
                groups.append(grp)
                seen_features.update(int(f) for f in simplex if f < n_features)
    for f in range(n_features):
        if f not in seen_features:
            groups.append([f"f{f}"])

    if task_type == "classification":
        model = XGBClassifier(
            interaction_constraints=groups,
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, verbosity=0,
        )
    else:
        model = XGBRegressor(
            interaction_constraints=groups,
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0,
        )

    model.fit(
        X_train_df, y_train,
        eval_set=[(X_val_df, y_val)],
        verbose=False,
    )
    if task_type == "classification" and hasattr(model, "predict_proba"):
        test_preds = model.predict_proba(X_test_df)[:, 1]
    else:
        test_preds = model.predict(X_test_df).astype(float)
    return {"predictions": test_preds, "model": model}


def fit_ebm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
) -> dict:
    """Fit EBM (Explainable Boosting Machine) baseline."""
    from interpret.glassbox import (
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
    )

    if task_type == "classification":
        model = ExplainableBoostingClassifier(
            interactions=10, max_bins=256, outer_bags=4, random_state=42,
        )
    else:
        model = ExplainableBoostingRegressor(
            interactions=10, max_bins=256, outer_bags=4, random_state=42,
        )

    model.fit(X_train, y_train)
    if task_type == "classification" and hasattr(model, "predict_proba"):
        test_preds = model.predict_proba(X_test)[:, 1]
    else:
        test_preds = model.predict(X_test).astype(float)
    return {"predictions": test_preds, "model": model}


# ====================================================================
# STEP 7: EVALUATION METRICS
# ====================================================================
def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> tuple:
    """Compute predictive performance score. Returns (score, metric_name, extra)."""
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    if task_type == "regression":
        var_y = np.var(y_true)
        r2 = 1 - np.mean((y_true - y_pred) ** 2) / var_y if var_y > 0 else 0.0
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return r2, "R2", rmse
    else:
        y_pred_binary = (y_pred > 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, y_pred)
            metric_name = "AUROC"
        except ValueError:
            auc = accuracy_score(y_true, y_pred_binary)
            metric_name = "accuracy"
        acc = accuracy_score(y_true, y_pred_binary)
        return auc, metric_name, acc


def compute_interaction_recovery(
    discovered_simplices: set,
    ground_truth_interactions: set,
) -> dict:
    """Compute precision, recall, F1 for interaction recovery."""
    if not ground_truth_interactions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(discovered_simplices & ground_truth_interactions)
    fp = len(discovered_simplices - ground_truth_interactions)
    fn = len(ground_truth_interactions - discovered_simplices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def parse_ground_truth_interactions(known_interactions: dict) -> set:
    """Parse ground-truth interactions from metadata into frozensets."""
    gt = set()
    if known_interactions is None:
        return gt

    for key in ["2-way", "3-way", "4-way"]:
        if key in known_interactions:
            for group in known_interactions[key]:
                gt.add(frozenset(group))

    return gt


# ====================================================================
# STEP 8: MAIN EXPERIMENT LOOP
# ====================================================================
@logger.catch
def run_experiment(
    data_path: Path,
    max_datasets: int = None,
    max_folds: int = None,
) -> dict:
    """Run the full SC-OTS experiment pipeline."""
    start_time = time.time()
    datasets = load_data(data_path)

    results = {}
    dataset_names = list(datasets.keys())
    if max_datasets is not None:
        dataset_names = dataset_names[:max_datasets]

    for ds_idx, ds_name in enumerate(dataset_names):
        ds_info = datasets[ds_name]
        X = ds_info["X"]
        y = ds_info["y"]
        folds = ds_info["folds"]
        task_type = ds_info["task_type"]
        n_features = ds_info["n_features"]
        feature_names = ds_info["feature_names"]
        known_interactions = ds_info["known_interactions"]

        logger.info(f"[{ds_idx+1}/{len(dataset_names)}] Processing {ds_name} "
                     f"(n={X.shape[0]}, p={X.shape[1]}, task={task_type})")

        results[ds_name] = {
            "task_type": task_type,
            "n_features": n_features,
            "n_samples": X.shape[0],
            "folds": {},
        }

        n_folds_to_run = max_folds if max_folds is not None else N_FOLDS
        unique_folds = sorted(np.unique(folds))[:n_folds_to_run]

        for fold_idx in unique_folds:
            fold_start = time.time()
            logger.info(f"  Fold {fold_idx}/{len(unique_folds)-1}")

            train_mask = folds != fold_idx
            test_mask = folds == fold_idx
            X_train_full, y_train_full = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # 80/20 train/val split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=fold_idx,
            )

            fold_results = {}

            # ──── SC-OTS Pipeline ────────────────────────────────
            t0 = time.time()
            simplices_filtered = {0: [], 1: [], 2: [], 3: []}
            try:
                # Step 1: dCor matrix
                dcor_matrix = compute_dcor_matrix(X_train)
                dissim_matrix = dcor_to_dissimilarity(dcor_matrix)

                # Step 2: Rips complex + persistence
                effective_max_dim = MAX_DIM_RIPS if n_features <= 30 else 2
                simplex_tree, persistence = build_rips_persistence(dissim_matrix, max_dim=effective_max_dim)

                # Step 3: Select persistence threshold
                threshold = select_persistence_threshold(
                    X_train, y_train, simplex_tree, persistence, task_type,
                )

                # Step 4: Extract and filter simplices
                simplices_raw = extract_persistent_simplices(simplex_tree, persistence, threshold)
                simplices_filtered = filter_simplices_by_mi(
                    simplices_raw, X_train, y_train, task_type,
                )

                # Fallback if everything filtered out
                has_higher = sum(len(simplices_filtered[d]) for d in [1, 2, 3])
                if has_higher == 0:
                    logger.warning(f"  All simplices filtered out for {ds_name} fold {fold_idx}, using raw")
                    simplices_filtered = simplices_raw

                # Step 5: Compute Betti numbers
                st_for_betti = gudhi.SimplexTree()
                for simplex_s, filt_s in simplex_tree.get_simplices():
                    st_for_betti.insert(simplex_s, filtration=filt_s)
                birth_vals_for_betti = []
                for dim, (birth, death) in persistence:
                    if dim >= 1 and np.isfinite(death) and (death - birth) >= threshold:
                        birth_vals_for_betti.append(death)
                if birth_vals_for_betti:
                    betti_eps = max(birth_vals_for_betti)
                else:
                    all_filts = [f for s, f in simplex_tree.get_skeleton(1) if len(s) == 2]
                    betti_eps = float(np.median(all_filts)) if all_filts else 0.5
                st_for_betti.prune_above_filtration(betti_eps)
                st_for_betti.persistence()
                betti = st_for_betti.betti_numbers()

                # Adaptive max_splits and time limit based on dataset size
                effective_max_splits = MAX_SPLITS_SCOTS
                effective_time_limit = MAX_SCOTS_TIME_PER_FOLD
                if n_features > 50:
                    effective_max_splits = min(20, MAX_SPLITS_SCOTS)
                    effective_time_limit = min(60, MAX_SCOTS_TIME_PER_FOLD)
                if X.shape[0] > 10000:
                    effective_time_limit = min(90, MAX_SCOTS_TIME_PER_FOLD)

                # Step 6: Fit SC-OTS
                scots_model = SCOTSModel(
                    simplices_by_dim=simplices_filtered,
                    task_type=task_type,
                    max_splits=effective_max_splits,
                    val_patience=VAL_PATIENCE,
                    l1_ratio=L1_RATIO,
                    alpha=ALPHA,
                    time_limit=effective_time_limit,
                )
                scots_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

                scots_time = time.time() - t0
                scots_preds = scots_model.predict(X_test)
                sc_score, sc_metric, sc_extra = compute_score(y_test, scots_preds, task_type)

                fold_results["SC-OTS"] = {
                    "score": float(sc_score),
                    "metric_name": sc_metric,
                    "total_splits": scots_model.n_splits_,
                    "n_trees": len(scots_model.trees_),
                    "betti_numbers": {
                        "b0": int(betti[0]) if len(betti) > 0 else 0,
                        "b1": int(betti[1]) if len(betti) > 1 else 0,
                        "b2": int(betti[2]) if len(betti) > 2 else 0,
                    },
                    "simplex_counts": {
                        f"dim_{d}": len(simplices_filtered[d]) for d in [0, 1, 2, 3]
                    },
                    "persistence_threshold": float(threshold),
                    "training_time_s": round(scots_time, 2),
                }
                if sc_metric == "AUROC":
                    fold_results["SC-OTS"]["accuracy"] = float(sc_extra)
                else:
                    fold_results["SC-OTS"]["rmse"] = float(sc_extra)

                # Interaction recovery for synthetic datasets
                if known_interactions is not None:
                    gt = parse_ground_truth_interactions(known_interactions)
                    discovered = {frozenset(s) for d in [1, 2, 3]
                                  for s in simplices_filtered.get(d, [])}
                    recovery = compute_interaction_recovery(discovered, gt)
                    fold_results["SC-OTS"]["interaction_recovery"] = recovery

                logger.info(f"    SC-OTS: {sc_metric}={sc_score:.4f}, splits={scots_model.n_splits_}, "
                             f"time={scots_time:.1f}s")

            except Exception:
                logger.exception(f"SC-OTS failed for {ds_name} fold {fold_idx}")
                fold_results["SC-OTS"] = {
                    "score": 0.0, "metric_name": "error", "total_splits": 0,
                    "n_trees": 0, "betti_numbers": {"b0": 0, "b1": 0, "b2": 0},
                    "simplex_counts": {"dim_0": 0, "dim_1": 0, "dim_2": 0, "dim_3": 0},
                    "persistence_threshold": 0.0, "training_time_s": 0.0,
                }

            # ──── FIGS Baseline ──────────────────────────────────
            t0 = time.time()
            try:
                figs_result = fit_figs(X_train, y_train, X_val, y_val, X_test, task_type)
                figs_time = time.time() - t0
                if "predictions" in figs_result:
                    fg_score, fg_metric, fg_extra = compute_score(y_test, figs_result["predictions"], task_type)
                    fold_results["FIGS"] = {
                        "score": float(fg_score),
                        "metric_name": fg_metric,
                        "total_splits": figs_result.get("best_max_rules", 0),
                        "training_time_s": round(figs_time, 2),
                    }
                    if fg_metric == "AUROC":
                        fold_results["FIGS"]["accuracy"] = float(fg_extra)
                    else:
                        fold_results["FIGS"]["rmse"] = float(fg_extra)
                else:
                    fold_results["FIGS"] = {"score": 0.0, "metric_name": "error", "total_splits": 0, "training_time_s": 0.0}
                logger.info(f"    FIGS:   {fold_results['FIGS']['metric_name']}={fold_results['FIGS']['score']:.4f}, time={figs_time:.1f}s")
            except Exception:
                logger.exception(f"FIGS failed for {ds_name} fold {fold_idx}")
                fold_results["FIGS"] = {"score": 0.0, "metric_name": "error", "total_splits": 0, "training_time_s": 0.0}

            # ──── XGBoost Default ────────────────────────────────
            t0 = time.time()
            try:
                xgb_result = fit_xgboost_default(X_train, y_train, X_val, y_val, X_test, task_type)
                xgb_time = time.time() - t0
                xg_score, xg_metric, xg_extra = compute_score(y_test, xgb_result["predictions"], task_type)
                fold_results["XGBoost"] = {
                    "score": float(xg_score),
                    "metric_name": xg_metric,
                    "training_time_s": round(xgb_time, 2),
                }
                if xg_metric == "AUROC":
                    fold_results["XGBoost"]["accuracy"] = float(xg_extra)
                else:
                    fold_results["XGBoost"]["rmse"] = float(xg_extra)
                logger.info(f"    XGBoost: {xg_metric}={xg_score:.4f}, time={xgb_time:.1f}s")
            except Exception:
                logger.exception(f"XGBoost failed for {ds_name} fold {fold_idx}")
                fold_results["XGBoost"] = {"score": 0.0, "metric_name": "error", "training_time_s": 0.0}

            # ──── XGBoost Constrained ────────────────────────────
            t0 = time.time()
            try:
                xgbc_result = fit_xgboost_constrained(
                    X_train, y_train, X_val, y_val, X_test,
                    task_type, simplices_filtered, n_features,
                )
                xgbc_time = time.time() - t0
                xgc_score, xgc_metric, xgc_extra = compute_score(y_test, xgbc_result["predictions"], task_type)
                fold_results["XGBoost_constrained"] = {
                    "score": float(xgc_score),
                    "metric_name": xgc_metric,
                    "training_time_s": round(xgbc_time, 2),
                }
                if xgc_metric == "AUROC":
                    fold_results["XGBoost_constrained"]["accuracy"] = float(xgc_extra)
                else:
                    fold_results["XGBoost_constrained"]["rmse"] = float(xgc_extra)
                logger.info(f"    XGB-C:  {xgc_metric}={xgc_score:.4f}, time={xgbc_time:.1f}s")
            except Exception:
                logger.exception(f"XGBoost_constrained failed for {ds_name} fold {fold_idx}")
                fold_results["XGBoost_constrained"] = {"score": 0.0, "metric_name": "error", "training_time_s": 0.0}

            # ──── EBM Baseline ───────────────────────────────────
            t0 = time.time()
            try:
                ebm_result = fit_ebm(X_train, y_train, X_test, task_type)
                ebm_time = time.time() - t0
                eb_score, eb_metric, eb_extra = compute_score(y_test, ebm_result["predictions"], task_type)
                fold_results["EBM"] = {
                    "score": float(eb_score),
                    "metric_name": eb_metric,
                    "training_time_s": round(ebm_time, 2),
                }
                if eb_metric == "AUROC":
                    fold_results["EBM"]["accuracy"] = float(eb_extra)
                else:
                    fold_results["EBM"]["rmse"] = float(eb_extra)
                logger.info(f"    EBM:    {eb_metric}={eb_score:.4f}, time={ebm_time:.1f}s")
            except Exception:
                logger.exception(f"EBM failed for {ds_name} fold {fold_idx}")
                fold_results["EBM"] = {"score": 0.0, "metric_name": "error", "training_time_s": 0.0}

            results[ds_name]["folds"][str(fold_idx)] = fold_results
            fold_elapsed = time.time() - fold_start
            logger.info(f"  Fold {fold_idx} completed in {fold_elapsed:.1f}s")

        # Incremental save after each dataset
        try:
            partial_path = WORKSPACE / "method_out_partial.json"
            partial_datasets = format_output_exp_gen_sol(results, datasets)
            partial_output = {"metadata": build_metadata(results), "datasets": partial_datasets}
            partial_path.write_text(json.dumps(partial_output))
            logger.info(f"  Incremental save: {len(results)} datasets -> {partial_path.name}")
        except Exception:
            logger.exception("Incremental save failed (non-fatal)")

    total_time = time.time() - start_time
    logger.info(f"Experiment completed in {total_time:.1f}s")

    return results


# ====================================================================
# STEP 9: OUTPUT FORMATTING
# ====================================================================
def format_output_exp_gen_sol(
    results: dict,
    datasets_info: dict,
) -> list:
    """Format results into exp_gen_sol_out.json schema."""
    output_datasets = []

    for ds_name, ds_result in results.items():
        ds_info = datasets_info.get(ds_name, {})
        X = ds_info.get("X", np.array([]))
        y = ds_info.get("y", np.array([]))
        folds = ds_info.get("folds", np.array([]))
        task_type = ds_result.get("task_type", "regression")

        examples = []
        for i in range(len(y)):
            fold_idx = int(folds[i]) if i < len(folds) else 0
            fold_key = str(fold_idx)

            input_str = json.dumps([round(float(v), 6) for v in X[i]])
            if task_type == "classification":
                output_str = str(int(y[i]))
            else:
                output_str = str(round(float(y[i]), 6))

            example = {
                "input": input_str,
                "output": output_str,
                "metadata_fold": fold_idx,
                "metadata_task_type": task_type,
            }

            # Add per-method scores as predictions
            fold_data = ds_result.get("folds", {}).get(fold_key, {})
            for method_name in ["SC-OTS", "FIGS", "XGBoost", "XGBoost_constrained", "EBM"]:
                method_data = fold_data.get(method_name, {})
                score = method_data.get("score", 0.0)
                safe_name = method_name.replace("-", "_").replace(" ", "_")
                example[f"predict_{safe_name}"] = str(round(score, 6))

            # Add SC-OTS specific metadata
            scots_data = fold_data.get("SC-OTS", {})
            if scots_data.get("betti_numbers"):
                example["metadata_betti"] = json.dumps(scots_data["betti_numbers"])
            if scots_data.get("simplex_counts"):
                example["metadata_simplex_counts"] = json.dumps(scots_data["simplex_counts"])
            if scots_data.get("interaction_recovery"):
                example["metadata_interaction_recovery"] = json.dumps(scots_data["interaction_recovery"])

            examples.append(example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    return output_datasets


def build_metadata(results: dict) -> dict:
    """Build top-level metadata for the output."""
    methods = ["SC-OTS", "FIGS", "XGBoost", "XGBoost_constrained", "EBM"]

    # Count wins
    wins = {m: 0 for m in methods}
    for ds_name, ds_result in results.items():
        fold_scores = {m: [] for m in methods}
        for fold_key, fold_data in ds_result.get("folds", {}).items():
            for m in methods:
                s = fold_data.get(m, {}).get("score", 0.0)
                fold_scores[m].append(s)

        mean_scores = {m: np.mean(fold_scores[m]) if fold_scores[m] else 0.0 for m in methods}
        winner = max(mean_scores, key=mean_scores.get)
        wins[winner] += 1

    # Aggregated results
    aggregated = {}
    for ds_name, ds_result in results.items():
        aggregated[ds_name] = {}
        for m in methods:
            scores = []
            times = []
            for fold_key, fold_data in ds_result.get("folds", {}).items():
                md = fold_data.get(m, {})
                scores.append(md.get("score", 0.0))
                times.append(md.get("training_time_s", 0.0))
            aggregated[ds_name][m] = {
                "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
                "std_score": round(float(np.std(scores)), 4) if scores else 0.0,
                "mean_time_s": round(float(np.mean(times)), 2) if times else 0.0,
            }

    # Topological analysis
    topo = {}
    for ds_name, ds_result in results.items():
        betti_all = {"b0": [], "b1": [], "b2": []}
        simplex_all = {"dim_0": [], "dim_1": [], "dim_2": [], "dim_3": []}
        thresholds = []
        recovery_all = {"precision": [], "recall": [], "f1": []}
        for fold_key, fold_data in ds_result.get("folds", {}).items():
            scots = fold_data.get("SC-OTS", {})
            bn = scots.get("betti_numbers", {})
            for k in betti_all:
                betti_all[k].append(bn.get(k, 0))
            sc = scots.get("simplex_counts", {})
            for k in simplex_all:
                simplex_all[k].append(sc.get(k, 0))
            thresholds.append(scots.get("persistence_threshold", 0.0))
            rec = scots.get("interaction_recovery")
            if rec:
                for k in recovery_all:
                    recovery_all[k].append(rec.get(k, 0.0))

        topo[ds_name] = {
            "mean_betti": {k: round(float(np.mean(v)), 2) if v else 0.0 for k, v in betti_all.items()},
            "mean_simplex_counts": {k: round(float(np.mean(v)), 2) if v else 0.0 for k, v in simplex_all.items()},
            "mean_persistence_threshold": round(float(np.mean(thresholds)), 4) if thresholds else 0.0,
        }
        if recovery_all["f1"]:
            topo[ds_name]["interaction_recovery_mean"] = {
                k: round(float(np.mean(v)), 4) for k, v in recovery_all.items()
            }

    return {
        "method_name": "SC-OTS v2",
        "description": "Simplicial-Constrained Oblique Tree Sums with Elastic Net Splits and Adaptive Stopping",
        "timestamp": datetime.now().isoformat(),
        "methods": methods,
        "overall_wins": wins,
        "aggregated_results": aggregated,
        "topological_analysis": topo,
        "hyperparameters": {
            "max_splits": MAX_SPLITS_SCOTS,
            "val_patience": VAL_PATIENCE,
            "l1_ratio": L1_RATIO,
            "alpha": ALPHA,
            "mi_threshold_ratio": MI_THRESHOLD_RATIO,
            "max_dim_rips": MAX_DIM_RIPS,
            "max_samples_dcor": MAX_SAMPLES_DCOR,
        },
    }


# ====================================================================
# MAIN
# ====================================================================
@logger.catch
def main():
    import os

    logger.info("=" * 60)
    logger.info("SC-OTS v2 Experiment Pipeline")
    logger.info("=" * 60)

    # Determine which data file to use
    use_mini = os.environ.get("USE_MINI", "0") == "1"
    max_datasets = int(os.environ.get("MAX_DATASETS", "0")) or None
    max_folds = int(os.environ.get("MAX_FOLDS", "0")) or None

    if use_mini:
        data_path = MINI_DATA_FILE
        logger.info("Using MINI data for testing")
    else:
        data_path = DATA_FILE
        logger.info(f"Using full data: {data_path}")

    if max_datasets:
        logger.info(f"Limiting to {max_datasets} datasets")
    if max_folds:
        logger.info(f"Limiting to {max_folds} folds")

    # Load data for metadata reference
    datasets_info = load_data(data_path)

    # Run experiment
    results = run_experiment(
        data_path=data_path,
        max_datasets=max_datasets,
        max_folds=max_folds,
    )

    # Build output
    metadata = build_metadata(results)
    output_datasets = format_output_exp_gen_sol(results, datasets_info)

    output = {
        "metadata": metadata,
        "datasets": output_datasets,
    }

    # Save
    OUTPUT_FILE.write_text(json.dumps(output))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Output written to {OUTPUT_FILE} ({size_mb:.1f} MB)")

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for ds_name, agg in metadata.get("aggregated_results", {}).items():
        scores_str = " | ".join(f"{m}: {agg[m]['mean_score']:.4f}±{agg[m]['std_score']:.4f}"
                                for m in metadata["methods"] if m in agg)
        logger.info(f"  {ds_name}: {scores_str}")

    logger.info(f"Overall wins: {metadata.get('overall_wins', {})}")


if __name__ == "__main__":
    main()
