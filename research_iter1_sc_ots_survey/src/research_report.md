# SC-OTS Survey

## Summary

Comprehensive implementation landscape survey for SC-OTS (Simplicial-Constrained Oblique Tree Sums). Covers RO-FIGS source code analysis (beam_size injection point identified at line 11 of Algorithm 1), FIGS base implementation from imodels, TDA library comparison (GUDHI recommended for simplex extraction), distance correlation computation via dcor, XGBoost interaction constraints API, InterpretML EBM baselines (supporting up to 3-way interactions), SHAP interaction values (pairwise) and shapiq for higher-order. Includes concrete code snippets, dependency list, integration blueprint, and risk analysis.

## Research Findings

## SC-OTS Implementation Landscape Survey

### 1. RO-FIGS Source Code & Oblique Split Mechanism

RO-FIGS (Random Oblique Fast Interpretable Greedy-Tree Sums) is the official implementation for the IEEE CITREx 2025 paper by Matjašec, Simidjievski, and Jamnik [1]. The repository at `github.com/um-k/rofigs` contains four source files: `rofigs.py` (main algorithm), `odt.py` (oblique decision tree), `constants.py` (hyperparameters), and `utils.py` (utilities) [1].

**Algorithm 1 (from the paper [2]) defines the critical injection point:**

```
1: Input: X, y, beam_size, min_imp_dec, max_splits
2: trees = []
3: while (get_max_imp_dec() > min_imp_dec or first_iteration) and count_total_splits() < max_splits:
4:   feat = select_random(beam_size)                    ← INJECTION POINT (initial)
5:   Φ = compute_linear_combination(X, y, feat)
6:   all_trees = join(trees, define_oblique_split(Φ))
7:   potential_splits = []
8:   for tree in all_trees:
9:     y_res = y - predict(all_trees except tree)
10:    for repetition in range(r):                      ← r=5 default
11:      feat = select_random(beam_size)                ← INJECTION POINT (main loop)
12:      for leaf in tree:
13:        Φ = compute_linear_combination(X, y_res, feat)
14:        potential_split = define_oblique_split(Φ, leaf)
15:        potential_splits.append(potential_split)
...
23:  trees.insert(best_split)
24: end while
25: Return: trees
```

The `select_random(beam_size)` calls at lines 4 and 11 are the exact points where SC-OTS replaces random feature selection with simplex-constrained selection [2]. In the source code, this manifests as:
```python
splitting_features = random.sample([i for i in range(X.shape[1])], self.beam_size)
```
[1]

**Oblique Split Optimization:** Each split learns weights via gradient descent using the spyct library (Stepišnik & Kocev [13]), imported as `GradSplitter`. The split takes form `w₁*F₁ + ... + wₖ*Fₖ ≤ t` [2]. The objective function combines L₁/₂ regularization with impurity minimization: `min_{w,b} ||w||_{1/2} + C * g(w,b)` where `||w||_{1/2} = (Σᵢ √|wᵢ|)²` [2]. However, the actual code delegates regularization to the GradSplitter via `regularization=1/self.C` (default C=10) [3].

**Default Hyperparameters:** Per-dataset optimized `(min_imp_dec, beam_size)` pairs in constants.py — e.g., blood=(5, 2), bioresponse=(0.05, 419) — with `RANDOM_STATE=12345` [4]. Worst-case complexity: O(i·r·m²·n²·d) where i=100 gradient iterations, r=5 repetitions [2].

**Dependencies:** spyct from GitLab, numpy 1.24, scikit-learn 1.3.2, scipy 1.10.1, pandas 2.0.3 [5, 13].

---

### 2. FIGS Base Implementation (imodels)

FIGS is in imodels v2.0.4 (Python ≥3.9) [6, 7]. Core: `FIGSClassifier(ClassifierMixin, FIGS)` and `FIGSRegressor(RegressorMixin, FIGS)`. Key parameters: `max_rules=12` (total splits), `max_trees`, `max_depth`, `min_impurity_decrease` [7]. The greedy loop maintains a priority queue of potential splits sorted by impurity reduction, using `_construct_node_with_stump()` which fits `DecisionTreeRegressor(max_depth=1)` [7].

**Integration Decision:** Extend RO-FIGS directly (not FIGS), since oblique splits are already implemented. The modification is surgical: replace `random.sample()` with simplex-based selection [1, 7].

---

### 3. TDA Library Comparison

**GUDHI v3.11.0 (RECOMMENDED)** — supports Python 3.10+ [8, 9]:
```python
rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=threshold)
st = rips.create_simplex_tree(max_dimension=3)
```
Critical methods: `get_filtration()`, `get_simplices()`, `get_skeleton(dim)`, `get_cofaces()`, `prune_above_filtration()`, `persistence()`, `find()` [10]. **GUDHI uniquely exposes actual simplex vertex lists**, not just persistence diagrams.

**giotto-tda v0.6.2** — outputs only persistence diagram triples `[birth, death, dim]`, NO individual simplex extraction [11, 26]. Useful for visualization only.

**ripser.py** — persistence diagrams + optional representative cocycles, but cocycles only represent generators, not all simplices at a threshold [12].

| Feature | GUDHI | giotto-tda | ripser |
|---|---|---|---|
| Extract simplices | ✓ | ✗ | Partial |
| Precomputed dist | ✓ | ✓ | ✓ |
| Threshold/prune | ✓ | ✗ | Via thresh |
| pip (Py 3.10/11) | ✓ | ✓ | ✓ |

---

### 4. Distance Correlation (dcor)

```python
dcor.distance_correlation(x, y, *, exponent=1, method='auto', compile_mode='auto')
```
O(n log n) for 1D inputs via AVL/mergesort methods [14, 15]. Pairwise matrix: `D[i,j] = 1 - dcor(X[:,i], X[:,j])` for all C(p,2) pairs. For p=50, n=5000: ~1,225 pairs, ~12-60 seconds [14, 16]. Batch optimization via `dcor.rowwise` with `CompileMode.COMPILE_PARALLEL` [16].

---

### 5. XGBoost Interaction Constraints

`xgb.XGBClassifier(interaction_constraints=[[0,1], [2,3,4]])` — features in same group CAN interact; overlapping groups create union of allowed interactions [18]. Natural mapping from simplicial complex: each simplex → one interaction group.

---

### 6. InterpretML EBM

Default: `ExplainableBoostingClassifier(interactions='3x', max_bins=1024, learning_rate=0.015)` [19, 20]. **KEY FINDING:** EBM DOES support 3-way interactions via `measure_interactions(X, y, interactions=combinations(range(n), 3))` [21]. Limitation: 3+ term interactions not graphed in global explanations.

---

### 7. SHAP & shapiq Interaction Values

Standard SHAP: `explainer.shap_interaction_values(X)` → (n_samples, n_features, n_features) pairwise matrix [22, 23]. **shapiq** extends to any-order: `TabularExplainer(model, data, index='k-SII', max_order=4)` computes 3-way and 4-way interactions, with TreeSHAP-IQ for tree models [24, 25].

---

### 8. Implementation Blueprint

**Stage 1:** Compute feature distance matrix via dcor (O(p²·n log n))
**Stage 2:** Build simplicial complex via GUDHI RipsComplex with persistence-based threshold
**Stage 3:** Modify RO-FIGS: replace `random.sample()` with simplex sampling for beam selection
**Stage 4:** Evaluate against FIGS, RO-FIGS, XGBoost±SC, EBM baselines using accuracy, model size, and interaction faithfulness metrics

### 9. Key Risks

1. RO-FIGS L₁/₂ vs L2 regularization discrepancy between paper and code [2, 3]
2. dcor cost for >100 features (~5-15 min for p=200) [16]
3. Rips complex size explosion: O(p⁴) for dim=3, mitigated by pruning [10]
4. spyct is a niche GitLab dependency requiring C compiler [5, 13]
5. SHAP limited to pairwise; shapiq needed for higher-order validation [22, 24]

## Sources

[1] [RO-FIGS GitHub Repository](https://github.com/um-k/rofigs) — Official RO-FIGS implementation with source code (rofigs.py, odt.py, constants.py, utils.py), showing beam_size feature selection and oblique split construction.

[2] [RO-FIGS Paper (arXiv)](https://arxiv.org/html/2504.06927v1) — Full paper with Algorithm 1 pseudocode, L₁/₂ regularization formulation, beam_size mechanics, split optimization, and O(i·r·m²·n²·d) complexity analysis.

[3] [RO-FIGS odt.py Source](https://raw.githubusercontent.com/um-k/rofigs/main/src/odt.py) — Oblique decision tree code showing GradSplitter integration with regularization=1/C parameter and weight/threshold extraction.

[4] [RO-FIGS Constants](https://raw.githubusercontent.com/um-k/rofigs/main/src/constants.py) — Per-dataset optimized (min_imp_dec, beam_size) pairs, RANDOM_STATE=12345, and 22 evaluation datasets.

[5] [RO-FIGS Requirements](https://raw.githubusercontent.com/um-k/rofigs/main/requirements.txt) — Dependencies: numpy 1.24.4, scikit-learn 1.3.2, scipy 1.10.1, pandas 2.0.3, spyct-tstep from GitLab.

[6] [imodels PyPI](https://pypi.org/project/imodels/) — imodels v2.0.4 (Nov 2025), Python ≥3.9, with FIGS and 20+ interpretable models.

[7] [FIGS Source Code](https://raw.githubusercontent.com/csinva/imodels/master/imodels/tree/figs.py) — Complete FIGS implementation: greedy potential_splits queue, _construct_node_with_stump(), max_rules parameter, residual-based tree growth.

[8] [GUDHI RipsComplex Docs](https://gudhi.inria.fr/python/latest/rips_complex_user.html) — RipsComplex API with distance_matrix input, max_edge_length, create_simplex_tree(max_dimension) for precomputed distances.

[9] [GUDHI v3.11.0 on Libraries.io](https://libraries.io/pypi/gudhi) — GUDHI v3.11.0 released Feb 21, 2025, pip installable with Python 3.10/3.11 wheels.

[10] [GUDHI SimplexTree Reference](https://gudhi.inria.fr/python/latest/simplex_tree_ref.html) — Complete API: get_filtration(), get_simplices(), get_skeleton(), get_cofaces(), prune_above_filtration(), persistence(), find().

[11] [giotto-tda VietorisRipsPersistence](https://giotto-ai.github.io/gtda-docs/latest/modules/generated/homology/gtda.homology.VietorisRipsPersistence.html) — Persistence diagram output only, NO individual simplex extraction. 3D input format for precomputed matrices.

[12] [ripser.py API Reference](https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html) — ripser() with maxdim, distance_matrix, do_cocycles. Returns diagrams and optional cocycles but not full simplex lists.

[13] [SPYCT Repository](https://github.com/knowledge-technologies/spyct) — Oblique Predictive Clustering Trees by Stepišnik & Kocev. Gradient-based split learning with Adam optimizer, used by RO-FIGS.

[14] [dcor.distance_correlation API](https://dcor.readthedocs.io/en/latest/functions/dcor.distance_correlation.html) — Full signature with method (auto/naive/avl/mergesort) and compile_mode. O(n log n) for 1D inputs.

[15] [dcor Usage Examples](https://dcor.readthedocs.io/en/latest/auto_examples/plot_dcor_usage.html) — Basic usage patterns, 1D and multidimensional support, method selection.

[16] [dcor Pairwise Performance Discussion](https://github.com/vnmabus/dcor/issues/32) — Efficient pairwise computation strategies: rowwise() with CompileMode.COMPILE_PARALLEL, symmetry exploitation.

[17] [scipy pdist API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) — Custom metric support via callable, but built-in 'correlation' is Pearson, NOT distance correlation.

[18] [XGBoost Interaction Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html) — interaction_constraints format, overlapping group semantics (union), feature index/name support.

[19] [InterpretML EBM Documentation](https://interpret.ml/docs/ebm.html) — Cyclic gradient boosting GAM with auto interaction detection, explain_global/local methods.

[20] [EBM Full Parameter API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html) — interactions='3x' default, max_bins=1024, max_interaction_bins=64, outer_bags=14, learning_rate=0.015.

[21] [EBM Custom Interactions Example](https://interpret.ml/docs/python/examples/custom-interactions.html) — 3-way interaction support via measure_interactions() with combinations(range(n), 3). Limitation: not graphed in global explanations.

[22] [SHAP Interaction Values Example](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html) — TreeExplainer, shap_interaction_values() → (n_samples, n_features, n_features), diagonal=main effects, off-diagonal=pairwise.

[23] [SHAP TreeExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) — Supports XGBoost/LightGBM/CatBoost/sklearn. Fast C++ implementation for interaction values.

[24] [shapiq GitHub Repository](https://github.com/mmschlk/shapiq) — Any-order Shapley interactions via k-SII. TabularExplainer with max_order=4 for 3-way/4-way. TreeSHAP-IQ for tree models.

[25] [shapiq Documentation](https://shapiq.readthedocs.io/en/latest/index.html) — TreeSHAP-IQ for tree-based models, supporting sklearn/XGBoost/LightGBM with any-order interactions.

[26] [giotto-tda PyPI](https://pypi.org/project/giotto-tda/) — v0.6.2 (May 2024), Python 3.8-3.12, actively maintained with pre-built wheels.

## Follow-up Questions

- Does the spyct GradSplitter truly implement L₁/₂ regularization (as claimed in the RO-FIGS paper) or L2 regularization (as suggested by the code's regularization=1/C parameter)? This discrepancy should be resolved by inspecting spyct's compiled C code or running controlled experiments.
- What is the optimal strategy for sampling simplices during SC-OTS beam selection — uniform random, weighted by filtration value (favoring tighter feature groups), or weighted by simplex dimension (favoring higher-order interactions)?
- Can the persistence diagram gap heuristic reliably select a filtration threshold for feature distance matrices computed via distance correlation, or does this require a custom threshold selection method tailored to the dcor metric space?
- How does shapiq's TreeSHAP-IQ computational cost scale with interaction order for validating 3-way and 4-way simplicial interactions, and is it feasible to use as part of a standard evaluation pipeline?

---
*Generated by AI Inventor Pipeline*
