# PH Foundations

## Summary

Comprehensive research on mathematical foundations for persistent-homology-based feature interaction discovery in tabular data. Covers five pillars: (1) feature dissimilarity metrics for Vietoris-Rips filtration—distance correlation, HSIC, and interaction information—with analysis of triangle inequality properties; (2) persistence threshold selection via largest-gap heuristic and persistence landscape statistical tests; (3) Betti number interpretation for feature interactions (β₀=independent groups, β₁=pairwise-only circular chains without synergy, β₂=interaction order ceilings); (4) opinion dynamics papers demonstrating qualitative differences between simplicial and pairwise models (discontinuous phase transitions, bistability, faster consensus); (5) prior work confirming novelty of using TDA to constrain tree model architecture rather than for post-hoc feature engineering.

## Research Findings

The mathematical and conceptual foundations for using persistent homology to discover feature interactions in tabular data rest on five interconnected pillars.

**PILLAR 1: Feature Dissimilarity Metrics for Vietoris-Rips Construction.** The primary candidate is distance correlation (dCor), introduced by Székely, Rizzo, and Bakirov (2007) [1]. Distance correlation satisfies 0 ≤ dCor(X,Y) ≤ 1, and critically, dCor(X,Y) = 0 if and only if X and Y are statistically independent—a property that Pearson correlation lacks [1]. The naive computation is O(n²) for n samples, but Huo and Székely (2016) developed an O(n log n) algorithm using AVL trees [2]. However, the raw dissimilarity d = 1 − dCor does NOT provably satisfy the triangle inequality. This is analogous to how d_a = 1 − |ρ| fails the triangle inequality for Pearson correlation—as demonstrated by explicit counterexamples in the gene expression clustering literature [3, 4]. The repair strategy follows the pattern established by van Dongen and Enright (2012) [5], who proved that d_r = √(1 − |ρ|) and d_s = √(1 − ρ²) both satisfy the triangle inequality for Pearson, Spearman, and cosine similarity. For distance correlation, √(1 − dCor) is the recommended transform: since dCor ∈ [0,1] and dCor is non-negative, the concavity argument for metric-preserving functions applies by analogy [6, 7]. **Critical caveat**: a formal proof that √(1 − dCor) satisfies the triangle inequality for arbitrary joint distributions has not been published—the existing proofs cover Pearson/Spearman only [3, 4]. This requires either a formal proof or empirical verification.

The alternative candidate HSIC (Hilbert-Schmidt Independence Criterion) [8] measures dependence via the squared Hilbert-Schmidt norm of the cross-covariance operator in RKHSs, computable in O(m²) time. When normalized via Centered Kernel Alignment (CKA), it yields values in [0,1] with CKA = 0 iff independence (with characteristic kernels) [9]. Its disadvantage is sensitivity to kernel bandwidth selection. Interaction information [10, 11] directly measures k-way synergy but scales combinatorially, making the VR approach (pairwise → topological lift) preferable for computational tractability.

The Vietoris-Rips complex is formally defined on metric spaces [12], and while it can be computed on any dissimilarity, the stability theorem for persistent homology—guaranteeing that small input perturbations cause small changes in persistence diagrams via bottleneck distance—requires a true metric [13, 14]. Without the triangle inequality, persistence diagrams are computable but lack formal robustness guarantees.

**PILLAR 2: Persistence Threshold Selection.** Two complementary approaches exist. The largest-gap heuristic examines persistence values (death − birth) and finds the biggest jump in sorted persistence values to separate signal from noise [15]. The more principled approach uses persistence landscapes (Bubenik 2015), which convert persistence diagrams into functional summaries in a separable Banach space, enabling permutation tests for statistical significance [16]. The stability theorem guarantees W_∞(Dgm(f), Dgm(g)) ≤ ‖f − g‖_∞ [13]. The recommended strategy: use persistence landscape permutation tests at α = 0.05, with largest-gap as a fast fallback.

**PILLAR 3: Betti Number Interpretation for Feature Interactions.** This is the novel interpretive contribution. β₀ counts connected components—in a feature interaction complex, β₀ = k means k independent groups of interacting features with no cross-group associations [17, 18]. β₁ counts 1-cycles (loops): a nonzero β₁ signals features forming pairwise interaction chains (A↔B, B↔C, C↔A) WITHOUT a simultaneous 3-way interaction (the triangle is hollow). This distinguishes circular pairwise dependencies from genuine higher-order synergistic interactions—a filled 2-simplex means A, B, C truly interact jointly, while a hollow triangle (contributing to β₁) means pairwise-only. β₂ counts 2-cycles (voids), signaling interaction order ceilings. No existing tree interpretability method provides this topological structural information.

**PILLAR 4: Opinion Dynamics Motivation.** Iacopini et al. (2019) demonstrated that simplicial contagion produces a DISCONTINUOUS (first-order) phase transition, while pairwise-only contagion produces a CONTINUOUS (second-order) transition, with a bistable region where healthy and endemic states coexist—impossible in pairwise models [19]. Battiston et al. (2020) [20] provide a comprehensive 109-page review documenting that pairwise networks are fundamentally insufficient for group phenomena. Zhang et al. (2023) [21] show higher-order interactions enhance synchronization in hypergraphs but have the opposite effect in simplicial complexes—the representation choice matters. Simplicial complexes are appropriate for feature interactions because of downward closure: if {A,B,C} is a simplex, all pairwise sub-interactions must exist [21]. Horstmeyer and Kuehn (2020) show peer pressure from 2-simplices accelerates consensus and fragmentation [22]. On reducibility, recent work [23] shows higher-order dynamics CAN be reduced to pairwise in the linear case but NOT in the nonlinear case—precisely the nonlinear synergistic effects where simplicial modeling adds value.

**PILLAR 5: Novelty Confirmation.** Prior work uses TDA to GENERATE topological features fed into classifiers [24, 25] or extracts feature interactions POST-HOC from trained forests [26, 27]. The iRF algorithm discovers interactions via iterative reweighting [27]. SC-OTS inverts this paradigm: discovering interaction structure FIRST via persistent homology to CONSTRAIN tree construction—a fundamentally different approach with no direct precedent.

**CRITICAL OPEN QUESTIONS AND CONTRADICTING EVIDENCE**: (1) The triangle inequality proof for √(1−dCor) is not established—this is extrapolated from Pearson correlation proofs and may not hold for all distributions [3, 4]. (2) The reducibility paper [23] shows that some higher-order dynamics ARE reducible to pairwise, meaning the simplicial approach would add no value in those cases—the irreducibility depends on nonlinearity of interactions. (3) Computational cost of persistent homology is O(k³) in the number of simplices [15], which grows combinatorially with feature count, potentially limiting applicability to datasets with >100 features without approximation strategies.

## Sources

[1] [Distance correlation - Wikipedia](https://en.wikipedia.org/wiki/Distance_correlation) — Definition of distance correlation, properties including dCor=0 iff independence, range [0,1], and computational aspects.

[2] [Fast Computing for Distance Covariance (Huo & Székely 2016)](https://arxiv.org/abs/1410.1503) — O(n log n) algorithm for distance covariance using AVL trees, making distance correlation practical for large datasets.

[3] [On triangle inequalities of correlation-based distances (BMC Bioinformatics 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9906874/) — Proves d_r = √(1−|ρ|) satisfies triangle inequality for Pearson, Spearman, cosine. Shows 1−|ρ| fails with counterexamples.

[4] [On triangle inequalities of correlation-based distances (Full Text)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05161-y) — Proposes and proves d_r = √(1−|ρ|) as a metric alternative to absolute correlation distance.

[5] [Metric distances derived from cosine similarity and correlations (van Dongen & Enright 2012)](https://arxiv.org/abs/1208.3145) — Foundational work proving d_o = √(0.5(1−ρ)) and d_s = √(1−ρ²) are valid metrics.

[6] [Introduction to Metric-Preserving Functions (Corazza)](https://www.researchgate.net/publication/255578827_Introduction_to_Metric-Preserving_Functions) — Theory of metric-preserving functions: concave, amenable, subadditive functions preserve metric properties.

[7] [Correlation-Based Metrics - mlfinlab documentation](https://random-docs.readthedocs.io/en/latest/codependence/correlation_based_metrics.html) — Angular distance formulas converting correlations to proper metrics with triangle inequality guarantees.

[8] [Measuring Statistical Dependence with Hilbert-Schmidt Norms (Gretton et al. 2005)](http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf) — Foundational HSIC paper: definition, O(m²) computation, HSIC=0 iff independence with universal kernels.

[9] [HSIC - Research Journal (Johnson)](https://jejjohnson.github.io/research_journal/appendix/similarity/hsic/) — Clear exposition of HSIC definition, kernel matrix formulation, CKA normalization to [0,1].

[10] [Interaction information - Wikipedia](https://en.wikipedia.org/wiki/Interaction_information) — Definition of interaction information; can be negative (synergy) or positive (redundancy); combinatorial scaling.

[11] [Multivariate information transmission (McGill 1954)](https://www.semanticscholar.org/paper/Multivariate-information-transmission-McGill/409ced53c7d854306dbadf4901fece1307a02637) — Original introduction of multivariate mutual information / interaction information.

[12] [Vietoris-Rips complex - Wikipedia](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex) — Formal definition of VR complex from metric space M and distance δ.

[13] [Persistent Homology: Theory and Practice (Edelsbrunner & Morozov 2012)](https://pub.ista.ac.at/~edels/Papers/2012-11-PHTheoryPractice.pdf) — Stability theorem: W_∞(Dgm(f), Dgm(g)) ≤ ‖f−g‖_∞. Bottleneck and Wasserstein distances.

[14] [On ℓp-Vietoris-Rips complexes (2024)](https://arxiv.org/abs/2411.01857) — Stability theorem for generalized VR complexes; triangle inequality essential for stability.

[15] [A roadmap for the computation of persistent homology (Otter et al. 2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6979512/) — Comprehensive PH computation guide: filtration construction, barcode interpretation, O(k³) complexity.

[16] [Statistical TDA using Persistence Landscapes (Bubenik 2015, JMLR)](https://www.jmlr.org/papers/volume16/bubenik15a/bubenik15a.pdf) — Persistence landscapes as Banach-space-valued random variables; permutation tests for significance.

[17] [Betti number - Wikipedia](https://en.wikipedia.org/wiki/Betti_number) — Formal definition: β₀ = connected components, β₁ = loops, β₂ = voids.

[18] [Mastering Betti Numbers in TDA - Number Analytics](https://www.numberanalytics.com/blog/mastering-betti-numbers-topological-data-analysis) — Practical interpretation of Betti numbers for data analysis and network structure.

[19] [Simplicial models of social contagion (Iacopini et al. 2019, Nature Comms)](https://www.nature.com/articles/s41467-019-10431-6) — 2-simplices cause discontinuous phase transitions and bistability, impossible in pairwise models.

[20] [Networks beyond pairwise interactions (Battiston et al. 2020, Physics Reports)](https://arxiv.org/abs/2006.01764) — 109-page review: pairwise networks insufficient for group phenomena across multiple dynamical processes.

[21] [Higher-order interactions shape dynamics differently in hypergraphs and simplicial complexes (Zhang et al. 2023)](https://www.nature.com/articles/s41467-023-37190-9) — Higher-order interactions enhance synchronization in hypergraphs but opposite in simplicial complexes.

[22] [An adaptive voter model on simplicial complexes (Horstmeyer & Kuehn 2020)](https://ar5iv.labs.arxiv.org/html/1909.05812) — Peer pressure from 2-simplices accelerates consensus and fragmentation transitions.

[23] [Reducibility of higher-order to pairwise interactions (2025)](https://arxiv.org/html/2601.05169) — HO dynamics reducible to pairwise in linear case but NOT in nonlinear case; nonlinearity determines irreducibility.

[24] [Topological data analysis and machine learning (2023 review)](https://www.tandfonline.com/doi/full/10.1080/23746149.2023.2202331) — Survey of TDA for ML: persistence images, Betti curves as feature engineering for classifiers.

[25] [Persistent homology classification algorithm (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280283/) — Using persistent homology features as inputs to classification pipelines.

[26] [Feature graphs for interpretable unsupervised tree ensembles (BioData Mining 2025)](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-025-00430-3) — Post-hoc feature interaction graphs from parent-child splits in random forests.

[27] [Iterative random forests for high-order interactions (Basu et al. 2018, PNAS)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5828575/) — iRF discovers up to 5th-6th order interactions post-hoc via iterative reweighting.

## Follow-up Questions

- Does √(1−dCor) provably satisfy the triangle inequality for arbitrary joint distributions, or do we need empirical verification on typical tabular datasets? The existing proofs cover Pearson/Spearman correlations but not distance correlation specifically.
- What is the computational cost of persistent homology up to dimension 3 for 50–200 features? The boundary matrix reduction is O(k³) in the number of simplices, which grows combinatorially—is a dimension-2 cap sufficient for practical feature counts?
- Are there fast approximations to distance correlation (e.g., the O(n log n) method of Huo & Székely 2016) that preserve the metric properties of the derived dissimilarity √(1−dCor)?
- How sensitive is the Vietoris-Rips complex to the choice of association metric—does switching from dCor to HSIC qualitatively change the resulting simplicial complex and its Betti numbers for typical tabular datasets?

---
*Generated by AI Inventor Pipeline*
