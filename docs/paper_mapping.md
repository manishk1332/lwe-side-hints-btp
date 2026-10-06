# Paper to Code Mapping

This document maps mathematical equations and concepts from:
*LWE with Side Information: Attacks and Concrete Security Estimation* (Albrecht et al., 2020)
to the corresponding C++ classes, functions, and unit tests.

| Paper Concept / Equation | Paper Ref | C++ Component | Test File |
|---|---|---|---|
| Kannan Embedding: $\bar{s} = (e, s, 1)$ | Eq. (7) | `KannanEmbedding::from_lwe` | `tests/lwe/test_embedding.cpp` |
| Homogenization: $\mathcal{E}(\mu, \Sigma) \subset \mathcal{E}(0, \Sigma + \mu^T \mu)$ | Eq. (8) | `Homogenization::apply` | `tests/dbdd/test_homogenization.cpp` |
| General GSA uSVP condition | Eq. (9) | `GSAIntersect::general_condition` | `tests/security/test_gsa.cpp` |
| Root-Hermite factor $\delta_\beta$ | Eq. (10) | `RootHermite::delta` | `tests/security/test_root_hermite.cpp` |
| Isotropic GSA Intersection condition | Eq. (11) | `GSAIntersect::condition` | `tests/security/test_gsa_intersect.cpp` |
| Perfect hint distribution update | Eq. (12), (13) | `PerfectHint::update_distribution` | `tests/hints/test_perfect_hint.cpp` |
| Perfect hint lattice intersection | Sec. 3.2 | `PerfectHint::update_lattice` | `tests/hints/test_perfect_hint.cpp` |
| Modular hint (smooth case) | Eq. (15)-(17) | `ModularHint::apply_smooth` | `tests/hints/test_modular_hint.cpp` |
| Modular hint (general discrete Gaussian) | Eq. (18)-(19) | `ModularHint::apply_general` | `tests/hints/test_modular_hint.cpp` |
| Approximate hint conditioning | Eq. (20)-(23) | `ApproximateHint::apply` | `tests/hints/test_approximate_hint.cpp` |
| A-posteriori approximate hint | Eq. (24)-(26) | `AposterioriApproximateHint::apply` | `tests/hints/test_aposteriori_hint.cpp` |
| Short-vector hint projection | Eq. (27)-(30) | `ShortVectorHint::apply` | `tests/hints/test_short_vector_hint.cpp` |
| Short-vector worthiness score | Eq. (31) | `ShortVectorWorthinessAnalyzer::score`| `tests/hints/test_worthiness.cpp` |
| Isotropization transformation $M = (\Sigma_h^{1/2})^\dagger$ | Sec. 3.3 | `Isotropization::compute_transform` | `tests/dbdd/test_isotropization.cpp` |
| Isotropized volume shortcut | Sec. 3.3 | `isotropized_volume` | `tests/dbdd/test_isotropization.cpp` |
| Probabilistic simulation predictor | Sec. 3.4 | `ProbabilisticEstimator::estimate` | `tests/security/test_probabilistic.cpp` |
| Greedy short-vector selection | Sec. 4.2 | `GreedyShortVectorSelector::select` | `tests/hints/test_greedy_selection.cpp` |
