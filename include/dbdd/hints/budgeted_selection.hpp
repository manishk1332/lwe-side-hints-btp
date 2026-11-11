#pragma once

// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 4.3, Equations (21)-(23) (approximate hints);
//                  thesis extension "Budgeted Hint Selection as D-Optimal Design"
//                  (innovation/proposal_3_budgeted_hint_selection.md).
// Mathematical Purpose: Given a pool of N candidate approximate hints <t, w_i> + e_i = l_i with independent
// noise e_i ~ N(0, sigma_i^2), choose at most k of them so that the isotropized lattice volume, and hence the
// required BKZ block size, is as favourable to the attacker as possible.
//
// Objective (in nats):
//     f(S) = ln Vol_iso(S) - ln Vol_iso(empty) = 0.5 * [ln rdet(Sigma) - ln rdet(Sigma_S)]
// Marginal gain of candidate i given the set S:
//     g_i(S) = 0.5 * ln(1 + w_i Sigma_S w_i^T / sigma_i^2)
// f is normalised, monotone and submodular, so the greedy rule satisfies f(greedy) >= (1 - 1/e) f(optimal), and
//     f(optimal) <= f(S) + (sum of the k largest remaining marginal gains at S)      for every set S.
// Approximate hints leave the lattice and its dimension unchanged, so beta for any set is obtained from
// GSAIntersect::estimate(d, ln Vol_iso(empty) + f(S)) without recomputing any determinant.

#include "dbdd/core/types.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dbdd {

class DBDDStateStatistics;

/**
 * @brief One candidate approximate hint <s_bar, v_bar> + e = 0 with e ~ N(0, variance).
 */
struct HintCandidate {
    VectorDouble v_bar;       ///< Homogenized hint vector; length must equal the ambient dimension.
    double variance = 1.0;    ///< Noise VARIANCE sigma^2 > 0 (not the standard deviation).
    std::string label;        ///< Human-readable description, used in reports only.
};

/**
 * @brief Outcome of one selection strategy.
 */
struct BudgetedSelectionResult {
    std::vector<size_t> selected_indices;  ///< Indices into the candidate pool, in order of selection.
    std::vector<double> step_gains;        ///< Realised marginal gain of each selected hint (nats).
    double total_gain = 0.0;               ///< f(S) in nats.
    double upper_bound_gain = 0.0;         ///< Certified upper bound on f(S*) over all sets of size <= k.
    size_t dimension = 0;                  ///< Lattice dimension d (unchanged by approximate hints).
    double ln_vol_iso_before = 0.0;        ///< ln Vol_iso before any hint.
    double beta_before = 0.0;              ///< Continuous GSA-Intersect beta before selection.
    double beta_after = 0.0;               ///< Continuous beta after integrating the selected set.
    double beta_opt_lower = 0.0;           ///< Continuous beta implied by upper_bound_gain: no set of size <= k does better.
    size_t beta_before_int = 0;            ///< Integer beta before selection.
    size_t beta_after_int = 0;             ///< Integer beta after selection.
    size_t beta_opt_lower_int = 0;         ///< Integer beta implied by upper_bound_gain.
    size_t gain_evaluations = 0;           ///< Number of marginal-gain evaluations performed by the strategy.
    double elapsed_ms = 0.0;               ///< Wall-clock time of the selection itself (certificate excluded).
};

/**
 * @brief Selection strategies for a budget of k approximate hints out of a candidate pool.
 *
 * The selector captures the prior once (dimension, covariance and ln Vol_iso) and never mutates it.
 */
class BudgetedHintSelector {
public:
    // Mathematical Purpose: Capture the prior DBDD statistics used by every strategy.
    //
    // Inputs:
    //     prior: basis-free DBDD state (d, Vol, mu, Sigma) before the candidate hints.
    // Preconditions:
    //     prior.covariance() is symmetric positive semidefinite.
    // Postconditions:
    //     ln Vol_iso(empty) is computed once; the prior is not modified.
    // Precision & Complexity:
    //     double precision; one symmetric eigendecomposition, O(d^3).
    explicit BudgetedHintSelector(const DBDDStateStatistics& prior);

    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return amb_dim_; }
    [[nodiscard]] double ln_vol_iso() const noexcept { return ln_vol_iso_; }

    // Mathematical Purpose: Baseline. Uniformly random k-subset of the pool.
    // Preconditions: k <= candidates.size(); every candidate has the ambient length and variance > 0.
    // Precision & Complexity: O(k (d^2 + N nnz)).
    [[nodiscard]] BudgetedSelectionResult select_random(
        const std::vector<HintCandidate>& candidates, size_t k, uint64_t seed) const;

    // Mathematical Purpose: Non-adaptive baseline. Sort once by g_i(empty) and take the k largest.
    // Optimal when the gains do not interact (diagonal prior, distinct coordinate hints).
    // Precision & Complexity: O(N log N + k (d^2 + N nnz)).
    [[nodiscard]] BudgetedSelectionResult select_static_top_k(
        const std::vector<HintCandidate>& candidates, size_t k) const;

    // Mathematical Purpose: Greedy rule. At each step add argmax_i g_i(S), using the rank-1 update
    //     u = Sigma_S w_j,  q_i <- q_i - (w_i . u)^2 / (q_j + sigma_j^2),  Sigma_S <- Sigma_S - u^T u / (q_j + sigma_j^2).
    // Postconditions:
    //     total_gain >= (1 - (1 - 1/k)^k) * f(optimal)   (Nemhauser-Wolsey-Fisher);
    //     upper_bound_gain = min( min over greedy prefixes of [f(S_j) + k largest remaining gains],
    //                             total_gain / (1 - (1 - 1/k)^k) )  >= f(optimal).
    // Ties are broken towards the smaller index. Stops early when the best gain is below 1e-12.
    // Precision & Complexity: O(k (d^2 + N nnz)).
    [[nodiscard]] BudgetedSelectionResult select_greedy(
        const std::vector<HintCandidate>& candidates, size_t k) const;

    // Mathematical Purpose: Minoux's lazy greedy. Keeps possibly stale gains in a max-heap; by submodularity a
    // stale gain is an upper bound on the current one, so a candidate whose refreshed gain is still on top is
    // the greedy choice. Returns the greedy set (up to floating-point ties) with fewer gain evaluations.
    // Precision & Complexity: O(k d^2 + (#refreshes) nnz^2).
    [[nodiscard]] BudgetedSelectionResult select_lazy_greedy(
        const std::vector<HintCandidate>& candidates, size_t k) const;

    // Mathematical Purpose: Pair-greedy lookahead. Among the m candidates with the largest current gain, pick
    // the pair (a, b) maximising the joint gain g_a(S) + g_b(S + {a}) and add both; if one slot remains, add
    // the best single candidate. This heuristic carries NO guarantee of beating the greedy rule.
    // Preconditions: m >= 2.
    // Precision & Complexity: O((k / 2) (m d nnz + m^2 nnz + N nnz + d^2)).
    [[nodiscard]] BudgetedSelectionResult select_lookahead2(
        const std::vector<HintCandidate>& candidates, size_t k, size_t m) const;

    // Mathematical Purpose: Exact optimum by enumeration of all C(N, k) subsets, evaluated in Gram space:
    //     f(S) = 0.5 * ln det(I + D_S^{-1/2} G_SS D_S^{-1/2}),   G = W Sigma W^T,  D = diag(sigma_i^2).
    // Preconditions: candidates.size() <= max_exact_pool_size().
    // Precision & Complexity: O(N^2 d^2 + C(N, k) k^3).
    [[nodiscard]] BudgetedSelectionResult select_exact(
        const std::vector<HintCandidate>& candidates, size_t k) const;

    // Mathematical Purpose: f(S) for an arbitrary index set, computed in Gram space (independent of the
    // incremental engine used by the strategies).
    // Preconditions: indices are distinct and in range.
    [[nodiscard]] double gain_of_set(
        const std::vector<HintCandidate>& candidates, const std::vector<size_t>& indices) const;

    // Mathematical Purpose: g = 0.5 * ln(1 + q / variance) for a directional prior variance q = w Sigma w^T.
    // Preconditions: variance > 0. Negative q (rounding) is treated as 0.
    [[nodiscard]] static double marginal_gain(double q, double variance);

    [[nodiscard]] static constexpr size_t max_exact_pool_size() noexcept { return 16; }

private:
    size_t dim_{0};
    size_t amb_dim_{0};
    double ln_vol_iso_{0.0};
    MatrixDouble covariance_;
};

} // namespace dbdd
