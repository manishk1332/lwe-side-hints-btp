#pragma once

// Paper Reference: Dachman-Soled et al. Section 4.5 / main_plan.txt Section 50
// Mathematical Purpose: Optimal and greedy polynomial-time subset selection for short-vector hints
// minimizing the attack hardness proxy (Eq. 31) and concrete BKZ block size beta.

#include "dbdd/core/types.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/hints/worthiness.hpp"
#include "dbdd/security/security_estimate.hpp"

#include <vector>
#include <cstddef>
#include <string>

namespace dbdd {

class DBDDInstance;
class DBDDStateStatistics;

/**
 * @brief Record of a single selection step during short-vector optimization.
 */
struct SelectionStep {
    VectorZZ vector;                      ///< Selected vector.
    size_t candidate_index = 0;           ///< Index in original candidate pool W.
    double incremental_variance = 0.0;    ///< Variance score along this vector.
    double incremental_proxy_ratio = 1.0; ///< Multiplicative hardness proxy change at this step.
    size_t beta_after = 0;                ///< Estimated integer beta after integration.
    double delta_after = 0.0;             ///< Root-Hermite delta after integration.
};

/**
 * @brief Comprehensive results of short-vector subset selection and integration.
 */
struct SelectionResult {
    std::vector<VectorZZ> selected_vectors; ///< Chosen worthy vectors V subset of W.
    std::vector<size_t> selected_indices;   ///< Indices of chosen vectors in initial candidate list.
    std::vector<SelectionStep> history;     ///< Progression history step-by-step.
    size_t candidates_evaluated = 0;        ///< Total number of candidate evaluations performed.
    size_t initial_dim = 0;                 ///< Lattice dimension before selection.
    size_t final_dim = 0;                   ///< Lattice dimension after integration.
    size_t initial_beta = 0;                ///< Initial required integer beta.
    size_t final_beta = 0;                  ///< Final required integer beta after integration.
    double total_proxy_ratio = 1.0;         ///< Cumulative multiplicative change in hardness proxy.
    bool terminated_early = false;          ///< Whether greedy loop stopped before max_k due to unworthiness.

    [[nodiscard]] int bikz_reduction() const noexcept {
        return static_cast<int>(initial_beta) - static_cast<int>(final_beta);
    }
};

/**
 * @brief Algorithms for selecting optimal or greedy subsets of candidate short vectors.
 */
class GreedyShortVectorSelector {
public:
    /**
     * @brief Exhaustive combinatorial search evaluating all binom(|W|, k) subsets of size k.
     * Finds the exact optimal subset minimizing the Eq. (31) hardness proxy.
     * @param dbdd The active DBDD instance.
     * @param candidates Pool of candidate short vectors W. Must satisfy |candidates| <= 10.
     * @param k Target subset size (k <= |candidates|).
     */
    [[nodiscard]] static SelectionResult select_exact(
        const DBDDInstance& dbdd,
        const std::vector<VectorZZ>& candidates,
        size_t k);

    /**
     * @brief Sequential greedy selection heuristic for candidate vectors W.
     * At each step, selects the candidate maximizing variance / minimizing hardness proxy,
     * verifies worthiness, and integrates the vector into dbdd.
     * @param dbdd The active DBDD instance (modified in place).
     * @param candidates Candidate pool W.
     * @param max_k Maximum number of vectors to select (0 defaults to dim - 1).
     * @param verify_beta If true, requires that beta does not increase at each step.
     */
    static SelectionResult select_greedy(
        DBDDInstance& dbdd,
        const std::vector<VectorZZ>& candidates,
        size_t max_k = 0,
        bool verify_beta = true);

    /**
     * @brief Sequential greedy selection on the lightweight predictor (Tier 2).
     */
    static SelectionResult select_greedy(
        DBDDStateStatistics& stats,
        const std::vector<VectorZZ>& candidates,
        size_t max_k = 0,
        bool verify_beta = true);

    /**
     * @brief High-level automated q-vector integration on a DBDDInstance.
     * Generates valid q-vectors, selects worthy ones greedily, and integrates them.
     * @param dbdd Instance to update in place.
     * @param q LWE modulus.
     * @param max_k Maximum number of q-vectors to integrate (0 -> up to m).
     * @param include_differences Whether to also consider differences q*(e_i - e_j).
     * @param verify_beta Whether to reject vectors if beta does not decrease.
     */
    static SelectionResult integrate_q_vectors(
        DBDDInstance& dbdd,
        const BigInt& q,
        size_t max_k = 0,
        bool include_differences = false,
        bool verify_beta = true);

    /**
     * @brief High-level automated q-vector integration on DBDDStateStatistics.
     */
    static SelectionResult integrate_q_vectors(
        DBDDStateStatistics& stats,
        const BigInt& q,
        size_t m,
        size_t max_k = 0,
        bool verify_beta = true);
};

} // namespace dbdd

