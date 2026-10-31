#pragma once

// Paper Reference: Albrecht et al. Section 3.4 & Figure 4 (CRYPTO 2020 / IACR ePrint 2020/292)
// Reference Sage: leaky-LWE-Estimator/framework/utils.sage (compute_beta_delta & conditional_chi_squared)
// Mathematical Purpose: Probabilistic BKZ simulation security estimator modeling projected secret
// vectors under spherical Gaussian distributions and computing detection probabilities.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/security/bkz_simulator.hpp"
#include "dbdd/security/security_estimate.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/security/lightweight_predictor.hpp"

#include <vector>
#include <string>
#include <cstddef>

namespace dbdd {

/**
 * @brief Configuration options for probabilistic security estimation.
 */
struct ProbabilisticOptions {
    size_t tours_per_beta = 1;          ///< Number of tours simulated per beta step (default 1).
    size_t max_beta = 0;                ///< Maximum beta to consider (0 defaults to dimension - 1).
    size_t number_targets = 1;          ///< Number of target short vectors (default 1).
    bool ignore_lift_proba = false;     ///< If true, ignores pull-back probability (tail block position only).
    bool lift_union_bound = false;      ///< If true, uses min() union bound instead of conditional Chi-squared.
    size_t integration_steps = 0;       ///< Steps for conditional Chi-squared integration (0 -> 5*(d1+d2)).
    double stop_remaining_proba = 0.001;///< Threshold of remaining probability to terminate progressive loop.
    SimulationModel simulation_model = SimulationModel::ChenNguyen; ///< Model used by BKZSimulator.
    double pruning_proba = 0.5;         ///< SVP pruning probability for simulator.
    bool legacy_sage_d2_precedence = false; ///< Emulates Sage utils.sage operator precedence if true.
    double rank_tol = 1e-12;            ///< Eigenvalue tolerance for isotropized volume from DBDDInstance.
};

/**
 * @brief Step record in the progressive probability distribution curve.
 */
struct ProbabilisticStep {
    double beta = 0.0;                  ///< BKZ block size at this step.
    double tour_probability = 0.0;      ///< Detection probability at this tour/block size.
    double step_probability = 0.0;      ///< Probability of first detection at this exact step: remaining * p.
    double cumulative_probability = 0.0;///< Cumulative detection probability up to this step.
    double remaining_probability = 1.0; ///< Probability of no detection up to this step.
};

/**
 * @brief Detailed result of progressive probabilistic simulation.
 */
struct ProbabilisticResult {
    double average_beta = 0.0;          ///< Expected required BKZ block size: E[beta] = sum beta * p_step.
    size_t integer_beta = 0;            ///< Ceil of average_beta (or minimum integer beta for target prob).
    double delta = 0.0;                 ///< Root-Hermite factor corresponding to integer_beta.
    double cumulative_probability = 0.0;///< Final cumulative detection probability reached.
    double remaining_probability = 1.0; ///< Final remaining failure probability.
    bool solvable = false;              ///< Whether the instance was solved before max_beta.
    std::vector<ProbabilisticStep> steps;///< Detailed probability progression curve.
    std::vector<std::string> warnings;  ///< Warning messages (e.g. underflow, unsolvable).

    [[nodiscard]] SecurityEstimate to_security_estimate(size_t dimension, double ln_volume) const;
};

/**
 * @brief Refined probabilistic simulation security estimator (Albrecht et al., Section 3.4).
 */
class ProbabilisticEstimator {
public:
    // =========================================================================
    // Special Distributions & Probability Functions
    // =========================================================================

    /**
     * @brief Evaluates the Cumulative Distribution Function (CDF) of the Chi-squared distribution chi^2_k(x).
     * Computes the regularized lower incomplete gamma function P(k/2, x/2).
     * @param df Degrees of freedom k (df >= 1).
     * @param x Evaluation point (x >= 0).
     */
    [[nodiscard]] static double chi_squared_cdf(size_t df, double x);

    /**
     * @brief Evaluates the Probability Density Function (PDF) of the Chi-squared distribution chi^2_k(x).
     * @param df Degrees of freedom k (df >= 1).
     * @param x Evaluation point (x >= 0).
     */
    [[nodiscard]] static double chi_squared_pdf(size_t df, double x);

    /**
     * @brief Computes the conditional Chi-squared probability (Section 3.4 & utils.sage).
     * Probability that a standard spherical Gaussian vector (Y, Z) in R^{d1+d2} satisfies
     * ||Y||^2 + ||Z||^2 <= lt knowing that ||Y||^2 <= l2, where Y in R^{d2} and Z in R^{d1}.
     * @param d1 Dimension of the new coordinate slice Z.
     * @param d2 Dimension of the previously conditioned subvector Y.
     * @param lt Target squared norm threshold.
     * @param l2 Prior conditioned squared norm threshold.
     * @param steps Discretization intervals (0 defaults to 5 * (d1 + d2)).
     */
    [[nodiscard]] static double conditional_chi_squared(
        size_t d1, size_t d2, double lt, double l2, size_t steps = 0);

    // =========================================================================
    // Single-Tour & Multi-Tour Detection Probabilities
    // =========================================================================

    /**
     * @brief Computes the single-tour detection probability for a given Gram-Schmidt log-profile.
     * Evaluates detection at the tail block position i = d - beta and lifting pull-back probabilities
     * across consecutive blocks i_j = d - j*(beta - 1) - 1.
     * @param log_profile Natural log-profile ln(||b_0^*||), ..., ln(||b_{d-1}^*||).
     * @param beta BKZ block size.
     * @param options Simulation options.
     */
    [[nodiscard]] static double single_tour_detection_probability(
        const std::vector<double>& log_profile,
        size_t beta,
        const ProbabilisticOptions& options = {});

    /**
     * @brief Computes the detection probability P_success(beta, T) = 1 - prod_{t=1}^T (1 - P_t)
     * for a fixed block size beta after T BKZ tours.
     * @param beta BKZ block size.
     * @param tours Number of BKZ tours.
     * @param dimension Lattice dimension d.
     * @param ln_volume Natural log-volume ln(Vol(Lambda)).
     * @param options Simulation options.
     */
    [[nodiscard]] static double estimate_probability(
        size_t beta,
        size_t tours,
        size_t dimension,
        double ln_volume,
        const ProbabilisticOptions& options = {});

    /**
     * @brief Finds the minimum integer block size beta in [2, dimension] such that
     * estimate_probability(beta, tours, dimension, ln_volume) >= target_p.
     * @param target_p Target success probability in (0, 1].
     * @param tours Number of tours per block size candidate.
     * @param dimension Lattice dimension.
     * @param ln_volume Natural log-volume.
     * @param options Simulation options.
     */
    [[nodiscard]] static size_t required_beta_for_probability(
        double target_p,
        size_t tours,
        size_t dimension,
        double ln_volume,
        const ProbabilisticOptions& options = {});

    // =========================================================================
    // Progressive BKZ Simulation (Albrecht et al., Section 3.4 & Figure 4)
    // =========================================================================

    /**
     * @brief Runs progressive BKZ simulation for increasing beta = 2, 3, ... until detection,
     * calculating the expected required block size E[beta] and full probability distribution.
     */
    [[nodiscard]] static ProbabilisticResult estimate_progressive(
        size_t dimension,
        double ln_volume,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static ProbabilisticResult estimate_progressive(
        const UsvpInstance& usvp,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static ProbabilisticResult estimate_progressive(
        const DBDDInstance& dbdd,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static ProbabilisticResult estimate_progressive(
        const DBDDStateStatistics& stats,
        const ProbabilisticOptions& options = {});

    // =========================================================================
    // Unified SecurityEstimate Converters
    // =========================================================================

    [[nodiscard]] static SecurityEstimate estimate(
        size_t dimension,
        double ln_volume,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static SecurityEstimate estimate(
        const UsvpInstance& usvp,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static SecurityEstimate estimate(
        const DBDDInstance& dbdd,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static SecurityEstimate estimate(
        const DBDDStateStatistics& stats,
        const ProbabilisticOptions& options = {});
};

} // namespace dbdd

