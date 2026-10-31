#pragma once

// Paper Reference: Albrecht et al. Section 3 / main_plan.txt Section 41 & 43
// Mathematical Purpose: Unified security estimator facade providing side-by-side
// evaluation of GSA-Intersect and Probabilistic Simulation estimators.

#include "dbdd/security/security_estimate.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/security/probabilistic_simulation.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/security/lightweight_predictor.hpp"

#include <string>
#include <cstddef>

namespace dbdd {

/**
 * @brief Comparative container contrasting GSA-Intersect and Probabilistic-Simulation estimates.
 */
struct SecurityComparison {
    SecurityEstimate gsa;           ///< Estimate obtained from GSA-Intersect.
    SecurityEstimate probabilistic; ///< Estimate obtained from Probabilistic-Simulation.
    double delta_beta = 0.0;        ///< Difference in estimated continuous beta: proba.beta - gsa.beta.
    int delta_bikz = 0;             ///< Difference in estimated integer bikz: proba.bikz - gsa.bikz.

    /**
     * @brief Generates a clean tabular summary string comparing both estimates.
     */
    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Unified facade for lattice cryptanalysis security estimation.
 */
class SecurityEstimator {
public:
    // =========================================================================
    // GSA-Intersect Estimators
    // =========================================================================

    [[nodiscard]] static SecurityEstimate gsa_intersect(
        size_t dimension,
        double ln_volume,
        double beta_min = 2.0,
        double beta_max = 2000.0);

    [[nodiscard]] static SecurityEstimate gsa_intersect(
        const UsvpInstance& usvp,
        double beta_min = 2.0,
        double beta_max = 2000.0);

    [[nodiscard]] static SecurityEstimate gsa_intersect(
        const DBDDInstance& dbdd,
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12);

    [[nodiscard]] static SecurityEstimate gsa_intersect(
        const DBDDStateStatistics& stats,
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12);

    // =========================================================================
    // Probabilistic Simulation Estimators
    // =========================================================================

    [[nodiscard]] static SecurityEstimate probabilistic(
        size_t dimension,
        double ln_volume,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static SecurityEstimate probabilistic(
        const UsvpInstance& usvp,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static SecurityEstimate probabilistic(
        const DBDDInstance& dbdd,
        const ProbabilisticOptions& options = {});

    [[nodiscard]] static SecurityEstimate probabilistic(
        const DBDDStateStatistics& stats,
        const ProbabilisticOptions& options = {});

    // =========================================================================
    // Comparative Security Analysis
    // =========================================================================

    [[nodiscard]] static SecurityComparison compare(
        size_t dimension,
        double ln_volume,
        const ProbabilisticOptions& options = {},
        double beta_min = 2.0,
        double beta_max = 2000.0);

    [[nodiscard]] static SecurityComparison compare(
        const UsvpInstance& usvp,
        const ProbabilisticOptions& options = {},
        double beta_min = 2.0,
        double beta_max = 2000.0);

    [[nodiscard]] static SecurityComparison compare(
        const DBDDInstance& dbdd,
        const ProbabilisticOptions& options = {},
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12);

    [[nodiscard]] static SecurityComparison compare(
        const DBDDStateStatistics& stats,
        const ProbabilisticOptions& options = {},
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12);
};

} // namespace dbdd

