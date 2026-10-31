#pragma once

// Paper Reference: Albrecht et al. Section 3.4 & Figure 4 (CRYPTO 2020 / IACR ePrint 2020/292)
// Chen & Nguyen, "BKZ 2.0: Better Lattice Security Estimates" (ASIACRYPT 2011, Section 3)
// Mathematical Purpose: Simulates BKZ reduction profile evolution independently of basis reduction.
// Predicts Gram-Schmidt orthogonalization lengths (||b_0^*||, ..., ||b_{d-1}^*||) after T tours
// of BKZ with block size beta, preserving exact lattice volume.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/lattice.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include <vector>
#include <cmath>
#include <cstddef>

namespace dbdd {

/**
 * @brief Strategy for evaluating local block reduction quality in BKZ simulation.
 */
enum class SimulationModel {
    ChenNguyen, ///< Standard CN11 model: empirical HKZ table for d <= 45, GH + pruning for d > 45.
    PureGH,     ///< Pure Gaussian heuristic for all d >= 2.
    GSA         ///< Pure Schnorr Geometric Series Assumption profile baseline.
};

/**
 * @brief Parameters governing the BKZ profile simulation.
 */
struct BKZSimulatorParameters {
    size_t block_size = 2;              ///< Local SVP block size beta (must be >= 2).
    size_t max_tours = 20;              ///< Maximum number of tours to simulate.
    double pruning_proba = 0.5;         ///< Success probability for extreme pruning in SVP (default: 0.5).
    double early_abort_threshold = 1e-8;///< Convergence threshold: max absolute profile change < threshold.
    SimulationModel model = SimulationModel::ChenNguyen; ///< Model selection.
    bool record_tours = false;          ///< Whether to record log-profile history after each tour.

    BKZSimulatorParameters() = default;

    BKZSimulatorParameters(size_t beta, size_t tours = 20, double p = 0.5,
                           double eps = 1e-8,
                           SimulationModel m = SimulationModel::ChenNguyen,
                           bool record = false)
        : block_size(beta), max_tours(tours), pruning_proba(p),
          early_abort_threshold(eps), model(m), record_tours(record) {}

    /**
     * @brief Validates parameter ranges.
     * @throws InvalidParameterException on violation.
     */
    void validate() const {
        if (block_size < 2) {
            throw InvalidParameterException("BKZSimulatorParameters: block_size beta must be >= 2");
        }
        if (pruning_proba <= 0.0 || pruning_proba > 1.0) {
            throw InvalidParameterException("BKZSimulatorParameters: pruning_proba must be in (0, 1]");
        }
        if (early_abort_threshold < 0.0) {
            throw InvalidParameterException("BKZSimulatorParameters: early_abort_threshold must be >= 0");
        }
    }
};

/**
 * @brief Results of BKZ profile simulation.
 */
struct BKZSimulationResult {
    std::vector<double> log_profile;            ///< Natural log-lengths [ln(||b_0^*||), ..., ln(||b_{d-1}^*||)].
    std::vector<double> profile;                ///< Gram-Schmidt lengths [||b_0^*||, ..., ||b_{d-1}^*||].
    std::vector<std::vector<double>> tour_log_profiles; ///< History of log profiles per tour (if recorded).
    size_t tours_completed = 0;                 ///< Number of tours simulated.
    bool converged = false;                     ///< Whether simulation converged before max_tours.
    double final_slope = 0.0;                   ///< Linear regression slope of ln(||b_i^*||) vs i.
    double log_volume = 0.0;                    ///< Invariant log-volume sum_{i=0}^{d-1} ln(||b_i^*||).

    [[nodiscard]] double volume() const {
        return std::exp(log_volume);
    }

    [[nodiscard]] double head_length() const {
        return profile.empty() ? 0.0 : profile.front();
    }

    [[nodiscard]] double tail_length() const {
        return profile.empty() ? 0.0 : profile.back();
    }

    /**
     * @brief Returns ln(||b_{d-k}^*||), the log-length of the head of the projected block of size k
     * at the tail of the basis.
     * @param k Block size at tail (1 <= k <= dimension).
     */
    [[nodiscard]] double projected_block_head(size_t k) const {
        if (log_profile.empty() || k == 0 || k > log_profile.size()) {
            throw IndexOutOfBoundsException("BKZSimulationResult::projected_block_head: invalid k");
        }
        return log_profile[log_profile.size() - k];
    }

    /**
     * @brief Converts natural log-profile to base-2 log-profile [log_2(||b_0^*||), ...].
     */
    [[nodiscard]] std::vector<double> to_log2_profile() const {
        const double inv_ln2 = 1.0 / std::log(2.0);
        std::vector<double> res(log_profile.size());
        for (size_t i = 0; i < log_profile.size(); ++i) {
            res[i] = log_profile[i] * inv_ln2;
        }
        return res;
    }
};

/**
 * @brief Predicts Gram-Schmidt orthogonalization profile evolution under BKZ reduction.
 */
class BKZSimulator {
public:
    /**
     * @brief Simulates BKZ reduction starting from an explicit initial natural log-profile.
     * @param initial_log_profile Vector of ln(||b_i^*||) for i = 0, ..., d-1.
     * @param params Simulation parameters.
     */
    [[nodiscard]] static BKZSimulationResult simulate(
        const std::vector<double>& initial_log_profile,
        const BKZSimulatorParameters& params);

    /**
     * @brief Simulates BKZ reduction starting from abstract dimension and natural log volume.
     * Initializes with standard LLL baseline profile under GSA with delta = delta(2).
     */
    [[nodiscard]] static BKZSimulationResult simulate(
        size_t dimension,
        double ln_volume,
        const BKZSimulatorParameters& params);

    /**
     * @brief Simulates BKZ reduction starting from abstract dimension and BigReal volume.
     */
    [[nodiscard]] static BKZSimulationResult simulate(
        size_t dimension,
        const BigReal& volume,
        const BKZSimulatorParameters& params);

    /**
     * @brief Simulates BKZ reduction starting from an actual integer lattice basis.
     */
    [[nodiscard]] static BKZSimulationResult simulate(
        const BasisZZ& basis,
        const BKZSimulatorParameters& params);

    /**
     * @brief Simulates BKZ reduction on a lattice object.
     */
    [[nodiscard]] static BKZSimulationResult simulate(
        const LatticeZZ& lattice,
        const BKZSimulatorParameters& params);

    /**
     * @brief Simulates BKZ reduction on a DBDD instance lattice.
     */
    [[nodiscard]] static BKZSimulationResult simulate(
        const DBDDInstance& dbdd,
        const BKZSimulatorParameters& params);

    /**
     * @brief Performs one single BKZ simulation tour on log_profile in place.
     * @param log_profile Vector of ln(||b_i^*||), modified in place preserving sum.
     * @param beta Local block size.
     * @param params Simulation parameters.
     * @return Number of local block updates performed during the tour.
     */
    static size_t simulate_tour(
        std::vector<double>& log_profile,
        size_t beta,
        const BKZSimulatorParameters& params);

    /**
     * @brief Returns the Chen-Nguyen normalized offset constant in natural log: c_ln[d].
     * For d <= 45: evaluated from empirical HKZ table r_k * ln(2).
     * For d >= 46: evaluated via (1/2)*ln(gamma_d) - ln(p)/d.
     * @param d Projected block dimension (d >= 1).
     * @param pruning_proba Pruning success probability (default: 0.5).
     */
    [[nodiscard]] static double chen_nguyen_constant_ln(size_t d, double pruning_proba = 0.5);

    /**
     * @brief Returns the Chen-Nguyen normalized offset constant in base-2 log: c_log2[d].
     */
    [[nodiscard]] static double chen_nguyen_constant_log2(size_t d, double pruning_proba = 0.5);

    /**
     * @brief Computes Gaussian heuristic shortest vector factor squared gamma_d:
     * gamma_d = (d / (2 * pi * e)) * (pi * d)^(1 / d).
     */
    [[nodiscard]] static double gh_factor_squared(size_t d);

    /**
     * @brief Computes Gaussian heuristic factor sqrt(gamma_d).
     */
    [[nodiscard]] static double gh_factor(size_t d);

    /**
     * @brief Generates standard initial LLL log-profile using GSA with delta = delta(2).
     */
    [[nodiscard]] static std::vector<double> initial_lll_log_profile(size_t dimension, double ln_volume);

    /**
     * @brief Computes linear regression slope of log_profile[i] with respect to index i.
     */
    [[nodiscard]] static double compute_profile_slope(const std::vector<double>& log_profile);
};

} // namespace dbdd

