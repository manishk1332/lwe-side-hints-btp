#pragma once

// Paper Reference: Schnorr (1987) / Schnorr-Euchner (1994) / Chen-Nguyen (2011) / Albrecht et al. Section 2
// Mathematical Purpose: Configuration parameters for Block Korkine-Zolotarev (BKZ) lattice reduction.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/reduction/svp.hpp"

namespace dbdd {

/**
 * @brief Parameters for BKZ lattice reduction.
 */
struct BKZParameters {
    size_t block_size{10};                        // Block size beta in [2, d]
    size_t max_tours{50};                         // Maximum number of BKZ tours
    size_t min_tours{1};                          // Minimum number of tours to execute
    BigRational delta{99, 100};                   // Lovász / insertion factor (default 0.99)
    double early_abort_threshold{1e-4};           // Profile convergence tolerance across tours
    double timeout_seconds{0.0};                  // Overall wall-clock timeout in seconds (0 = unlimited)
    SVPParameters svp_params;                     // Configuration for local block SVP oracle
    bool log_tours{false};                        // Whether to record tour profiles

    BKZParameters() = default;

    explicit BKZParameters(size_t beta, size_t max_t = 50, double d = 0.99)
        : block_size(beta), max_tours(max_t), delta(BigReal(d).to_bigrational()) {
        validate();
    }

    void validate() const {
        if (block_size < 2) {
            throw InvalidParameterException("BKZParameters: block_size must be >= 2, got " + std::to_string(block_size));
        }
        if (max_tours == 0) {
            throw InvalidParameterException("BKZParameters: max_tours must be >= 1");
        }
        if (min_tours > max_tours) {
            throw InvalidParameterException("BKZParameters: min_tours cannot exceed max_tours");
        }
        if (delta <= BigRational(1, 4) || delta > BigRational(1)) {
            throw InvalidParameterException("BKZParameters: delta must be in (1/4, 1]");
        }
        if (timeout_seconds < 0.0) {
            throw InvalidParameterException("BKZParameters: timeout_seconds cannot be negative");
        }
        svp_params.validate();
    }
};

} // namespace dbdd

