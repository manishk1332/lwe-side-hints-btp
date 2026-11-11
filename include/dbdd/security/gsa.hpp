#pragma once

// Paper Reference: Dachman-Soled et al. Section 3.1 & 3.2, Eq. (9), Eq. (11)
// Mathematical Purpose: Models lattice reduction profiles under Schnorr's Geometric Series Assumption (GSA).
// Predicts Gram-Schmidt vector lengths, local projected block head lengths, and intersection conditions.

#include "dbdd/core/types.hpp"
#include <vector>

namespace dbdd {

class GSAModel {
public:
    /**
     * @brief Computes natural log of Gram-Schmidt norm ln(||b_i^*||) under GSA.
     * ln(||b_i^*||) = (d - 2i + 1) * ln(delta_beta) + ln(Vol) / d
     * @param beta BKZ block size (>= 2.0).
     * @param d Lattice dimension (>= 1).
     * @param ln_vol Natural log of lattice volume ln(Vol).
     * @param i 1-based index (1 <= i <= d).
     */
    [[nodiscard]] static double log_gs_norm(double beta, size_t d, double ln_vol, size_t i);

    /**
     * @brief Computes Gram-Schmidt norm ||b_i^*|| under GSA.
     * @param beta BKZ block size (>= 2.0).
     * @param d Lattice dimension (>= 1).
     * @param ln_vol Natural log of lattice volume ln(Vol).
     * @param i 1-based index (1 <= i <= d).
     */
    [[nodiscard]] static double gs_norm(double beta, size_t d, double ln_vol, size_t i);

    /**
     * @brief Returns predicted log-profile [ln(||b_1^*||), ..., ln(||b_d^*||)].
     */
    [[nodiscard]] static std::vector<double> log_profile(double beta, size_t d, double ln_vol);

    /**
     * @brief Returns predicted Gram-Schmidt profile [||b_1^*||, ..., ||b_d^*||].
     */
    [[nodiscard]] static std::vector<double> profile(double beta, size_t d, double ln_vol);

    /**
     * @brief Computes GSA decay factor alpha = delta_beta^{-2}.
     */
    [[nodiscard]] static double slope(double beta);

    /**
     * @brief Computes log GSA decay factor ln(alpha) = -2 * ln(delta_beta).
     */
    [[nodiscard]] static double log_slope(double beta);

    /**
     * @brief Computes ln(||b_{d - beta + 1}^*||), the log length of the first vector
     * of the projected block of size beta at the end of the basis.
     * ln(||b_{d - beta + 1}^*||) = (2*beta - d - 1) * ln(delta_beta) + ln(Vol) / d.
     */
    [[nodiscard]] static double log_projected_block_head(double beta, size_t d, double ln_vol);

    /**
     * @brief Computes predicted log projected secret norm.
     * For isotropic secret (||s|| = sqrt(d), passed as s_norm <= 0): 0.5 * ln(beta).
     * For general secret: 0.5 * ln(beta / d) + ln(||s||).
     * @param beta BKZ block size.
     * @param d Lattice dimension.
     * @param s_norm Secret norm (if <= 0, isotropic assumption ||s|| = sqrt(d) is used).
     */
    [[nodiscard]] static double log_projected_target_norm(double beta, size_t d, double s_norm = -1.0);

    /**
     * @brief Tests if projected target length is <= projected block head length.
     * Equivalent to log_projected_block_head >= log_projected_target_norm.
     */
    [[nodiscard]] static bool intersects(double beta, size_t d, double ln_vol, double s_norm = -1.0);
};

} // namespace dbdd

