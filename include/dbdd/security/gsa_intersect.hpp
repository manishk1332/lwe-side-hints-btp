#pragma once

// Paper Reference: Albrecht et al. Section 3.1 & 3.2, Eq. (9), Eq. (11)
// Mathematical Purpose: Numerically stable log-domain GSA-Intersect security estimator.
// Predicts required BKZ block size beta ("bikz") to solve the uSVP lattice instance.

#include "dbdd/core/types.hpp"
#include "dbdd/security/security_estimate.hpp"
#include <cstddef>

namespace dbdd {

class UsvpInstance;
class DBDDInstance;

class GSAIntersect {
public:
    /**
     * @brief Computes log-domain isotropic objective function F(beta, d, ln_vol):
     * F(beta, d, ln_vol) = (2*beta - d - 1) * ln(delta_beta) + ln_vol / d - 0.5 * ln(beta).
     * An attack is predicted to succeed when F >= 0.
     */
    [[nodiscard]] static double objective(double beta, size_t d, double ln_vol);

    /**
     * @brief Computes log-domain general objective function F_gen(beta, d, ln_vol, s_norm):
     * F_gen = (2*beta - d - 1) * ln(delta_beta) + ln_vol / d - 0.5 * ln(beta / d) - ln(||s||).
     */
    [[nodiscard]] static double objective_general(double beta, size_t d, double ln_vol, double s_norm);

    /**
     * @brief Tests if isotropic condition F(beta, d, ln_vol) >= 0 is met.
     */
    [[nodiscard]] static bool condition(double beta, size_t d, double ln_vol);

    /**
     * @brief Tests if general condition F_gen(beta, d, ln_vol, s_norm) >= 0 is met.
     */
    [[nodiscard]] static bool condition_general(double beta, size_t d, double ln_vol, double s_norm);

    /**
     * @brief Continuous root-finding for F(beta) = 0 via bracketed Brent's method.
     * @param d Lattice dimension.
     * @param ln_vol Natural log of lattice volume.
     * @param beta_min Minimum block size bracket (default: 2.0).
     * @param beta_max Maximum block size bracket (default: 2000.0).
     * @param tol Convergence tolerance (default: 1e-8).
     * @return Continuous beta root. If F(beta_min) >= 0, returns beta_min. If F(beta_max) < 0, returns beta_max + 1.0.
     */
    [[nodiscard]] static double estimate_continuous_beta(
        size_t d,
        double ln_vol,
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double tol = 1e-8);

    /**
     * @brief Continuous root-finding for general secret norm ||s||.
     */
    [[nodiscard]] static double estimate_continuous_beta_general(
        size_t d,
        double ln_vol,
        double s_norm,
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double tol = 1e-8);

    /**
     * @brief Integer block size estimate: beta_int = min { beta in Z_>=beta_min : F(beta) >= 0 }.
     * By strict monotonicity of F, this equals ceil(beta_continuous).
     */
    [[nodiscard]] static size_t estimate_integer_beta(
        size_t d,
        double ln_vol,
        size_t beta_min = 2,
        size_t beta_max = 2000);

    /**
     * @brief Integer block size estimate for general secret norm ||s||.
     */
    [[nodiscard]] static size_t estimate_integer_beta_general(
        size_t d,
        double ln_vol,
        double s_norm,
        size_t beta_min = 2,
        size_t beta_max = 2000);

    /**
     * @brief Produces a comprehensive SecurityEstimate container for given (d, ln_vol).
     */
    [[nodiscard]] static SecurityEstimate estimate(
        size_t d,
        double ln_vol,
        double beta_min = 2.0,
        double beta_max = 2000.0);

    /**
     * @brief Estimates security for an isotropized UsvpInstance.
     */
    [[nodiscard]] static SecurityEstimate estimate(
        const UsvpInstance& usvp,
        double beta_min = 2.0,
        double beta_max = 2000.0);

    /**
     * @brief Estimates security for a DBDDInstance using the restricted-determinant volume shortcut.
     */
    [[nodiscard]] static SecurityEstimate estimate(
        const DBDDInstance& dbdd,
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12);
};

} // namespace dbdd

