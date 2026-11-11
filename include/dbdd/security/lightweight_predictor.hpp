#pragma once

// Paper Reference: Dachman-Soled et al. Section 3 / main_plan.txt Sections 39-40
// Mathematical Purpose: Basis-free lightweight security predictors for fast parameter sweeps
// and hint optimization without constructing or reducing lattice bases.
// Implements Tier 2 (DBDDStateStatistics) and Tier 3 (DiagonalCovariance).

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/security/security_estimate.hpp"
#include <vector>
#include <optional>

namespace dbdd {

class DBDDInstance;
class PerfectHint;
class ModularHint;
class ApproximateHint;
class AposterioriApproximateHint;
using AposterioriHint = AposterioriApproximateHint;
class ShortVectorHint;

/**
 * @brief Tier 2: Dense basis-free predictor maintaining (d, Vol, mu, Sigma).
 * Executes analytical rank-1 updates and volume scaling without basis reductions.
 */
class DBDDStateStatistics {
public:
    DBDDStateStatistics(
        size_t dim,
        size_t amb_dim,
        BigReal volume,
        VectorDouble mean,
        MatrixDouble covariance);

    DBDDStateStatistics(
        size_t dim,
        BigReal volume,
        VectorDouble mean,
        MatrixDouble covariance);

    static DBDDStateStatistics from_dbdd(const DBDDInstance& dbdd);

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return amb_dim_; }

    // Volumes
    [[nodiscard]] const BigReal& volume() const noexcept { return volume_; }
    [[nodiscard]] const BigReal& log_volume() const noexcept { return log_volume_; }

    // Distribution
    [[nodiscard]] const VectorDouble& mean() const noexcept { return mean_; }
    [[nodiscard]] const MatrixDouble& covariance() const noexcept { return covariance_; }

    // State mutators
    void set_dimension(size_t d) noexcept { dim_ = d; }
    void set_volume(BigReal vol);
    void set_log_volume(BigReal log_vol);
    void set_distribution(VectorDouble mean, MatrixDouble cov);

    // =========================================================================
    // Analytical Hint Integrations (Basis-Free)
    // =========================================================================

    /**
     * @brief Integrates a perfect hint <s, v> = l.
     * Drops dimension by 1, updates volume by dual_norm, and updates (mu, Sigma) via Eq. (12, 13).
     * @param v Hint vector in ambient dimension.
     * @param l Hint scalar value.
     * @param dual_norm Primitive dual norm of v (defaults to 1.0 for coordinate reveals).
     */
    void integrate_perfect_hint(const VectorDouble& v, double l, double dual_norm = 1.0);
    void integrate_perfect_hint(const PerfectHint& hint, double dual_norm = 1.0);

    /**
     * @brief Integrates a modular hint <s, v> = l (mod k).
     * Preserves dimension, scales volume by k, and leaves (mu, Sigma) unchanged in smooth regime.
     */
    void integrate_modular_hint(const VectorDouble& v, double l, double k);
    void integrate_modular_hint(const ModularHint& hint);

    /**
     * @brief Integrates an approximate hint <s, v> = l + e, where e ~ N(0, sigma^2).
     * Preserves dimension and volume, updates (mu, Sigma) via Eq. (22, 23).
     */
    void integrate_approximate_hint(const VectorDouble& v, double l, double sigma);
    void integrate_approximate_hint(const ApproximateHint& hint);

    /**
     * @brief Integrates an a-posteriori approximate hint.
     */
    void integrate_aposteriori_hint(const AposterioriHint& hint);

    /**
     * @brief Integrates a short vector hint v in Lambda.
     * Drops dimension by 1, divides volume by ||v||, and projects (mu, Sigma) orthogonal to v.
     */
    void integrate_short_vector_hint(const VectorDouble& v);
    void integrate_short_vector_hint(const ShortVectorHint& hint);

    // =========================================================================
    // Homogenization & Isotropized Security Estimation
    // =========================================================================

    /**
     * @brief Homogenizes distribution state in-place: mu_h = 0, Sigma_h = Sigma + mu^T * mu.
     */
    void homogenize();

    [[nodiscard]] bool is_homogenized(double tol = 1e-12) const;

    /**
     * @brief Returns homogenized covariance envelope Sigma_h = Sigma + mu^T * mu.
     */
    [[nodiscard]] MatrixDouble homogenized_covariance() const;

    /**
     * @brief Computes natural log of isotropized volume ln Vol(Lambda_iso) via shortcut:
     * ln Vol_iso = ln Vol - 0.5 * ln rdet(Sigma_h).
     */
    [[nodiscard]] double isotropized_log_volume(double rank_tol = 1e-12) const;
    [[nodiscard]] BigReal isotropized_volume(double rank_tol = 1e-12) const;

    /**
     * @brief Estimates attack complexity using GSA-Intersect security estimator.
     */
    [[nodiscard]] SecurityEstimate estimate_attack_gsa(
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12) const;

    [[nodiscard]] size_t estimate_beta_gsa(double rank_tol = 1e-12) const;

private:
    size_t dim_{0};
    size_t amb_dim_{0};
    BigReal volume_;
    BigReal log_volume_;
    VectorDouble mean_;
    MatrixDouble covariance_;
};

/**
 * @brief Tier 3: Super-lightweight diagonal predictor for canonical coordinate hints.
 * Operates in O(1) time per hint update and O(m) time for isotropized volume evaluation.
 */
class DiagonalCovariance {
public:
    DiagonalCovariance(
        size_t dim,
        size_t amb_dim,
        BigReal log_volume,
        std::vector<double> variances,
        std::vector<double> mean = {});

    static DiagonalCovariance from_state_statistics(
        const DBDDStateStatistics& stats,
        double off_diag_tol = 1e-10);

    static DiagonalCovariance from_dbdd(
        const DBDDInstance& dbdd,
        double off_diag_tol = 1e-10);

    static DiagonalCovariance from_lwe_parameters(
        size_t n,
        size_t m,
        double q,
        double sigma_e,
        double sigma_s);

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return amb_dim_; }

    // Volumes
    [[nodiscard]] const BigReal& log_volume() const noexcept { return log_volume_; }

    // Coordinate Variances and Means
    [[nodiscard]] const std::vector<double>& variances() const noexcept { return variances_; }
    [[nodiscard]] double variance(size_t i) const;
    [[nodiscard]] const std::vector<double>& mean() const noexcept { return mean_; }
    [[nodiscard]] double mean(size_t i) const;

    // =========================================================================
    // O(1) Canonical Coordinate Hint Integrations
    // =========================================================================

    /**
     * @brief Perfect hint on coordinate i: s_i = l.
     * Drops dimension by 1, sets variance(i) = 0, mean(i) = l.
     */
    void integrate_coordinate_perfect_hint(size_t coord_idx, double l);

    /**
     * @brief Modular hint on coordinate i: s_i = l (mod k).
     * Scales volume by k.
     */
    void integrate_coordinate_modular_hint(size_t coord_idx, double l, double k);

    /**
     * @brief Approximate hint on coordinate i: s_i = l + e, e ~ N(0, sigma^2).
     * 1D Gaussian Kalman update: var' = (var * sigma^2) / (var + sigma^2).
     */
    void integrate_coordinate_approximate_hint(size_t coord_idx, double l, double sigma);

    // =========================================================================
    // O(m) Isotropized Security Estimation
    // =========================================================================

    void homogenize();

    [[nodiscard]] double isotropized_log_volume(double tol = 1e-12) const;

    [[nodiscard]] SecurityEstimate estimate_attack_gsa(
        double beta_min = 2.0,
        double beta_max = 2000.0) const;

    [[nodiscard]] size_t estimate_beta_gsa() const;

private:
    size_t dim_{0};
    size_t amb_dim_{0};
    BigReal log_volume_;
    std::vector<double> variances_;
    std::vector<double> mean_;
};

} // namespace dbdd
