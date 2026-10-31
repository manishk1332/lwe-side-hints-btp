#pragma once

// Paper Reference: Albrecht et al. Section 2, Section 3 / docs/mathematical_conventions.md Section 3
// Mathematical Purpose: Encapsulates an ellipsoid E(mu, Sigma) = { x in R^d : (x - mu) Sigma^dagger (x - mu)^T <= 1 }.
// Enforces covariance symmetry Sigma = 1/2 (Sigma + Sigma^T) and positive semidefiniteness.
// Computes Moore-Penrose pseudoinverse Sigma^dagger with degenerate subspace support.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <vector>

namespace dbdd {

/**
 * @brief Representation of a multidimensional ellipsoid E(mu, Sigma).
 * 
 * In the DBDD framework, the secret target vector s_bar is known a priori to follow a
 * distribution with mean mu and covariance Sigma. The search space is bounded by an
 * ellipsoid centered at mu with shape determined by the Moore-Penrose pseudoinverse Sigma^dagger.
 */
class Ellipsoid {
public:
    /**
     * @brief Constructs an empty (dimension 0) ellipsoid.
     */
    Ellipsoid();

    /**
     * @brief Constructs an ellipsoid with given mean vector and covariance matrix.
     * Enforces symmetry and validates positive semidefiniteness.
     * 
     * @param mean Center mean vector mu in R^d.
     * @param covariance Covariance matrix Sigma in R^{d x d}.
     * @param psd_tolerance Tolerance for validating non-negative eigenvalues (default 1e-9).
     * @param rank_tolerance Eigenvalue cutoff for Moore-Penrose pseudoinverse (default 1e-12).
     * @throws DimensionMismatchException if dimensions do not match or covariance is non-square.
     * @throws NonPositiveSemidefiniteException if any eigenvalue is < -psd_tolerance.
     */
    Ellipsoid(
        VectorDouble mean,
        MatrixDouble covariance,
        double psd_tolerance = 1e-9,
        double rank_tolerance = 1e-12);

    /**
     * @brief Constructs an ellipsoid centered at zero with given covariance matrix.
     */
    static Ellipsoid from_covariance(
        MatrixDouble covariance,
        double psd_tolerance = 1e-9,
        double rank_tolerance = 1e-12);

    /**
     * @brief Constructs a standard spherical ellipsoid E(0, I_d).
     */
    static Ellipsoid standard(size_t dim);

    /**
     * @brief Constructs an axis-aligned ellipsoid with diagonal covariance.
     */
    static Ellipsoid diagonal(
        VectorDouble mean,
        VectorDouble variances,
        double psd_tolerance = 1e-9,
        double rank_tolerance = 1e-12);

    // Dimensions and rank
    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t rank() const noexcept { return rank_; }
    [[nodiscard]] bool empty() const noexcept { return dim_ == 0; }
    [[nodiscard]] bool is_full_rank() const noexcept { return rank_ == dim_; }
    [[nodiscard]] bool is_degenerate() const noexcept { return rank_ < dim_; }

    // Statistical & geometric accessors
    [[nodiscard]] const VectorDouble& mean() const noexcept { return mean_; }
    [[nodiscard]] const MatrixDouble& covariance() const noexcept { return covariance_; }
    [[nodiscard]] const MatrixDouble& pseudoinverse() const noexcept { return pinv_; }
    [[nodiscard]] const VectorDouble& eigenvalues() const noexcept { return eigenvalues_; }

    /**
     * @brief Computes squared Mahalanobis distance d_M^2(x) = (x - mu) Sigma^dagger (x - mu)^T.
     * @param x Point in R^d.
     * @return Squared distance value.
     */
    [[nodiscard]] double mahalanobis_distance_squared(const VectorDouble& x) const;

    /**
     * @brief Computes Mahalanobis distance d_M(x) = sqrt(d_M^2(x)).
     * @param x Point in R^d.
     * @return Distance value.
     */
    [[nodiscard]] double mahalanobis_distance(const VectorDouble& x) const;

    /**
     * @brief Checks whether point x lies in the affine support subspace mu + Im(Sigma).
     * For degenerate ellipsoids (e.g. Kannan embedding), variance along null directions is 0.
     * @param x Point in R^d.
     * @param tol Maximum allowed norm of residual in ker(Sigma).
     */
    [[nodiscard]] bool is_in_support(const VectorDouble& x, double tol = 1e-7) const;

    /**
     * @brief Projects point x orthogonally onto the affine support subspace mu + Im(Sigma).
     * @param x Point in R^d.
     * @return Point projected into support.
     */
    [[nodiscard]] VectorDouble project_onto_support(const VectorDouble& x) const;

    /**
     * @brief Computes orthogonal projector P_supp = Sigma * Sigma^dagger onto the support subspace.
     */
    [[nodiscard]] MatrixDouble support_projector() const;

    /**
     * @brief Checks whether point x is contained within the ellipsoid:
     * d_M^2(x) <= threshold (and x lies in the support subspace if check_support is true).
     * 
     * @param x Point in R^d.
     * @param threshold Squared radius threshold (default 1.0 for unit ellipsoid).
     * @param check_support If true, verifies that x - mu has no component in ker(Sigma).
     * @param support_tol Tolerance for support check.
     */
    [[nodiscard]] bool contains(
        const VectorDouble& x,
        double threshold = 1.0,
        bool check_support = true,
        double support_tol = 1e-7) const;

    /**
     * @brief Volume of the ellipsoid.
     * For full-rank ellipsoids: Vol = (pi^(d/2) / Gamma(d/2 + 1)) * sqrt(det(Sigma)).
     * For degenerate ellipsoids (rank < dim), returns 0.0 in ambient space.
     */
    [[nodiscard]] double volume() const;

    /**
     * @brief Log-volume of the ellipsoid.
     * For full-rank ellipsoids: ln Vol(E).
     * For degenerate ellipsoids: restricted log-volume on the support subspace of dimension rank.
     */
    [[nodiscard]] double log_volume() const;

    // Tolerance accessors
    [[nodiscard]] double psd_tolerance() const noexcept { return psd_tol_; }
    [[nodiscard]] double rank_tolerance() const noexcept { return rank_tol_; }

    bool operator==(const Ellipsoid& rhs) const noexcept;
    bool operator!=(const Ellipsoid& rhs) const noexcept;

private:
    void decompose_and_validate();

    size_t dim_{0};
    size_t rank_{0};
    VectorDouble mean_;
    MatrixDouble covariance_;
    MatrixDouble pinv_;
    VectorDouble eigenvalues_;
    double psd_tol_{1e-9};
    double rank_tol_{1e-12};
};

} // namespace dbdd
