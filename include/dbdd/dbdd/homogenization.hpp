#pragma once

// Paper Reference: Dachman-Soled et al. Section 2 / Equation (8) / docs/mathematical_conventions.md Section 5
// Mathematical Purpose: Homogenization of the DBDD target distribution.
// Embeds the affine target ellipsoid E(mu, Sigma) into an origin-centered ellipsoid envelope E(0, Sigma_h),
// where Sigma_h = Sigma + mu^T * mu.

#include "dbdd/core/types.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/dbdd/ellipsoid.hpp"
#include <utility>

namespace dbdd {

class DBDDInstance;

class Homogenization {
public:
    /**
     * @brief Computes homogenized distribution: mu_h = 0 and Sigma_h = Sigma + mu^T * mu.
     * @param mean Prior mean row vector mu in R^d.
     * @param cov Prior covariance matrix Sigma in R^{d x d} (symmetric PSD).
     * @return std::pair<VectorDouble, MatrixDouble> {mu_h, Sigma_h}.
     */
    static std::pair<VectorDouble, MatrixDouble> homogenize(
        const VectorDouble& mean,
        const MatrixDouble& cov);

    /**
     * @brief Constructs the homogenized origin-centered ellipsoid E(0, Sigma + mu^T * mu).
     */
    static Ellipsoid homogenize(const Ellipsoid& ell);

    /**
     * @brief Applies homogenization in-place to a DBDD instance.
     * Updates mean to 0 and covariance to Sigma + mu^T * mu while strictly
     * preserving the lattice basis, volume, dimension, and target vector.
     */
    static void apply(DBDDInstance& instance);

    /**
     * @brief Numerically verifies the inclusion E(mu, Sigma) subset { x : x Sigma_h^dagger x^T <= threshold }
     * by sampling boundary points of orig_ell and testing their containment in homog_ell.
     * By the Sherman-Morrison theorem and Kannan embedding geometry, for any point in E(mu, Sigma),
     * its homogenized Mahalanobis distance squared satisfies x Sigma_h^dagger x^T <= 2.0.
     * @param orig_ell Original affine ellipsoid E(mu, Sigma).
     * @param homog_ell Homogenized origin-centered ellipsoid E(0, Sigma_h).
     * @param threshold Maximum allowed Mahalanobis distance squared (default: 2.0).
     * @param num_samples Number of random spherical boundary samples (default: 50).
     */
    static bool verify_inclusion(
        const Ellipsoid& orig_ell,
        const Ellipsoid& homog_ell,
        double threshold = 2.0,
        size_t num_samples = 50);
};

} // namespace dbdd
