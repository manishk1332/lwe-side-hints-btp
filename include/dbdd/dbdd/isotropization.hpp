#pragma once

// Paper Reference: Albrecht et al. Section 2, Section 3.3 / docs/mathematical_conventions.md Section 6
// Mathematical Purpose: Isotropization of homogenized DBDD instances.
// Computes transformation matrix M = (Sigma_h^{1/2})^dagger and converts the instance into an
// isotropic uSVP instance with normalized covariance M * Sigma_h * M = I (or projector).

#include "dbdd/core/types.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"

namespace dbdd {

class DBDDInstance;

class Isotropization {
public:
    /**
     * @brief Computes transformation matrix M = (Sigma_h^{1/2})^dagger.
     * @param cov_h Homogenized covariance matrix in R^{d x d} (symmetric PSD).
     * @param rank_tol Tolerance threshold for non-zero eigenvalues.
     * @return Symmetric matrix M in R^{d x d}.
     */
    static MatrixDouble compute_transform(
        const MatrixDouble& cov_h,
        double rank_tol = 1e-12);

    /**
     * @brief Computes matrix square root Sigma_h^{1/2}.
     */
    static MatrixDouble compute_matrix_sqrt(
        const MatrixDouble& cov_h,
        double rank_tol = 1e-12);

    /**
     * @brief Computes transformed covariance Sigma' = M^T * Sigma_h * M = M * Sigma_h * M.
     * Guaranteed to be identity on the active span, or orthogonal projector if singular.
     */
    static MatrixDouble compute_isotropized_covariance(
        const MatrixDouble& cov_h,
        const MatrixDouble& M);

    /**
     * @brief Fully converts a DBDDInstance into an isotropic UsvpInstance.
     * Automatically homogenizes the instance if it is not already centered.
     */
    static UsvpInstance isotropize(
        const DBDDInstance& instance,
        double rank_tol = 1e-12);

    /**
     * @brief Applies isotropization in-place to the covariance and mean of a DBDDInstance.
     * Covariance becomes identity on active span, mean remains 0.
     */
    static void apply(
        DBDDInstance& instance,
        double rank_tol = 1e-12);
};

} // namespace dbdd
