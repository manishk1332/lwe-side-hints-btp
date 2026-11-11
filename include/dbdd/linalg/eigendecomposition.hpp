#pragma once

// Paper Reference: Dachman-Soled et al. Section 2, Section 3.3 / docs/mathematical_conventions.md Section 6
// Mathematical Purpose: Spectral decomposition of symmetric positive-semidefinite matrices Sigma_h,
// matrix square root, pseudoinverse square root M = (Sigma_h^{1/2})^dagger, and multi-precision Jacobi solver.

#include "dbdd/core/types.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <vector>

namespace dbdd {

/**
 * @brief Spectral decomposition of symmetric matrices using double precision (Eigen backend).
 * Represents Sigma = Q * Lambda_D * Q^T with eigenvalues sorted in descending order.
 */
class SymmetricEigendecomposition {
public:
    /**
     * @brief Computes spectral decomposition of a symmetric matrix.
     * Enforces symmetry A <- 0.5 * (A + A^T).
     * @param A Square symmetric matrix in R^{d x d}.
     * @param rank_tol Tolerance threshold for considering an eigenvalue positive (default 1e-12).
     */
    static SymmetricEigendecomposition decompose(
        const MatrixDouble& A,
        double rank_tol = 1e-12);

    SymmetricEigendecomposition() = default;

    // Dimensions and rank
    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t rank() const noexcept { return rank_; }
    [[nodiscard]] double rank_tolerance() const noexcept { return rank_tol_; }
    [[nodiscard]] bool is_full_rank() const noexcept { return rank_ == dim_; }
    [[nodiscard]] bool empty() const noexcept { return dim_ == 0; }

    // Spectral components
    [[nodiscard]] const VectorDouble& eigenvalues() const noexcept { return eigenvalues_; }
    [[nodiscard]] const MatrixDouble& eigenvectors() const noexcept { return eigenvectors_; }

    // Matrix functions
    [[nodiscard]] MatrixDouble matrix_sqrt() const;
    [[nodiscard]] MatrixDouble matrix_inv_sqrt() const;
    [[nodiscard]] MatrixDouble projector() const;

    // Determinants on active support
    [[nodiscard]] BigReal restricted_determinant() const;
    [[nodiscard]] BigReal log_restricted_determinant() const;

private:
    size_t dim_{0};
    size_t rank_{0};
    double rank_tol_{1e-12};
    VectorDouble eigenvalues_;
    MatrixDouble eigenvectors_;
};

/**
 * @brief High-precision MPFR spectral decomposition of symmetric matrices using the cyclic Jacobi method.
 * Guarantees numerical stability and orthogonality across arbitrary precision (128, 256, 512 bits).
 */
class SymmetricEigendecompositionRR {
public:
    static SymmetricEigendecompositionRR decompose(
        const MatrixRR& A,
        double rank_tol = 1e-12,
        size_t max_sweeps = 100);

    SymmetricEigendecompositionRR() = default;

    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t rank() const noexcept { return rank_; }
    [[nodiscard]] double rank_tolerance() const noexcept { return rank_tol_; }

    [[nodiscard]] const VectorRR& eigenvalues() const noexcept { return eigenvalues_; }
    [[nodiscard]] const MatrixRR& eigenvectors() const noexcept { return eigenvectors_; }

    [[nodiscard]] MatrixRR matrix_sqrt() const;
    [[nodiscard]] MatrixRR matrix_inv_sqrt() const;

    [[nodiscard]] BigReal restricted_determinant() const;
    [[nodiscard]] BigReal log_restricted_determinant() const;

private:
    size_t dim_{0};
    size_t rank_{0};
    double rank_tol_{1e-12};
    VectorRR eigenvalues_;
    MatrixRR eigenvectors_;
};

} // namespace dbdd
