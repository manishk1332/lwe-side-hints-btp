#pragma once

// Paper Reference: Dachman-Soled et al. Section 2, Section 3.1, Section 3.2 / docs/mathematical_conventions.md Section 8
// Mathematical Purpose: Explicit dual lattice basis construction and primal-dual involution.
// For basis B in Q^{d x m}:
//     Dual basis D in Q^{d x m} satisfies:
//         B * D^T = I_d
//         rows of D lie in span(Lambda)
// Formula:
//     D = (B * B^T)^{-1} * B
// For full-rank (d == m): D = (B^{-1})^T
// Involution:
//     primal_from_dual(D) = (D * D^T)^{-1} * D = B
// Invariants:
//     Vol(Lambda*) = 1 / Vol(Lambda)

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"

namespace dbdd {

/**
 * @brief Computes the dual basis matrix D for basis matrix B in rational arithmetic.
 * Satisfies B * D^T = I_d.
 * 
 * @param B Full-rank or subspace basis matrix (d x m)
 * @return MatrixQQ Dual basis matrix D (d x m)
 */
inline MatrixQQ compute_dual_basis(const MatrixQQ& B) {
    if (B.empty()) {
        return MatrixQQ(0, 0);
    }
    size_t d = B.rows();
    size_t m = B.cols();
    if (d > m) {
        throw DimensionMismatchException("Basis rank cannot exceed ambient dimension");
    }

    if (d == m) {
        // Full rank shortcut: D = (B^{-1})^T
        return B.inverse().transpose();
    } else {
        // Subspace dual in span(Lambda): D = (B * B^T)^{-1} * B
        MatrixQQ G = B * B.transpose(); // (d x d)
        MatrixQQ G_inv = G.inverse();    // (d x d)
        return G_inv * B;                // (d x m)
    }
}

/**
 * @brief Computes the dual basis matrix D for an integer basis matrix B.
 */
inline MatrixQQ compute_dual_basis(const MatrixZZ& B) {
    return compute_dual_basis(B.cast<BigRational>());
}

/**
 * @brief Computes the dual basis for any Basis<T> in rational arithmetic.
 */
template <typename T>
BasisQQ dual_basis(const Basis<T>& basis) {
    MatrixQQ D = compute_dual_basis(basis.matrix().template cast<BigRational>());
    return BasisQQ(std::move(D), false);
}

/**
 * @brief Reconstructs primal basis B from dual basis D:
 * B = (D * D^T)^{-1} * D.
 * Satisfies primal_from_dual(dual_basis(B)) == B.
 */
inline BasisQQ primal_from_dual(const BasisQQ& dual_basis) {
    const MatrixQQ& D = dual_basis.matrix();
    if (D.empty()) {
        return BasisQQ();
    }
    size_t d = D.rows();
    size_t m = D.cols();

    if (d == m) {
        return BasisQQ(D.inverse().transpose(), false);
    } else {
        MatrixQQ G = D * D.transpose();
        MatrixQQ G_inv = G.inverse();
        return BasisQQ(G_inv * D, false);
    }
}

} // namespace dbdd
