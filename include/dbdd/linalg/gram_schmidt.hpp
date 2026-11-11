#pragma once

// Paper Reference: Dachman-Soled et al. Section 2 / docs/algorithm_notes.md Section 7
// Mathematical Purpose: Gram-Schmidt orthogonalization (classical, modified, and exact rational)
// and state maintenance with O(m + d) local update on adjacent vector swaps.
//
// Invariants guaranteed:
//     <b_i*, b_j*> = 0 for i != j (Orthogonality)
//     b_i = b_i* + sum_{j < i} mu_{i,j} b_j* (Triangular reconstruction)
//     ||b_i*||^2 = <b_i*, b_i*> = B_i (Squared norm consistency)

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <vector>
#include <cmath>
#include <type_traits>

namespace dbdd {

/**
 * @brief Gram-Schmidt State storing orthogonalized vectors, coefficients, and squared norms.
 * 
 * Reused across LLL, BKZ, SVP enumeration, and security estimators.
 */
template <typename T>
struct GramSchmidtState {
    Matrix<T> basis;               // Original basis B (d x m)
    Matrix<T> mu;                  // Lower triangular coefficient matrix (d x d), mu[i][i] = 1
    std::vector<Vector<T>> bstar;  // Orthogonalized vectors b_0*, ..., b_{d-1}*
    std::vector<T> bstar_norm_sq;  // Squared norms B_i = ||b_i*||^2

    [[nodiscard]] size_t dimension() const noexcept { return basis.rows(); }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return basis.cols(); }
};

using GramSchmidtStateZZ = GramSchmidtState<BigInt>;
using GramSchmidtStateQQ = GramSchmidtState<BigRational>;
using GramSchmidtStateRR = GramSchmidtState<BigReal>;
using GramSchmidtStateDouble = GramSchmidtState<double>;

/**
 * @brief Classical Gram-Schmidt (CGS) reference implementation.
 */
template <typename T>
GramSchmidtState<T> compute_gram_schmidt_cgs(const Matrix<T>& B) {
    size_t d = B.rows();
    size_t m = B.cols();
    if (d > m) {
        throw DimensionMismatchException("Basis rank cannot exceed ambient dimension");
    }

    GramSchmidtState<T> state;
    state.basis = B;
    state.mu = Matrix<T>::identity(d);
    state.bstar.reserve(d);
    state.bstar_norm_sq.reserve(d);

    for (size_t i = 0; i < d; ++i) {
        Vector<T> bi = B.row(i);
        Vector<T> bi_star = bi;

        for (size_t j = 0; j < i; ++j) {
            T dot_prod = bi.dot(state.bstar[j]);
            const T& Bj = state.bstar_norm_sq[j];
            if (Bj == T{}) {
                throw SingularMatrixException("Linearly dependent vectors encountered during Gram-Schmidt");
            }
            T mu_ij = dot_prod / Bj;
            state.mu(i, j) = mu_ij;
            bi_star -= mu_ij * state.bstar[j];
        }

        T norm_sq = bi_star.squared_norm();
        if (norm_sq == T{}) {
            throw SingularMatrixException("Basis row is linearly dependent on preceding rows");
        }

        state.bstar.push_back(bi_star);
        state.bstar_norm_sq.push_back(norm_sq);
    }

    return state;
}

/**
 * @brief Modified Gram-Schmidt (MGS) implementation.
 * 
 * Numerically superior to CGS in floating-point domains by preventing catastrophic cancellation.
 */
template <typename T>
GramSchmidtState<T> compute_gram_schmidt_mgs(const Matrix<T>& B) {
    size_t d = B.rows();
    size_t m = B.cols();
    if (d > m) {
        throw DimensionMismatchException("Basis rank cannot exceed ambient dimension");
    }

    GramSchmidtState<T> state;
    state.basis = B;
    state.mu = Matrix<T>::identity(d);
    state.bstar.resize(d);
    state.bstar_norm_sq.resize(d);

    // Copy original rows into bstar
    for (size_t i = 0; i < d; ++i) {
        state.bstar[i] = B.row(i);
    }

    for (size_t j = 0; j < d; ++j) {
        T Bj = state.bstar[j].squared_norm();
        if (Bj == T{}) {
            throw SingularMatrixException("Linearly dependent vector in Modified Gram-Schmidt");
        }
        state.bstar_norm_sq[j] = Bj;

        for (size_t i = j + 1; i < d; ++i) {
            // In MGS: mu_ij = <b_i, b_j*> / ||b_j*||^2 where b_i is the original row
            T dot_prod = B.row(i).dot(state.bstar[j]);
            T mu_ij = dot_prod / Bj;
            state.mu(i, j) = mu_ij;
            state.bstar[i] -= mu_ij * state.bstar[j];
        }
    }

    return state;
}

/**
 * @brief Exact rational Gram-Schmidt computation for exact lattices.
 */
inline GramSchmidtStateQQ compute_gram_schmidt_exact(const MatrixQQ& B) {
    return compute_gram_schmidt_mgs<BigRational>(B);
}

inline GramSchmidtStateQQ compute_gram_schmidt_exact(const MatrixZZ& B) {
    return compute_gram_schmidt_mgs<BigRational>(B.cast<BigRational>());
}

/**
 * @brief Performs an O(m + d) local Gram-Schmidt update when rows k and k+1 are swapped.
 * 
 * Lovász / Cohen Alg. 2.6.3 update:
 *   B_k' = B_{k+1} + mu^2 * B_k
 *   t = B_k / B_k'
 *   mu' = mu * t
 *   (b_k*)' = b_{k+1}* + mu * b_k*
 *   (b_{k+1}*)' = b_k* * (B_{k+1} / B_k') - mu' * b_{k+1}*
 *   B_{k+1}' = B_k * B_{k+1} / B_k'
 */
template <typename T>
void update_gram_schmidt_swap(GramSchmidtState<T>& state, size_t k) {
    size_t d = state.dimension();
    if (k + 1 >= d) {
        throw IndexOutOfBoundsException("Swap index out of range in update_gram_schmidt_swap");
    }

    const T mu = state.mu(k + 1, k);
    const T Bk = state.bstar_norm_sq[k];
    const T Bk1 = state.bstar_norm_sq[k + 1];

    const T Bk_new = Bk1 + mu * mu * Bk;
    if (Bk_new == T{}) {
        throw NumericalPrecisionException("Division by zero in Gram-Schmidt swap update");
    }

    const T t = Bk / Bk_new;
    const T mu_prime = mu * t;

    // New orthogonal vectors
    Vector<T> bk_star_new = state.bstar[k + 1] + mu * state.bstar[k];
    Vector<T> bk1_star_new = state.bstar[k] * (Bk1 / Bk_new) - state.bstar[k + 1] * mu_prime;

    const T Bk1_new = (Bk * Bk1) / Bk_new;

    // Update state for indices k and k + 1
    state.basis.swap_rows(k, k + 1);
    state.bstar[k] = std::move(bk_star_new);
    state.bstar[k + 1] = std::move(bk1_star_new);
    state.bstar_norm_sq[k] = Bk_new;
    state.bstar_norm_sq[k + 1] = Bk1_new;
    state.mu(k + 1, k) = mu_prime;

    // Swap columns j < k
    for (size_t j = 0; j < k; ++j) {
        std::swap(state.mu(k, j), state.mu(k + 1, j));
    }

    // Update rows i > k + 1
    for (size_t i = k + 2; i < d; ++i) {
        const T mu_ik = state.mu(i, k);
        const T mu_ik1 = state.mu(i, k + 1);

        state.mu(i, k) = mu_ik * mu_prime + mu_ik1 * (Bk1 / Bk_new);
        state.mu(i, k + 1) = mu_ik - mu_ik1 * mu;
    }
}

/**
 * @brief Updates Gram-Schmidt state after elementary row operation:
 * b_i <- b_i - factor * b_j for j < i.
 * b_k* and B_k are invariant; only mu is updated.
 */
template <typename T>
void update_gram_schmidt_row_add(GramSchmidtState<T>& state, size_t i, size_t j, const T& factor) {
    if (j >= i || i >= state.dimension()) {
        throw IndexOutOfBoundsException("Invalid indices for Gram-Schmidt row addition");
    }

    state.basis.row_add_multiple(i, j, -factor);
    state.mu(i, j) -= factor;
    for (size_t l = 0; l < j; ++l) {
        state.mu(i, l) -= factor * state.mu(j, l);
    }
}

} // namespace dbdd
