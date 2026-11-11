#pragma once

// Paper Reference: Lenstra, Lenstra, Lovász (1982) / Dachman-Soled et al. Section 2
// Mathematical Purpose: Generic LLL parameters, reduction result container,
// and rigorous invariant verification functions (unimodularity, volume, size-reduction,
// Lovász condition, Hermite bound).

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include <optional>
#include <cmath>

namespace dbdd {

/**
 * @brief Parameters for LLL lattice reduction.
 */
struct LLLParams {
    BigRational delta{3, 4};               // Lovász reduction parameter delta in (1/4, 1]
    BigRational eta{1, 2};                 // Size-reduction parameter eta in [1/2, 1)
    size_t max_iterations{0};              // Safety bound (0 = unlimited)
    prec_t precision{256};                 // MPFR precision in bits for floating reduction
    size_t refresh_interval{50};           // Swap count threshold for MGS refresh in LLLFloat

    LLLParams() = default;

    explicit LLLParams(double delta_val, double eta_val = 0.5, size_t max_iter = 0,
                       prec_t prec = 256, size_t refresh = 50)
        : delta(BigReal(delta_val).to_bigrational()),
          eta(BigReal(eta_val).to_bigrational()),
          max_iterations(max_iter),
          precision(prec),
          refresh_interval(refresh) {
        validate();
    }

    explicit LLLParams(BigRational delta_val, BigRational eta_val = BigRational(1, 2),
                       size_t max_iter = 0, prec_t prec = 256, size_t refresh = 50)
        : delta(std::move(delta_val)),
          eta(std::move(eta_val)),
          max_iterations(max_iter),
          precision(prec),
          refresh_interval(refresh) {
        validate();
    }

    void validate() const {
        if (delta <= BigRational(1, 4) || delta > BigRational(1)) {
            throw InvalidParameterException("LLL delta must be in (1/4, 1], got " + delta.to_string());
        }
        if (eta < BigRational(1, 2) || eta >= BigRational(1)) {
            throw InvalidParameterException("LLL eta must be in [1/2, 1), got " + eta.to_string());
        }
    }
};

/**
 * @brief Result of an LLL reduction.
 * 
 * Contains the reduced basis, unimodular transformation matrix U in GL_d(Z)
 * such that B' = U * B, and reduction profiling statistics.
 */
template <typename T>
struct LLLReductionResult {
    Basis<T> basis;                                     // Reduced basis B'
    MatrixZZ transformation;                            // Unimodular matrix U in GL_d(Z) such that B' = U * B
    size_t iterations{0};                               // Total outer loop iterations
    size_t swaps{0};                                    // Total row swaps performed
    bool converged{true};                               // Whether reduction completed within max_iterations
    std::optional<GramSchmidtStateQQ> exact_gs_state;  // Cached rational Gram-Schmidt state
};

// ============================================================================
// Invariant Verification Helpers
// ============================================================================

/**
 * @brief Verifies whether the Gram-Schmidt state is size-reduced:
 * |mu_{i,j}| <= eta for all 0 <= j < i < d.
 */
template <typename T>
bool is_size_reduced(const GramSchmidtState<T>& state, const T& eta) {
    size_t d = state.dimension();
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if constexpr (std::is_same_v<T, BigRational>) {
                if (state.mu(i, j).abs() > eta) {
                    return false;
                }
            } else if constexpr (std::is_same_v<T, BigReal>) {
                if (state.mu(i, j).abs() > eta + BigReal(1e-12)) {
                    return false;
                }
            } else {
                if (std::abs(state.mu(i, j)) > eta + 1e-9) {
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool is_size_reduced(const GramSchmidtStateQQ& state, const BigRational& eta = BigRational(1, 2)) {
    return is_size_reduced<BigRational>(state, eta);
}

/**
 * @brief Verifies whether the Gram-Schmidt state satisfies the Lovász condition:
 * ||b_{k+1}*||^2 >= (delta - mu_{k+1,k}^2) ||b_k*||^2 for all 0 <= k < d - 1.
 */
template <typename T>
bool is_lovasz_satisfied(const GramSchmidtState<T>& state, const T& delta) {
    size_t d = state.dimension();
    if (d <= 1) return true;
    for (size_t k = 0; k < d - 1; ++k) {
        const T& Bk = state.bstar_norm_sq[k];
        const T& Bk1 = state.bstar_norm_sq[k + 1];
        const T mu = state.mu(k + 1, k);
        T rhs = (delta - mu * mu) * Bk;

        if constexpr (std::is_same_v<T, BigRational>) {
            if (Bk1 < rhs) {
                return false;
            }
        } else if constexpr (std::is_same_v<T, BigReal>) {
            if (Bk1 < rhs - BigReal(1e-12)) {
                return false;
            }
        } else {
            if (Bk1 < rhs - 1e-9) {
                return false;
            }
        }
    }
    return true;
}

inline bool is_lovasz_satisfied(const GramSchmidtStateQQ& state, const BigRational& delta = BigRational(3, 4)) {
    return is_lovasz_satisfied<BigRational>(state, delta);
}

/**
 * @brief Checks if a Gram-Schmidt state is fully LLL-reduced.
 */
template <typename T>
bool is_lll_reduced(const GramSchmidtState<T>& state, const T& delta, const T& eta) {
    return is_size_reduced<T>(state, eta) && is_lovasz_satisfied<T>(state, delta);
}

/**
 * @brief Checks if an integer basis is LLL-reduced using exact Gram-Schmidt.
 */
inline bool is_lll_reduced(const BasisZZ& basis,
                           const BigRational& delta = BigRational(3, 4),
                           const BigRational& eta = BigRational(1, 2)) {
    if (basis.dimension() <= 1) return true;
    auto gs = compute_gram_schmidt_exact(basis.matrix());
    return is_lll_reduced(gs, delta, eta);
}

/**
 * @brief Checks if a rational basis is LLL-reduced using exact Gram-Schmidt.
 */
inline bool is_lll_reduced(const BasisQQ& basis,
                           const BigRational& delta = BigRational(3, 4),
                           const BigRational& eta = BigRational(1, 2)) {
    if (basis.dimension() <= 1) return true;
    auto gs = compute_gram_schmidt_exact(basis.matrix());
    return is_lll_reduced(gs, delta, eta);
}

/**
 * @brief Verifies unimodular transformation invariants:
 * 1. B' = U * B
 * 2. det(U) = +/- 1
 */
inline bool verify_unimodular_transformation(const BasisZZ& original,
                                            const BasisZZ& reduced,
                                            const MatrixZZ& U) {
    size_t d = original.dimension();
    if (U.rows() != d || U.cols() != d) return false;
    if (U * original.matrix() != reduced.matrix()) return false;
    BigInt det = U.determinant();
    return (det == BigInt(1) || det == BigInt(-1));
}

inline bool verify_unimodular_transformation(const BasisQQ& original,
                                            const BasisQQ& reduced,
                                            const MatrixZZ& U) {
    size_t d = original.dimension();
    if (U.rows() != d || U.cols() != d) return false;
    if (U.template cast<BigRational>() * original.matrix() != reduced.matrix()) return false;
    BigInt det = U.determinant();
    return (det == BigInt(1) || det == BigInt(-1));
}

/**
 * @brief Verifies volume invariance: Vol(B') == Vol(B).
 */
inline bool verify_volume_invariance(const BasisZZ& original, const BasisZZ& reduced) {
    return compute_volume_exact(original) == compute_volume_exact(reduced);
}

inline bool verify_volume_invariance(const BasisQQ& original, const BasisQQ& reduced) {
    return compute_volume_exact(original) == compute_volume_exact(reduced);
}

/**
 * @brief Verifies the Hermite bound:
 * ||b_0|| <= (2 / sqrt(4 delta - 1))^{d - 1} * Vol(Lambda)^{1/d}.
 * 
 * Computed strictly in the log domain to prevent numerical overflow:
 * ln(||b_0||) <= (d - 1) * (ln(2) - 0.5 * ln(4 delta - 1)) + (1/d) * ln(Vol).
 */
inline bool verify_hermite_bound(const BasisZZ& reduced,
                                 const BigRational& delta = BigRational(3, 4)) {
    size_t d = reduced.dimension();
    if (d <= 1) return true;

    BigInt norm_sq_b0 = reduced[0].squared_norm();
    if (norm_sq_b0 == BigInt(0)) return false;

    BigReal b0_log_norm = BigReal(norm_sq_b0).log() / BigReal(2.0);
    BigReal vol = compute_volume(reduced);
    if (vol <= BigReal(0.0)) return false;

    BigReal log_vol = vol.log();
    BigReal delta_real = delta.to_mpfr();
    BigReal term = BigReal(4.0) * delta_real - BigReal(1.0);
    if (term <= BigReal(0.0)) return false;

    BigReal log_alpha_half = BigReal(2.0).log() - term.log() / BigReal(2.0);
    BigReal bound_log = BigReal(static_cast<int64_t>(d - 1)) * log_alpha_half +
                        log_vol / BigReal(static_cast<int64_t>(d));

    // Allow tiny numerical slack for floating log comparisons
    return b0_log_norm <= bound_log + BigReal(1e-9);
}

} // namespace dbdd
