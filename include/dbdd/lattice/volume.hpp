#pragma once

// Paper Reference: Dachman-Soled et al. Section 2 / docs/mathematical_conventions.md Section 1
// Mathematical Purpose: Exact, intrinsic, and log-domain lattice volume computation.
// For full-rank basis: Vol(Lambda) = |det(B)|
// For lower-rank subspace lattice: Vol(Lambda) = sqrt(det(B * B^T))
// From Gram-Schmidt: Vol(Lambda) = prod_{i=0}^{d-1} ||b_i*||
// In log domain: ln Vol(Lambda) = (1/2) * sum_{i=0}^{d-1} ln(B_i)

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include <cmath>

namespace dbdd {

/**
 * @brief Computes exact integer volume |det(B)| for a full-rank integer basis.
 * Throws DimensionMismatchException if basis is not full-rank.
 */
inline BigInt compute_volume_exact(const BasisZZ& basis) {
    if (!basis.is_full_rank()) {
        throw DimensionMismatchException("Exact integer volume requires a full-rank square basis (d == m)");
    }
    BigInt det = basis.matrix().determinant();
    return det.abs();
}

/**
 * @brief Computes exact rational volume |det(B)| for a full-rank rational basis.
 * Throws DimensionMismatchException if basis is not full-rank.
 */
inline BigRational compute_volume_exact(const BasisQQ& basis) {
    if (!basis.is_full_rank()) {
        throw DimensionMismatchException("Exact rational volume requires a full-rank square basis (d == m)");
    }
    BigRational det = basis.matrix().determinant();
    return det.abs();
}

/**
 * @brief Computes log-volume from a Gram-Schmidt state:
 * ln Vol(Lambda) = (1/2) * sum_{i=0}^{d-1} ln(B_i)
 */
template <typename T>
BigReal compute_log_volume_from_gs(const GramSchmidtState<T>& gs) {
    size_t d = gs.dimension();
    if (d == 0) {
        return BigReal(0.0);
    }
    BigReal sum_log(0.0);
    for (size_t i = 0; i < d; ++i) {
        const T& Bi = gs.bstar_norm_sq[i];
        if constexpr (std::is_same_v<T, BigReal>) {
            sum_log += Bi.log();
        } else if constexpr (std::is_same_v<T, BigRational> || std::is_same_v<T, BigInt>) {
            sum_log += Bi.to_mpfr().log();
        } else {
            sum_log += BigReal(std::log(static_cast<double>(Bi)));
        }
    }
    return sum_log * BigReal(0.5);
}

/**
 * @brief Computes volume from a Gram-Schmidt state:
 * Vol(Lambda) = prod_{i=0}^{d-1} sqrt(B_i) = exp(ln Vol(Lambda))
 */
template <typename T>
BigReal compute_volume_from_gs(const GramSchmidtState<T>& gs) {
    BigReal log_vol = compute_log_volume_from_gs(gs);
    return log_vol.exp();
}

/**
 * @brief Computes log-volume of a lattice basis in the log domain:
 * ln Vol(Lambda) = (1/2) * sum ln(B_i) via Gram-Schmidt.
 */
template <typename T>
BigReal compute_log_volume(const Basis<T>& basis) {
    if (basis.empty()) {
        return BigReal(0.0);
    }
    if constexpr (std::is_same_v<T, BigReal> || std::is_same_v<T, double>) {
        auto gs = compute_gram_schmidt_mgs(basis.matrix());
        return compute_log_volume_from_gs(gs);
    } else {
        auto gs = compute_gram_schmidt_exact(basis.matrix().template cast<BigRational>());
        return compute_log_volume_from_gs(gs);
    }
}

/**
 * @brief Computes intrinsic volume for any basis (full rank or subspace):
 * Vol(Lambda) = sqrt(det(B * B^T))
 */
template <typename T>
BigReal compute_volume(const Basis<T>& basis) {
    if (basis.empty()) {
        return BigReal(1.0);
    }
    if (basis.is_full_rank()) {
        if constexpr (std::is_same_v<T, BigInt>) {
            return BigReal(compute_volume_exact(basis));
        } else if constexpr (std::is_same_v<T, BigRational>) {
            return compute_volume_exact(basis).to_mpfr();
        }
    }
    // Subspace or general floating basis:
    // Gram matrix G = B * B^T is (d x d) symmetric positive definite
    auto G = basis.matrix() * basis.matrix().transpose();
    if constexpr (std::is_same_v<T, BigInt> || std::is_same_v<T, BigRational>) {
        auto G_qq = G.template cast<BigRational>();
        BigRational det = G_qq.determinant();
        return det.to_mpfr().sqrt();
    } else {
        auto G_rr = G.template cast<BigReal>();
        BigReal det = G_rr.determinant();
        return det.sqrt();
    }
}

// =============================================================================
// Volume Shortcut for Isotropization (Phase 12 / Dachman-Soled et al. Section 3.3)
// =============================================================================

/**
 * @brief Computes restricted determinant rdet(Sigma_h) = prod_{lambda_i > rank_tol} lambda_i.
 */
BigReal compute_restricted_determinant(const MatrixDouble& cov, double rank_tol = 1e-12);

/**
 * @brief Computes log restricted determinant ln rdet(Sigma_h) = sum_{lambda_i > rank_tol} ln(lambda_i).
 */
BigReal compute_log_restricted_determinant(const MatrixDouble& cov, double rank_tol = 1e-12);

/**
 * @brief Computes isotropized lattice volume via the shortcut formula:
 * Vol(Lambda_iso) = Vol(Lambda) / sqrt(rdet(Sigma_h))
 */
BigReal compute_isotropized_volume(
    const BigReal& original_volume,
    const MatrixDouble& cov,
    double rank_tol = 1e-12);

/**
 * @brief Computes isotropized lattice log-volume via the log-domain shortcut:
 * ln Vol(Lambda_iso) = ln Vol(Lambda) - (1/2) * ln rdet(Sigma_h)
 */
BigReal compute_isotropized_log_volume(
    const BigReal& original_log_volume,
    const MatrixDouble& cov,
    double rank_tol = 1e-12);

} // namespace dbdd
