#include "dbdd/reduction/lll_float.hpp"
#include <cmath>

namespace dbdd {

// Paper Reference: Lenstra, Lenstra, Lovász (1982) / Dachman-Soled et al. Section 2
// Mathematical Purpose: Floating-point LLL reduction using MPFR (BigReal) and double.
// Inputs:
//     basis: Input integer lattice basis B in Z^{d x m} (row-basis convention)
// Outputs:
//     LLLReductionResult containing reduced basis B' and unimodular matrix U in GL_d(Z)
// Preconditions:
//     Basis is full rank in its span (d <= m, linearly independent rows)
// Postconditions:
//     B' = U * B with det(U) = +/- 1
//     Vol(B') = Vol(B)
//     Lattice equivalence preserved exactly in integer domain
// Precision & Complexity:
//     High precision MPFR (128-512 bits) or IEEE double (53 bits)

LLLFloat::LLLFloat(LLLParams params) : params_(std::move(params)) {
    params_.validate();
}

template <typename FloatT>
LLLReductionResult<BigInt> LLLFloat::reduce_impl(const BasisZZ& basis, prec_t prec) const {
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();

    LLLReductionResult<BigInt> res;
    if (d == 0) {
        res.basis = BasisZZ();
        res.transformation = MatrixZZ(0, 0);
        res.converged = true;
        return res;
    }
    if (d == 1) {
        res.basis = basis;
        res.transformation = MatrixZZ::identity(1);
        res.converged = true;
        return res;
    }

    if (d > m) {
        throw DimensionMismatchException("Basis rank cannot exceed ambient dimension (d > m)");
    }

    BasisZZ current_basis = basis;
    MatrixZZ U = MatrixZZ::identity(d);

    auto to_float_matrix = [&](const BasisZZ& b) -> Matrix<FloatT> {
        Matrix<FloatT> M(d, m);
        for (size_t r = 0; r < d; ++r) {
            for (size_t c = 0; c < m; ++c) {
                if constexpr (std::is_same_v<FloatT, BigReal>) {
                    M(r, c) = BigReal(b.matrix()(r, c), prec);
                } else {
                    M(r, c) = b.matrix()(r, c).to_double();
                }
            }
        }
        return M;
    };

    auto to_float_val = [&](const BigInt& val) -> FloatT {
        if constexpr (std::is_same_v<FloatT, BigReal>) {
            return BigReal(val, prec);
        } else {
            return val.to_double();
        }
    };

    auto round_float = [](const FloatT& val) -> BigInt {
        if constexpr (std::is_same_v<FloatT, BigReal>) {
            return val.round().to_bigint();
        } else {
            return BigInt(std::to_string(std::llround(val)));
        }
    };

    FloatT delta_f{};
    FloatT eta_f{};
    if constexpr (std::is_same_v<FloatT, BigReal>) {
        delta_f = params_.delta.to_mpfr(prec);
        eta_f = params_.eta.to_mpfr(prec);
    } else {
        delta_f = params_.delta.to_double();
        eta_f = params_.eta.to_double();
    }

    GramSchmidtState<FloatT> state = compute_gram_schmidt_mgs<FloatT>(to_float_matrix(current_basis));

    const size_t max_iter = params_.max_iterations;
    const size_t refresh_interval = params_.refresh_interval;
    size_t swaps_since_refresh = 0;

    auto size_reduce_step = [&](size_t k, size_t j) {
        const FloatT& mu_kj = state.mu(k, j);
        bool needs_reduction = false;
        if constexpr (std::is_same_v<FloatT, BigReal>) {
            needs_reduction = (mu_kj.abs() > eta_f);
        } else {
            needs_reduction = (std::abs(mu_kj) > eta_f);
        }

        if (needs_reduction) {
            BigInt q = round_float(mu_kj);
            if (q != BigInt(0)) {
                FloatT q_f = to_float_val(q);
                current_basis.row_add_multiple(k, j, -q);
                U.row_add_multiple(k, j, -q);
                update_gram_schmidt_row_add(state, k, j, q_f);
            }
        }
    };

    size_t k = 1;
    bool in_verification = false;

    while (k < d) {
        res.iterations++;
        if (max_iter > 0 && res.iterations > max_iter) {
            res.converged = false;
            break;
        }

        // Step 1: Size-reduce b_k against b_{k-1}
        size_reduce_step(k, k - 1);

        // Step 2: Check Lovász condition
        const FloatT& Bk = state.bstar_norm_sq[k];
        const FloatT& Bk1 = state.bstar_norm_sq[k - 1];
        const FloatT mu_k_k1 = state.mu(k, k - 1);
        FloatT threshold = (delta_f - mu_k_k1 * mu_k_k1) * Bk1;

        bool lovasz_ok = false;
        if constexpr (std::is_same_v<FloatT, BigReal>) {
            lovasz_ok = (Bk >= threshold - BigReal(1e-15, prec));
        } else {
            lovasz_ok = (Bk >= threshold - 1e-12);
        }

        if (lovasz_ok) {
            // Size-reduce b_k against b_{k-2}, ..., b_0
            for (size_t j = k - 1; j-- > 0; ) {
                size_reduce_step(k, j);
            }
            k++;
        } else {
            // Swap rows k-1 and k
            current_basis.swap_rows(k - 1, k);
            U.swap_rows(k - 1, k);
            update_gram_schmidt_swap(state, k - 1);
            res.swaps++;
            swaps_since_refresh++;

            if (refresh_interval > 0 && swaps_since_refresh >= refresh_interval) {
                state = compute_gram_schmidt_mgs<FloatT>(to_float_matrix(current_basis));
                swaps_since_refresh = 0;
            }

            k = (k > 1) ? (k - 1) : 1;
        }

        // When reaching k == d, perform a verification with freshly computed MGS state
        if (k == d && !in_verification) {
            state = compute_gram_schmidt_mgs<FloatT>(to_float_matrix(current_basis));
            swaps_since_refresh = 0;
            in_verification = true;

            // Check if any size-reduction or Lovász violation remains
            size_t violation_k = d;
            for (size_t i = 0; i < d && violation_k == d; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    bool vio = false;
                    if constexpr (std::is_same_v<FloatT, BigReal>) {
                        vio = (state.mu(i, j).abs() > eta_f + BigReal(1e-12, prec));
                    } else {
                        vio = (std::abs(state.mu(i, j)) > eta_f + 1e-9);
                    }
                    if (vio) {
                        violation_k = i;
                        break;
                    }
                }
            }

            for (size_t i = 1; i < d && violation_k == d; ++i) {
                const FloatT& Bi = state.bstar_norm_sq[i];
                const FloatT& Bi1 = state.bstar_norm_sq[i - 1];
                const FloatT mui = state.mu(i, i - 1);
                FloatT th = (delta_f - mui * mui) * Bi1;
                bool vio = false;
                if constexpr (std::is_same_v<FloatT, BigReal>) {
                    vio = (Bi < th - BigReal(1e-12, prec));
                } else {
                    vio = (Bi < th - 1e-9);
                }
                if (vio) {
                    violation_k = i;
                    break;
                }
            }

            if (violation_k < d) {
                k = violation_k;
                in_verification = false;
            }
        }
    }

    res.basis = std::move(current_basis);
    res.transformation = std::move(U);
    return res;
}

LLLReductionResult<BigInt> LLLFloat::reduce(const BasisZZ& basis) const {
    return reduce_mpfr(basis, params_.precision);
}

LLLReductionResult<BigInt> LLLFloat::reduce_mpfr(const BasisZZ& basis, prec_t prec) const {
    return reduce_impl<BigReal>(basis, prec);
}

LLLReductionResult<BigInt> LLLFloat::reduce_double(const BasisZZ& basis) const {
    return reduce_impl<double>(basis, 53);
}

LatticeZZ LLLFloat::reduce(const LatticeZZ& lattice) const {
    auto res = reduce(lattice.basis());
    return LatticeZZ(std::move(res.basis), lattice.volume());
}

} // namespace dbdd
