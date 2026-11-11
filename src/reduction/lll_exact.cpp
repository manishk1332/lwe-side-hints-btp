#include "dbdd/reduction/lll_exact.hpp"

namespace dbdd {

// Paper Reference: Lenstra, Lenstra, Lovász (1982) / Cohen Alg. 2.6.3 / Dachman-Soled et al. Section 2
// Mathematical Purpose: Exact rational LLL reduction over BigInt and BigRational bases.
// Inputs:
//     basis: Input lattice basis B in K^{d x m} (row-basis convention)
// Outputs:
//     LLLReductionResult containing reduced basis B', unimodular U in GL_d(Z), and GS state
// Preconditions:
//     Basis is full rank in its span (d <= m, linearly independent rows)
// Postconditions:
//     B' = U * B with det(U) = +/- 1
//     Vol(B') = Vol(B)
//     |mu_{i,j}| <= 1/2 for all j < i
//     ||b_k*||^2 >= (delta - mu_{k,k-1}^2) ||b_{k-1}*||^2 for all k >= 1
// Precision & Complexity:
//     Exact BigRational arithmetic, polynomial time O(d^4 m log B_max)

LLLExact::LLLExact(LLLParams params) : params_(std::move(params)) {
    params_.validate();
}

LLLReductionResult<BigInt> LLLExact::reduce(const BasisZZ& basis) const {
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
        res.exact_gs_state = compute_gram_schmidt_exact(basis.matrix());
        return res;
    }

    if (d > m) {
        throw DimensionMismatchException("Basis rank cannot exceed ambient dimension (d > m)");
    }

    // Initialize exact Gram-Schmidt state and unimodular transformation matrix
    GramSchmidtStateQQ state = compute_gram_schmidt_exact(basis.matrix());
    MatrixZZ U = MatrixZZ::identity(d);

    const BigRational& delta = params_.delta;
    const BigRational& eta = params_.eta;
    const size_t max_iter = params_.max_iterations;

    auto size_reduce_step = [&](size_t k, size_t j) {
        const BigRational& mu_kj = state.mu(k, j);
        if (mu_kj.abs() > eta) {
            BigInt q = mu_kj.round();
            if (q != BigInt(0)) {
                BigRational q_rat(q);
                update_gram_schmidt_row_add(state, k, j, q_rat);
                U.row_add_multiple(k, j, -q);
            }
        }
    };

    size_t k = 1;
    while (k < d) {
        res.iterations++;
        if (max_iter > 0 && res.iterations > max_iter) {
            res.converged = false;
            break;
        }

        // Step 1: Size-reduce b_k against b_{k-1}
        size_reduce_step(k, k - 1);

        // Step 2: Check Lovász condition
        const BigRational& Bk = state.bstar_norm_sq[k];
        const BigRational& Bk1 = state.bstar_norm_sq[k - 1];
        const BigRational mu_k_k1 = state.mu(k, k - 1);
        BigRational threshold = (delta - mu_k_k1 * mu_k_k1) * Bk1;

        if (Bk >= threshold) {
            // Lovász condition satisfied: size-reduce b_k against b_{k-2}, ..., b_0
            for (size_t j = k - 1; j-- > 0; ) {
                size_reduce_step(k, j);
            }
            k++;
        } else {
            // Lovász condition violated: swap rows k-1 and k
            update_gram_schmidt_swap(state, k - 1);
            U.swap_rows(k - 1, k);
            res.swaps++;
            k = (k > 1) ? (k - 1) : 1;
        }
    }

    // Reconstruct exact integer reduced basis: B' = U * B
    res.basis = BasisZZ(U * basis.matrix(), false);
    res.transformation = std::move(U);
    res.exact_gs_state = std::move(state);
    return res;
}

LLLReductionResult<BigRational> LLLExact::reduce(const BasisQQ& basis) const {
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();

    LLLReductionResult<BigRational> res;
    if (d == 0) {
        res.basis = BasisQQ();
        res.transformation = MatrixZZ(0, 0);
        res.converged = true;
        return res;
    }
    if (d == 1) {
        res.basis = basis;
        res.transformation = MatrixZZ::identity(1);
        res.converged = true;
        res.exact_gs_state = compute_gram_schmidt_exact(basis.matrix());
        return res;
    }

    if (d > m) {
        throw DimensionMismatchException("Basis rank cannot exceed ambient dimension (d > m)");
    }

    GramSchmidtStateQQ state = compute_gram_schmidt_exact(basis.matrix());
    MatrixZZ U = MatrixZZ::identity(d);

    const BigRational& delta = params_.delta;
    const BigRational& eta = params_.eta;
    const size_t max_iter = params_.max_iterations;

    auto size_reduce_step = [&](size_t k, size_t j) {
        const BigRational& mu_kj = state.mu(k, j);
        if (mu_kj.abs() > eta) {
            BigInt q = mu_kj.round();
            if (q != BigInt(0)) {
                BigRational q_rat(q);
                update_gram_schmidt_row_add(state, k, j, q_rat);
                U.row_add_multiple(k, j, -q);
            }
        }
    };

    size_t k = 1;
    while (k < d) {
        res.iterations++;
        if (max_iter > 0 && res.iterations > max_iter) {
            res.converged = false;
            break;
        }

        size_reduce_step(k, k - 1);

        const BigRational& Bk = state.bstar_norm_sq[k];
        const BigRational& Bk1 = state.bstar_norm_sq[k - 1];
        const BigRational mu_k_k1 = state.mu(k, k - 1);
        BigRational threshold = (delta - mu_k_k1 * mu_k_k1) * Bk1;

        if (Bk >= threshold) {
            for (size_t j = k - 1; j-- > 0; ) {
                size_reduce_step(k, j);
            }
            k++;
        } else {
            update_gram_schmidt_swap(state, k - 1);
            U.swap_rows(k - 1, k);
            res.swaps++;
            k = (k > 1) ? (k - 1) : 1;
        }
    }

    res.basis = BasisQQ(state.basis, false);
    res.transformation = std::move(U);
    res.exact_gs_state = std::move(state);
    return res;
}

LatticeZZ LLLExact::reduce(const LatticeZZ& lattice) const {
    auto res = reduce(lattice.basis());
    return LatticeZZ(std::move(res.basis), lattice.volume());
}

LatticeQQ LLLExact::reduce(const LatticeQQ& lattice) const {
    auto res = reduce(lattice.basis());
    return LatticeQQ(std::move(res.basis), lattice.volume());
}

} // namespace dbdd
