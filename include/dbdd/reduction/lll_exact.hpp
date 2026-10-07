#pragma once

// Paper Reference: Lenstra, Lenstra, Lovász (1982) / Cohen Alg. 2.6.3 / Albrecht et al. Section 2
// Mathematical Purpose: Exact rational LLL reduction over BigInt and BigRational bases.
// Guarantees exact size-reduction, strict Lovász condition adherence,
// volume preservation, and unimodular transformation tracking U in GL_d(Z).

#include "dbdd/reduction/lll.hpp"
#include "dbdd/lattice/lattice.hpp"

namespace dbdd {

/**
 * @brief Exact rational LLL reduction engine.
 * 
 * Implements Cohen Algorithm 2.6.3 with exact arithmetic, zero precision loss,
 * and O(m + d) local Gram-Schmidt updates per vector swap and size reduction.
 */
class LLLExact {
public:
    explicit LLLExact(LLLParams params = LLLParams{});

    [[nodiscard]] const LLLParams& params() const noexcept { return params_; }
    void set_params(LLLParams params) {
        params.validate();
        params_ = std::move(params);
    }

    /**
     * @brief Reduces an integer lattice basis using exact rational arithmetic.
     */
    [[nodiscard]] LLLReductionResult<BigInt> reduce(const BasisZZ& basis) const;

    /**
     * @brief Reduces a rational lattice basis using exact rational arithmetic.
     */
    [[nodiscard]] LLLReductionResult<BigRational> reduce(const BasisQQ& basis) const;

    /**
     * @brief Reduces a lattice and preserves its cached volume.
     */
    [[nodiscard]] LatticeZZ reduce(const LatticeZZ& lattice) const;
    [[nodiscard]] LatticeQQ reduce(const LatticeQQ& lattice) const;

private:
    LLLParams params_;
};

// ============================================================================
// Free-function convenience API
// ============================================================================

inline LLLReductionResult<BigInt> lll_exact(const BasisZZ& basis,
                                           const LLLParams& params = LLLParams{}) {
    return LLLExact(params).reduce(basis);
}

inline LLLReductionResult<BigRational> lll_exact(const BasisQQ& basis,
                                                const LLLParams& params = LLLParams{}) {
    return LLLExact(params).reduce(basis);
}

inline LatticeZZ lll_exact(const LatticeZZ& lattice,
                          const LLLParams& params = LLLParams{}) {
    return LLLExact(params).reduce(lattice);
}

inline LatticeQQ lll_exact(const LatticeQQ& lattice,
                          const LLLParams& params = LLLParams{}) {
    return LLLExact(params).reduce(lattice);
}

} // namespace dbdd
