#pragma once

// Paper Reference: Lenstra, Lenstra, Lovász (1982) / Albrecht et al. Section 2
// Mathematical Purpose: High-precision floating-point LLL reduction using MPFR (BigReal)
// and double precision. Maintains exact integer representations of the basis and unimodular
// transformation matrix, with floating-point Gram-Schmidt orthogonalization guidance
// and periodic MGS refreshes.

#include "dbdd/reduction/lll.hpp"
#include "dbdd/lattice/lattice.hpp"

namespace dbdd {

/**
 * @brief Floating-point LLL reduction engine.
 * 
 * Tracks basis B in ZZ and unimodular matrix U in GL_d(Z) with exact integer operations,
 * while utilizing floating-point Gram-Schmidt arithmetic (MPFR BigReal or double)
 * for fast reduction guidance. Periodic Gram-Schmidt refreshes eliminate numerical drift.
 */
class LLLFloat {
public:
    explicit LLLFloat(LLLParams params = LLLParams{});

    [[nodiscard]] const LLLParams& params() const noexcept { return params_; }
    void set_params(LLLParams params) {
        params.validate();
        params_ = std::move(params);
    }

    /**
     * @brief Reduces an integer lattice basis using MPFR BigReal (default precision).
     */
    [[nodiscard]] LLLReductionResult<BigInt> reduce(const BasisZZ& basis) const;

    /**
     * @brief Reduces an integer lattice basis using MPFR BigReal with specified precision bits.
     */
    [[nodiscard]] LLLReductionResult<BigInt> reduce_mpfr(const BasisZZ& basis,
                                                        prec_t prec = 256) const;

    /**
     * @brief Reduces an integer lattice basis using double precision (hardware speed).
     */
    [[nodiscard]] LLLReductionResult<BigInt> reduce_double(const BasisZZ& basis) const;

    /**
     * @brief Reduces a lattice and preserves its cached volume.
     */
    [[nodiscard]] LatticeZZ reduce(const LatticeZZ& lattice) const;

private:
    template <typename FloatT>
    LLLReductionResult<BigInt> reduce_impl(const BasisZZ& basis, prec_t prec) const;

    LLLParams params_;
};

// ============================================================================
// Free-function convenience API
// ============================================================================

inline LLLReductionResult<BigInt> lll_float(const BasisZZ& basis,
                                           const LLLParams& params = LLLParams{}) {
    return LLLFloat(params).reduce(basis);
}

inline LatticeZZ lll_float(const LatticeZZ& lattice,
                          const LLLParams& params = LLLParams{}) {
    return LLLFloat(params).reduce(lattice);
}

} // namespace dbdd
