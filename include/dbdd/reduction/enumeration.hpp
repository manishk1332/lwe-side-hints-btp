#pragma once

// Paper Reference: Fincke & Pohst (1985) / Schnorr & Euchner (1994) / Dachman-Soled et al. Section 3
// Mathematical Purpose: Exact SVP enumeration using Fincke-Pohst and Schnorr-Euchner
// algorithms with dynamic branch-and-bound radius shrinking and projected block sublattice support.

#include "dbdd/reduction/svp.hpp"

namespace dbdd {

/**
 * @brief Concrete SVPOracle implementation using exact coordinate enumeration.
 * 
 * Supports both full basis SVP and projected block sublattices for BKZ tours.
 */
class EnumerationSVP : public SVPOracle {
public:
    EnumerationSVP() = default;

    /**
     * @brief Solves exact SVP on the full lattice basis.
     */
    SVPResult solve(
        const BasisZZ& basis,
        const SVPParameters& params = {}) override;

    /**
     * @brief Solves exact SVP on a projected sub-block [start_idx, start_idx + block_size - 1].
     */
    SVPResult solve_projected(
        const GramSchmidtStateDouble& gs,
        size_t start_idx,
        size_t block_size,
        const SVPParameters& params = {}) override;

    /**
     * @brief Static convenience helper for full basis SVP.
     */
    static SVPResult find_shortest_vector(
        const BasisZZ& basis,
        const SVPParameters& params = {});

    /**
     * @brief Static convenience helper for projected block SVP.
     */
    static SVPResult find_shortest_vector_projected(
        const GramSchmidtStateDouble& gs,
        size_t start_idx,
        size_t block_size,
        const SVPParameters& params = {});
};

} // namespace dbdd

