#pragma once

// Paper Reference: Dachman-Soled et al. Section 4.5 / main_plan.txt Section 51
// Mathematical Purpose: Generates canonical and difference q-vectors for LWE lattices
// and validates their membership in transformed (subspace) lattices.

#include "dbdd/core/types.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/lattice/lattice.hpp"

#include <vector>
#include <cstddef>

namespace dbdd {

class DBDDInstance;

/**
 * @brief Utilities for generating and validating LWE q-vectors (q*e_i and q*(e_i - e_j)).
 */
class QVectorGenerator {
public:
    /**
     * @brief Generates canonical q-vectors: q * e_i in Z^ambient_dim for i in [0, m - 1].
     * @param q LWE modulus.
     * @param m Number of LWE error coordinates.
     * @param ambient_dim Ambient dimension of the lattice (e.g. m + n + 1).
     */
    [[nodiscard]] static std::vector<VectorZZ> generate_canonical(
        const BigInt& q,
        size_t m,
        size_t ambient_dim);

    /**
     * @brief Generates pairwise difference q-vectors: q * (e_i - e_j) for 0 <= i < j < m.
     */
    [[nodiscard]] static std::vector<VectorZZ> generate_differences(
        const BigInt& q,
        size_t m,
        size_t ambient_dim);

    /**
     * @brief Tests if a candidate vector v belongs to lattice Lambda.
     * Accurately supports subspace lattices (d < ambient_dim) by verifying that
     * coordinates x = v B^T (B B^T)^{-1} have integer entries AND x * B == v.
     */
    [[nodiscard]] static bool is_in_lattice(
        const LatticeZZ& lattice,
        const VectorZZ& v);

    /**
     * @brief Filters a candidate vector list, returning only those vectors that belong to lattice Lambda.
     */
    [[nodiscard]] static std::vector<VectorZZ> filter_in_lattice(
        const LatticeZZ& lattice,
        const std::vector<VectorZZ>& candidates);

    /**
     * @brief Generates all valid q-vectors for an active DBDDInstance.
     * Automatically filters generated vectors against dbdd.lattice().
     * @param dbdd The active DBDD instance.
     * @param q The LWE modulus.
     * @param include_differences Whether to also generate and filter difference vectors q*(e_i - e_j).
     */
    [[nodiscard]] static std::vector<VectorZZ> generate_valid(
        const DBDDInstance& dbdd,
        const BigInt& q,
        bool include_differences = false);
};

} // namespace dbdd

