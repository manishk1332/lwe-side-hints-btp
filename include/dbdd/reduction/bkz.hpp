#pragma once

// Paper Reference: Schnorr (1987) / Schnorr-Euchner (1994) / Chen-Nguyen (2011) / Dachman-Soled et al. Section 2 & 3
// Mathematical Purpose: Block Korkine-Zolotarev (BKZ) reduction engine with decoupled SVP oracle,
// exact unimodular completion insertion, tour profile tracking, and end-to-end secret recovery.

#include "dbdd/reduction/bkz_parameters.hpp"
#include "dbdd/reduction/bkz_result.hpp"
#include "dbdd/reduction/svp.hpp"
#include "dbdd/reduction/enumeration.hpp"
#include "dbdd/lattice/lattice.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include <memory>
#include <optional>

namespace dbdd {

/**
 * @brief Block Korkine-Zolotarev (BKZ) reduction engine.
 */
class BKZReduction {
public:
    explicit BKZReduction(
        BKZParameters params = BKZParameters{},
        std::shared_ptr<SVPOracle> oracle = std::make_shared<EnumerationSVP>());

    [[nodiscard]] const BKZParameters& params() const noexcept { return params_; }
    void set_params(BKZParameters params) {
        params.validate();
        params_ = std::move(params);
    }

    [[nodiscard]] const SVPOracle& oracle() const noexcept { return *oracle_; }
    void set_oracle(std::shared_ptr<SVPOracle> oracle) {
        if (!oracle) {
            throw InvalidParameterException("BKZReduction: oracle cannot be null");
        }
        oracle_ = std::move(oracle);
    }

    /**
     * @brief Reduces an integer lattice basis using BKZ reduction.
     */
    [[nodiscard]] BKZResult reduce(const BasisZZ& basis) const;

    /**
     * @brief Reduces a lattice and preserves its cached volume.
     */
    [[nodiscard]] LatticeZZ reduce(const LatticeZZ& lattice) const;

    /**
     * @brief Executes a single BKZ tour on the basis.
     * @param basis Lattice basis (mutated in-place).
     * @param U Unimodular transformation matrix (mutated in-place).
     * @param total_svp_calls Accumulated SVP oracle calls.
     * @param total_svp_nodes Accumulated SVP search nodes.
     * @return Number of insertions made during the tour.
     */
    size_t run_tour(
        BasisZZ& basis,
        MatrixZZ& U,
        size_t& total_svp_calls,
        size_t& total_svp_nodes) const;

    /**
     * @brief Attempts to recover LWE secret from a reduced basis.
     * Tests basis rows against LWE instance relation b = s * A^T + e (mod q).
     */
    static std::optional<VectorZZ> recover_lwe_secret(
        const BasisZZ& reduced_basis,
        const LWEInstance& lwe);

    /**
     * @brief Attempts to recover LWE secret from a DBDD instance.
     */
    static std::optional<VectorZZ> recover_dbdd_secret(
        const BasisZZ& reduced_basis,
        const DBDDInstance& dbdd);

private:
    BKZParameters params_;
    std::shared_ptr<SVPOracle> oracle_;
};

// ============================================================================
// Free-Function Convenience API
// ============================================================================

inline BKZResult bkz_reduce(
    const BasisZZ& basis,
    const BKZParameters& params = BKZParameters{}) {
    return BKZReduction(params).reduce(basis);
}

inline LatticeZZ bkz_reduce(
    const LatticeZZ& lattice,
    const BKZParameters& params = BKZParameters{}) {
    return BKZReduction(params).reduce(lattice);
}

} // namespace dbdd

