#pragma once

// Paper Reference: Dachman-Soled et al. Section 3.5 / main_plan.txt Section 22
// Mathematical Purpose: Rational linear secret reconstruction after projected solving.
// Maps candidate vectors w' in Lambda' from lattice reduction back to ground truth secret s in Z^n.

#include "dbdd/core/types.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include <vector>

namespace dbdd {

/**
 * @brief Reconstructs the original LWE secret vector s in Z^n and full target vector s_bar = (e, s, 1)
 * from a candidate vector w' found in a projected lattice Lambda' = Lambda * P.
 */
class SecretReconstructor {
public:
    explicit SecretReconstructor(LWEInstance lwe);

    void record_projection(VectorZZ v);
    void record_projections(const std::vector<VectorZZ>& V);

    [[nodiscard]] const LWEInstance& lwe_instance() const noexcept { return lwe_; }
    [[nodiscard]] const std::vector<VectorZZ>& projections() const noexcept { return projections_; }
    [[nodiscard]] size_t num_projections() const noexcept { return projections_.size(); }

    /**
     * @brief Extracts candidate secret s in Z^n from candidate projected target w'.
     * Resolves orientation (+/- sign), extracts secret coordinates, and reconstructs
     * projected error coordinates via the LWE modular relation.
     */
    [[nodiscard]] VectorZZ recover_secret(const VectorDouble& candidate_w) const;
    [[nodiscard]] VectorZZ recover_secret(const VectorZZ& candidate_w) const;

    /**
     * @brief Reconstructs the complete ground-truth target vector s_bar = (e, s, 1) in Z^{m+n+1}.
     */
    [[nodiscard]] VectorZZ recover_target(const VectorDouble& candidate_w) const;
    [[nodiscard]] VectorZZ recover_target(const VectorZZ& candidate_w) const;

    /**
     * @brief Verifies whether a candidate secret s satisfies b = s * A^T + e (mod q) with small error.
     */
    [[nodiscard]] bool verify_recovery(const VectorZZ& secret) const;

private:
    LWEInstance lwe_;
    std::vector<VectorZZ> projections_;
};

} // namespace dbdd
