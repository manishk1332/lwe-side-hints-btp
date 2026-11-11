#pragma once

// Paper Reference: Dachman-Soled et al. Section 3.5 / Equations (27)-(30) / docs/mathematical_conventions.md
// Mathematical Purpose: Short-vector hint v in Lambda orthogonal projection.
// Projects lattice Lambda' = Lambda * P_v^perp and distribution (mu', Sigma') = (mu P, P^T Sigma P).
// Drops lattice dimension by 1 (or k) and contracts volume by ||v|| (or sqrt(det(V V^T))).

#include "dbdd/core/types.hpp"
#include "dbdd/hints/hint.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lattice/basis.hpp"
#include <vector>
#include <string>

namespace dbdd {

class DBDDInstance;

/**
 * @brief Represents a short-vector hint where a known non-zero lattice vector v in Lambda
 * (or multiple vectors V in Lambda^{k x m}) is projected out of the instance.
 *
 * Distinct model requirement (Section 3.5):
 * - Projects lattice: Lambda' = Lambda * P_v^perp, dropping dimension by 1 (or k).
 * - Contracts volume: Vol(Lambda') = Vol(Lambda) / ||v|| (or Vol / sqrt(det(V V^T))).
 * - Projects distribution: mu' = mu * P, Sigma' = P^T * Sigma * P.
 * - Projects target vector: s_bar' = s_bar * P in Lambda'.
 */
class ShortVectorHint : public Hint {
public:
    explicit ShortVectorHint(VectorZZ v);
    explicit ShortVectorHint(MatrixZZ V);

    static ShortVectorHint from_vector(VectorZZ v);
    static ShortVectorHint from_matrix(MatrixZZ V);

    void apply(DBDDInstance& instance) const override;
    [[nodiscard]] HintType type() const override { return HintType::ShortVector; }
    [[nodiscard]] std::string description() const override;

    [[nodiscard]] const VectorZZ& hint_vector() const;
    [[nodiscard]] const MatrixZZ& hint_matrix() const;
    [[nodiscard]] bool is_multi_vector() const noexcept { return is_multi_; }
    [[nodiscard]] size_t num_vectors() const noexcept;

    // =========================================================================
    // Static Mathematical Helper Routines
    // =========================================================================

    /**
     * @brief Computes orthogonal projector P_v^perp = I - (v^T v) / ||v||^2 in Q^{m x m}.
     */
    static MatrixQQ orthogonal_projector(const VectorZZ& v);

    /**
     * @brief Computes orthogonal projector P_V^perp = I - V^T (V V^T)^{-1} V in Q^{m x m}.
     */
    static MatrixQQ orthogonal_projector(const MatrixZZ& V);

    /**
     * @brief Projects Gaussian distribution: mu' = mu * P, Sigma' = P^T * cov * P.
     */
    static std::pair<VectorDouble, MatrixDouble> project_distribution(
        const VectorDouble& mean,
        const MatrixDouble& cov,
        const MatrixDouble& P);

    /**
     * @brief Projects integer basis B along primitive vector v in Lambda.
     * When projected basis has integer entries (e.g. for q-vectors), returns BasisZZ with dimension d - 1.
     */
    static BasisZZ project_basis(const BasisZZ& basis, const VectorZZ& v);

    /**
     * @brief Projects integer basis B along primitive vector v in Lambda, returning exact rational basis BasisQQ.
     */
    static BasisQQ project_basis_rational(const BasisZZ& basis, const VectorZZ& v);

    /**
     * @brief Projects integer basis B along multiple primitive vectors V in Lambda.
     */
    static BasisZZ project_basis(const BasisZZ& basis, const MatrixZZ& V);
    static BasisQQ project_basis_rational(const BasisZZ& basis, const MatrixZZ& V);

    /**
     * @brief Computes projected lattice volume Vol(Lambda') = Vol(Lambda) / ||v||.
     */
    static BigReal projected_volume(const BigReal& vol, const VectorZZ& v);

    /**
     * @brief Computes projected lattice volume Vol(Lambda') = Vol(Lambda) / sqrt(det(V V^T)).
     */
    static BigReal projected_volume(const BigReal& vol, const MatrixZZ& V);

private:
    VectorZZ v_;
    MatrixZZ V_;
    bool is_multi_{false};
};

} // namespace dbdd
