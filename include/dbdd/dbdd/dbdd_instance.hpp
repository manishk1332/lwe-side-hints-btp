#pragma once

// Paper Reference: Albrecht et al. Section 2, Section 3 / docs/mathematical_conventions.md
// Mathematical Purpose: Encapsulates the full DBDD instance (Lambda, mu, Sigma) with lattice basis,
// target distributional priors, ground-truth verification, and structured extension points
// for hint transformations (Phases 06-10) and basis transformations (Phases 11-12).

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/lattice.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/dbdd/ellipsoid.hpp"
#include "dbdd/hints/hint.hpp"
#include <optional>

namespace dbdd {

/**
 * @brief Full DBDD Instance representing the tuple (Lambda, mu, Sigma).
 * 
 * Manages the explicit lattice basis B in Z^{d x m}, target vector prior distribution
 * with center mean mu and covariance Sigma, ground-truth validation mechanics,
 * and dispatch for hints and basis transformations.
 */
class DBDDInstance {
public:
    /**
     * @brief Factory constructor from a Kannan embedding.
     * Preserves the Kannan lattice, cached volume q^m, target s_bar = (e, s, 1),
     * and coordinates x = (k, -s, 1).
     */
    static DBDDInstance from_kannan(const KannanEmbedding& kannan);

    /**
     * @brief Direct constructor with explicit LatticeZZ.
     */
    DBDDInstance(
        LatticeZZ lattice,
        VectorDouble mean,
        MatrixDouble covariance,
        std::optional<VectorZZ> target = std::nullopt,
        std::optional<VectorZZ> target_coords = std::nullopt);

    /**
     * @brief Direct constructor with BasisZZ.
     */
    DBDDInstance(
        BasisZZ basis,
        VectorDouble mean,
        MatrixDouble covariance,
        std::optional<VectorZZ> target = std::nullopt,
        std::optional<VectorZZ> target_coords = std::nullopt);

    /**
     * @brief Direct constructor with basis matrix.
     */
    DBDDInstance(
        MatrixZZ basis_matrix,
        VectorDouble mean,
        MatrixDouble covariance,
        std::optional<VectorZZ> target = std::nullopt,
        std::optional<VectorZZ> target_coords = std::nullopt);

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return lattice_.dimension(); }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return lattice_.ambient_dimension(); }

    // Lattice & Basis accessors (Row-basis convention)
    [[nodiscard]] const LatticeZZ& lattice() const noexcept { return lattice_; }
    [[nodiscard]] const BasisZZ& basis() const noexcept { return lattice_.basis(); }
    [[nodiscard]] const MatrixZZ& matrix() const noexcept { return lattice_.basis().matrix(); }

    // Volume accessors (cached)
    [[nodiscard]] BigReal volume() const { return lattice_.volume(); }
    [[nodiscard]] BigReal log_volume() const { return lattice_.log_volume(); }
    [[nodiscard]] BigInt exact_volume_int() const { return lattice_.exact_volume_int(); }

    // Statistical & Ellipsoid accessors
    [[nodiscard]] const VectorDouble& mean() const noexcept { return ellipsoid_.mean(); }
    [[nodiscard]] const MatrixDouble& covariance() const noexcept { return ellipsoid_.covariance(); }
    [[nodiscard]] const Ellipsoid& ellipsoid() const noexcept { return ellipsoid_; }

    // Target accessors & ground truth
    [[nodiscard]] bool has_target() const noexcept { return target_.has_value(); }
    [[nodiscard]] const VectorZZ& target() const;
    [[nodiscard]] VectorDouble target_double() const;
    [[nodiscard]] bool has_target_coordinates() const noexcept { return target_coords_.has_value(); }
    [[nodiscard]] const VectorZZ& target_coordinates() const;

    void set_target(VectorZZ target, std::optional<VectorZZ> target_coords = std::nullopt);

    // Target verification & containment
    [[nodiscard]] bool target_in_lattice() const;
    [[nodiscard]] double target_mahalanobis_distance_squared() const;
    [[nodiscard]] bool target_in_ellipsoid(double threshold = 1.0) const;
    [[nodiscard]] double chi2_threshold(double confidence = 0.999) const;
    [[nodiscard]] bool verify_target(double confidence = 0.999) const;

    // Mutators for lattice and distribution state
    void set_lattice(LatticeZZ new_lattice);
    void set_distribution(VectorDouble new_mean, MatrixDouble new_covariance);

    // Generic hint applicator
    void integrate_hint(const Hint& hint);

    // =========================================================================
    // Extension points for Hint Integration (Phases 06-10)
    // =========================================================================

    /**
     * @brief Integrates a perfect hint <s, v> = l.
     * Paper Reference: Section 3.1
     * Scheduled for implementation in Phase 06.
     */
    void integrate_perfect_hint(const VectorZZ& v, const BigInt& l);
    void integrate_perfect_hint(const VectorDouble& v, double l);

    /**
     * @brief Integrates a modular hint <s, v> = l (mod k).
     * Paper Reference: Section 3.2
     * Scheduled for implementation in Phase 07.
     */
    void integrate_modular_hint(const VectorZZ& v, const BigInt& l, const BigInt& k);
    void integrate_modular_hint(const VectorDouble& v, double l, double k);

    /**
     * @brief Integrates an approximate hint <s, v> = l + e, e ~ N(0, sigma^2).
     * Paper Reference: Section 3.3
     * Scheduled for implementation in Phase 08.
     */
    void integrate_approximate_hint(const VectorDouble& v, double l, double variance);
    void integrate_approximate_hint(const VectorZZ& v, const BigInt& l, double variance);

    /**
     * @brief Integrates an a-posteriori approximate hint.
     * Paper Reference: Section 3.3
     * Scheduled for implementation in Phase 09.
     */
    void integrate_aposteriori_hint(const VectorDouble& v, double l, double variance);

    /**
     * @brief Integrates a short vector hint v in Lambda*.
     * Paper Reference: Section 3.5
     * Scheduled for implementation in Phase 10.
     */
    void integrate_short_vector_hint(const VectorZZ& v);

    // =========================================================================
    // Extension points for Transformations (Phases 11-12)
    // =========================================================================

    /**
     * @brief Homogenizes the instance: mu_h = 0, Sigma_h = Sigma + mu^T mu.
     * Paper Reference: Section 2
     * Scheduled for implementation in Phase 11.
     */
    void homogenize();

    /**
     * @brief Isotropizes the instance so that covariance becomes proportional to identity on span.
     * Paper Reference: Section 2 / Section 4
     * Scheduled for implementation in Phase 12.
     */
    void isotropize();

    /**
     * @brief Converts DBDD state to a uSVP lattice basis for reduction.
     * Scheduled for implementation in Phase 12.
     */
    [[nodiscard]] BasisZZ to_usvp() const;

private:
    LatticeZZ lattice_;
    Ellipsoid ellipsoid_;
    std::optional<VectorZZ> target_;
    std::optional<VectorZZ> target_coords_;
};

} // namespace dbdd
