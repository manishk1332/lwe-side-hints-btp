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
#include "dbdd/security/security_estimate.hpp"
#include <optional>

namespace dbdd {

class UsvpInstance;
class DBDDStateStatistics;
class DiagonalCovariance;

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
     * Paper Reference: Albrecht et al. Section 3.3 / Equations (24)-(26)
     * Directly updates the 1D marginal along direction v to have mean mu_post and variance var_post.
     * Guarantees zero mutations to the lattice basis, volume, dimension, or target vector.
     * @param v Direction vector in R^n (or R^d).
     * @param mu_post Target posterior mean of the projection along v.
     * @param var_post Target posterior variance of the projection along v (>= 0).
     */
    void integrate_aposteriori_hint(const VectorDouble& v, double mu_post, double var_post);

    /**
     * @brief Integer overload for integrating an a-posteriori approximate hint.
     * @param v Direction vector in Z^n (or Z^d).
     * @param mu_post Target posterior mean of the projection along v.
     * @param var_post Target posterior variance of the projection along v (>= 0).
     */
    void integrate_aposteriori_hint(const VectorZZ& v, double mu_post, double var_post);

    /**
     * @brief Integrates a short vector hint v in Lambda.
     * Paper Reference: Albrecht et al. Section 3.5 / Equations (27)-(30)
     * Projects the lattice Lambda' = Lambda * P_v^perp, drops dimension by 1,
     * contracts volume by ||v||, and projects distribution (mu', Sigma').
     * @param v Non-zero lattice vector in Lambda.
     */
    void integrate_short_vector_hint(const VectorZZ& v);

    /**
     * @brief Multi-vector overload for integrating short vector hints V in Lambda^{k x m}.
     * Projects the lattice orthogonal to span(V), dropping dimension by k and volume by sqrt(det(V V^T)).
     * @param V Matrix whose linearly independent rows belong to Lambda.
     */
    void integrate_short_vector_hint(const MatrixZZ& V);

    // =========================================================================
    // Extension points for Transformations (Phases 11-12)
    // =========================================================================

    /**
     * @brief Homogenizes the instance: mu_h = 0, Sigma_h = Sigma + mu^T mu.
     * Paper Reference: Albrecht et al. Section 2 / Equation (8)
     * Embeds the target distribution into an origin-centered ellipsoid envelope.
     * Preserves lattice basis, volume, dimension, and target vector.
     */
    void homogenize();

    /**
     * @brief Checks whether the instance is origin-centered (homogenized).
     * @param tol Euclidean norm tolerance for the mean vector (default: 1e-12).
     */
    [[nodiscard]] bool is_homogenized(double tol = 1e-12) const;

    /**
     * @brief Isotropizes the instance so that covariance becomes identity on active span.
     * Paper Reference: Albrecht et al. Section 2, Section 3.3
     */
    void isotropize(double rank_tol = 1e-12);

    /**
     * @brief Checks whether the instance is isotropized (covariance is identity on active support).
     */
    [[nodiscard]] bool is_isotropized(double tol = 1e-7) const;

    /**
     * @brief Converts DBDD state to a discrete uSVP lattice basis for lattice reduction.
     * Discretizes B_iso = B * M by scaling and rounding.
     * @param scale_factor Scaling factor applied before integer rounding (default: 1.0).
     */
    [[nodiscard]] BasisZZ to_usvp(double scale_factor = 1.0) const;

    /**
     * @brief Converts DBDD state to an isotropic UsvpInstance (continuous transformed lattice).
     */
    [[nodiscard]] UsvpInstance to_usvp_instance(double rank_tol = 1e-12) const;

    // =========================================================================
    // Security Estimation (Phase 13: GSA-Intersect)
    // =========================================================================

    /**
     * @brief Estimates attack complexity using GSA-Intersect security estimator.
     * Uses restricted-determinant volume shortcut on homogenized covariance.
     */
    [[nodiscard]] SecurityEstimate estimate_attack_gsa(
        double beta_min = 2.0,
        double beta_max = 2000.0,
        double rank_tol = 1e-12) const;

    /**
     * @brief Estimates required integer BKZ block size beta ("bikz").
     */
    [[nodiscard]] size_t estimate_beta_gsa(double rank_tol = 1e-12) const;

    // =========================================================================
    // Lightweight Predictors (Phase 14)
    // =========================================================================

    /**
     * @brief Converts instance to lightweight dense statistics predictor (Tier 2).
     */
    [[nodiscard]] DBDDStateStatistics to_state_statistics() const;

    /**
     * @brief Converts instance to super-lightweight diagonal predictor (Tier 3).
     * @throws InvalidParameterException if covariance is not diagonal.
     */
    [[nodiscard]] DiagonalCovariance to_diagonal_covariance(double off_diag_tol = 1e-10) const;

private:
    LatticeZZ lattice_;
    Ellipsoid ellipsoid_;
    std::optional<VectorZZ> target_;
    std::optional<VectorZZ> target_coords_;
};

} // namespace dbdd
