#pragma once

// Paper Reference: Albrecht et al. Section 3.3 / Equations (24)-(26) / docs/mathematical_conventions.md
// Mathematical Purpose: A-posteriori approximate hint conditioning on direct posterior
// distribution <s, v> ~ N(mu_post, var_post).
// Updates prior Gaussian distribution (mu, Sigma) along direction v_bar while strictly preserving
// the lattice basis Lambda: Lambda' = Lambda, Vol(Lambda') = Vol(Lambda), dim(Lambda') = d.

#include "dbdd/core/types.hpp"
#include "dbdd/hints/hint.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <utility>
#include <string>

namespace dbdd {

class DBDDInstance;

/**
 * @brief Represents an a-posteriori approximate hint where side-channel leakage,
 * decryption failures, or statistical classifiers directly yield a posterior distribution
 * on the scalar projection <s, v> characterized by posterior mean mu_post and posterior variance var_post.
 *
 * Distinct model requirement (Section 3.3): This is NOT modeled as independent additive noise.
 * Instead, it directly replaces/updates the 1D marginal along direction v with (mu_post, var_post).
 * The lattice basis, dimension, volume, and target coordinates remain strictly unchanged.
 */
class AposterioriApproximateHint : public Hint {
public:
    /**
     * @brief Constructor for integer direction vector v with posterior parameters.
     * @param v Direction vector in Z^n.
     * @param mu_post Known posterior mean of <s, v>.
     * @param var_post Known posterior variance of <s, v> (must be >= 0).
     */
    AposterioriApproximateHint(VectorZZ v, double mu_post, double var_post);

    /**
     * @brief Constructor for floating-point direction vector v with posterior parameters.
     * @param v Direction vector in R^n.
     * @param mu_post Known posterior mean of <s, v>.
     * @param var_post Known posterior variance of <s, v> (must be >= 0).
     */
    AposterioriApproximateHint(VectorDouble v, double mu_post, double var_post);

    // Factory methods
    static AposterioriApproximateHint from_secret(VectorZZ v, double mu_post, double var_post);
    static AposterioriApproximateHint from_secret(VectorDouble v, double mu_post, double var_post);
    static AposterioriApproximateHint from_homogenized(VectorZZ v_bar, double mu_post, double var_post);
    static AposterioriApproximateHint from_homogenized(VectorDouble v_bar, double mu_post, double var_post);

    // Hint interface implementation
    void apply(DBDDInstance& instance) const override;
    [[nodiscard]] HintType type() const override { return HintType::Aposteriori; }
    [[nodiscard]] std::string description() const override;

    // Accessors
    [[nodiscard]] const VectorZZ& hint_vector_int() const;
    [[nodiscard]] const VectorDouble& hint_vector_double() const;
    [[nodiscard]] double posterior_mean() const noexcept { return mu_post_; }
    [[nodiscard]] double posterior_variance() const noexcept { return var_post_; }
    [[nodiscard]] bool is_integer() const noexcept { return is_integer_; }
    [[nodiscard]] bool is_already_homogenized() const noexcept { return is_already_homogenized_; }

    /**
     * @brief Generates the homogenized hint vector v_bar in ZZ^{ambient_dim}.
     * If v is already homogenized, validates its dimension.
     * If v has length ambient_dim - 1 (representing (e_LWE, s)), forms (v, 0).
     * If v has length n < ambient_dim - 1 (representing secret s), forms (0_m, v, 0).
     */
    [[nodiscard]] VectorZZ homogenize_int(size_t ambient_dim) const;

    /**
     * @brief Generates the homogenized hint vector v_bar in RR^{ambient_dim}.
     */
    [[nodiscard]] VectorDouble homogenize_double(size_t ambient_dim) const;

    // =========================================================================
    // Static Mathematical Helper Routines
    // =========================================================================

    /**
     * @brief Paper Reference: Albrecht et al. Section 3.3 / Equations (24)-(26)
     * Mathematical Purpose: Updates mean mu and covariance Sigma given a posterior
     * distribution along direction v_bar with mean mu_post and variance var_post.
     *
     * Formulas:
     *     mu_prior = <mean, v_bar>
     *     w = v_bar * cov (1 x d row vector)
     *     sigma_prior_sq = w * v_bar^T
     *     delta_sigma_sq = sigma_prior_sq - var_post
     *     mu' = mean + ((mu_post - mu_prior) / sigma_prior_sq) * w
     *     Sigma' = cov - (delta_sigma_sq / (sigma_prior_sq * sigma_prior_sq)) * (w^T * w)
     *
     * Inputs:
     *     mean: Prior mean row vector mu in R^d
     *     cov: Prior covariance matrix Sigma in R^{d x d} (symmetric PSD)
     *     v_bar: Direction vector in R^d
     *     mu_post: Posterior mean of <x, v_bar>
     *     var_post: Posterior variance of <x, v_bar> (must be >= 0)
     * Outputs:
     *     std::pair<VectorDouble, MatrixDouble> {mu', Sigma'}
     * Preconditions:
     *     cov is symmetric PSD, dimensions match, var_post >= 0, sigma_prior_sq > 0.
     * Postconditions:
     *     Sigma' is symmetric PSD, <mu', v_bar> == mu_post, v_bar * Sigma' * v_bar^T == var_post.
     * Precision & Complexity:
     *     Floating-point dense lin-alg, O(d^2) operations.
     */
    static std::pair<VectorDouble, MatrixDouble> update_distribution(
        const VectorDouble& mean,
        const MatrixDouble& cov,
        const VectorDouble& v_bar,
        double mu_post,
        double var_post);

private:
    VectorZZ v_int_;
    VectorDouble v_double_;
    double mu_post_{0.0};
    double var_post_{0.0};
    bool is_integer_{true};
    bool is_already_homogenized_{false};
};

} // namespace dbdd
