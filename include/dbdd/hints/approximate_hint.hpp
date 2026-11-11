#pragma once

// Paper Reference: Dachman-Soled et al. Section 3.3 / Equations (20)-(23) / docs/mathematical_conventions.md
// Mathematical Purpose: Approximate linear hint <s, v> + e = l integration where e ~ N(0, sigma_e^2).
// Homogenizes to <s_bar, v_bar> + e = 0 against embedded target s_bar = (e_LWE, s, 1).
// Updates target prior distribution (mu, Sigma) via Gaussian conditioning formulas (Eq. 20-23).
// Guarantees zero mutations to the lattice basis: Lambda' = Lambda, Vol(Lambda') = Vol(Lambda), dim(Lambda') = d.

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
 * @brief Represents an approximate (noisy) linear hint <s, v> + e = l with e ~ N(0, sigma_e^2).
 * 
 * In the DBDD framework, this hint modifies only the Gaussian distribution (mu, Sigma)
 * of the embedded target vector s_bar = (e_LWE, s, 1). The lattice basis, dimension,
 * volume, and target coordinates remain strictly unchanged.
 */
class ApproximateHint : public Hint {
public:
    /**
     * @brief Constructor for integer approximate hint <s, v> + e = l.
     * @param v Hint coefficient vector in Z^n.
     * @param l Observed noisy inner product in Z.
     * @param variance Known noise variance sigma_e^2 > 0.
     */
    ApproximateHint(VectorZZ v, BigInt l, double variance);

    /**
     * @brief Constructor for floating-point approximate hint <s, v> + e = l.
     * @param v Hint coefficient vector in R^n.
     * @param l Observed noisy inner product in R.
     * @param variance Known noise variance sigma_e^2 > 0.
     */
    ApproximateHint(VectorDouble v, double l, double variance);

    // Factory methods
    static ApproximateHint from_secret(VectorZZ v, BigInt l, double variance);
    static ApproximateHint from_secret(VectorDouble v, double l, double variance);
    static ApproximateHint from_homogenized(VectorZZ v_bar, double variance);
    static ApproximateHint from_homogenized(VectorDouble v_bar, double variance);

    // Hint interface implementation
    void apply(DBDDInstance& instance) const override;
    [[nodiscard]] HintType type() const override { return HintType::Approximate; }
    [[nodiscard]] std::string description() const override;

    // Accessors
    [[nodiscard]] const VectorZZ& hint_vector_int() const;
    [[nodiscard]] const BigInt& hint_scalar_int() const;
    [[nodiscard]] const VectorDouble& hint_vector_double() const;
    [[nodiscard]] double hint_scalar_double() const;
    [[nodiscard]] double noise_variance() const noexcept { return variance_; }
    [[nodiscard]] bool is_integer() const noexcept { return is_integer_; }
    [[nodiscard]] bool is_already_homogenized() const noexcept { return is_already_homogenized_; }

    /**
     * @brief Generates the homogenized hint vector v_bar in ZZ^{ambient_dim}.
     * If v is already homogenized, validates its dimension.
     * If v has length ambient_dim - 1 (representing (e_LWE, s)), forms (v, -l).
     * If v has length n < ambient_dim - 1 (representing secret s), forms (0_m, v, -l).
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
     * @brief Paper Reference: Dachman-Soled et al. Section 3.3 / Equations (20)-(23)
     * Mathematical Purpose: Updates mean mu and covariance Sigma conditioned on
     * noisy observation <x, v_bar> + e = 0, where e ~ N(0, variance).
     * 
     * Formulas:
     *     w = v_bar * Sigma (1 x d)
     *     denom = v_bar * Sigma * v_bar^T + variance
     *     mu' = mu - (<mu, v_bar> / denom) * w
     *     Sigma' = Sigma - (w^T * w) / denom
     * 
     * Inputs:
     *     mean: Prior mean row vector mu in R^d
     *     cov: Prior covariance matrix Sigma in R^{d x d} (symmetric PSD)
     *     v_bar: Homogenized hint vector in R^d
     *     variance: Observation noise variance sigma_e^2 > 0
     * Outputs:
     *     std::pair<VectorDouble, MatrixDouble> {mu', Sigma'}
     * Preconditions:
     *     cov is symmetric PSD, dimensions match, variance > 0.
     * Postconditions:
     *     Sigma' is symmetric PSD, Sigma - Sigma' is PSD (uncertainty never increases).
     * Precision & Complexity:
     *     Floating-point dense lin-alg, O(d^2) operations.
     */
    static std::pair<VectorDouble, MatrixDouble> update_distribution(
        const VectorDouble& mean,
        const MatrixDouble& cov,
        const VectorDouble& v_bar,
        double variance);

private:
    VectorZZ v_int_;
    BigInt l_int_;
    VectorDouble v_double_;
    double l_double_{0.0};
    double variance_{1.0};
    bool is_integer_{true};
    bool is_already_homogenized_{false};
};

} // namespace dbdd
