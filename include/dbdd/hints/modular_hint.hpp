#pragma once

// Paper Reference: Albrecht et al. Section 3.2 / Equations (15)-(19) / docs/mathematical_conventions.md
// Mathematical Purpose: Modular linear hint <s, v> = l (mod k) integration.
// Homogenizes to <s_bar, v_bar> = 0 (mod k) against embedded target s_bar = (e, s, 1).
// Handles both:
//   1. Smooth case (k^2 << v_bar * Sigma * v_bar^T): prior distribution (mu, Sigma) is unchanged.
//   2. General non-smooth case: discrete Gaussian 1D coset update on (mu, Sigma) via Equations (18)-(19).
// In both cases, the primal lattice is sparsified to Lambda' = {x in Lambda : <x, v_bar> = 0 (mod k)}
// using dual lattice densification (adjoining v_bar_V / k) and exact linear dependency elimination.

#include "dbdd/core/types.hpp"
#include "dbdd/hints/hint.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/lattice.hpp"
#include <utility>
#include <string>

namespace dbdd {

class DBDDInstance;

/**
 * @brief Represents a modular linear hint <s, v> = l (mod k).
 * 
 * In the DBDD framework, this hint restricts the secret search space to a periodic coset modulo k.
 * Homogenization converts it to an orthogonal modular constraint <s_bar, v_bar> = 0 (mod k)
 * against the embedded target vector s_bar = (e, s, 1).
 */
class ModularHint : public Hint {
public:
    /**
     * @brief Constructor for integer modular hint <s, v> = l (mod k).
     * @param v Hint coefficient vector.
     * @param l Known scalar residue.
     * @param k Modulus (must be >= 2).
     * @param smoothness_threshold Factor tau defining smooth regime when sigma_v >= tau * k (default: 2.0).
     */
    ModularHint(VectorZZ v, BigInt l, BigInt k, double smoothness_threshold = 2.0);

    /**
     * @brief Constructor for floating-point modular hint <s, v> = l (mod k).
     * @param v Hint coefficient vector.
     * @param l Known scalar residue.
     * @param k Modulus (must be >= 2.0).
     * @param smoothness_threshold Factor tau defining smooth regime (default: 2.0).
     */
    ModularHint(VectorDouble v, double l, double k, double smoothness_threshold = 2.0);

    // Factory methods
    static ModularHint from_secret(VectorZZ v, BigInt l, BigInt k, double smoothness_threshold = 2.0);
    static ModularHint from_secret(VectorDouble v, double l, double k, double smoothness_threshold = 2.0);
    static ModularHint from_homogenized(VectorZZ v_bar, BigInt k, double smoothness_threshold = 2.0);
    static ModularHint from_homogenized(VectorDouble v_bar, double k, double smoothness_threshold = 2.0);

    // Hint interface implementation
    void apply(DBDDInstance& instance) const override;
    [[nodiscard]] HintType type() const override { return HintType::Modular; }
    [[nodiscard]] std::string description() const override;

    // Accessors
    [[nodiscard]] const VectorZZ& hint_vector_int() const;
    [[nodiscard]] const BigInt& hint_scalar_int() const;
    [[nodiscard]] const BigInt& modulus_int() const;
    [[nodiscard]] const VectorDouble& hint_vector_double() const;
    [[nodiscard]] double hint_scalar_double() const;
    [[nodiscard]] double modulus_double() const;
    [[nodiscard]] bool is_integer() const noexcept { return is_integer_; }
    [[nodiscard]] bool is_already_homogenized() const noexcept { return is_already_homogenized_; }
    [[nodiscard]] double smoothness_threshold() const noexcept { return smoothness_threshold_; }

    /**
     * @brief Generates the homogenized hint vector v_bar in ZZ^{ambient_dim}.
     * If v is already homogenized, validates its dimension.
     * If v has length ambient_dim - 1 (representing (e, s)), forms (v, -l).
     * If v has length n < ambient_dim - 1 (representing secret s), forms (0_m, v, -l).
     */
    [[nodiscard]] VectorZZ homogenize_int(size_t ambient_dim) const;

    /**
     * @brief Generates the homogenized hint vector v_bar in RR^{ambient_dim}.
     */
    [[nodiscard]] VectorDouble homogenize_double(size_t ambient_dim) const;

    /**
     * @brief Tests whether the hint falls in the smooth regime for the given DBDD instance:
     * sigma_v >= smoothness_threshold * k.
     */
    [[nodiscard]] bool is_smooth(const DBDDInstance& instance) const;

    // =========================================================================
    // Static Mathematical Helper Routines
    // =========================================================================

    /**
     * @brief Paper Reference: Albrecht et al. Section 3.2
     * Tests whether directional variance sigma_v^2 falls in the smooth regime:
     * sqrt(sigma_v_sq) >= threshold * k.
     */
    static bool check_smoothness(double directional_variance, double k, double threshold = 2.0);

    /**
     * @brief Paper Reference: Albrecht et al. Section 3.2
     * Mathematical Purpose: Computes 1D discrete Gaussian mean and variance over coset {l + j * k : j in Z}.
     * 
     * Uses numerically stable exponential summation centered at j_0 = round((mu - l) / k).
     * Normalizes weights by the peak term exp(-Delta_0^2 / (2 * sigma_sq)) to prevent underflow/overflow.
     * 
     * @param mu Prior mean of scalar projection.
     * @param sigma_sq Prior variance of scalar projection.
     * @param k Periodicity / modulus.
     * @param l Coset shift (default 0.0 for homogenized constraints).
     * @return std::pair<double, double> {mu_scalar, sigma_scalar_sq}
     */
    static std::pair<double, double> discrete_gaussian_1d_coset(
        double mu,
        double sigma_sq,
        double k,
        double l = 0.0);

    /**
     * @brief Paper Reference: Albrecht et al. Section 3.2 / Equations (18) and (19)
     * Mathematical Purpose: Directional Gaussian distribution update along v_bar.
     * 
     * Formulas:
     *     w = v_bar * Sigma (1 x d)
     *     sigma_Y^2 = v_bar * Sigma * v_bar^T
     *     mu_Y = <mu, v_bar>
     *     Compute (mu_scalar, sigma_scalar^2) via discrete_gaussian_1d_coset
     *     mu' = mu + ((mu_scalar - mu_Y) / sigma_Y^2) * w
     *     Sigma' = Sigma - ((sigma_Y^2 - sigma_scalar^2) / (sigma_Y^2)^2) * w^T * w
     * 
     * Inputs:
     *     mean: Prior mean vector mu in R^d
     *     cov: Prior covariance matrix Sigma in R^{d x d} (symmetric PSD)
     *     v_bar: Homogenized hint vector in R^d
     *     k: Modulus
     * Outputs:
     *     std::pair<VectorDouble, MatrixDouble> {mu', Sigma'}
     * Preconditions:
     *     cov is symmetric PSD, dimensions match, k > 0.
     * Postconditions:
     *     Sigma' is symmetric PSD, Sigma - Sigma' is PSD (uncertainty never increases).
     */
    static std::pair<VectorDouble, MatrixDouble> update_distribution(
        const VectorDouble& mean,
        const MatrixDouble& cov,
        const VectorDouble& v_bar,
        double k);

    /**
     * @brief Paper Reference: Albrecht et al. Section 3.2
     * Mathematical Purpose: Sparsifies primal lattice basis Lambda -> Lambda' = {x in Lambda : <x, v_bar> = 0 (mod k)}.
     * 
     * Algorithm (Dual-lattice densification & dependency elimination):
     *     1. Let u = v_bar * B^T in Z^d. If gcd(u, k) == k, all basis vectors already satisfy congruence -> return B.
     *     2. Compute dual basis D in Q^{d x m} satisfying B * D^T = I_d.
     *     3. Compute adjoined dual vector w = v_bar_V / k = (u / k) * D in Q^m.
     *     4. Form dependency vector c = (-u_1, ..., -u_d, k) in Z^{d+1} satisfying c * [D; w] = 0.
     *     5. Divide c by g = gcd(c) to obtain primitive vector a = c / g.
     *     6. Complete a to unimodular U in GL_{d+1}(Z) with last row a.
     *     7. Compute U * [D; w]; last row is exactly 0. Extract first d rows as full-rank dual basis D'_raw in Q^{d x m}.
     *     8. LLL-reduce D'_raw using exact arithmetic -> D'_red.
     *     9. Primalize: B'_raw = primal_from_dual(D'_red).
     *    10. LLL-reduce primal integer basis -> B'.
     * 
     * Inputs:
     *     basis: Original primal basis B of rank d
     *     v_bar: Homogenized integer vector
     *     k: Modulus
     * Outputs:
     *     BasisZZ of dimension d spanning the sparsified lattice Lambda'
     */
    static BasisZZ sparsify_lattice(
        const BasisZZ& basis,
        const VectorZZ& v_bar,
        const BigInt& k);

    /**
     * @brief Paper Reference: Albrecht et al. Section 3.2
     * Updates lattice state and cached volume: Vol(Lambda') = Vol(Lambda) * (k / gcd(k, u)).
     */
    static LatticeZZ update_lattice(
        const LatticeZZ& lattice,
        const VectorZZ& v_bar,
        const BigInt& k);

private:
    VectorZZ v_int_;
    BigInt l_int_;
    BigInt k_int_;
    VectorDouble v_double_;
    double l_double_{0.0};
    double k_double_{2.0};
    double smoothness_threshold_{2.0};
    bool is_integer_{true};
    bool is_already_homogenized_{false};
};

} // namespace dbdd
