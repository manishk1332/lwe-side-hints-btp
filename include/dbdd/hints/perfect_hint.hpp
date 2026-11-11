#pragma once

// Paper Reference: Dachman-Soled et al. Section 3.1 / Equations (12), (13) / docs/mathematical_conventions.md
// Mathematical Purpose: Perfect linear hint <s, v> = l integration.
// Homogenizes to <s_bar, v_bar> = 0, updates prior distribution (mu, Sigma) via
// conditional Gaussian formulas, and restricts lattice Lambda -> Lambda \cap v_bar^\perp
// via dual-lattice projection and inversion.

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
 * @brief Represents an exact (noiseless) linear hint <s, v> = l.
 * 
 * In the DBDD framework, this hint restricts the secret search space to an affine hyperplane.
 * Homogenization converts it to an orthogonal constraint <s_bar, v_bar> = 0 against the
 * embedded target vector s_bar = (e, s, 1).
 */
class PerfectHint : public Hint {
public:
    /**
     * @brief Constructor for integer hint relation <s, v> = l.
     * @param v Hint coefficient vector.
     * @param l Known scalar inner product.
     */
    PerfectHint(VectorZZ v, BigInt l);

    /**
     * @brief Constructor for floating-point hint relation <s, v> = l.
     * @param v Hint coefficient vector.
     * @param l Known scalar inner product.
     */
    PerfectHint(VectorDouble v, double l);

    // Factory methods
    static PerfectHint from_secret(VectorZZ v, BigInt l);
    static PerfectHint from_secret(VectorDouble v, double l);
    static PerfectHint from_homogenized(VectorZZ v_bar);
    static PerfectHint from_homogenized(VectorDouble v_bar);

    // Hint interface implementation
    void apply(DBDDInstance& instance) const override;
    [[nodiscard]] HintType type() const override { return HintType::Perfect; }
    [[nodiscard]] std::string description() const override;

    // Accessors
    [[nodiscard]] const VectorZZ& hint_vector_int() const;
    [[nodiscard]] const BigInt& hint_scalar_int() const;
    [[nodiscard]] const VectorDouble& hint_vector_double() const;
    [[nodiscard]] double hint_scalar_double() const;
    [[nodiscard]] bool is_integer() const noexcept { return is_integer_; }
    [[nodiscard]] bool is_already_homogenized() const noexcept { return is_already_homogenized_; }

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

    // =========================================================================
    // Static Mathematical Helper Routines
    // =========================================================================

    /**
     * @brief Paper Reference: Dachman-Soled et al. Equations (12) and (13)
     * Mathematical Purpose: Updates mean mu and covariance Sigma conditioned on <x, v_bar> = 0.
     * 
     * Formulas:
     *     Sigma' = Sigma - (v_bar * Sigma)^T * (v_bar * Sigma) / (v_bar * Sigma * v_bar^T)
     *     mu' = mu - (<mu, v_bar> / (v_bar * Sigma * v_bar^T)) * (v_bar * Sigma)
     * 
     * Inputs:
     *     mean: Row vector mu in R^d
     *     cov: Covariance matrix Sigma in R^{d x d} (symmetric PSD)
     *     v_bar: Homogenized hint row vector in R^d
     * Outputs:
     *     std::pair<VectorDouble, MatrixDouble> containing {mu', Sigma'}
     * Preconditions:
     *     cov is symmetric PSD, dimensions match.
     * Postconditions:
     *     Sigma' is symmetric PSD with rank(Sigma') = rank(Sigma) - 1 (if not degenerate),
     *     v_bar * Sigma' = 0, and <mu', v_bar> = 0.
     * Precision & Complexity:
     *     Floating-point dense lin-alg, O(d^2) operations.
     */
    static std::pair<VectorDouble, MatrixDouble> update_distribution(
        const VectorDouble& mean,
        const MatrixDouble& cov,
        const VectorDouble& v_bar);

    /**
     * @brief Paper Reference: Dachman-Soled et al. Section 3.1
     * Mathematical Purpose: Restricts the lattice Lambda to the orthogonal hyperplane Lambda \cap v_bar^\perp.
     * 
     * Algorithm (Dual-lattice hyperplane intersection):
     *     1. Compute dual basis D in Q^{d x m} such that B * D^T = I_d.
     *     2. Project D onto v_bar^\perp: D_proj = D * P_{v_bar}^\perp.
     *     3. Find primitive integer null vector a = (v_bar * B^T) / gcd(v_bar * B^T) in Z^d.
     *     4. Construct unimodular completion U in GL_d(Z) with final row a.
     *     5. Compute U * D_proj; the final row is 0. Extract first d-1 rows as D_sub.
     *     6. Run exact LLL on D_sub to obtain reduced dual basis D'_red.
     *     7. Invert D'_red to obtain primal basis B' = primal_from_dual(D'_red).
     *     8. LLL-reduce B' and update cached volume via Vol(Lambda') = Vol(Lambda) * ||v_bar*_{Lambda^*}||.
     * 
     * Inputs:
     *     lattice: Original lattice Lambda of rank d
     *     v_bar: Homogenized integer vector defining the hyperplane
     * Outputs:
     *     LatticeZZ of dimension d - 1 spanning Lambda \cap v_bar^\perp
     * Preconditions:
     *     lattice.dimension() >= 2, v_bar is not orthogonal to lattice.
     * Postconditions:
     *     dim(Lambda') = d - 1, Lambda' is a sublattice of Lambda, all vectors orthogonal to v_bar.
     * Precision & Complexity:
     *     Exact rational arithmetic, polynomial time via exact LLL.
     */
    static LatticeZZ update_lattice(
        const LatticeZZ& lattice,
        const VectorZZ& v_bar);

    /**
     * @brief Computes a reduced primal basis for basis matrix B intersected with v_bar^\perp.
     */
    static BasisZZ intersect_hyperplane(
        const BasisZZ& basis,
        const VectorZZ& v_bar);

    /**
     * @brief Computes the Euclidean norm of the primitive dual vector parallel to v_bar:
     * w = a * D = (v_bar * B^T / gcd(v_bar * B^T)) * D in Lambda^*.
     */
    static BigReal primitive_dual_norm(
        const BasisZZ& primal_basis,
        const VectorZZ& v_bar);

    /**
     * @brief Completes a primitive integer vector a in Z^d to a unimodular matrix U in GL_d(Z)
     * such that det(U) = +/- 1 and row d-1 of U equals a.
     * Uses iterated 2x2 extended Euclidean transformations.
     */
    static MatrixZZ complete_primitive_vector_unimodular(const VectorZZ& a);

private:
    VectorZZ v_int_;
    BigInt l_int_;
    VectorDouble v_double_;
    double l_double_{0.0};
    bool is_integer_{true};
    bool is_already_homogenized_{false};
};

} // namespace dbdd
