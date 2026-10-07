#pragma once

// Paper Reference: Albrecht et al. Section 2, Eq. (7) / docs/paper_mapping.md / docs/mathematical_conventions.md Section 4
// Mathematical Purpose: Exact Kannan embedding transforming an LWE instance (A, b) into an initial DBDD lattice basis:
//     B = [ q I_m   0     0 ]
//         [  A^T   -I_n   0 ]
//         [   b     0     1 ]
// Target vector:
//     s_bar = (e, s, 1)
// Invariants:
//     1. Target Invariant: (k, -s, 1) * B = (e, s, 1) = s_bar, where k = (e + s * A^T - b) / q in Z^m
//     2. Volume Invariant: Vol(Lambda) = q^m
//     3. Covariance Invariant: rank(Sigma) = d - 1 = m + n, Sigma is symmetric PSD with Sigma(d-1, d-1) = 0

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/lattice.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include <cstddef>
#include <optional>

namespace dbdd {

class KannanEmbedding {
public:
    // Factory methods matching paper mapping
    static KannanEmbedding from_lwe(
        const LWEInstance& instance,
        double sigma_e, double sigma_s,
        double mu_e = 0.0, double mu_s = 0.0);

    static KannanEmbedding from_lwe(
        const LWEInstance& instance,
        const Distribution& error_dist,
        const Distribution& secret_dist);

    // Direct constructor
    KannanEmbedding(
        const MatrixZZ& A, const VectorZZ& b, const BigInt& q,
        double sigma_e, double sigma_s,
        double mu_e = 0.0, double mu_s = 0.0,
        std::optional<VectorZZ> secret = std::nullopt,
        std::optional<VectorZZ> error = std::nullopt);

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return d_; }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return d_; }
    [[nodiscard]] size_t m() const noexcept { return m_; }
    [[nodiscard]] size_t n() const noexcept { return n_; }
    [[nodiscard]] const BigInt& q() const noexcept { return q_; }

    // Basis & Lattice accessors (Row-basis convention: rows are basis vectors)
    [[nodiscard]] const BasisZZ& basis() const noexcept { return basis_; }
    [[nodiscard]] const LatticeZZ& lattice() const noexcept { return lattice_; }
    [[nodiscard]] const MatrixZZ& matrix() const noexcept { return basis_.matrix(); }

    // Target vector accessors
    [[nodiscard]] bool has_target() const noexcept { return target_.has_value(); }
    [[nodiscard]] const VectorZZ& target() const;
    [[nodiscard]] const VectorZZ& target_coordinates() const;

    // Statistical accessors for DBDD ellipsoid
    [[nodiscard]] const VectorDouble& mean() const noexcept { return mean_; }
    [[nodiscard]] const MatrixDouble& covariance() const noexcept { return covariance_; }

    // Volume accessors (Cached Vol(Lambda) = q^m)
    [[nodiscard]] BigInt exact_volume() const { return BigInt::pow(q_, static_cast<unsigned long>(m_)); }
    [[nodiscard]] BigReal volume() const { return lattice_.volume(); }
    [[nodiscard]] BigReal log_volume() const { return lattice_.log_volume(); }

    // Verification methods
    [[nodiscard]] bool verify_target_invariant() const;
    [[nodiscard]] bool verify_covariance_invariants() const;

    // Static helper: computes integer vector k = (e + s * A^T - b) / q
    static VectorZZ compute_k_vector(
        const MatrixZZ& A, const VectorZZ& b, const BigInt& q,
        const VectorZZ& s, const VectorZZ& e);

private:
    size_t m_;
    size_t n_;
    size_t d_;
    BigInt q_;
    BasisZZ basis_;
    LatticeZZ lattice_;
    std::optional<VectorZZ> target_;
    std::optional<VectorZZ> coords_;
    VectorDouble mean_;
    MatrixDouble covariance_;
};

} // namespace dbdd
