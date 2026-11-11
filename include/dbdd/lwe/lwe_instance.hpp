#pragma once

// Paper Reference: Dachman-Soled et al. Section 2 / docs/mathematical_conventions.md Section 4
// Mathematical Purpose: Representation and generation of LWE instances:
//     - Public parameters: n (dimension), m (samples), q (modulus)
//     - Public matrix: A in Z_q^{m x n}
//     - Target row vector: b = s * A^T + e mod q in Z_q^m
//     - Encapsulated test-only ground truth: secret s in Z^n, error e in Z^m
// Preconditions:
//     - n > 0, m > 0, q >= 2
//     - A has dimensions m x n, b has dimension m
// Postconditions:
//     - verify_secret(candidate_s) validates reconstructed secrets against noise bounds or ground truth

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lwe/distribution.hpp"
#include <cstddef>
#include <optional>
#include <random>

namespace dbdd {

class LWEInstance {
public:
    // Public instance constructor (attacker's view, without ground truth)
    LWEInstance(size_t n, size_t m, BigInt q, MatrixZZ A, VectorZZ b);

    // Full instance constructor (with test-only ground truth secret and error)
    LWEInstance(size_t n, size_t m, BigInt q, MatrixZZ A, VectorZZ b,
                VectorZZ secret, VectorZZ error);

    // Static instance generators
    static LWEInstance generate_random(
        size_t n, size_t m, const BigInt& q,
        Distribution& secret_dist, Distribution& error_dist,
        std::mt19937_64& rng);

    static LWEInstance generate_random(
        size_t n, size_t m, const BigInt& q,
        Distribution& secret_dist, Distribution& error_dist,
        uint64_t seed = 42);

    // Dimension and parameter accessors
    [[nodiscard]] size_t n() const noexcept { return n_; }
    [[nodiscard]] size_t m() const noexcept { return m_; }
    [[nodiscard]] const BigInt& q() const noexcept { return q_; }

    // Public matrix and vector accessors (Row-basis convention: b is row vector 1 x m)
    [[nodiscard]] const MatrixZZ& A() const noexcept { return A_; }
    [[nodiscard]] const VectorZZ& b() const noexcept { return b_; }

    // Ground truth status and accessors (Strict separation from public API)
    [[nodiscard]] bool has_ground_truth() const noexcept {
        return secret_.has_value() && error_.has_value();
    }

    [[nodiscard]] const VectorZZ& test_only_secret() const;
    [[nodiscard]] const VectorZZ& test_only_error() const;

    // Secret verification methods
    [[nodiscard]] bool verify_secret(const VectorZZ& candidate_s) const;
    [[nodiscard]] bool verify_secret(const VectorZZ& candidate_s, double max_noise_norm) const;

    // Static sampling helpers
    static VectorZZ sample_secret(size_t n, Distribution& dist, std::mt19937_64& rng);
    static VectorZZ sample_error(size_t m, Distribution& dist, std::mt19937_64& rng);
    static MatrixZZ sample_uniform_matrix(size_t rows, size_t cols, const BigInt& q, std::mt19937_64& rng);

private:
    void validate_dimensions() const;

    size_t n_;
    size_t m_;
    BigInt q_;
    MatrixZZ A_;
    VectorZZ b_;
    std::optional<VectorZZ> secret_;
    std::optional<VectorZZ> error_;
};

} // namespace dbdd
