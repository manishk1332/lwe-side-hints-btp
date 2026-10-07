#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/reduction/lll_exact.hpp"

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("KannanEmbedding: Toy Parameters (n=3, m=3, q=11)", "[lwe][embedding]") {
    size_t n = 3;
    size_t m = 3;
    BigInt q(11);
    size_t d = m + n + 1; // 7

    DiscreteGaussian noise_dist(1.5, 42);
    DiscreteGaussian secret_dist(2.0, 43);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 1001);
    auto kannan = KannanEmbedding::from_lwe(instance, noise_dist, secret_dist);

    SECTION("Dimensions and Structure") {
        REQUIRE(kannan.dimension() == d);
        REQUIRE(kannan.ambient_dimension() == d);
        REQUIRE(kannan.m() == m);
        REQUIRE(kannan.n() == n);
        REQUIRE(kannan.q() == q);

        const auto& B = kannan.matrix();
        REQUIRE(B.rows() == d);
        REQUIRE(B.cols() == d);

        // Check Block 0: q * I_m
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < d; ++j) {
                if (i == j) {
                    REQUIRE(B(i, j) == q);
                } else {
                    REQUIRE(B(i, j) == BigInt(0));
                }
            }
        }

        // Check Block 1: A^T and -I_n
        const auto& A = instance.A();
        for (size_t i = 0; i < n; ++i) {
            size_t row = m + i;
            for (size_t j = 0; j < m; ++j) {
                REQUIRE(B(row, j) == A(j, i));
            }
            for (size_t j = m; j < m + n; ++j) {
                if (j == m + i) {
                    REQUIRE(B(row, j) == BigInt(-1));
                } else {
                    REQUIRE(B(row, j) == BigInt(0));
                }
            }
            REQUIRE(B(row, d - 1) == BigInt(0));
        }

        // Check Block 2: b and 1
        const auto& b = instance.b();
        size_t row_b = m + n;
        for (size_t j = 0; j < m; ++j) {
            REQUIRE(B(row_b, j) == b[j]);
        }
        for (size_t j = m; j < m + n; ++j) {
            REQUIRE(B(row_b, j) == BigInt(0));
        }
        REQUIRE(B(row_b, d - 1) == BigInt(1));
    }

    SECTION("Target Invariant: (k, -s, 1) * B = (e, s, 1) = s_bar") {
        REQUIRE(kannan.has_target());
        const auto& target = kannan.target();
        const auto& coords = kannan.target_coordinates();

        REQUIRE(target.size() == d);
        REQUIRE(coords.size() == d);

        const auto& s = instance.test_only_secret();
        const auto& e = instance.test_only_error();

        // Check target coordinate components
        for (size_t j = 0; j < m; ++j) {
            REQUIRE(target[j] == e[j]);
        }
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(target[m + i] == s[i]);
        }
        REQUIRE(target[d - 1] == BigInt(1));

        // Check coords components
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(coords[m + i] == -s[i]);
        }
        REQUIRE(coords[d - 1] == BigInt(1));

        // Exact vector equation in Z^d
        VectorZZ actual_target = coords * kannan.basis().matrix();
        REQUIRE(actual_target == target);
        REQUIRE(kannan.verify_target_invariant());
    }

    SECTION("Volume Invariant: Vol(Lambda) = q^m = 11^3 = 1331") {
        BigInt expected_vol = BigInt::pow(q, 3); // 1331
        REQUIRE(expected_vol == BigInt(1331));

        // Exact determinant check via Bareiss algorithm on basis
        BigInt computed_vol = compute_volume_exact(kannan.basis());
        REQUIRE(computed_vol == expected_vol);

        REQUIRE(kannan.exact_volume() == expected_vol);
        REQUIRE_THAT(kannan.volume().to_double(), WithinAbs(1331.0, 1e-10));
    }

    SECTION("Covariance Invariants: Rank, Symmetry, PSD") {
        REQUIRE(kannan.verify_covariance_invariants());
        const auto& Sigma = kannan.covariance();

        REQUIRE(Sigma.rank() == d - 1); // 6
        REQUIRE(Sigma(d - 1, d - 1) == 0.0);

        double var_e = 1.5 * 1.5;
        double var_s = 2.0 * 2.0;

        for (size_t j = 0; j < m; ++j) {
            REQUIRE_THAT(Sigma(j, j), WithinAbs(var_e, 1e-12));
        }
        for (size_t i = 0; i < n; ++i) {
            REQUIRE_THAT(Sigma(m + i, m + i), WithinAbs(var_s, 1e-12));
        }
    }

    SECTION("Integration with Phase 03: LLL Reduction on Kannan Basis") {
        auto res = lll_exact(kannan.basis());
        REQUIRE(res.converged);
        REQUIRE(res.basis.dimension() == d);

        // Volume must be strictly preserved under unimodular LLL transformations
        BigInt reduced_vol = compute_volume_exact(res.basis);
        REQUIRE(reduced_vol == BigInt(1331));
    }
}

TEST_CASE("KannanEmbedding: Literature Benchmark Parameters (n=70, m=70, q=3301, sigma=20)", "[lwe][embedding]") {
    size_t n = 70;
    size_t m = 70;
    BigInt q(3301);
    double sigma = 20.0;
    size_t d = m + n + 1; // 141

    DiscreteGaussian noise_dist(sigma, 2024);
    DiscreteGaussian secret_dist(sigma, 2025);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 9999);
    auto kannan = KannanEmbedding::from_lwe(instance, noise_dist, secret_dist);

    SECTION("Dimensions and Exact Target Invariant") {
        REQUIRE(kannan.dimension() == d);
        REQUIRE(kannan.verify_target_invariant());
    }

    SECTION("Volume Invariant for Large Dimension") {
        BigInt expected_vol = BigInt::pow(q, static_cast<unsigned long>(m));
        REQUIRE(kannan.exact_volume() == expected_vol);

        BigReal expected_log_vol = BigReal(q).log() * BigReal(static_cast<double>(m));
        REQUIRE_THAT(kannan.log_volume().to_double(), WithinAbs(expected_log_vol.to_double(), 1e-8));
    }

    SECTION("Covariance Invariants") {
        REQUIRE(kannan.verify_covariance_invariants());
        REQUIRE(kannan.covariance().rank() == d - 1); // 140
    }
}

TEST_CASE("KannanEmbedding: Error Handling & Public Instances", "[lwe][embedding]") {
    size_t n = 2;
    size_t m = 2;
    BigInt q(17);

    MatrixZZ A(2, 2, {BigInt(1), BigInt(2), BigInt(3), BigInt(4)});
    VectorZZ b({BigInt(5), BigInt(6)});

    SECTION("Standard deviation <= 0 throws InvalidParameterException") {
        REQUIRE_THROWS_AS(KannanEmbedding(A, b, q, 0.0, 1.0), InvalidParameterException);
        REQUIRE_THROWS_AS(KannanEmbedding(A, b, q, 1.0, -0.5), InvalidParameterException);
    }

    SECTION("Public instance without ground truth has no target") {
        LWEInstance pub_inst(n, m, q, A, b);
        auto kannan = KannanEmbedding::from_lwe(pub_inst, 1.0, 1.0);

        REQUIRE_FALSE(kannan.has_target());
        REQUIRE_THROWS_AS(kannan.target(), InvalidParameterException);
        REQUIRE_THROWS_AS(kannan.target_coordinates(), InvalidParameterException);
        REQUIRE_FALSE(kannan.verify_target_invariant());
    }
}
