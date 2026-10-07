#include <catch2/catch_test_macros.hpp>
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"

using namespace dbdd;

TEST_CASE("LWEInstance: parameter validation", "[lwe][instance]") {
    BigInt q(11);
    MatrixZZ A = MatrixZZ::zero(3, 2);
    VectorZZ b = VectorZZ::zero(3);

    SECTION("Invalid n or m throws InvalidParameterException") {
        REQUIRE_THROWS_AS(LWEInstance(0, 3, q, A, b), InvalidParameterException);
        REQUIRE_THROWS_AS(LWEInstance(2, 0, q, A, b), InvalidParameterException);
    }

    SECTION("Invalid modulus q < 2 throws InvalidParameterException") {
        REQUIRE_THROWS_AS(LWEInstance(2, 3, BigInt(1), A, b), InvalidParameterException);
        REQUIRE_THROWS_AS(LWEInstance(2, 3, BigInt(0), A, b), InvalidParameterException);
    }

    SECTION("Matrix A dimensions mismatch throws DimensionMismatchException") {
        MatrixZZ bad_A = MatrixZZ::zero(4, 2);
        REQUIRE_THROWS_AS(LWEInstance(2, 3, q, bad_A, b), DimensionMismatchException);
    }

    SECTION("Vector b dimension mismatch throws DimensionMismatchException") {
        VectorZZ bad_b = VectorZZ::zero(2);
        REQUIRE_THROWS_AS(LWEInstance(2, 3, q, A, bad_b), DimensionMismatchException);
    }

    SECTION("Ground-truth secret or error dimension mismatch throws DimensionMismatchException") {
        VectorZZ s_ok({BigInt(1), BigInt(2)});
        VectorZZ e_ok({BigInt(0), BigInt(0), BigInt(0)});
        VectorZZ s_bad({BigInt(1)});
        VectorZZ e_bad({BigInt(0), BigInt(0)});

        REQUIRE_THROWS_AS(LWEInstance(2, 3, q, A, b, s_bad, e_ok), DimensionMismatchException);
        REQUIRE_THROWS_AS(LWEInstance(2, 3, q, A, b, s_ok, e_bad), DimensionMismatchException);
    }

    SECTION("Inconsistent ground-truth throws InvalidParameterException") {
        // A is zero, so s * A^T = 0
        // If s = (1, 2) and e = (1, 0, 0), then b must be (1, 0, 0) mod 11
        // But b is (0, 0, 0), so it should throw
        VectorZZ s({BigInt(1), BigInt(2)});
        VectorZZ e({BigInt(1), BigInt(0), BigInt(0)});
        REQUIRE_THROWS_AS(LWEInstance(2, 3, q, A, b, s, e), InvalidParameterException);
    }
}

TEST_CASE("LWEInstance: random generation and ground-truth verification", "[lwe][instance]") {
    size_t n = 5;
    size_t m = 8;
    BigInt q(97);

    DiscreteGaussian noise_dist(2.0, 101);
    UniformTernary secret_dist(2.0 / 3.0, 202);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist);

    REQUIRE(instance.n() == n);
    REQUIRE(instance.m() == m);
    REQUIRE(instance.q() == q);
    REQUIRE(instance.A().rows() == m);
    REQUIRE(instance.A().cols() == n);
    REQUIRE(instance.b().size() == m);
    REQUIRE(instance.has_ground_truth());

    const auto& s = instance.test_only_secret();
    const auto& e = instance.test_only_error();
    REQUIRE(s.size() == n);
    REQUIRE(e.size() == m);

    // Verify row-vector multiplication b = s * A^T + e mod q
    VectorZZ s_AT = s * instance.A().transpose();
    for (size_t j = 0; j < m; ++j) {
        BigInt expected_b = BigInt::mod_floor(s_AT[j] + e[j], q);
        BigInt actual_b = BigInt::mod_floor(instance.b()[j], q);
        REQUIRE(expected_b == actual_b);
    }

    SECTION("verify_secret with ground-truth secret") {
        REQUIRE(instance.verify_secret(s));

        // Perturbed secret fails
        VectorZZ s_pert = s;
        s_pert[0] += BigInt(1);
        REQUIRE_FALSE(instance.verify_secret(s_pert));

        // Dimension mismatch throws
        VectorZZ s_wrong_dim(n + 1, BigInt(0));
        REQUIRE_THROWS_AS(instance.verify_secret(s_wrong_dim), DimensionMismatchException);
    }

    SECTION("verify_secret with bounded noise threshold") {
        REQUIRE(instance.verify_secret(s, 10.0));

        VectorZZ s_pert = s;
        s_pert[0] += BigInt(1);
        REQUIRE_FALSE(instance.verify_secret(s_pert, 5.0));
    }
}

TEST_CASE("LWEInstance: public instance without ground truth", "[lwe][instance]") {
    size_t n = 3;
    size_t m = 4;
    BigInt q(31);

    MatrixZZ A = MatrixZZ::identity(m);
    A = MatrixZZ(m, n, {
        BigInt(1), BigInt(2), BigInt(3),
        BigInt(4), BigInt(5), BigInt(6),
        BigInt(7), BigInt(8), BigInt(9),
        BigInt(10), BigInt(11), BigInt(12)
    });
    VectorZZ b({BigInt(5), BigInt(10), BigInt(15), BigInt(20)});

    LWEInstance pub_inst(n, m, q, A, b);
    REQUIRE_FALSE(pub_inst.has_ground_truth());

    REQUIRE_THROWS_AS(pub_inst.test_only_secret(), InvalidParameterException);
    REQUIRE_THROWS_AS(pub_inst.test_only_error(), InvalidParameterException);
}
