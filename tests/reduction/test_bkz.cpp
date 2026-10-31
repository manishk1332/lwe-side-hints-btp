#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/reduction/bkz.hpp"
#include "dbdd/reduction/lll_float.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <vector>

using namespace dbdd;
using Catch::Approx;

TEST_CASE("BKZReduction - Base Cases and Validation", "[reduction][bkz]") {
    SECTION("Dimension 1 Base Case") {
        MatrixZZ B(1, 2);
        B(0, 0) = BigInt(3); B(0, 1) = BigInt(4);
        BasisZZ basis(B);

        BKZParameters params(2);
        auto res = bkz_reduce(basis, params);

        REQUIRE(res.converged);
        REQUIRE(res.basis.dimension() == 1);
        REQUIRE(res.transformation == MatrixZZ::identity(1));
        REQUIRE(res.basis.row(0) == basis.row(0));
    }

    SECTION("Parameter Validation") {
        MatrixZZ B_empty(0, 0);
        BasisZZ empty_basis(B_empty);
        REQUIRE_THROWS_AS(bkz_reduce(empty_basis), DimensionMismatchException);

        BKZParameters invalid_params;
        invalid_params.block_size = 1;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        invalid_params.block_size = 5;
        invalid_params.max_tours = 0;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        invalid_params.max_tours = 10;
        invalid_params.min_tours = 15;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        invalid_params.min_tours = 1;
        invalid_params.delta = BigRational(1, 5);
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        invalid_params.delta = BigRational(99, 100);
        invalid_params.timeout_seconds = -1.0;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        BKZReduction bkz;
        REQUIRE_THROWS_AS(bkz.set_oracle(nullptr), InvalidParameterException);
    }
}

TEST_CASE("BKZReduction - Invariants and Unimodularity", "[reduction][bkz]") {
    const size_t d = 6;
    MatrixZZ B(d, d, BigInt(0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            B(i, j) = BigInt(static_cast<long long>((i + 1) * (j + 2) + ((i == j) ? 25 : 0)));
        }
    }
    BasisZZ basis(B);
    BigReal orig_vol(compute_volume_exact(basis));

    SECTION("BKZ-2 LLL Size-Reduction and Unimodularity") {
        BKZParameters params(2, 10, 0.99);
        auto res = bkz_reduce(basis, params);

        REQUIRE(res.converged);
        REQUIRE(res.basis.dimension() == d);

        // Exact algebraic identity: B' = U * B
        MatrixZZ computed_B = res.transformation * basis.matrix();
        REQUIRE(res.basis.matrix() == computed_B);

        // Volume preservation: Vol(B') = Vol(B)
        BigReal red_vol(compute_volume_exact(res.basis));
        REQUIRE(red_vol.to_double() == Approx(orig_vol.to_double()).epsilon(1e-9));

        // Size reduction check on output basis
        MatrixDouble B_double(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                B_double(i, j) = res.basis.matrix()(i, j).to_double();
            }
        }
        auto gs = compute_gram_schmidt_mgs<double>(B_double);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < i; ++j) {
                REQUIRE(std::abs(gs.mu(i, j)) <= 0.5 + 1e-7);
            }
        }
    }

    SECTION("BKZ-4 Invariants and Profiles") {
        BKZParameters params(4, 10, 0.99);
        params.log_tours = true;
        auto res = bkz_reduce(basis, params);

        REQUIRE(res.basis.dimension() == d);
        REQUIRE(res.basis.matrix() == res.transformation * basis.matrix());

        BigReal red_vol(compute_volume_exact(res.basis));
        REQUIRE(red_vol.to_double() == Approx(orig_vol.to_double()).epsilon(1e-9));

        // Tour profiles recorded
        REQUIRE(res.tour_profiles.size() == res.tours_completed);
        REQUIRE(res.final_profile.size() == d);
        REQUIRE(res.total_svp_calls > 0);
    }
}

TEST_CASE("BKZReduction - Profile Progression across Block Sizes", "[reduction][bkz]") {
    // 10-dimensional test lattice
    const size_t d = 10;
    MatrixZZ B(d, d, BigInt(0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            B(i, j) = BigInt(static_cast<long long>(((i + 1) * 17 + (j + 1) * 31) % 40 + ((i == j) ? 45 : 0)));
        }
    }
    BasisZZ basis(B);

    BKZParameters params_beta2(2, 10, 0.99);
    auto res_beta2 = bkz_reduce(basis, params_beta2);

    BKZParameters params_beta5(5, 10, 0.99);
    auto res_beta5 = bkz_reduce(basis, params_beta5);

    // Higher block size beta yields <= first vector norm ||b_0||
    BigInt norm0_beta2 = res_beta2.basis.row(0).squared_norm();
    BigInt norm0_beta5 = res_beta5.basis.row(0).squared_norm();
    REQUIRE(norm0_beta5 <= norm0_beta2);
}

TEST_CASE("BKZReduction - End-to-End LWE Secret Recovery", "[reduction][bkz][lwe]") {
    // LWE instance: n = 3, m = 6, q = 23
    const size_t n = 3;
    const size_t m = 6;
    const BigInt q(23);

    DiscreteGaussian secret_dist(1.2, 101);
    DiscreteGaussian noise_dist(1.0, 102);

    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 4004);
    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);

    SECTION("Direct BKZ Attack on Kannan Lattice") {
        BKZParameters params(4, 15, 0.99);
        auto bkz_res = bkz_reduce(kannan.basis(), params);

        auto recovered_s = BKZReduction::recover_lwe_secret(bkz_res.basis, lwe);
        REQUIRE(recovered_s.has_value());
        REQUIRE(*recovered_s == lwe.test_only_secret());
    }

    SECTION("BKZ Attack on DBDD Instance with Perfect Hint") {
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

        // Provide 1 perfect hint revealing secret coordinate s_0
        VectorZZ v_hint(dbdd.ambient_dimension(), BigInt(0));
        v_hint[m] = BigInt(1); // s_0 is at index m
        dbdd.integrate_perfect_hint(v_hint, lwe.test_only_secret()[0]);

        // Run BKZ on the hint-integrated DBDD lattice
        BKZParameters params(4, 15, 0.99);
        auto bkz_res = bkz_reduce(dbdd.lattice().basis(), params);

        // Verify target vector recovery
        auto recovered_target = BKZReduction::recover_dbdd_secret(bkz_res.basis, dbdd);
        REQUIRE(recovered_target.has_value());
        REQUIRE(*recovered_target == dbdd.target());
    }
}
