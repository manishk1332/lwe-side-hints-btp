#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/dbdd/homogenization.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/core/exceptions.hpp"

#include <Eigen/Dense>

using namespace dbdd;
using Catch::Approx;

TEST_CASE("Homogenization: Analytical 1D and 2D Exact Match", "[dbdd][homogenization]") {
    SECTION("1D Scalar Test") {
        VectorDouble mean({3.0});
        MatrixDouble cov(1, 1);
        cov(0, 0) = 4.0;

        auto [mu_h, cov_h] = Homogenization::homogenize(mean, cov);

        REQUIRE(mu_h.size() == 1);
        REQUIRE(mu_h[0] == Approx(0.0).margin(1e-15));
        REQUIRE(cov_h.rows() == 1);
        REQUIRE(cov_h.cols() == 1);
        // Sigma_h = 4.0 + 3.0 * 3.0 = 13.0
        REQUIRE(cov_h(0, 0) == Approx(13.0).epsilon(1e-14));
    }

    SECTION("2D Correlated Covariance with Non-Zero Mean") {
        VectorDouble mean({1.5, -2.0});
        MatrixDouble cov(2, 2);
        cov(0, 0) = 2.0; cov(0, 1) = 0.5;
        cov(1, 0) = 0.5; cov(1, 1) = 3.0;

        auto [mu_h, cov_h] = Homogenization::homogenize(mean, cov);

        REQUIRE(mu_h.size() == 2);
        REQUIRE(mu_h[0] == Approx(0.0).margin(1e-15));
        REQUIRE(mu_h[1] == Approx(0.0).margin(1e-15));

        // Expected outer product:
        // [ 1.5 * 1.5,   1.5 * -2.0 ] = [ 2.25, -3.00 ]
        // [ -2.0 * 1.5, -2.0 * -2.0 ] = [ -3.00,  4.00 ]
        // Expected Sigma_h:
        // [ 2.0 + 2.25,  0.5 - 3.0 ] = [ 4.25, -2.50 ]
        // [ 0.5 - 3.0,   3.0 + 4.0 ] = [ -2.50,  7.00 ]
        REQUIRE(cov_h(0, 0) == Approx(4.25).epsilon(1e-14));
        REQUIRE(cov_h(0, 1) == Approx(-2.50).epsilon(1e-14));
        REQUIRE(cov_h(1, 0) == Approx(-2.50).epsilon(1e-14));
        REQUIRE(cov_h(1, 1) == Approx(7.00).epsilon(1e-14));
    }

    SECTION("Dimension Mismatch Exceptions") {
        VectorDouble mean({1.0, 2.0});
        MatrixDouble non_square(2, 3);
        REQUIRE_THROWS_AS(Homogenization::homogenize(mean, non_square), DimensionMismatchException);

        MatrixDouble wrong_dim(3, 3);
        REQUIRE_THROWS_AS(Homogenization::homogenize(mean, wrong_dim), DimensionMismatchException);
    }
}

TEST_CASE("Homogenization: Ellipsoid Envelope Inclusion Invariant", "[dbdd][homogenization]") {
    SECTION("2D Non-trivial Ellipsoid Inclusion") {
        VectorDouble mean({1.2, -0.8});
        MatrixDouble cov(2, 2);
        cov(0, 0) = 3.0; cov(0, 1) = 1.0;
        cov(1, 0) = 1.0; cov(1, 1) = 2.0;

        Ellipsoid orig(mean, cov);
        Ellipsoid homog = Homogenization::homogenize(orig);

        REQUIRE(homog.mean().size() == 2);
        REQUIRE(homog.mean()[0] == Approx(0.0).margin(1e-15));
        REQUIRE(homog.mean()[1] == Approx(0.0).margin(1e-15));
        // Center mean is strictly inside homog ellipsoid
        REQUIRE(homog.mahalanobis_distance_squared(orig.mean()) < 1.0);
        // Universal inclusion envelope E(mu, Sigma) subset E(0, 2 Sigma_h)
        REQUIRE(Homogenization::verify_inclusion(orig, homog, 2.0, 100));
    }

    SECTION("3D Correlated Ellipsoid Inclusion") {
        VectorDouble mean({0.5, -1.0, 2.0});
        MatrixDouble cov(3, 3);
        cov(0, 0) = 4.0; cov(0, 1) = 0.5; cov(0, 2) = -0.2;
        cov(1, 0) = 0.5; cov(1, 1) = 3.0; cov(1, 2) = 0.1;
        cov(2, 0) = -0.2; cov(2, 1) = 0.1; cov(2, 2) = 2.0;

        Ellipsoid orig(mean, cov);
        Ellipsoid homog = Homogenization::homogenize(orig);

        REQUIRE(Homogenization::verify_inclusion(orig, homog, 2.0, 200));
    }

    SECTION("Degenerate Ellipsoid Support Inclusion") {
        // 3D space with rank 2 covariance (subspace ellipsoid)
        VectorDouble mean({1.0, 2.0, 0.0});
        MatrixDouble cov(3, 3);
        cov(0, 0) = 2.0; cov(0, 1) = 0.0; cov(0, 2) = 0.0;
        cov(1, 0) = 0.0; cov(1, 1) = 3.0; cov(1, 2) = 0.0;
        cov(2, 0) = 0.0; cov(2, 1) = 0.0; cov(2, 2) = 0.0;

        Ellipsoid orig(mean, cov);
        REQUIRE(orig.rank() == 2);

        Ellipsoid homog = Homogenization::homogenize(orig);
        REQUIRE(Homogenization::verify_inclusion(orig, homog, 2.0, 100));
    }

    SECTION("Zero-Mean Ellipsoid Exact Inclusion (Threshold 1.0)") {
        VectorDouble mean({0.0, 0.0});
        MatrixDouble cov(2, 2);
        cov(0, 0) = 2.0; cov(0, 1) = 0.3;
        cov(1, 0) = 0.3; cov(1, 1) = 1.5;

        Ellipsoid orig(mean, cov);
        Ellipsoid homog = Homogenization::homogenize(orig);

        // When mu = 0, Sigma_h = Sigma and threshold 1.0 holds exactly
        REQUIRE(Homogenization::verify_inclusion(orig, homog, 1.0, 100));
    }
}

TEST_CASE("Homogenization: Rank Increment Property", "[dbdd][homogenization]") {
    SECTION("Kannan Embedding Style: mu not in col(Sigma) strictly increments rank") {
        // 4D space with rank 3 covariance (last coordinate has 0 variance)
        VectorDouble mean({0.0, 0.0, 0.0, 1.0});
        MatrixDouble cov = MatrixDouble::identity(4);
        cov(3, 3) = 0.0; // rank 3

        Ellipsoid orig(mean, cov);
        REQUIRE(orig.dimension() == 4);
        REQUIRE(orig.rank() == 3);
        REQUIRE(orig.is_degenerate());

        Ellipsoid homog = Homogenization::homogenize(orig);
        REQUIRE(homog.dimension() == 4);
        REQUIRE(homog.rank() == 4); // Incremented from 3 to 4!
        REQUIRE(homog.is_full_rank());
    }

    SECTION("mu in col(Sigma): rank remains unchanged") {
        // 3D space with rank 2 covariance, mean in span
        VectorDouble mean({1.0, 2.0, 0.0});
        MatrixDouble cov = MatrixDouble::identity(3);
        cov(2, 2) = 0.0; // rank 2, null direction is (0, 0, 1), perpendicular to mean

        Ellipsoid orig(mean, cov);
        REQUIRE(orig.rank() == 2);

        Ellipsoid homog = Homogenization::homogenize(orig);
        REQUIRE(homog.rank() == 2); // Rank unchanged
    }

    SECTION("Full Rank Covariance: rank remains full") {
        VectorDouble mean({1.0, -1.0});
        MatrixDouble cov = MatrixDouble::identity(2);

        Ellipsoid orig(mean, cov);
        REQUIRE(orig.rank() == 2);

        Ellipsoid homog = Homogenization::homogenize(orig);
        REQUIRE(homog.rank() == 2);
    }
}

TEST_CASE("Homogenization: Covariance Symmetry and PSD Guarantees", "[dbdd][homogenization]") {
    VectorDouble mean({2.5, -1.3, 0.7});
    MatrixDouble cov(3, 3);
    cov(0, 0) = 5.0; cov(0, 1) = 1.2; cov(0, 2) = 0.3;
    cov(1, 0) = 1.2; cov(1, 1) = 3.0; cov(1, 2) = -0.5;
    cov(2, 0) = 0.3; cov(2, 1) = -0.5; cov(2, 2) = 2.0;

    auto [mu_h, cov_h] = Homogenization::homogenize(mean, cov);

    // 1. Strict symmetry check
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(cov_h(i, j) == Approx(cov_h(j, i)).margin(1e-15));
        }
    }

    // 2. Eigenvalue positivity check on Sigma_h
    Eigen::MatrixXd em = cov_h.to_eigen();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(em);
    REQUIRE(solver.info() == Eigen::Success);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(solver.eigenvalues()(i) >= -1e-12);
    }

    // 3. Difference matrix Sigma_h - Sigma = mu^T * mu must be positive semidefinite
    MatrixDouble diff = cov_h - cov;
    Eigen::MatrixXd diff_em = diff.to_eigen();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> diff_solver(diff_em);
    REQUIRE(diff_solver.info() == Eigen::Success);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(diff_solver.eigenvalues()(i) >= -1e-12);
    }
}

TEST_CASE("Homogenization: Zero Lattice Mutation Invariant", "[dbdd][homogenization]") {
    size_t n = 3;
    size_t m = 3;
    BigInt q(11);
    DiscreteGaussian noise_dist(1.0, 111);
    DiscreteGaussian secret_dist(1.0, 222);

    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 12345);
    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    // Snapshot pre-homogenization state
    const MatrixZZ orig_basis = dbdd.matrix();
    const BigReal orig_vol = dbdd.volume();
    const size_t orig_dim = dbdd.dimension();
    const size_t orig_amb = dbdd.ambient_dimension();
    const VectorZZ orig_target = dbdd.target();
    const VectorZZ orig_coords = dbdd.target_coordinates();

    REQUIRE_FALSE(dbdd.is_homogenized());

    // Apply homogenization
    dbdd.homogenize();

    // Verify all lattice invariants are strictly preserved
    REQUIRE(dbdd.is_homogenized());
    REQUIRE(dbdd.dimension() == orig_dim);
    REQUIRE(dbdd.ambient_dimension() == orig_amb);
    REQUIRE(dbdd.volume() == orig_vol);
    REQUIRE(dbdd.matrix() == orig_basis);
    REQUIRE(dbdd.target() == orig_target);
    REQUIRE(dbdd.target_coordinates() == orig_coords);

    // Verify mean is zero
    REQUIRE(dbdd.mean().squared_norm() == Approx(0.0).margin(1e-15));
}

TEST_CASE("Homogenization: LWE Kannan Target Containment Post-Homogenization", "[dbdd][homogenization]") {
    SECTION("Toy LWE Instance (n=3, m=3, q=11)") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(11);
        DiscreteGaussian noise_dist(1.0, 42);
        DiscreteGaussian secret_dist(1.0, 43);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 44);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        REQUIRE(dbdd.verify_target());

        // Homogenize
        dbdd.homogenize();

        // Target (e, s, 1) must remain verified post-homogenization
        REQUIRE(dbdd.verify_target());

        // Mahalanobis distance squared of target: s_bar * Sigma_h^dagger * s_bar^T <= chi2_threshold
        double mahalanobis_sq = dbdd.target_mahalanobis_distance_squared();
        double threshold = dbdd.chi2_threshold();
        REQUIRE(mahalanobis_sq <= threshold);
    }

    SECTION("Medium LWE Instance (n=8, m=8, q=97)") {
        size_t n = 8;
        size_t m = 8;
        BigInt q(97);
        DiscreteGaussian noise_dist(1.5, 991);
        DiscreteGaussian secret_dist(1.5, 992);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 999);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        REQUIRE(dbdd.verify_target());

        dbdd.homogenize();

        REQUIRE(dbdd.verify_target());
        REQUIRE(dbdd.target_mahalanobis_distance_squared() <= dbdd.chi2_threshold());
    }
}
