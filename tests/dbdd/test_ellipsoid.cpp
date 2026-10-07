#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/dbdd/ellipsoid.hpp"
#include <cmath>
#include <numbers>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Ellipsoid: Symmetrization and Basic Properties", "[dbdd][ellipsoid]") {
    SECTION("Asymmetric input matrix is strictly symmetrized") {
        VectorDouble mean({1.0, 2.0});
        // Non-symmetric matrix with positive eigenvalues for symmetrized version:
        // [ 2  1 ]
        // [ 3  4 ] -> Symmetrized: [ 2 2 ] with trace=6, det=4 > 0
        //                          [ 2 4 ]
        MatrixDouble cov(2, 2, {2.0, 1.0, 3.0, 4.0});

        Ellipsoid ell(mean, cov);

        REQUIRE(ell.dimension() == 2);
        REQUIRE(ell.rank() == 2);
        REQUIRE(ell.is_full_rank());
        REQUIRE_FALSE(ell.is_degenerate());

        const auto& sym_cov = ell.covariance();
        REQUIRE_THAT(sym_cov(0, 0), WithinAbs(2.0, 1e-12));
        REQUIRE_THAT(sym_cov(0, 1), WithinAbs(2.0, 1e-12));
        REQUIRE_THAT(sym_cov(1, 0), WithinAbs(2.0, 1e-12));
        REQUIRE_THAT(sym_cov(1, 1), WithinAbs(4.0, 1e-12));
    }

    SECTION("Empty ellipsoid behaves correctly") {
        Ellipsoid ell;
        REQUIRE(ell.dimension() == 0);
        REQUIRE(ell.rank() == 0);
        REQUIRE(ell.empty());
    }

    SECTION("Standard spherical ellipsoid") {
        auto ell = Ellipsoid::standard(3);
        REQUIRE(ell.dimension() == 3);
        REQUIRE(ell.rank() == 3);
        REQUIRE(ell.is_full_rank());

        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(ell.mean()[i] == 0.0);
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(ell.covariance()(i, j) == (i == j ? 1.0 : 0.0));
            }
        }
    }
}

TEST_CASE("Ellipsoid: Positive Semidefiniteness and Eigenvalue Validation", "[dbdd][ellipsoid]") {
    SECTION("Strictly non-PSD matrix throws NonPositiveSemidefiniteException") {
        VectorDouble mean({0.0, 0.0});
        // Matrix with eigenvalues 3 and -1:
        // [ 1  2 ]
        // [ 2  1 ]
        MatrixDouble non_psd(2, 2, {1.0, 2.0, 2.0, 1.0});

        REQUIRE_THROWS_AS(Ellipsoid(mean, non_psd), NonPositiveSemidefiniteException);
    }

    SECTION("Matrix with small negative numerical noise is clamped to zero") {
        VectorDouble mean({0.0, 0.0});
        // Eigenvalues 2.0 and -1e-11 (within psd_tolerance 1e-9)
        MatrixDouble near_psd(2, 2, {2.0, 0.0, 0.0, -1e-11});

        Ellipsoid ell(mean, near_psd, 1e-9);
        REQUIRE(ell.dimension() == 2);
        REQUIRE(ell.rank() == 1); // 1 non-zero eigenvalue
        REQUIRE(ell.eigenvalues()[0] >= 0.0);
        REQUIRE(ell.eigenvalues()[1] >= 0.0);
    }

    SECTION("Positive semidefinite with exact zero eigenvalue") {
        VectorDouble mean({0.0, 0.0, 0.0});
        MatrixDouble psd = MatrixDouble::diagonal(VectorDouble({4.0, 0.0, 9.0}));

        Ellipsoid ell(mean, psd);
        REQUIRE(ell.dimension() == 3);
        REQUIRE(ell.rank() == 2);
        REQUIRE(ell.is_degenerate());
    }
}

TEST_CASE("Ellipsoid: Moore-Penrose Pseudoinverse Rigorous Verification", "[dbdd][ellipsoid]") {
    SECTION("Full-rank invertible covariance satisfies Sigma * Sigma^dagger = I") {
        VectorDouble mean({0.0, 0.0});
        MatrixDouble cov(2, 2, {5.0, 2.0, 2.0, 3.0}); // det = 11

        Ellipsoid ell(mean, cov);
        const auto& Sigma = ell.covariance();
        const auto& pinv = ell.pseudoinverse();

        MatrixDouble prod = Sigma * pinv;
        REQUIRE_THAT(prod(0, 0), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(prod(0, 1), WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(prod(1, 0), WithinAbs(0.0, 1e-10));
        REQUIRE_THAT(prod(1, 1), WithinAbs(1.0, 1e-10));
    }

    SECTION("Degenerate rank-deficient covariance satisfies all 4 Penrose equations") {
        // Construct 3x3 rank-2 covariance: diag(4, 25, 0)
        VectorDouble mean({1.0, 2.0, 3.0});
        MatrixDouble cov = MatrixDouble::diagonal(VectorDouble({4.0, 25.0, 0.0}));

        Ellipsoid ell(mean, cov);
        REQUIRE(ell.rank() == 2);

        const auto& Sigma = ell.covariance();
        const auto& pinv = ell.pseudoinverse();

        // Check explicit entries of pseudoinverse
        REQUIRE_THAT(pinv(0, 0), WithinAbs(0.25, 1e-12));
        REQUIRE_THAT(pinv(1, 1), WithinAbs(0.04, 1e-12));
        REQUIRE_THAT(pinv(2, 2), WithinAbs(0.0, 1e-12));

        // Penrose Condition 1: Sigma * pinv * Sigma = Sigma
        MatrixDouble c1 = Sigma * pinv * Sigma;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE_THAT(c1(i, j), WithinAbs(Sigma(i, j), 1e-10));
            }
        }

        // Penrose Condition 2: pinv * Sigma * pinv = pinv
        MatrixDouble c2 = pinv * Sigma * pinv;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE_THAT(c2(i, j), WithinAbs(pinv(i, j), 1e-10));
            }
        }

        // Penrose Condition 3: (Sigma * pinv)^T = Sigma * pinv
        MatrixDouble c3 = Sigma * pinv;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE_THAT(c3(i, j), WithinAbs(c3(j, i), 1e-10));
            }
        }

        // Penrose Condition 4: (pinv * Sigma)^T = pinv * Sigma
        MatrixDouble c4 = pinv * Sigma;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE_THAT(c4(i, j), WithinAbs(c4(j, i), 1e-10));
            }
        }
    }
}

TEST_CASE("Ellipsoid: Mahalanobis Distance and Containment", "[dbdd][ellipsoid]") {
    SECTION("Standard sphere distance and containment") {
        auto ell = Ellipsoid::standard(3);

        VectorDouble center({0.0, 0.0, 0.0});
        REQUIRE_THAT(ell.mahalanobis_distance_squared(center), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(ell.mahalanobis_distance(center), WithinAbs(0.0, 1e-12));
        REQUIRE(ell.contains(center, 1.0));

        VectorDouble boundary({1.0, 0.0, 0.0});
        REQUIRE_THAT(ell.mahalanobis_distance_squared(boundary), WithinAbs(1.0, 1e-12));
        REQUIRE(ell.contains(boundary, 1.0));

        VectorDouble interior({0.5, 0.5, 0.5}); // norm^2 = 0.75
        REQUIRE_THAT(ell.mahalanobis_distance_squared(interior), WithinAbs(0.75, 1e-12));
        REQUIRE(ell.contains(interior, 1.0));

        VectorDouble exterior({1.0, 1.0, 0.0}); // norm^2 = 2.0
        REQUIRE_THAT(ell.mahalanobis_distance_squared(exterior), WithinAbs(2.0, 1e-12));
        REQUIRE_FALSE(ell.contains(exterior, 1.0));
    }

    SECTION("Axis-aligned diagonal ellipsoid") {
        VectorDouble mean({1.0, 2.0});
        VectorDouble vars({4.0, 9.0});
        auto ell = Ellipsoid::diagonal(mean, vars);

        // x = (3, 5) -> delta = (2, 3) -> d_M^2 = 2^2/4 + 3^2/9 = 1 + 1 = 2
        VectorDouble x({3.0, 5.0});
        REQUIRE_THAT(ell.mahalanobis_distance_squared(x), WithinAbs(2.0, 1e-12));
        REQUIRE_FALSE(ell.contains(x, 1.0));
        REQUIRE(ell.contains(x, 2.0));
    }

    SECTION("Degenerate subspace support check") {
        // mu = (0, 0, 1), Sigma = diag(1, 1, 0) (Kannan-like homogeneous structure)
        VectorDouble mean({0.0, 0.0, 1.0});
        VectorDouble vars({1.0, 1.0, 0.0});
        auto ell = Ellipsoid::diagonal(mean, vars);

        REQUIRE(ell.is_degenerate());
        REQUIRE(ell.rank() == 2);

        // Point strictly in support: coordinate 2 matches mean (z = 1)
        VectorDouble in_pt({0.6, 0.8, 1.0});
        REQUIRE(ell.is_in_support(in_pt));
        REQUIRE_THAT(ell.mahalanobis_distance_squared(in_pt), WithinAbs(0.6*0.6 + 0.8*0.8, 1e-12)); // 1.0
        REQUIRE(ell.contains(in_pt, 1.0, /*check_support=*/true));

        // Point outside support: coordinate 2 deviates from mean (z = 1.5)
        VectorDouble out_pt({0.6, 0.8, 1.5});
        REQUIRE_FALSE(ell.is_in_support(out_pt));
        // Even with huge threshold, support check fails containment!
        REQUIRE_FALSE(ell.contains(out_pt, 100.0, /*check_support=*/true));
        // If check_support is explicitly disabled, Mahalanobis ignores nullspace
        REQUIRE(ell.contains(out_pt, 1.0, /*check_support=*/false));

        // Projection onto support restores coordinate 2 to mean
        VectorDouble proj = ell.project_onto_support(out_pt);
        REQUIRE_THAT(proj[0], WithinAbs(0.6, 1e-12));
        REQUIRE_THAT(proj[1], WithinAbs(0.8, 1e-12));
        REQUIRE_THAT(proj[2], WithinAbs(1.0, 1e-12));
        REQUIRE(ell.is_in_support(proj));
    }
}

TEST_CASE("Ellipsoid: Volume and Log-Volume", "[dbdd][ellipsoid]") {
    SECTION("Full-rank 3D sphere volume") {
        auto ell = Ellipsoid::standard(3);
        double expected_vol = (4.0 / 3.0) * std::numbers::pi;
        REQUIRE_THAT(ell.volume(), WithinAbs(expected_vol, 1e-10));
        REQUIRE_THAT(ell.log_volume(), WithinAbs(std::log(expected_vol), 1e-10));
    }

    SECTION("Degenerate ellipsoid has zero ambient volume and valid restricted log-volume") {
        VectorDouble mean({0.0, 0.0, 0.0});
        VectorDouble vars({4.0, 9.0, 0.0}); // 2D ellipse in 3D
        auto ell = Ellipsoid::diagonal(mean, vars);

        REQUIRE(ell.volume() == 0.0);

        // Restricted 2D volume: pi * a * b = pi * 2 * 3 = 6 pi
        double expected_restricted_vol = 6.0 * std::numbers::pi;
        REQUIRE_THAT(ell.log_volume(), WithinAbs(std::log(expected_restricted_vol), 1e-10));
    }
}

TEST_CASE("Ellipsoid: Error Handling & Dimension Checks", "[dbdd][ellipsoid]") {
    VectorDouble mean2({0.0, 0.0});
    VectorDouble mean3({0.0, 0.0, 0.0});
    MatrixDouble cov2 = MatrixDouble::identity(2);
    MatrixDouble non_sq(2, 3, 0.0);

    SECTION("Non-square covariance throws DimensionMismatchException") {
        REQUIRE_THROWS_AS(Ellipsoid(mean2, non_sq), DimensionMismatchException);
    }

    SECTION("Mean and covariance dimension mismatch throws DimensionMismatchException") {
        REQUIRE_THROWS_AS(Ellipsoid(mean3, cov2), DimensionMismatchException);
    }

    SECTION("Mahalanobis distance with mismatched vector throws DimensionMismatchException") {
        Ellipsoid ell(mean2, cov2);
        REQUIRE_THROWS_AS(ell.mahalanobis_distance_squared(mean3), DimensionMismatchException);
        REQUIRE_THROWS_AS(ell.is_in_support(mean3), DimensionMismatchException);
        REQUIRE_THROWS_AS(ell.project_onto_support(mean3), DimensionMismatchException);
    }
}
