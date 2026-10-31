#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/linalg/eigendecomposition.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/dbdd/isotropization.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/arithmetic/precision.hpp"
#include "dbdd/core/exceptions.hpp"

#include <Eigen/Dense>

using namespace dbdd;
using Catch::Approx;

TEST_CASE("Symmetric Eigendecomposition and Matrix Roots", "[dbdd][isotropization]") {
    SECTION("Known Diagonal Matrix") {
        MatrixDouble diag(3, 3, 0.0);
        diag(0, 0) = 4.0;
        diag(1, 1) = 9.0;
        diag(2, 2) = 16.0;

        auto eigen = SymmetricEigendecomposition::decompose(diag);

        REQUIRE(eigen.dimension() == 3);
        REQUIRE(eigen.rank() == 3);
        REQUIRE(eigen.is_full_rank());

        // Eigenvalues sorted descending: 16, 9, 4
        REQUIRE(eigen.eigenvalues()[0] == Approx(16.0));
        REQUIRE(eigen.eigenvalues()[1] == Approx(9.0));
        REQUIRE(eigen.eigenvalues()[2] == Approx(4.0));

        // Matrix square root: diag(4, 3, 2)
        MatrixDouble sqrt_m = eigen.matrix_sqrt();
        REQUIRE(sqrt_m(0, 0) == Approx(2.0));
        REQUIRE(sqrt_m(1, 1) == Approx(3.0));
        REQUIRE(sqrt_m(2, 2) == Approx(4.0));

        // Pseudoinverse square root: diag(1/2, 1/3, 1/4)
        MatrixDouble inv_sqrt_m = eigen.matrix_inv_sqrt();
        REQUIRE(inv_sqrt_m(0, 0) == Approx(0.5));
        REQUIRE(inv_sqrt_m(1, 1) == Approx(1.0 / 3.0));
        REQUIRE(inv_sqrt_m(2, 2) == Approx(0.25));

        // Restricted determinant: 16 * 9 * 4 = 576
        REQUIRE(eigen.restricted_determinant().to_double() == Approx(576.0));
        REQUIRE(eigen.log_restricted_determinant().to_double() == Approx(std::log(576.0)));
    }

    SECTION("Non-Diagonal 2D Symmetric Matrix") {
        MatrixDouble A(2, 2);
        A(0, 0) = 2.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 2.0;

        auto eigen = SymmetricEigendecomposition::decompose(A);
        REQUIRE(eigen.rank() == 2);
        REQUIRE(eigen.eigenvalues()[0] == Approx(3.0));
        REQUIRE(eigen.eigenvalues()[1] == Approx(1.0));

        // Verify (A^{1/2})^2 == A
        MatrixDouble sqrt_A = eigen.matrix_sqrt();
        MatrixDouble sq = sqrt_A * sqrt_A;
        REQUIRE(sq(0, 0) == Approx(2.0).epsilon(1e-12));
        REQUIRE(sq(0, 1) == Approx(1.0).epsilon(1e-12));
        REQUIRE(sq(1, 0) == Approx(1.0).epsilon(1e-12));
        REQUIRE(sq(1, 1) == Approx(2.0).epsilon(1e-12));

        // Verify M * A^{1/2} == I_2
        MatrixDouble M = eigen.matrix_inv_sqrt();
        MatrixDouble prod = M * sqrt_A;
        REQUIRE(prod(0, 0) == Approx(1.0).epsilon(1e-12));
        REQUIRE(prod(0, 1) == Approx(0.0).margin(1e-12));
        REQUIRE(prod(1, 0) == Approx(0.0).margin(1e-12));
        REQUIRE(prod(1, 1) == Approx(1.0).epsilon(1e-12));
    }
}

TEST_CASE("Isotropization: Full-Rank Covariance Yields Identity", "[dbdd][isotropization]") {
    SECTION("2D Correlated Covariance") {
        MatrixDouble cov(2, 2);
        cov(0, 0) = 5.0; cov(0, 1) = 2.0;
        cov(1, 0) = 2.0; cov(1, 1) = 3.0;

        MatrixDouble M = Isotropization::compute_transform(cov);

        // M is symmetric: M^T == M
        REQUIRE(M(0, 1) == Approx(M(1, 0)).margin(1e-15));

        // M * cov * M == I_2
        MatrixDouble cov_iso = Isotropization::compute_isotropized_covariance(cov, M);
        REQUIRE(cov_iso(0, 0) == Approx(1.0).epsilon(1e-12));
        REQUIRE(cov_iso(0, 1) == Approx(0.0).margin(1e-12));
        REQUIRE(cov_iso(1, 0) == Approx(0.0).margin(1e-12));
        REQUIRE(cov_iso(1, 1) == Approx(1.0).epsilon(1e-12));
    }

    SECTION("3D Correlated Covariance") {
        MatrixDouble cov(3, 3);
        cov(0, 0) = 4.0; cov(0, 1) = 0.5; cov(0, 2) = -0.3;
        cov(1, 0) = 0.5; cov(1, 1) = 3.0; cov(1, 2) = 0.2;
        cov(2, 0) = -0.3; cov(2, 1) = 0.2; cov(2, 2) = 2.0;

        MatrixDouble M = Isotropization::compute_transform(cov);
        MatrixDouble cov_iso = Isotropization::compute_isotropized_covariance(cov, M);

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                REQUIRE(cov_iso(i, j) == Approx(expected).margin(1e-12));
            }
        }
    }
}

TEST_CASE("Isotropization: Singular and Degenerate Covariance Handling", "[dbdd][isotropization]") {
    // 3D space with rank 2 covariance
    MatrixDouble cov(3, 3, 0.0);
    cov(0, 0) = 4.0;
    cov(1, 1) = 9.0;
    cov(2, 2) = 0.0; // rank 2

    MatrixDouble M = Isotropization::compute_transform(cov);
    REQUIRE(M(0, 0) == Approx(0.5));
    REQUIRE(M(1, 1) == Approx(1.0 / 3.0));
    REQUIRE(M(2, 2) == Approx(0.0).margin(1e-15));

    // Transformed covariance is the orthogonal projection onto active span: diag(1, 1, 0)
    MatrixDouble cov_iso = Isotropization::compute_isotropized_covariance(cov, M);
    REQUIRE(cov_iso(0, 0) == Approx(1.0).epsilon(1e-12));
    REQUIRE(cov_iso(1, 1) == Approx(1.0).epsilon(1e-12));
    REQUIRE(cov_iso(2, 2) == Approx(0.0).margin(1e-12));

    // Idempotent: Pi^2 == Pi
    MatrixDouble cov_iso_sq = cov_iso * cov_iso;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            REQUIRE(cov_iso_sq(i, j) == Approx(cov_iso(i, j)).margin(1e-12));
        }
    }

    // Annihilates null direction v = (0, 0, 1)
    VectorDouble null_vec({0.0, 0.0, 1.0});
    VectorDouble vM = null_vec * M;
    REQUIRE(vM.squared_norm() == Approx(0.0).margin(1e-15));
}

TEST_CASE("Isotropized Volume Shortcut Equivalence", "[dbdd][isotropization]") {
    // 3D integer basis
    MatrixZZ b_mat(3, 3);
    b_mat(0, 0) = BigInt(3); b_mat(0, 1) = BigInt(1); b_mat(0, 2) = BigInt(0);
    b_mat(1, 0) = BigInt(1); b_mat(1, 1) = BigInt(4); b_mat(1, 2) = BigInt(1);
    b_mat(2, 0) = BigInt(0); b_mat(2, 1) = BigInt(1); b_mat(2, 2) = BigInt(2);
    BasisZZ basis(b_mat);

    BigReal orig_vol = compute_volume(basis);
    BigReal orig_log_vol = compute_log_volume(basis);

    // Symmetric PSD covariance
    MatrixDouble cov(3, 3);
    cov(0, 0) = 2.0; cov(0, 1) = 0.4; cov(0, 2) = 0.1;
    cov(1, 0) = 0.4; cov(1, 1) = 3.0; cov(1, 2) = 0.2;
    cov(2, 0) = 0.1; cov(2, 1) = 0.2; cov(2, 2) = 1.5;

    // Direct basis transformation B_iso = B * M
    MatrixDouble M = Isotropization::compute_transform(cov);
    MatrixDouble B_double = b_mat.cast<double>();
    MatrixDouble B_iso = B_double * M;
    BasisDouble basis_iso(B_iso);

    BigReal direct_vol = compute_volume(basis_iso);
    BigReal direct_log_vol = compute_log_volume(basis_iso);

    // Volume shortcut: Vol(Lambda_iso) = Vol(Lambda) / sqrt(rdet(cov))
    BigReal shortcut_vol = compute_isotropized_volume(orig_vol, cov);
    BigReal shortcut_log_vol = compute_isotropized_log_volume(orig_log_vol, cov);

    // Verify exact equivalence between direct Gram-Schmidt volume and shortcut formula
    REQUIRE(direct_vol.to_double() == Approx(shortcut_vol.to_double()).epsilon(1e-10));
    REQUIRE(direct_log_vol.to_double() == Approx(shortcut_log_vol.to_double()).epsilon(1e-10));
}

TEST_CASE("Multi-Precision Validation (128, 256, 512 bits)", "[dbdd][isotropization]") {
    for (mpfr_prec_t prec : {128, 256, 512}) {
        ScopedPrecision sp(prec);

        MatrixRR A(2, 2);
        A(0, 0) = BigReal(5.0); A(0, 1) = BigReal(2.0);
        A(1, 0) = BigReal(2.0); A(1, 1) = BigReal(3.0);

        auto eigen_rr = SymmetricEigendecompositionRR::decompose(A);
        REQUIRE(eigen_rr.dimension() == 2);
        REQUIRE(eigen_rr.rank() == 2);

        // Eigenvalues of [[5, 2], [2, 3]] are 4 + sqrt(5) ~ 6.236068 and 4 - sqrt(5) ~ 1.763932
        BigReal expected_l1 = BigReal(4.0) + BigReal(5.0).sqrt();
        BigReal expected_l2 = BigReal(4.0) - BigReal(5.0).sqrt();

        REQUIRE((eigen_rr.eigenvalues()[0] - expected_l1).abs().to_double() < 1e-25);
        REQUIRE((eigen_rr.eigenvalues()[1] - expected_l2).abs().to_double() < 1e-25);

        // Matrix square root (A^{1/2})^2 == A to multi-precision
        MatrixRR sqrt_A = eigen_rr.matrix_sqrt();
        MatrixRR sq = sqrt_A * sqrt_A;
        REQUIRE((sq(0, 0) - A(0, 0)).abs().to_double() < 1e-25);
        REQUIRE((sq(0, 1) - A(0, 1)).abs().to_double() < 1e-25);
        REQUIRE((sq(1, 0) - A(1, 0)).abs().to_double() < 1e-25);
        REQUIRE((sq(1, 1) - A(1, 1)).abs().to_double() < 1e-25);

        // Determinant = 15 - 4 = 11
        REQUIRE((eigen_rr.restricted_determinant() - BigReal(11.0)).abs().to_double() < 1e-25);
    }
}

TEST_CASE("UsvpInstance and Target Normalization on LWE", "[dbdd][isotropization]") {
    SECTION("Toy LWE Instance (n=3, m=3, q=11, d=7)") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(11);
        size_t d = m + n + 1; // 7

        DiscreteGaussian noise_dist(1.0, 11);
        DiscreteGaussian secret_dist(1.0, 22);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 33);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        // Ensure homogenized first
        dbdd.homogenize();
        REQUIRE(dbdd.is_homogenized());

        // Convert to UsvpInstance
        UsvpInstance usvp = dbdd.to_usvp_instance();

        REQUIRE(usvp.dimension() == d);
        REQUIRE(usvp.ambient_dimension() == d);
        REQUIRE(usvp.has_target());

        // Theoretical expectation: E[||s_bar * M||^2] = d = 7
        double s_norm_sq = usvp.target_squared_norm();
        REQUIRE(s_norm_sq > 0.5 * static_cast<double>(d));
        REQUIRE(s_norm_sq < 2.0 * static_cast<double>(d));

        // Volume consistency: direct vs shortcut
        BigReal direct_vol = compute_volume(usvp.basis());
        REQUIRE(direct_vol.to_double() == Approx(usvp.volume().to_double()).epsilon(1e-10));

        // Discretization to integer basis
        BasisZZ int_basis = dbdd.to_usvp(100.0);
        REQUIRE(int_basis.dimension() == d);
        REQUIRE(int_basis.ambient_dimension() == d);
        REQUIRE(int_basis.is_full_rank());

        // In-place isotropization on DBDDInstance
        dbdd.isotropize();
        REQUIRE(dbdd.is_isotropized());
        REQUIRE(dbdd.is_homogenized());
    }

    SECTION("Medium LWE Instance (n=8, m=8, q=97, d=17)") {
        size_t n = 8;
        size_t m = 8;
        BigInt q(97);
        size_t d = m + n + 1; // 17

        DiscreteGaussian noise_dist(1.5, 91);
        DiscreteGaussian secret_dist(1.5, 92);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 93);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        dbdd.homogenize();
        UsvpInstance usvp = dbdd.to_usvp_instance();

        REQUIRE(usvp.dimension() == d);
        REQUIRE(usvp.has_target());

        // Target normalized squared norm
        double s_norm_sq = usvp.target_squared_norm();
        REQUIRE(s_norm_sq > 0.5 * static_cast<double>(d));
        REQUIRE(s_norm_sq < 2.0 * static_cast<double>(d));

        dbdd.isotropize();
        REQUIRE(dbdd.is_isotropized());
    }
}
