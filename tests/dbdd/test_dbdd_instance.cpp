#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lattice/volume.hpp"

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("DBDDInstance: Toy Parameters (n=3, m=3, q=11, d=7)", "[dbdd][instance]") {
    size_t n = 3;
    size_t m = 3;
    BigInt q(11);
    size_t d = m + n + 1; // 7

    double sigma_e = 1.5;
    double sigma_s = 2.0;

    DiscreteGaussian noise_dist(sigma_e, 101);
    DiscreteGaussian secret_dist(sigma_s, 102);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 2002);
    auto kannan = KannanEmbedding::from_lwe(instance, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    SECTION("Dimensions and Structure") {
        REQUIRE(dbdd.dimension() == d);
        REQUIRE(dbdd.ambient_dimension() == d);
        REQUIRE(dbdd.lattice().dimension() == d);
        REQUIRE(dbdd.lattice().ambient_dimension() == d);
        REQUIRE(dbdd.ellipsoid().dimension() == d);
        REQUIRE(dbdd.ellipsoid().rank() == d - 1); // 6
        REQUIRE(dbdd.ellipsoid().is_degenerate());
    }

    SECTION("Volume Preservation") {
        BigInt expected_vol = BigInt::pow(q, 3); // 1331
        REQUIRE(dbdd.exact_volume_int() == expected_vol);
        REQUIRE_THAT(dbdd.volume().to_double(), WithinAbs(1331.0, 1e-10));
    }

    SECTION("Statistical Parameters and Covariance Structure") {
        const auto& mean = dbdd.mean();
        const auto& cov = dbdd.covariance();

        for (size_t j = 0; j < m + n; ++j) {
            REQUIRE(mean[j] == 0.0);
        }
        REQUIRE(mean[d - 1] == 1.0);

        for (size_t j = 0; j < m; ++j) {
            REQUIRE_THAT(cov(j, j), WithinAbs(sigma_e * sigma_e, 1e-12));
        }
        for (size_t i = 0; i < n; ++i) {
            REQUIRE_THAT(cov(m + i, m + i), WithinAbs(sigma_s * sigma_s, 1e-12));
        }
        REQUIRE_THAT(cov(d - 1, d - 1), WithinAbs(0.0, 1e-12));
    }

    SECTION("Target Vector Invariant and Containment") {
        REQUIRE(dbdd.has_target());
        REQUIRE(dbdd.has_target_coordinates());

        const auto& target = dbdd.target();
        REQUIRE(target.size() == d);
        REQUIRE(target[d - 1] == BigInt(1));

        // Exact lattice containment
        REQUIRE(dbdd.target_in_lattice());

        // Target Mahalanobis distance
        const auto& s = instance.test_only_secret();
        const auto& e = instance.test_only_error();

        double expected_dist_sq = 0.0;
        for (size_t j = 0; j < m; ++j) {
            double ej = e[j].to_double();
            expected_dist_sq += (ej * ej) / (sigma_e * sigma_e);
        }
        for (size_t i = 0; i < n; ++i) {
            double si = s[i].to_double();
            expected_dist_sq += (si * si) / (sigma_s * sigma_s);
        }

        double actual_dist_sq = dbdd.target_mahalanobis_distance_squared();
        REQUIRE_THAT(actual_dist_sq, WithinAbs(expected_dist_sq, 1e-10));

        // Target satisfies high-confidence chi^2 threshold
        double thresh_999 = dbdd.chi2_threshold(0.999);
        REQUIRE(thresh_999 > 6.0); // degrees of freedom = 6
        REQUIRE(dbdd.target_in_ellipsoid(thresh_999));
        REQUIRE(dbdd.verify_target(0.999));
    }
}

TEST_CASE("DBDDInstance: Literature Benchmark Parameters (n=70, m=70, q=3301, sigma=20)", "[dbdd][instance]") {
    size_t n = 70;
    size_t m = 70;
    BigInt q(3301);
    double sigma = 20.0;
    size_t d = m + n + 1; // 141

    DiscreteGaussian noise_dist(sigma, 501);
    DiscreteGaussian secret_dist(sigma, 502);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 7777);
    auto kannan = KannanEmbedding::from_lwe(instance, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    SECTION("Dimensions and Rank") {
        REQUIRE(dbdd.dimension() == d);
        REQUIRE(dbdd.ambient_dimension() == d);
        REQUIRE(dbdd.ellipsoid().rank() == d - 1); // 140
    }

    SECTION("Volume Invariants") {
        BigInt expected_vol = BigInt::pow(q, static_cast<unsigned long>(m));
        REQUIRE(dbdd.exact_volume_int() == expected_vol);

        BigReal expected_log_vol = BigReal(q).log() * BigReal(static_cast<double>(m));
        REQUIRE_THAT(dbdd.log_volume().to_double(), WithinAbs(expected_log_vol.to_double(), 1e-8));
    }

    SECTION("Target Verification on Benchmark") {
        REQUIRE(dbdd.has_target());
        REQUIRE(dbdd.target_in_lattice());

        double dist_sq = dbdd.target_mahalanobis_distance_squared();
        double thresh_999 = dbdd.chi2_threshold(0.999);

        // Chi-squared distribution with 140 degrees of freedom has mean 140, stddev sqrt(280) approx 16.7
        // Threshold for 99.9% is approx 194. dist_sq should comfortably fall below threshold
        REQUIRE(dist_sq < thresh_999);
        REQUIRE(dbdd.verify_target(0.999));
    }
}

TEST_CASE("DBDDInstance: Public Instance Without Target", "[dbdd][instance]") {
    size_t n = 2;
    size_t m = 2;
    BigInt q(17);

    MatrixZZ A(2, 2, {BigInt(1), BigInt(2), BigInt(3), BigInt(4)});
    VectorZZ b({BigInt(5), BigInt(6)});

    LWEInstance pub_inst(n, m, q, A, b);
    auto kannan = KannanEmbedding::from_lwe(pub_inst, 1.0, 1.0);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    REQUIRE_FALSE(dbdd.has_target());
    REQUIRE_FALSE(dbdd.has_target_coordinates());

    REQUIRE_THROWS_AS(dbdd.target(), InvalidParameterException);
    REQUIRE_THROWS_AS(dbdd.target_coordinates(), InvalidParameterException);
    REQUIRE_THROWS_AS(dbdd.target_double(), InvalidParameterException);
    REQUIRE_THROWS_AS(dbdd.target_in_lattice(), InvalidParameterException);
    REQUIRE_FALSE(dbdd.verify_target());
}

TEST_CASE("DBDDInstance: Direct Instantiation & Setters", "[dbdd][instance]") {
    BasisZZ basis(MatrixZZ::identity(3));
    LatticeZZ lattice(basis);
    VectorDouble mean({0.0, 0.0, 0.0});
    MatrixDouble cov = MatrixDouble::identity(3);

    DBDDInstance dbdd(lattice, mean, cov);
    REQUIRE(dbdd.dimension() == 3);
    REQUIRE_FALSE(dbdd.has_target());

    SECTION("Setting target vector") {
        VectorZZ target({BigInt(1), BigInt(0), BigInt(0)});
        VectorZZ coords({BigInt(1), BigInt(0), BigInt(0)});
        dbdd.set_target(target, coords);

        REQUIRE(dbdd.has_target());
        REQUIRE(dbdd.has_target_coordinates());
        REQUIRE(dbdd.target_in_lattice());
        REQUIRE(dbdd.target_in_ellipsoid(1.0));
        REQUIRE(dbdd.verify_target());
    }

    SECTION("Dimension mismatch throws DimensionMismatchException") {
        VectorDouble mean_wrong({0.0, 0.0});
        REQUIRE_THROWS_AS(DBDDInstance(lattice, mean_wrong, cov), DimensionMismatchException);

        VectorZZ bad_target({BigInt(1), BigInt(2)});
        REQUIRE_THROWS_AS(dbdd.set_target(bad_target), DimensionMismatchException);
    }
}

TEST_CASE("DBDDInstance: Extension Points for Future Phases", "[dbdd][instance]") {
    auto ell = Ellipsoid::standard(3);
    BasisZZ basis(MatrixZZ::identity(3));
    DBDDInstance dbdd(basis, ell.mean(), ell.covariance());

    VectorZZ v_zz({BigInt(1), BigInt(0), BigInt(0)});
    VectorDouble v_d({1.0, 0.0, 0.0});
    BigInt l_zz(0);

    // Phase 06: integrate_perfect_hint is now implemented
    dbdd.integrate_perfect_hint(v_zz, l_zz);
    REQUIRE(dbdd.dimension() == 2);

    // Phase 07: integrate_modular_hint is now implemented
    // Phase 08: integrate_approximate_hint is now implemented
    dbdd.integrate_approximate_hint(v_d, 0.0, 1.0);
    REQUIRE(dbdd.dimension() == 2);

    // Phase 09: integrate_aposteriori_hint is now implemented
    VectorDouble v_apost({0.0, 1.0, 0.0});
    dbdd.integrate_aposteriori_hint(v_apost, 0.0, 0.5);
    REQUIRE(dbdd.dimension() == 2);

    // Phase 10: integrate_short_vector_hint is now implemented
    VectorZZ v_sv({BigInt(0), BigInt(1), BigInt(0)});
    dbdd.integrate_short_vector_hint(v_sv);
    REQUIRE(dbdd.dimension() == 1);

    // Phase 11: homogenize is now implemented
    dbdd.homogenize();
    REQUIRE(dbdd.is_homogenized());
    REQUIRE(dbdd.mean().squared_norm() == 0.0);

    BasisZZ basis2(MatrixZZ::identity(2));
    DBDDInstance dbdd2(basis2, VectorDouble({1.0, 2.0}), MatrixDouble::identity(2));
    REQUIRE_FALSE(dbdd2.is_homogenized());
    dbdd2.homogenize();
    REQUIRE(dbdd2.is_homogenized());
    REQUIRE(dbdd2.mean().squared_norm() == 0.0);

    // Phase 12: isotropize and to_usvp are now implemented
    REQUIRE_FALSE(dbdd2.is_isotropized());
    dbdd2.isotropize();
    REQUIRE(dbdd2.is_isotropized());
    REQUIRE(dbdd2.is_homogenized());

    auto usvp_basis = dbdd2.to_usvp(1.0);
    REQUIRE(usvp_basis.dimension() == dbdd2.dimension());
    REQUIRE(usvp_basis.ambient_dimension() == dbdd2.ambient_dimension());

    auto usvp_inst = dbdd2.to_usvp_instance();
    REQUIRE(usvp_inst.dimension() == dbdd2.dimension());
    REQUIRE(usvp_inst.ambient_dimension() == dbdd2.ambient_dimension());
}
