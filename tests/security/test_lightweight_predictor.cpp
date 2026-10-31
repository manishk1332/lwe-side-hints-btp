#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/short_vector_hint.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/core/exceptions.hpp"

#include <chrono>
#include <cmath>

using namespace dbdd;
using Catch::Approx;

TEST_CASE("LightweightPredictor - Exact Equivalence with DBDDInstance", "[security][lightweight]") {
    const size_t n = 3;
    const size_t m = 3;
    const BigInt q(11);

    DiscreteGaussian noise_dist(1.5, 101);
    DiscreteGaussian secret_dist(2.0, 102);

    auto lwe_inst = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 2002);
    auto kannan = KannanEmbedding::from_lwe(lwe_inst, noise_dist, secret_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
    DBDDStateStatistics stats = dbdd.to_state_statistics();

    SECTION("Initial State Matching") {
        REQUIRE(stats.dimension() == dbdd.dimension());
        REQUIRE(stats.ambient_dimension() == dbdd.ambient_dimension());
        REQUIRE(stats.volume().to_double() == Approx(dbdd.volume().to_double()).epsilon(1e-12));
        REQUIRE(stats.log_volume().to_double() == Approx(dbdd.log_volume().to_double()).epsilon(1e-12));

        for (size_t i = 0; i < dbdd.ambient_dimension(); ++i) {
            REQUIRE(stats.mean()[i] == Approx(dbdd.mean()[i]).margin(1e-12));
            for (size_t j = 0; j < dbdd.ambient_dimension(); ++j) {
                REQUIRE(stats.covariance()(i, j) == Approx(dbdd.covariance()(i, j)).margin(1e-12));
            }
        }

        // Isotropized volume shortcut matching
        const double iso_dbdd = dbdd.to_usvp_instance().log_volume().to_double();
        const double iso_stats = stats.isotropized_log_volume();
        REQUIRE(iso_stats == Approx(iso_dbdd).epsilon(1e-9));

        // Predicted beta matching
        REQUIRE(stats.estimate_beta_gsa() == dbdd.estimate_beta_gsa());
        REQUIRE(stats.estimate_attack_gsa().beta_continuous == Approx(dbdd.estimate_attack_gsa().beta_continuous).epsilon(1e-9));
    }

    SECTION("Perfect Hints Trajectory Equivalence") {
        const auto& s_bar = dbdd.target();
        size_t amb_dim = dbdd.ambient_dimension();

        // Reveal coordinate s_0 at index 3
        VectorZZ v0_zz(amb_dim, BigInt(0));
        v0_zz[3] = BigInt(1);
        PerfectHint hint0(v0_zz, s_bar[3]);
        VectorZZ v0_bar = hint0.homogenize_int(amb_dim);
        double w0_norm = PerfectHint::primitive_dual_norm(dbdd.lattice().basis(), v0_bar).to_double();

        dbdd.integrate_hint(hint0);
        stats.integrate_perfect_hint(hint0, w0_norm);

        REQUIRE(stats.dimension() == dbdd.dimension());
        REQUIRE(stats.isotropized_log_volume() == Approx(dbdd.to_usvp_instance().log_volume().to_double()).epsilon(1e-8));
        REQUIRE(stats.estimate_beta_gsa() == dbdd.estimate_beta_gsa());

        // Reveal coordinate s_1 at index 4
        VectorZZ v1_zz(amb_dim, BigInt(0));
        v1_zz[4] = BigInt(1);
        PerfectHint hint1(v1_zz, s_bar[4]);
        VectorZZ v1_bar = hint1.homogenize_int(amb_dim);
        double w1_norm = PerfectHint::primitive_dual_norm(dbdd.lattice().basis(), v1_bar).to_double();

        dbdd.integrate_hint(hint1);
        stats.integrate_perfect_hint(hint1, w1_norm);

        REQUIRE(stats.dimension() == dbdd.dimension());
        REQUIRE(stats.isotropized_log_volume() == Approx(dbdd.to_usvp_instance().log_volume().to_double()).epsilon(1e-8));
        REQUIRE(stats.estimate_beta_gsa() == dbdd.estimate_beta_gsa());

        // Covariance matrices match to machine precision
        for (size_t i = 0; i < amb_dim; ++i) {
            for (size_t j = 0; j < amb_dim; ++j) {
                REQUIRE(stats.covariance()(i, j) == Approx(dbdd.covariance()(i, j)).margin(1e-9));
            }
        }
    }

    SECTION("Approximate Hints Equivalence") {
        size_t amb_dim = dbdd.ambient_dimension();
        VectorDouble v(amb_dim, 0.0);
        v[0] = 1.0; v[1] = -1.0; v[3] = 2.0;
        double l = 1.5;
        double sigma = 0.8;

        ApproximateHint hint(v, l, sigma * sigma);
        dbdd.integrate_hint(hint);
        stats.integrate_approximate_hint(hint);

        REQUIRE(stats.dimension() == dbdd.dimension());
        REQUIRE(stats.volume().to_double() == Approx(dbdd.volume().to_double()).epsilon(1e-12));

        for (size_t i = 0; i < amb_dim; ++i) {
            REQUIRE(stats.mean()[i] == Approx(dbdd.mean()[i]).margin(1e-9));
            for (size_t j = 0; j < amb_dim; ++j) {
                REQUIRE(stats.covariance()(i, j) == Approx(dbdd.covariance()(i, j)).margin(1e-9));
            }
        }

        REQUIRE(stats.estimate_beta_gsa() == dbdd.estimate_beta_gsa());
    }

    SECTION("Modular Hints Equivalence") {
        size_t amb_dim = dbdd.ambient_dimension();
        VectorZZ v_zz(amb_dim, BigInt(0));
        v_zz[0] = BigInt(1);
        BigInt l(1);
        BigInt k(2);

        ModularHint hint(v_zz, l, k);
        dbdd.integrate_hint(hint);
        stats.integrate_modular_hint(hint);

        REQUIRE(stats.dimension() == dbdd.dimension());
        REQUIRE(stats.volume().to_double() == Approx(dbdd.volume().to_double()).epsilon(1e-9));
        REQUIRE(stats.estimate_beta_gsa() == dbdd.estimate_beta_gsa());
    }
}

TEST_CASE("DiagonalCovariance - Equivalence and O(1) Updates", "[security][diagonal]") {
    const size_t n = 10;
    const size_t m = 10;
    const double q = 127.0;
    const double sigma_e = 2.0;
    const double sigma_s = 2.5;

    auto diag = DiagonalCovariance::from_lwe_parameters(n, m, q, sigma_e, sigma_s);

    // Build equivalent DBDDStateStatistics for cross-validation
    const size_t d = m + n + 1;
    VectorDouble mu(d, 0.0);
    mu[d - 1] = 1.0;
    MatrixDouble cov(d, d, 0.0);
    for (size_t i = 0; i < m; ++i) cov(i, i) = sigma_e * sigma_e;
    for (size_t i = 0; i < n; ++i) cov(m + i, m + i) = sigma_s * sigma_s;
    cov(d - 1, d - 1) = 0.0;

    DBDDStateStatistics stats(d, BigReal(std::pow(q, m)), mu, cov);

    SECTION("Initial Equivalence") {
        REQUIRE(diag.dimension() == stats.dimension());
        REQUIRE(diag.ambient_dimension() == stats.ambient_dimension());
        REQUIRE(diag.isotropized_log_volume() == Approx(stats.isotropized_log_volume()).epsilon(1e-9));
        REQUIRE(diag.estimate_beta_gsa() == stats.estimate_beta_gsa());
    }

    SECTION("Coordinate Perfect Hint Equivalence") {
        // Reveal s_0 at index m = 10
        diag.integrate_coordinate_perfect_hint(m, 2.0);

        VectorDouble v0(d, 0.0);
        v0[m] = 1.0;
        v0[d - 1] = -2.0;
        stats.integrate_perfect_hint(v0, 0.0, 1.0);

        REQUIRE(diag.dimension() == stats.dimension());
        REQUIRE(diag.variance(m) == 0.0);
        REQUIRE(diag.mean(m) == 2.0);
        REQUIRE(diag.isotropized_log_volume() == Approx(stats.isotropized_log_volume()).epsilon(1e-9));
        REQUIRE(diag.estimate_beta_gsa() == stats.estimate_beta_gsa());
    }

    SECTION("Coordinate Approximate Hint Equivalence") {
        // Approximate hint on coordinate 0: s_0 + e = 1.0, sigma = 1.2
        const double l = 1.0;
        const double sigma = 1.2;

        diag.integrate_coordinate_approximate_hint(0, l, sigma);

        VectorDouble v(d, 0.0);
        v[0] = 1.0;
        stats.integrate_approximate_hint(v, l, sigma);

        REQUIRE(diag.variance(0) == Approx(stats.covariance()(0, 0)).epsilon(1e-9));
        REQUIRE(diag.mean(0) == Approx(stats.mean()[0]).epsilon(1e-9));
        REQUIRE(diag.isotropized_log_volume() == Approx(stats.isotropized_log_volume()).epsilon(1e-9));
        REQUIRE(diag.estimate_beta_gsa() == stats.estimate_beta_gsa());
    }

    SECTION("Coordinate Modular Hint Equivalence") {
        diag.integrate_coordinate_modular_hint(1, 0.0, 3.0);
        VectorDouble v(d, 0.0);
        v[1] = 1.0;
        stats.integrate_modular_hint(v, 0.0, 3.0);

        REQUIRE(diag.log_volume().to_double() == Approx(stats.log_volume().to_double()).epsilon(1e-9));
        REQUIRE(diag.isotropized_log_volume() == Approx(stats.isotropized_log_volume()).epsilon(1e-9));
        REQUIRE(diag.estimate_beta_gsa() == stats.estimate_beta_gsa());
    }
}

TEST_CASE("LightweightPredictor - Throughput and Stress Benchmark", "[security][benchmark]") {
    const size_t n = 100;
    const size_t m = 100;
    const double q = 3301.0;
    const double sigma_e = 3.2;
    const double sigma_s = 3.2;

    SECTION("DBDDStateStatistics: 500 Hint Integrations") {
        const size_t d = m + n + 1;
        VectorDouble mu(d, 0.0);
        mu[d - 1] = 1.0;
        MatrixDouble cov(d, d, 0.0);
        for (size_t i = 0; i < m; ++i) cov(i, i) = sigma_e * sigma_e;
        for (size_t i = 0; i < n; ++i) cov(m + i, m + i) = sigma_s * sigma_s;
        cov(d - 1, d - 1) = 0.0;

        DBDDStateStatistics stats(d, BigReal(std::pow(q, m)), mu, cov);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < 200; ++i) {
            VectorDouble v(d, 0.0);
            v[i % (d - 1)] = 1.0;
            stats.integrate_approximate_hint(v, 0.0, 1.0);
        }
        for (size_t i = 0; i < 50; ++i) {
            VectorDouble v(d, 0.0);
            v[m + i] = 1.0;
            stats.integrate_perfect_hint(v, 0.0, 1.0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 250 dense rank-1 hint integrations on 201x201 matrix must complete in < 250 ms
        REQUIRE(elapsed_ms < 250);
        REQUIRE(stats.dimension() == d - 50);
        REQUIRE(stats.estimate_beta_gsa() > 0);
    }

    SECTION("DiagonalCovariance: 10,000 Hint Integrations") {
        auto diag = DiagonalCovariance::from_lwe_parameters(n, m, q, sigma_e, sigma_s);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t rep = 0; rep < 50; ++rep) {
            for (size_t i = 0; i < 200; ++i) {
                diag.integrate_coordinate_approximate_hint(i, 0.0, 1.5);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 10,000 O(1) hint integrations must complete in < 15 ms
        REQUIRE(elapsed_ms < 15);
    }
}

TEST_CASE("LightweightPredictor - Error Handling", "[security][lightweight]") {
    VectorDouble mu(3, 0.0);
    MatrixDouble cov(3, 3, 0.0);
    cov(0, 0) = 1.0; cov(1, 1) = 1.0; cov(2, 2) = 1.0;
    cov(0, 1) = 0.5; cov(1, 0) = 0.5; // Non-diagonal!

    DBDDStateStatistics stats(3, BigReal(10.0), mu, cov);

    // Non-diagonal matrix cannot be converted to DiagonalCovariance
    REQUIRE_THROWS_AS(DiagonalCovariance::from_state_statistics(stats), InvalidParameterException);

    auto diag = DiagonalCovariance::from_lwe_parameters(2, 2, 11.0, 1.0, 1.0);
    // Index out of bounds
    REQUIRE_THROWS_AS(diag.variance(100), IndexOutOfBoundsException);
    REQUIRE_THROWS_AS(diag.integrate_coordinate_perfect_hint(100, 0.0), IndexOutOfBoundsException);
    REQUIRE_THROWS_AS(diag.integrate_coordinate_approximate_hint(100, 0.0, 1.0), IndexOutOfBoundsException);

    // Invalid parameters
    REQUIRE_THROWS_AS(diag.integrate_coordinate_approximate_hint(0, 0.0, -1.0), InvalidParameterException);
    REQUIRE_THROWS_AS(diag.integrate_coordinate_modular_hint(0, 0.0, 0.0), InvalidParameterException);
}
