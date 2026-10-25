#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include <random>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Approximate Hint Construction and Homogenization", "[approximate_hint][homogenize]") {
    SECTION("Constructor parameter validation") {
        VectorZZ v({BigInt(1), BigInt(2)});
        REQUIRE_THROWS_AS(ApproximateHint(v, BigInt(1), 0.0), InvalidParameterException); // variance <= 0
        REQUIRE_THROWS_AS(ApproximateHint(v, BigInt(1), -1.5), InvalidParameterException); // variance < 0
        REQUIRE_THROWS_AS(ApproximateHint(VectorZZ{}, BigInt(1), 1.0), InvalidParameterException); // empty v

        VectorDouble vd({1.0, 2.0});
        REQUIRE_THROWS_AS(ApproximateHint(vd, 1.0, 0.0), InvalidParameterException);
        REQUIRE_THROWS_AS(ApproximateHint(VectorDouble{}, 1.0, 1.0), InvalidParameterException);
    }

    SECTION("Homogenization from secret vector") {
        size_t n = 3;
        size_t m = 4;
        size_t ambient_dim = m + n + 1; // 8

        VectorZZ v({BigInt(2), BigInt(-1), BigInt(5)});
        BigInt l(7);
        double var = 0.5;

        ApproximateHint hint(v, l, var);
        REQUIRE(hint.type() == HintType::Approximate);
        REQUIRE(!hint.is_already_homogenized());
        REQUIRE(hint.noise_variance() == var);

        VectorZZ v_bar_int = hint.homogenize_int(ambient_dim);
        REQUIRE(v_bar_int.size() == ambient_dim);

        // Layout: (0_m, v, -l)
        for (size_t i = 0; i < m; ++i) {
            REQUIRE(v_bar_int[i] == BigInt(0));
        }
        for (size_t j = 0; j < n; ++j) {
            REQUIRE(v_bar_int[m + j] == v[j]);
        }
        REQUIRE(v_bar_int[ambient_dim - 1] == -l);

        VectorDouble v_bar_double = hint.homogenize_double(ambient_dim);
        REQUIRE(v_bar_double.size() == ambient_dim);
        for (size_t i = 0; i < ambient_dim; ++i) {
            REQUIRE(v_bar_double[i] == v_bar_int[i].to_double());
        }
    }

    SECTION("Already homogenized vector validation") {
        VectorDouble v_bar({0.0, 0.0, 1.0, -1.0, 2.0});
        auto hint = ApproximateHint::from_homogenized(v_bar, 1.0);
        REQUIRE(hint.is_already_homogenized());
        REQUIRE(hint.homogenize_double(5).size() == 5);
        REQUIRE_THROWS_AS(hint.homogenize_double(6), DimensionMismatchException);
    }
}

TEST_CASE("Hand-Computed 2D and 3D Gaussian Conditioning Exact Match", "[approximate_hint][analytical]") {
    SECTION("2D Zero-Mean Identity Covariance") {
        // mu = (0, 0), Sigma = I_2, v_bar = (1, 0), var = 1.0
        VectorDouble mean({0.0, 0.0});
        MatrixDouble cov = MatrixDouble::identity(2);
        VectorDouble v_bar({1.0, 0.0});
        double var = 1.0;

        // w = (1, 0)
        // denom = 1*1 + 1 = 2
        // mu' = (0, 0)
        // Sigma' = [1 - 1/2, 0; 0, 1] = [0.5, 0; 0, 1.0]
        auto [mu_p, cov_p] = ApproximateHint::update_distribution(mean, cov, v_bar, var);

        REQUIRE_THAT(mu_p[0], WithinAbs(0.0, 1e-14));
        REQUIRE_THAT(mu_p[1], WithinAbs(0.0, 1e-14));

        REQUIRE_THAT(cov_p(0, 0), WithinAbs(0.5, 1e-14));
        REQUIRE_THAT(cov_p(0, 1), WithinAbs(0.0, 1e-14));
        REQUIRE_THAT(cov_p(1, 0), WithinAbs(0.0, 1e-14));
        REQUIRE_THAT(cov_p(1, 1), WithinAbs(1.0, 1e-14));
    }

    SECTION("2D Correlated Covariance with Non-Zero Mean") {
        VectorDouble mean({2.0, -1.0});
        MatrixDouble cov(2, 2);
        cov(0, 0) = 2.0; cov(0, 1) = 1.0;
        cov(1, 0) = 1.0; cov(1, 1) = 2.0;

        VectorDouble v_bar({1.0, 1.0});
        double var = 2.0;

        // w = v_bar * cov = (1, 1) * [2, 1; 1, 2] = (3.0, 3.0)
        // var_v = w * v_bar^T = 3 + 3 = 6.0
        // denom = 6.0 + 2.0 = 8.0
        // mean_dot_v = 2*1 + (-1)*1 = 1.0
        // alpha = 1.0 / 8.0 = 0.125
        // mu' = (2.0 - 0.125*3, -1.0 - 0.125*3) = (1.625, -1.375)
        // w^T * w / denom = [9/8, 9/8; 9/8, 9/8] = [1.125, 1.125; 1.125, 1.125]
        // cov' = [2 - 1.125, 1 - 1.125; 1 - 1.125, 2 - 1.125] = [0.875, -0.125; -0.125, 0.875]

        auto [mu_p, cov_p] = ApproximateHint::update_distribution(mean, cov, v_bar, var);

        REQUIRE_THAT(mu_p[0], WithinAbs(1.625, 1e-14));
        REQUIRE_THAT(mu_p[1], WithinAbs(-1.375, 1e-14));

        REQUIRE_THAT(cov_p(0, 0), WithinAbs(0.875, 1e-14));
        REQUIRE_THAT(cov_p(0, 1), WithinAbs(-0.125, 1e-14));
        REQUIRE_THAT(cov_p(1, 0), WithinAbs(-0.125, 1e-14));
        REQUIRE_THAT(cov_p(1, 1), WithinAbs(0.875, 1e-14));
    }
}

TEST_CASE("Covariance Invariants: Symmetry, Contraction, and PSD", "[approximate_hint][covariance]") {
    size_t d = 6;
    VectorDouble mean(d);
    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (size_t i = 0; i < d; ++i) mean[i] = dist(rng);

    // Create a random symmetric positive-definite covariance matrix
    MatrixDouble A(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            A(i, j) = dist(rng);
        }
    }
    MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d) * 0.5;

    VectorDouble v_bar(d);
    for (size_t i = 0; i < d; ++i) v_bar[i] = dist(rng);
    double var = 0.75;

    auto [mu_p, cov_p] = ApproximateHint::update_distribution(mean, cov, v_bar, var);

    SECTION("Symmetry is strictly preserved") {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                REQUIRE_THAT(cov_p(i, j), WithinAbs(cov_p(j, i), 1e-14));
            }
        }
    }

    SECTION("cov_p is positive semidefinite") {
        for (int trial = 0; trial < 30; ++trial) {
            VectorDouble x(d);
            for (size_t i = 0; i < d; ++i) x[i] = dist(rng);

            double quad = 0.0;
            for (size_t i = 0; i < d; ++i) {
                double row_sum = 0.0;
                for (size_t j = 0; j < d; ++j) row_sum += x[j] * cov_p(j, i);
                quad += row_sum * x[i];
            }
            REQUIRE(quad >= -1e-12);
        }
    }

    SECTION("Monotonic uncertainty reduction: cov - cov_p is PSD") {
        MatrixDouble diff(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                diff(i, j) = cov(i, j) - cov_p(i, j);
            }
        }

        for (int trial = 0; trial < 30; ++trial) {
            VectorDouble x(d);
            for (size_t i = 0; i < d; ++i) x[i] = dist(rng);

            double quad = 0.0;
            for (size_t i = 0; i < d; ++i) {
                double row_sum = 0.0;
                for (size_t j = 0; j < d; ++j) row_sum += x[j] * diff(j, i);
                quad += row_sum * x[i];
            }
            REQUIRE(quad >= -1e-12);
        }
    }
}

TEST_CASE("Asymptotic Limiting Behaviors", "[approximate_hint][limits]") {
    size_t d = 4;
    VectorDouble mean({1.0, -0.5, 2.0, 0.0});
    MatrixDouble cov = MatrixDouble::identity(d) * 3.0;
    VectorDouble v_bar({1.0, 2.0, -1.0, 0.5});

    SECTION("Small noise limit (var -> 0) approaches PerfectHint conditioning") {
        auto [perf_mu, perf_cov] = PerfectHint::update_distribution(mean, cov, v_bar);
        auto [approx_mu, approx_cov] = ApproximateHint::update_distribution(mean, cov, v_bar, 1e-9);

        for (size_t i = 0; i < d; ++i) {
            REQUIRE_THAT(approx_mu[i], WithinAbs(perf_mu[i], 1e-7));
            for (size_t j = 0; j < d; ++j) {
                REQUIRE_THAT(approx_cov(i, j), WithinAbs(perf_cov(i, j), 1e-7));
            }
        }
    }

    SECTION("Huge noise limit (var -> infty) leaves distribution unchanged") {
        double huge_var = 1e12;
        auto [approx_mu, approx_cov] = ApproximateHint::update_distribution(mean, cov, v_bar, huge_var);

        for (size_t i = 0; i < d; ++i) {
            REQUIRE_THAT(approx_mu[i], WithinAbs(mean[i], 1e-10));
            for (size_t j = 0; j < d; ++j) {
                REQUIRE_THAT(approx_cov(i, j), WithinAbs(cov(i, j), 1e-10));
            }
        }
    }
}

TEST_CASE("Zero Lattice Mutation Invariant", "[approximate_hint][lattice]") {
    size_t n = 3;
    size_t m = 3;
    BigInt q(11);

    DiscreteGaussian noise_dist(1.0, 111);
    DiscreteGaussian secret_dist(2.0, 222);

    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 333);
    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    // Record initial lattice state
    size_t init_dim = dbdd.dimension();
    size_t init_amb = dbdd.ambient_dimension();
    BigReal init_vol = dbdd.volume();
    MatrixZZ init_basis_mat = dbdd.basis().matrix();
    VectorZZ init_target = dbdd.target();
    VectorZZ init_coords = dbdd.target_coordinates();

    // Apply approximate hint
    VectorDouble v_d({1.0, 0.0, 1.0});
    double l_d = 2.5;
    double var = 0.5;

    dbdd.integrate_approximate_hint(v_d, l_d, var);

    // Verify ZERO mutations to lattice and target
    REQUIRE(dbdd.dimension() == init_dim);
    REQUIRE(dbdd.ambient_dimension() == init_amb);
    REQUIRE(dbdd.volume() == init_vol);
    REQUIRE(dbdd.basis().matrix() == init_basis_mat);
    REQUIRE(dbdd.target() == init_target);
    REQUIRE(dbdd.target_coordinates() == init_coords);
    REQUIRE(dbdd.target_in_lattice());
}

TEST_CASE("LWE Kannan Embedding Integration (Toy & Medium Parameters)", "[approximate_hint][lwe]") {
    SECTION("Toy LWE: n=3, m=3, q=11") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(11);
        double sigma_e = 1.0;
        double sigma_s = 2.0;

        DiscreteGaussian noise_dist(sigma_e, 444);
        DiscreteGaussian secret_dist(sigma_s, 555);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 666);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto instance = DBDDInstance::from_kannan(kannan);

        const auto& s = lwe.test_only_secret();

        // Simulate noisy measurement on secret: l = <s, v> + e_obs
        VectorZZ v({BigInt(1), BigInt(-1), BigInt(2)});
        double true_inner = s.dot(v).to_double();
        double obs_var = 0.25; // sigma_obs = 0.5
        std::mt19937_64 rng(777);
        std::normal_distribution<double> obs_noise(0.0, std::sqrt(obs_var));
        double l_obs = true_inner + obs_noise(rng);

        double init_trace = 0.0;
        for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
            init_trace += instance.covariance()(i, i);
        }

        instance.integrate_approximate_hint(v, BigInt(static_cast<long long>(std::round(l_obs))), obs_var);

        double new_trace = 0.0;
        for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
            new_trace += instance.covariance()(i, i);
        }

        // Trace of covariance strictly decreased
        REQUIRE(new_trace < init_trace);
        REQUIRE(instance.target_in_lattice());
        REQUIRE(instance.verify_target(0.999));
    }

    SECTION("Sequential Approximate Hints Progressively Shrink Covariance") {
        size_t n = 4;
        size_t m = 4;
        BigInt q(23);

        DiscreteGaussian noise_dist(1.0, 12);
        DiscreteGaussian secret_dist(2.0, 34);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 56);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto instance = DBDDInstance::from_kannan(kannan);

        const auto& s = lwe.test_only_secret();
        std::mt19937_64 rng(78);
        double obs_var = 0.16; // sigma_obs = 0.4
        std::normal_distribution<double> obs_noise(0.0, 0.4);

        double prev_trace = 0.0;
        for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
            prev_trace += instance.covariance()(i, i);
        }

        // Apply 4 coordinate hints on s_0, s_1, s_2, s_3
        for (size_t coord = 0; coord < n; ++coord) {
            VectorDouble v(n, 0.0);
            v[coord] = 1.0;
            double l_obs = s[coord].to_double() + obs_noise(rng);

            instance.integrate_approximate_hint(v, l_obs, obs_var);

            double cur_trace = 0.0;
            for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
                cur_trace += instance.covariance()(i, i);
            }

            REQUIRE(cur_trace < prev_trace);
            REQUIRE(instance.target_in_lattice());
            REQUIRE(instance.verify_target(0.999));
            prev_trace = cur_trace;
        }
    }
}
