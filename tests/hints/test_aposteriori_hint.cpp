#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/hints/aposteriori_hint.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include <random>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("A-Posteriori Hint Construction and Homogenization", "[aposteriori_hint][homogenize]") {
    SECTION("Constructor parameter validation") {
        VectorZZ v({BigInt(1), BigInt(2)});
        // Negative posterior variance is disallowed
        REQUIRE_THROWS_AS(AposterioriApproximateHint(v, 0.0, -0.5), InvalidParameterException);
        // Empty hint vector is disallowed
        REQUIRE_THROWS_AS(AposterioriApproximateHint(VectorZZ{}, 0.0, 1.0), InvalidParameterException);

        VectorDouble vd({1.0, 2.0});
        REQUIRE_THROWS_AS(AposterioriApproximateHint(vd, 1.0, -1.0), InvalidParameterException);
        REQUIRE_THROWS_AS(AposterioriApproximateHint(VectorDouble{}, 1.0, 1.0), InvalidParameterException);

        // Variance 0.0 is allowed (perfect certainty boundary)
        REQUIRE_NOTHROW(AposterioriApproximateHint(v, 1.5, 0.0));
        REQUIRE_NOTHROW(AposterioriApproximateHint(vd, 1.5, 0.0));
    }

    SECTION("Homogenization from secret vector") {
        size_t n = 3;
        size_t m = 4;
        size_t ambient_dim = m + n + 1; // 8

        VectorZZ v({BigInt(2), BigInt(-1), BigInt(5)});
        double mu_post = 3.25;
        double var_post = 0.5;

        AposterioriApproximateHint hint(v, mu_post, var_post);
        REQUIRE(hint.type() == HintType::Aposteriori);
        REQUIRE(!hint.is_already_homogenized());
        REQUIRE(hint.posterior_mean() == mu_post);
        REQUIRE(hint.posterior_variance() == var_post);

        VectorZZ v_bar_int = hint.homogenize_int(ambient_dim);
        REQUIRE(v_bar_int.size() == ambient_dim);

        // Layout: (0_m, v, 0)
        for (size_t i = 0; i < m; ++i) {
            REQUIRE(v_bar_int[i] == BigInt(0));
        }
        for (size_t j = 0; j < n; ++j) {
            REQUIRE(v_bar_int[m + j] == v[j]);
        }
        REQUIRE(v_bar_int[ambient_dim - 1] == BigInt(0));

        VectorDouble v_bar_double = hint.homogenize_double(ambient_dim);
        REQUIRE(v_bar_double.size() == ambient_dim);
        for (size_t i = 0; i < ambient_dim; ++i) {
            REQUIRE(v_bar_double[i] == v_bar_int[i].to_double());
        }
    }

    SECTION("Already homogenized vector validation") {
        VectorDouble v_bar({0.0, 0.0, 1.0, -1.0, 2.0});
        auto hint = AposterioriApproximateHint::from_homogenized(v_bar, 2.0, 0.25);
        REQUIRE(hint.is_already_homogenized());
        REQUIRE(hint.homogenize_double(5).size() == 5);
        REQUIRE_THROWS_AS(hint.homogenize_double(6), DimensionMismatchException);
    }
}

TEST_CASE("Hand-Computed 1D and 2D Analytical Match", "[aposteriori_hint][analytical]") {
    SECTION("1D Scalar Gaussian Conditioning") {
        // Prior: mu = 5.0, Sigma = [4.0], v_bar = [1.0]
        // Target posterior: mu_post = 2.0, var_post = 1.0
        // mu_prior = 5.0, w = 4.0, sigma_prior^2 = 4.0
        // Delta_sigma^2 = 4.0 - 1.0 = 3.0
        // mu' = 5.0 + ((2.0 - 5.0) / 4.0) * 4.0 = 2.0
        // Sigma' = 4.0 - (3.0 / 16.0) * 16.0 = 1.0
        VectorDouble mean({5.0});
        MatrixDouble cov(1, 1);
        cov(0, 0) = 4.0;
        VectorDouble v_bar({1.0});

        auto [mu_p, cov_p] = AposterioriApproximateHint::update_distribution(
            mean, cov, v_bar, 2.0, 1.0);

        REQUIRE_THAT(mu_p[0], WithinAbs(2.0, 1e-14));
        REQUIRE_THAT(cov_p(0, 0), WithinAbs(1.0, 1e-14));
    }

    SECTION("2D Correlated Covariance with Non-Zero Mean") {
        // mu = (1.0, 2.0)
        // cov = [3.0, 1.0; 1.0, 2.0]
        // v_bar = (1.0, -1.0)
        // mu_prior = 1.0 - 2.0 = -1.0
        // w = v_bar * cov = (1, -1) * [3, 1; 1, 2] = (2.0, -1.0)
        // sigma_prior^2 = w * v_bar^T = 2.0(1.0) + (-1.0)(-1.0) = 3.0
        //
        // Choose mu_post = 4.0, var_post = 0.5
        // Delta_sigma^2 = 3.0 - 0.5 = 2.5
        // mu' = (1.0, 2.0) + ((4.0 - (-1.0)) / 3.0) * (2.0, -1.0)
        //     = (1.0, 2.0) + (5.0 / 3.0) * (2.0, -1.0)
        //     = (13.0 / 3.0, 1.0 / 3.0)
        //
        // w^T * w = [4.0, -2.0; -2.0, 1.0]
        // Delta / (sigma_prior^2)^2 = 2.5 / 9.0
        // (Delta / 9.0) * w^T * w = [10.0/9.0, -5.0/9.0; -5.0/9.0, 2.5/9.0]
        // cov' = [3 - 10/9, 1 - (-5/9); 1 - (-5/9), 2 - 2.5/9]
        //      = [17.0/9.0, 14.0/9.0; 14.0/9.0, 15.5/9.0]
        VectorDouble mean({1.0, 2.0});
        MatrixDouble cov(2, 2);
        cov(0, 0) = 3.0; cov(0, 1) = 1.0;
        cov(1, 0) = 1.0; cov(1, 1) = 2.0;

        VectorDouble v_bar({1.0, -1.0});
        double mu_post = 4.0;
        double var_post = 0.5;

        auto [mu_p, cov_p] = AposterioriApproximateHint::update_distribution(
            mean, cov, v_bar, mu_post, var_post);

        REQUIRE_THAT(mu_p[0], WithinAbs(13.0 / 3.0, 1e-14));
        REQUIRE_THAT(mu_p[1], WithinAbs(1.0 / 3.0, 1e-14));

        REQUIRE_THAT(cov_p(0, 0), WithinAbs(17.0 / 9.0, 1e-14));
        REQUIRE_THAT(cov_p(0, 1), WithinAbs(14.0 / 9.0, 1e-14));
        REQUIRE_THAT(cov_p(1, 0), WithinAbs(14.0 / 9.0, 1e-14));
        REQUIRE_THAT(cov_p(1, 1), WithinAbs(15.5 / 9.0, 1e-14));

        // Verify 1D marginal along v_bar matches target exactly
        double post_proj_mean = mu_p[0] * v_bar[0] + mu_p[1] * v_bar[1];
        REQUIRE_THAT(post_proj_mean, WithinAbs(mu_post, 1e-14));

        double post_proj_var = v_bar[0] * (cov_p(0, 0) * v_bar[0] + cov_p(0, 1) * v_bar[1]) +
                               v_bar[1] * (cov_p(1, 0) * v_bar[0] + cov_p(1, 1) * v_bar[1]);
        REQUIRE_THAT(post_proj_var, WithinAbs(var_post, 1e-14));
    }
}

TEST_CASE("Exact Posterior 1D Marginal Matching Across Dimensions", "[aposteriori_hint][projection_matching]") {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist_val(-3.0, 3.0);
    std::uniform_real_distribution<double> dist_var(0.05, 5.0);

    for (size_t d : {2, 3, 5, 8, 12}) {
        VectorDouble mean(d);
        for (size_t i = 0; i < d; ++i) mean[i] = dist_val(rng);

        // Generate random symmetric positive-definite covariance matrix
        MatrixDouble A(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                A(i, j) = dist_val(rng);
            }
        }
        MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d) * 0.5;

        // Non-zero direction vector
        VectorDouble v_bar(d);
        for (size_t i = 0; i < d; ++i) v_bar[i] = dist_val(rng);
        if (v_bar.norm() < 1e-3) v_bar[0] = 1.0;

        double mu_post = dist_val(rng);
        double var_post = dist_var(rng);

        auto [mu_p, cov_p] = AposterioriApproximateHint::update_distribution(
            mean, cov, v_bar, mu_post, var_post);

        // 1. Verify posterior projection mean matches mu_post exactly
        double actual_mu_proj = 0.0;
        for (size_t i = 0; i < d; ++i) actual_mu_proj += mu_p[i] * v_bar[i];
        REQUIRE_THAT(actual_mu_proj, WithinAbs(mu_post, 1e-12));

        // 2. Verify posterior projection variance matches var_post exactly
        double actual_var_proj = 0.0;
        for (size_t i = 0; i < d; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < d; ++j) {
                row_sum += cov_p(i, j) * v_bar[j];
            }
            actual_var_proj += v_bar[i] * row_sum;
        }
        REQUIRE_THAT(actual_var_proj, WithinAbs(var_post, 1e-12));
    }
}

TEST_CASE("Identity Invariant: Unchanged when posterior matches prior", "[aposteriori_hint][identity]") {
    size_t d = 5;
    std::mt19937_64 rng(101);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    VectorDouble mean(d);
    for (size_t i = 0; i < d; ++i) mean[i] = dist(rng);

    MatrixDouble A(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            A(i, j) = dist(rng);
        }
    }
    MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d);

    VectorDouble v_bar(d);
    for (size_t i = 0; i < d; ++i) v_bar[i] = dist(rng);

    // Compute prior projection
    double mu_prior = 0.0;
    for (size_t i = 0; i < d; ++i) mu_prior += mean[i] * v_bar[i];

    double sigma_prior_sq = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < d; ++j) row_sum += cov(i, j) * v_bar[j];
        sigma_prior_sq += v_bar[i] * row_sum;
    }

    // Pass prior parameters as posterior parameters
    auto [mu_p, cov_p] = AposterioriApproximateHint::update_distribution(
        mean, cov, v_bar, mu_prior, sigma_prior_sq);

    for (size_t i = 0; i < d; ++i) {
        REQUIRE_THAT(mu_p[i], WithinAbs(mean[i], 1e-14));
        for (size_t j = 0; j < d; ++j) {
            REQUIRE_THAT(cov_p(i, j), WithinAbs(cov(i, j), 1e-14));
        }
    }
}

TEST_CASE("Boundary Case: Perfect Certainty (var_post = 0) Matches PerfectHint", "[aposteriori_hint][perfect_boundary]") {
    size_t d = 4;
    VectorDouble mean({1.5, -0.5, 2.0, 1.0});
    MatrixDouble cov = MatrixDouble::identity(d) * 2.5;
    cov(0, 1) = 0.5; cov(1, 0) = 0.5;

    VectorDouble v_bar({1.0, 0.0, -1.0, 2.0});

    // When mu_post = 0 and var_post = 0, update matches PerfectHint::update_distribution conditioned on <x, v_bar> = 0
    auto [perf_mu, perf_cov] = PerfectHint::update_distribution(mean, cov, v_bar);
    auto [apost_mu, apost_cov] = AposterioriApproximateHint::update_distribution(
        mean, cov, v_bar, 0.0, 0.0);

    for (size_t i = 0; i < d; ++i) {
        REQUIRE_THAT(apost_mu[i], WithinAbs(perf_mu[i], 1e-14));
        for (size_t j = 0; j < d; ++j) {
            REQUIRE_THAT(apost_cov(i, j), WithinAbs(perf_cov(i, j), 1e-14));
        }
    }
}

TEST_CASE("Covariance Invariants: Symmetry, Contraction, and PSD", "[aposteriori_hint][covariance]") {
    size_t d = 6;
    VectorDouble mean(d);
    std::mt19937_64 rng(555);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (size_t i = 0; i < d; ++i) mean[i] = dist(rng);

    MatrixDouble A(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            A(i, j) = dist(rng);
        }
    }
    MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d) * 0.5;

    VectorDouble v_bar(d);
    for (size_t i = 0; i < d; ++i) v_bar[i] = dist(rng);

    double sigma_prior_sq = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < d; ++j) row_sum += cov(i, j) * v_bar[j];
        sigma_prior_sq += v_bar[i] * row_sum;
    }

    // Choose var_post smaller than prior (contracting information)
    double var_post = 0.3 * sigma_prior_sq;
    double mu_post = 1.5;

    auto [mu_p, cov_p] = AposterioriApproximateHint::update_distribution(
        mean, cov, v_bar, mu_post, var_post);

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

TEST_CASE("Zero Lattice Mutation Invariant", "[aposteriori_hint][lattice]") {
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

    // Apply a-posteriori hint
    VectorDouble v_d({1.0, 0.0, 1.0});
    double mu_post = 2.0;
    double var_post = 0.25;

    dbdd.integrate_aposteriori_hint(v_d, mu_post, var_post);

    // Verify ZERO mutations to lattice and target
    REQUIRE(dbdd.dimension() == init_dim);
    REQUIRE(dbdd.ambient_dimension() == init_amb);
    REQUIRE(dbdd.volume() == init_vol);
    REQUIRE(dbdd.basis().matrix() == init_basis_mat);
    REQUIRE(dbdd.target() == init_target);
    REQUIRE(dbdd.target_coordinates() == init_coords);
    REQUIRE(dbdd.target_in_lattice());
}

TEST_CASE("LWE Kannan Embedding Integration (Toy & Medium Parameters)", "[aposteriori_hint][lwe]") {
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

        // Direction along secret: v in Z^3
        VectorZZ v({BigInt(1), BigInt(-1), BigInt(2)});
        double true_inner = s.dot(v).to_double();

        // Side-channel leakage yields posterior centered close to ground truth with small variance
        double post_var = 0.1;
        double mu_post = true_inner; // accurate posterior mean

        double init_trace = 0.0;
        for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
            init_trace += instance.covariance()(i, i);
        }

        instance.integrate_aposteriori_hint(v, mu_post, post_var);

        double new_trace = 0.0;
        for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
            new_trace += instance.covariance()(i, i);
        }

        // Covariance trace contracted
        REQUIRE(new_trace < init_trace);
        REQUIRE(instance.target_in_lattice());
        REQUIRE(instance.verify_target(0.999));
    }

    SECTION("Medium LWE (n=8, m=8, q=97) Sequential Hints Progressively Shrink Covariance") {
        size_t n = 8;
        size_t m = 8;
        BigInt q(97);

        DiscreteGaussian noise_dist(1.0, 77);
        DiscreteGaussian secret_dist(2.5, 88);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 99);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto instance = DBDDInstance::from_kannan(kannan);

        const auto& s = lwe.test_only_secret();

        double prev_trace = 0.0;
        for (size_t i = 0; i < instance.ambient_dimension(); ++i) {
            prev_trace += instance.covariance()(i, i);
        }

        // Apply a-posteriori hints on multiple secret coordinates
        for (size_t coord = 0; coord < n; ++coord) {
            VectorDouble v(n, 0.0);
            v[coord] = 1.0;
            double mu_post = s[coord].to_double();
            double var_post = 0.05; // precise leakage

            instance.integrate_aposteriori_hint(v, mu_post, var_post);

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
