#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/security/probabilistic_simulation.hpp"
#include "dbdd/security/security_estimator.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/security/root_hermite.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/dbdd/homogenization.hpp"
#include "dbdd/dbdd/isotropization.hpp"
#include "dbdd/hints/perfect_hint.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace dbdd;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Probabilistic - Chi-Squared Distribution CDF & PDF", "[security][probabilistic]") {
    SECTION("Parameter Validation & Boundary Cases") {
        REQUIRE_THROWS_AS(ProbabilisticEstimator::chi_squared_cdf(0, 1.0), InvalidParameterException);
        REQUIRE_THROWS_AS(ProbabilisticEstimator::chi_squared_pdf(0, 1.0), InvalidParameterException);

        REQUIRE(ProbabilisticEstimator::chi_squared_cdf(1, 0.0) == 0.0);
        REQUIRE(ProbabilisticEstimator::chi_squared_cdf(10, -5.0) == 0.0);
        REQUIRE(ProbabilisticEstimator::chi_squared_pdf(5, 0.0) == 0.0);
        REQUIRE(ProbabilisticEstimator::chi_squared_pdf(5, -1.0) == 0.0);
    }

    SECTION("Analytical Special Cases (df = 1 and df = 2)") {
        // df = 1: CDF(x) = erf(sqrt(x / 2))
        for (double x : {0.2, 0.5, 1.0, 2.0, 4.0, 9.0}) {
            double expected = std::erf(std::sqrt(0.5 * x));
            REQUIRE_THAT(ProbabilisticEstimator::chi_squared_cdf(1, x), WithinAbs(expected, 1e-12));
        }

        // df = 2: CDF(x) = 1 - exp(-x / 2)
        // Median of chi^2_2 is 2 * ln(2) ~ 1.3862943611198906
        double median_2 = 2.0 * std::log(2.0);
        REQUIRE_THAT(ProbabilisticEstimator::chi_squared_cdf(2, median_2), WithinAbs(0.5, 1e-12));

        for (double x : {0.1, 1.0, 2.5, 5.0, 10.0}) {
            double expected = 1.0 - std::exp(-0.5 * x);
            REQUIRE_THAT(ProbabilisticEstimator::chi_squared_cdf(2, x), WithinAbs(expected, 1e-12));
        }
    }

    SECTION("Numerical Accuracy Across Higher Degrees of Freedom") {
        // df = 10, x = 5.0 -> exact regularized lower incomplete gamma P(5, 2.5) ~ 0.108821996
        REQUIRE_THAT(ProbabilisticEstimator::chi_squared_cdf(10, 5.0), WithinAbs(0.108821996, 1e-6));

        // df = 4, x = 4.0: P(2, 2) = 1 - e^-2 * (1 + 2) = 1 - 3*e^-2 ~ 0.59399415
        double expected_4 = 1.0 - 3.0 * std::exp(-2.0);
        REQUIRE_THAT(ProbabilisticEstimator::chi_squared_cdf(4, 4.0), WithinAbs(expected_4, 1e-10));

        // df = 50, x = 50.0: mean of chi^2_50 is 50. Median is ~ 49.34, so CDF at 50 is ~ 0.5266
        double cdf_50 = ProbabilisticEstimator::chi_squared_cdf(50, 50.0);
        REQUIRE(cdf_50 > 0.50);
        REQUIRE(cdf_50 < 0.55);
        REQUIRE_THAT(cdf_50, WithinAbs(0.5266, 0.005));

        // Strict Monotonicity in x
        double prev = 0.0;
        for (double x = 1.0; x <= 30.0; x += 1.0) {
            double cur = ProbabilisticEstimator::chi_squared_cdf(15, x);
            REQUIRE(cur >= prev);
            prev = cur;
        }

        // Asymptotic convergence to 1
        REQUIRE_THAT(ProbabilisticEstimator::chi_squared_cdf(10, 100.0), WithinAbs(1.0, 1e-12));
    }

    SECTION("PDF Non-Negativity & Consistency") {
        // PDF should be strictly positive for x > 0
        for (size_t df : {1, 3, 10, 30}) {
            for (double x : {0.5, 2.0, 5.0, 15.0}) {
                REQUIRE(ProbabilisticEstimator::chi_squared_pdf(df, x) > 0.0);
            }
        }

        // df = 2: PDF(x) = 0.5 * exp(-x / 2)
        for (double x : {0.5, 1.0, 3.0}) {
            double expected = 0.5 * std::exp(-0.5 * x);
            REQUIRE_THAT(ProbabilisticEstimator::chi_squared_pdf(2, x), WithinAbs(expected, 1e-12));
        }
    }
}

TEST_CASE("Probabilistic - Conditional Chi-Squared Distribution", "[security][probabilistic]") {
    SECTION("Validation & Edge Cases") {
        REQUIRE_THROWS_AS(ProbabilisticEstimator::conditional_chi_squared(0, 0, 1.0, 1.0), InvalidParameterException);

        REQUIRE(ProbabilisticEstimator::conditional_chi_squared(5, 5, 0.0, 10.0) == 0.0);
        REQUIRE(ProbabilisticEstimator::conditional_chi_squared(5, 5, -2.0, 10.0) == 0.0);
        REQUIRE(ProbabilisticEstimator::conditional_chi_squared(5, 5, 10.0, 0.0) == 0.0);

        // Degenerate d1 = 0
        REQUIRE(ProbabilisticEstimator::conditional_chi_squared(0, 5, 15.0, 10.0) == 1.0);
        REQUIRE(ProbabilisticEstimator::conditional_chi_squared(0, 5, 5.0, 10.0) < 1.0);

        // Degenerate d2 = 0
        double p_d2_zero = ProbabilisticEstimator::conditional_chi_squared(6, 0, 8.0, 10.0);
        REQUIRE_THAT(p_d2_zero, WithinAbs(ProbabilisticEstimator::chi_squared_cdf(6, 8.0), 1e-12));
    }

    SECTION("Mathematical Monotonicity & Invariants") {
        const size_t d1 = 10;
        const size_t d2 = 20;
        const double l2 = 25.0;

        // Monotonicity in target bound lt: larger lt must yield higher probability
        double prev_p = 0.0;
        for (double lt = 5.0; lt <= 40.0; lt += 5.0) {
            double p = ProbabilisticEstimator::conditional_chi_squared(d1, d2, lt, l2);
            REQUIRE(p >= prev_p);
            REQUIRE(p >= 0.0);
            REQUIRE(p <= 1.0);
            prev_p = p;
        }

        // When lt is tiny compared to conditioned variables, probability should be very small
        double p_tiny = ProbabilisticEstimator::conditional_chi_squared(d1, d2, 1.0, l2);
        REQUIRE(p_tiny < 1e-4);

        // When lt is very large, probability approaches 1
        double p_huge = ProbabilisticEstimator::conditional_chi_squared(d1, d2, 200.0, l2);
        REQUIRE_THAT(p_huge, WithinAbs(1.0, 1e-4));
    }

    SECTION("Underflow Handling") {
        // Extremely small l2 in high dimension -> PE2 underflow, returns 1.0 without throwing
        double p_underflow = ProbabilisticEstimator::conditional_chi_squared(20, 100, 50.0, 0.01);
        REQUIRE(p_underflow == 1.0);
    }
}

TEST_CASE("Probabilistic - Single-Tour Detection Probability", "[security][probabilistic]") {
    SECTION("Parameter Validation") {
        std::vector<double> profile = {3.0, 2.5, 2.0, 1.5, 1.0};
        REQUIRE_THROWS_AS(ProbabilisticEstimator::single_tour_detection_probability(std::vector<double>{1.0}, 2), DimensionMismatchException);
        REQUIRE_THROWS_AS(ProbabilisticEstimator::single_tour_detection_probability(profile, 1), InvalidParameterException);
        REQUIRE_THROWS_AS(ProbabilisticEstimator::single_tour_detection_probability(profile, 10), InvalidParameterException);
    }

    SECTION("Monotonicity with Block Size beta") {
        const size_t dim = 40;
        const double ln_vol = 100.0;
        auto initial_profile = BKZSimulator::initial_lll_log_profile(dim, ln_vol);

        ProbabilisticOptions opts;
        opts.ignore_lift_proba = true; // Base tail block detection only

        double prev_p = 0.0;
        for (size_t beta = 2; beta <= 25; ++beta) {
            double p = ProbabilisticEstimator::single_tour_detection_probability(initial_profile, beta, opts);
            REQUIRE(p >= 0.0);
            REQUIRE(p <= 1.0);
            REQUIRE(p >= prev_p - 1e-5); // Monotonic up to numerical limits near saturation (p ~ 0.999999)
            prev_p = p;
        }
    }

    SECTION("Effect of Consecutive Lifting Pull-Back") {
        const size_t dim = 40;
        const double ln_vol = 80.0;
        auto profile = BKZSimulator::initial_lll_log_profile(dim, ln_vol);

        ProbabilisticOptions opts_no_lift;
        opts_no_lift.ignore_lift_proba = true;

        ProbabilisticOptions opts_with_lift;
        opts_with_lift.ignore_lift_proba = false;

        for (size_t beta : {5, 10, 15, 20}) {
            double p_no_lift = ProbabilisticEstimator::single_tour_detection_probability(profile, beta, opts_no_lift);
            double p_with_lift = ProbabilisticEstimator::single_tour_detection_probability(profile, beta, opts_with_lift);

            // Pull-back requires simultaneous satisfaction of prior events, so p_with_lift <= p_no_lift
            REQUIRE(p_with_lift <= p_no_lift + 1e-12);
        }
    }
}

TEST_CASE("Probabilistic - Fixed-Beta Multi-Tour Estimation", "[security][probabilistic]") {
    const size_t dim = 50;
    const double ln_vol = 120.0;

    SECTION("Monotonicity with Tours T") {
        const size_t beta = 20;
        double p_tour1 = ProbabilisticEstimator::estimate_probability(beta, 1, dim, ln_vol);
        double p_tour5 = ProbabilisticEstimator::estimate_probability(beta, 5, dim, ln_vol);
        double p_tour15 = ProbabilisticEstimator::estimate_probability(beta, 15, dim, ln_vol);

        REQUIRE(p_tour1 >= 0.0);
        REQUIRE(p_tour5 >= p_tour1 - 1e-12);
        REQUIRE(p_tour15 >= p_tour5 - 1e-12);
        REQUIRE(p_tour15 <= 1.0);
    }

    SECTION("Monotonicity with Block Size beta") {
        const size_t tours = 5;
        double p_b10 = ProbabilisticEstimator::estimate_probability(10, tours, dim, ln_vol);
        double p_b25 = ProbabilisticEstimator::estimate_probability(25, tours, dim, ln_vol);
        double p_b40 = ProbabilisticEstimator::estimate_probability(40, tours, dim, ln_vol);

        REQUIRE(p_b25 >= p_b10);
        REQUIRE(p_b40 >= p_b25);
    }

    SECTION("Required Beta for Target Probability Inversion") {
        size_t b_50 = ProbabilisticEstimator::required_beta_for_probability(0.50, 5, dim, ln_vol);
        size_t b_99 = ProbabilisticEstimator::required_beta_for_probability(0.99, 5, dim, ln_vol);

        REQUIRE(b_50 >= 2);
        REQUIRE(b_99 >= b_50);

        double p_at_b50 = ProbabilisticEstimator::estimate_probability(b_50, 5, dim, ln_vol);
        REQUIRE(p_at_b50 >= 0.50);

        if (b_50 > 2) {
            double p_below = ProbabilisticEstimator::estimate_probability(b_50 - 1, 5, dim, ln_vol);
            REQUIRE(p_below < 0.50);
        }
    }
}

TEST_CASE("Probabilistic - Progressive Simulation & Paper Reproduction", "[security][probabilistic]") {
    SECTION("Toy LWE Instance (n=m=3, q=11)") {
        const size_t n = 3;
        const size_t m = 3;
        const size_t q = 11;
        const size_t dim = m + n + 1; // 7

        DiscreteGaussian secret_dist(1.0, 42);
        DiscreteGaussian noise_dist(1.0, 43);
        LWEInstance lwe = LWEInstance::generate_random(n, m, BigInt(q), secret_dist, noise_dist, 44);
        KannanEmbedding kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

        ProbabilisticOptions opts;
        opts.simulation_model = SimulationModel::ChenNguyen;

        auto result = ProbabilisticEstimator::estimate_progressive(dbdd, opts);

        REQUIRE(result.solvable);
        REQUIRE(result.average_beta >= 2.0);
        REQUIRE(result.average_beta <= static_cast<double>(dim));
        REQUIRE(result.integer_beta >= 2);
        REQUIRE(result.integer_beta <= dim);
        REQUIRE(result.cumulative_probability >= 0.99);
        REQUIRE(!result.steps.empty());

        // Steps monotonicity
        double prev_cum = 0.0;
        for (const auto& step : result.steps) {
            REQUIRE(step.cumulative_probability >= prev_cum - 1e-12);
            prev_cum = step.cumulative_probability;
        }
    }

    SECTION("Dachman-Soled et al. Section 3.4 & Figure 4 Phenomenon") {
        // In Section 3.4 & Figure 4, Dachman-Soled et al. show that for small/intermediate block sizes,
        // GSA-Intersect underestimates the required block size (beta_real > beta_gsa), because
        // GSA-Intersect only checks intersection at tail block and ignores pull-back lifting probability.
        // The probabilistic simulator with lifting correctly requires larger beta.
        const size_t dim = 50;
        const double ln_vol = 140.0;

        auto gsa_est = GSAIntersect::estimate(dim, ln_vol);

        ProbabilisticOptions opts;
        opts.ignore_lift_proba = false; // Lifting enabled
        auto prob_res = ProbabilisticEstimator::estimate_progressive(dim, ln_vol, opts);

        REQUIRE(prob_res.solvable);
        REQUIRE(prob_res.average_beta >= 2.0);
        REQUIRE(prob_res.integer_beta >= 2);

        // Verification of Figure 4 effect: probabilistic simulation with lifting
        // accounts for vector pull-back, so required beta is >= GSA prediction in this regime.
        REQUIRE(prob_res.integer_beta >= gsa_est.beta);
    }

    SECTION("Conversion to SecurityEstimate Container") {
        const size_t dim = 40;
        const double ln_vol = 85.0;

        SecurityEstimate est = ProbabilisticEstimator::estimate(dim, ln_vol);
        REQUIRE(est.method == "Probabilistic-Simulation");
        REQUIRE(est.dimension == dim);
        REQUIRE_THAT(est.log_volume, WithinAbs(ln_vol, 1e-12));
        REQUIRE(est.beta >= 2);
        REQUIRE(est.beta_continuous >= 2.0);
        REQUIRE(est.delta_beta > 1.0);
        REQUIRE(est.delta() == est.delta_beta);
        REQUIRE(est.beta_integer() == est.beta);
        REQUIRE(est.bikz() == est.beta);
        REQUIRE(est.solved);
        REQUIRE(est.success_probability > 0.95);
        REQUIRE(!est.confidence_metadata.empty());
    }
}

TEST_CASE("Probabilistic - Hints & Multiple Representations", "[security][probabilistic]") {
    const size_t n = 4;
    const size_t m = 4;
    const size_t q = 17;

    DiscreteGaussian secret_dist(1.0, 101);
    DiscreteGaussian noise_dist(1.0, 102);
    LWEInstance lwe = LWEInstance::generate_random(n, m, BigInt(q), secret_dist, noise_dist, 103);
    KannanEmbedding kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    SECTION("Consistency Across Representations (DBDD, Usvp, Stats)") {
        UsvpInstance usvp = Isotropization::isotropize(dbdd);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        ProbabilisticOptions opts;
        opts.stop_remaining_proba = 0.01;

        auto est_dbdd = ProbabilisticEstimator::estimate(dbdd, opts);
        auto est_usvp = ProbabilisticEstimator::estimate(usvp, opts);
        auto est_stats = ProbabilisticEstimator::estimate(stats, opts);

        // All representations should predict identical or nearly identical block size
        REQUIRE(est_dbdd.beta == est_usvp.beta);
        REQUIRE(est_dbdd.beta == est_stats.beta);
        REQUIRE_THAT(est_dbdd.beta_continuous, WithinAbs(est_usvp.beta_continuous, 1e-3));
        REQUIRE_THAT(est_dbdd.beta_continuous, WithinAbs(est_stats.beta_continuous, 1e-3));
    }

    SECTION("Impact of Perfect Hint on Probabilistic Security") {
        ProbabilisticOptions opts;
        auto est_before = ProbabilisticEstimator::estimate(dbdd, opts);

        // Add a perfect hint revealing target[0]
        VectorZZ v(dbdd.dimension(), BigInt(0));
        v[0] = BigInt(1);
        BigInt l = dbdd.target()[0];
        dbdd.integrate_perfect_hint(v, l);

        auto est_after = ProbabilisticEstimator::estimate(dbdd, opts);

        // Dimension dropped by 1 and security level decreases or stays equal
        REQUIRE(est_after.dimension == est_before.dimension - 1);
        REQUIRE(est_after.beta_continuous <= est_before.beta_continuous + 1e-6);
        REQUIRE(est_after.beta <= est_before.beta);
    }
}

TEST_CASE("Unified SecurityEstimator API & Comparative Analysis", "[security][unified_estimator]") {
    const size_t dim = 60;
    const double ln_vol = 150.0;

    SECTION("Direct Methods Dispatch") {
        SecurityEstimate est_gsa = SecurityEstimator::gsa_intersect(dim, ln_vol);
        REQUIRE(est_gsa.method == "GSA-Intersect");
        REQUIRE(est_gsa.dimension == dim);
        REQUIRE(est_gsa.beta >= 2);

        SecurityEstimate est_prob = SecurityEstimator::probabilistic(dim, ln_vol);
        REQUIRE(est_prob.method == "Probabilistic-Simulation");
        REQUIRE(est_prob.dimension == dim);
        REQUIRE(est_prob.beta >= 2);
    }

    SECTION("SecurityComparison Container & Tabular Output") {
        SecurityComparison comp = SecurityEstimator::compare(dim, ln_vol);

        REQUIRE(comp.gsa.method == "GSA-Intersect");
        REQUIRE(comp.probabilistic.method == "Probabilistic-Simulation");
        REQUIRE_THAT(comp.delta_beta, WithinAbs(comp.probabilistic.beta_continuous - comp.gsa.beta_continuous, 1e-12));
        REQUIRE(comp.delta_bikz == static_cast<int>(comp.probabilistic.beta) - static_cast<int>(comp.gsa.beta));

        std::string report = comp.to_string();
        REQUIRE(!report.empty());
        REQUIRE(report.find("Security Estimation Comparison") != std::string::npos);
        REQUIRE(report.find("GSA-Intersect") != std::string::npos);
        REQUIRE(report.find("Probabilistic-Simulation") != std::string::npos);
        REQUIRE(report.find("Continuous Beta") != std::string::npos);
        REQUIRE(report.find("Integer bikz") != std::string::npos);
    }

    SECTION("Comparison on DBDDInstance & DBDDStateStatistics") {
        const size_t n = 5;
        const size_t m = 5;
        const size_t q = 23;
        DiscreteGaussian secret_dist(1.0, 991);
        DiscreteGaussian noise_dist(1.0, 992);
        LWEInstance lwe = LWEInstance::generate_random(n, m, BigInt(q), secret_dist, noise_dist, 993);
        KannanEmbedding kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        SecurityComparison comp_dbdd = SecurityEstimator::compare(dbdd);
        SecurityComparison comp_stats = SecurityEstimator::compare(stats);

        REQUIRE(comp_dbdd.gsa.beta == comp_stats.gsa.beta);
        REQUIRE(comp_dbdd.probabilistic.beta == comp_stats.probabilistic.beta);
        REQUIRE(!comp_dbdd.to_string().empty());
    }
}
