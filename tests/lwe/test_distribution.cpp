#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/lwe/distribution.hpp"
#include <cmath>
#include <vector>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("DiscreteGaussian: parameter validation and moments", "[lwe][distribution]") {
    SECTION("Invalid sigma throws InvalidParameterException") {
        REQUIRE_THROWS_AS(DiscreteGaussian(0.0), InvalidParameterException);
        REQUIRE_THROWS_AS(DiscreteGaussian(-2.5), InvalidParameterException);
        REQUIRE_THROWS_AS(DiscreteGaussian::from_gaussian_parameter(0.0), InvalidParameterException);
        REQUIRE_THROWS_AS(DiscreteGaussian::from_gaussian_parameter(-1.0), InvalidParameterException);
    }

    SECTION("Parameter conversion from gaussian parameter s") {
        double s = 10.0;
        auto dist = DiscreteGaussian::from_gaussian_parameter(s);
        double expected_sigma = s / std::sqrt(2.0 * std::numbers::pi);
        REQUIRE_THAT(dist.sigma(), WithinAbs(expected_sigma, 1e-12));
        REQUIRE_THAT(dist.gaussian_parameter_s(), WithinAbs(s, 1e-12));
        REQUIRE_THAT(dist.mean(), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(dist.variance(), WithinAbs(expected_sigma * expected_sigma, 1e-12));
        REQUIRE_THAT(dist.standard_deviation(), WithinAbs(expected_sigma, 1e-12));
    }

    SECTION("PMF symmetry and normalization") {
        double sigma = 3.2;
        DiscreteGaussian dist(sigma);

        REQUIRE(dist.probability(0) > dist.probability(1));
        REQUIRE_THAT(dist.probability(1), WithinAbs(dist.probability(-1), 1e-12));
        REQUIRE_THAT(dist.probability(3), WithinAbs(dist.probability(-3), 1e-12));

        double sum_prob = dist.probability(0);
        int64_t bound = static_cast<int64_t>(std::ceil(10.0 * sigma));
        for (int64_t k = 1; k <= bound; ++k) {
            sum_prob += 2.0 * dist.probability(k);
        }
        REQUIRE_THAT(sum_prob, WithinAbs(1.0, 1e-10));
        REQUIRE(dist.probability(bound + 10) == 0.0);
    }

    SECTION("Empirical statistics over 10^5 samples (sigma = 3.2)") {
        double sigma = 3.2;
        DiscreteGaussian dist(sigma, 12345);

        const size_t N = 100000;
        std::vector<int64_t> samples(N);
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            samples[i] = dist.sample();
            sum += static_cast<double>(samples[i]);
        }

        double mean = sum / static_cast<double>(N);
        double sq_sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double diff = static_cast<double>(samples[i]) - mean;
            sq_sum += diff * diff;
        }
        double var = sq_sum / static_cast<double>(N - 1);

        // Standard error of mean: sigma / sqrt(N) approx 3.2 / 316.2 approx 0.0101
        // 4 * SE = 0.0405
        REQUIRE_THAT(mean, WithinAbs(0.0, 0.05));

        // Standard error of variance: sigma^2 * sqrt(2 / N) approx 10.24 * 0.00447 approx 0.0458
        // 4 * SE = 0.183
        REQUIRE_THAT(var, WithinAbs(sigma * sigma, 0.20));
    }

    SECTION("Deterministic seeding") {
        DiscreteGaussian dist1(4.0, 9999);
        DiscreteGaussian dist2(4.0, 9999);

        for (int i = 0; i < 50; ++i) {
            REQUIRE(dist1.sample() == dist2.sample());
        }
    }
}

TEST_CASE("CenteredBinomial: parameter validation and moments", "[lwe][distribution]") {
    SECTION("Invalid eta throws InvalidParameterException") {
        REQUIRE_THROWS_AS(CenteredBinomial(0), InvalidParameterException);
    }

    SECTION("PMF matches exact binomial formula for eta = 2") {
        CenteredBinomial cbd2(2);
        REQUIRE_THAT(cbd2.mean(), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(cbd2.variance(), WithinAbs(1.0, 1e-12));

        // For eta = 2: P(0) = 6/16 = 0.375, P(+/-1) = 4/16 = 0.25, P(+/-2) = 1/16 = 0.0625
        REQUIRE_THAT(cbd2.probability(0), WithinAbs(6.0 / 16.0, 1e-12));
        REQUIRE_THAT(cbd2.probability(1), WithinAbs(4.0 / 16.0, 1e-12));
        REQUIRE_THAT(cbd2.probability(-1), WithinAbs(4.0 / 16.0, 1e-12));
        REQUIRE_THAT(cbd2.probability(2), WithinAbs(1.0 / 16.0, 1e-12));
        REQUIRE_THAT(cbd2.probability(-2), WithinAbs(1.0 / 16.0, 1e-12));
        REQUIRE(cbd2.probability(3) == 0.0);
        REQUIRE(cbd2.probability(-3) == 0.0);
    }

    SECTION("Empirical statistics over 10^5 samples (eta = 3)") {
        CenteredBinomial cbd3(3, 54321);
        const size_t N = 100000;
        double sum = 0.0;
        double sq_sum = 0.0;

        for (size_t i = 0; i < N; ++i) {
            int64_t x = cbd3.sample();
            REQUIRE(x >= -3);
            REQUIRE(x <= 3);
            sum += static_cast<double>(x);
            sq_sum += static_cast<double>(x * x);
        }

        double mean = sum / static_cast<double>(N);
        double var = (sq_sum - sum * mean) / static_cast<double>(N - 1);

        // Theoretical variance: eta / 2 = 1.5
        REQUIRE_THAT(mean, WithinAbs(0.0, 0.03));
        REQUIRE_THAT(var, WithinAbs(1.5, 0.05));
    }
}

TEST_CASE("UniformTernary: parameter validation and moments", "[lwe][distribution]") {
    SECTION("Invalid parameter p throws InvalidParameterException") {
        REQUIRE_THROWS_AS(UniformTernary(-0.1), InvalidParameterException);
        REQUIRE_THROWS_AS(UniformTernary(1.1), InvalidParameterException);
    }

    SECTION("Balanced ternary p = 2/3") {
        UniformTernary dist(2.0 / 3.0);
        REQUIRE_THAT(dist.mean(), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(dist.variance(), WithinAbs(2.0 / 3.0, 1e-12));

        REQUIRE_THAT(dist.probability(1), WithinAbs(1.0 / 3.0, 1e-12));
        REQUIRE_THAT(dist.probability(-1), WithinAbs(1.0 / 3.0, 1e-12));
        REQUIRE_THAT(dist.probability(0), WithinAbs(1.0 / 3.0, 1e-12));
        REQUIRE(dist.probability(2) == 0.0);
    }

    SECTION("Empirical sampling for p = 0.5") {
        UniformTernary dist(0.5, 7777);
        const size_t N = 100000;
        int count_neg = 0, count_zero = 0, count_pos = 0;

        for (size_t i = 0; i < N; ++i) {
            int64_t x = dist.sample();
            if (x == -1) ++count_neg;
            else if (x == 0) ++count_zero;
            else if (x == 1) ++count_pos;
            else FAIL("Sample outside ternary support");
        }

        // Expected counts: 25000, 50000, 25000
        double freq_neg = static_cast<double>(count_neg) / static_cast<double>(N);
        double freq_zero = static_cast<double>(count_zero) / static_cast<double>(N);
        double freq_pos = static_cast<double>(count_pos) / static_cast<double>(N);

        REQUIRE_THAT(freq_neg, WithinAbs(0.25, 0.01));
        REQUIRE_THAT(freq_zero, WithinAbs(0.50, 0.01));
        REQUIRE_THAT(freq_pos, WithinAbs(0.25, 0.01));
    }
}

TEST_CASE("ConfigurableDiscrete: custom PMF and moments", "[lwe][distribution]") {
    SECTION("Validation of invalid inputs") {
        REQUIRE_THROWS_AS(ConfigurableDiscrete({}, {}), InvalidParameterException);
        REQUIRE_THROWS_AS(ConfigurableDiscrete({1, 2}, {0.5}), DimensionMismatchException);
        REQUIRE_THROWS_AS(ConfigurableDiscrete({1, 2}, {-0.1, 0.5}), InvalidParameterException);
        REQUIRE_THROWS_AS(ConfigurableDiscrete({1, 2}, {0.0, 0.0}), InvalidParameterException);
    }

    SECTION("Custom PMF moments and sampling") {
        // Values: -3, -1, 2, 5 with weights: 0.1, 0.4, 0.3, 0.2
        // Mean = -0.3 - 0.4 + 0.6 + 1.0 = 0.9
        // Variance = 0.1*( -3.9)^2 + 0.4*(-1.9)^2 + 0.3*(1.1)^2 + 0.2*(4.1)^2 = 6.69
        ConfigurableDiscrete dist({{-3, 0.1}, {-1, 0.4}, {2, 0.3}, {5, 0.2}}, 8888);

        REQUIRE_THAT(dist.mean(), WithinAbs(0.9, 1e-12));
        REQUIRE_THAT(dist.variance(), WithinAbs(6.69, 1e-12));

        REQUIRE_THAT(dist.probability(-3), WithinAbs(0.1, 1e-12));
        REQUIRE_THAT(dist.probability(-1), WithinAbs(0.4, 1e-12));
        REQUIRE_THAT(dist.probability(2), WithinAbs(0.3, 1e-12));
        REQUIRE_THAT(dist.probability(5), WithinAbs(0.2, 1e-12));
        REQUIRE(dist.probability(0) == 0.0);

        const size_t N = 100000;
        double sum = 0.0;
        double sq_sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            int64_t x = dist.sample();
            sum += static_cast<double>(x);
            sq_sum += static_cast<double>(x * x);
        }

        double emp_mean = sum / static_cast<double>(N);
        double emp_var = (sq_sum - sum * emp_mean) / static_cast<double>(N - 1);

        REQUIRE_THAT(emp_mean, WithinAbs(0.9, 0.05));
        REQUIRE_THAT(emp_var, WithinAbs(6.69, 0.15));
    }

    SECTION("sample_vector creates VectorZZ of correct size") {
        DiscreteGaussian dg(2.0);
        VectorZZ vec = dg.sample_vector(10);
        REQUIRE(vec.size() == 10);
    }
}
