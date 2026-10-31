#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/security/root_hermite.hpp"
#include "dbdd/arithmetic/precision.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>

using namespace dbdd;
using Catch::Approx;

TEST_CASE("RootHermite - Literature Reference Values (beta >= 50)", "[security][root_hermite]") {
    // Reference values computed from Equation (10) to 12 decimal places
    struct TestCase {
        double beta;
        double expected_delta;
    };

    const TestCase test_cases[] = {
        {50.0,  1.012064863555},
        {100.0, 1.009258721036},
        {150.0, 1.007456501192},
        {200.0, 1.006282606911},
        {250.0, 1.005457102327},
        {300.0, 1.004842344941},
        {350.0, 1.004364890621},
        {400.0, 1.003982128196},
        {450.0, 1.003667611812},
        {500.0, 1.003404026785}
    };

    for (const auto& tc : test_cases) {
        CAPTURE(tc.beta);
        const double computed = RootHermite::delta(tc.beta);
        REQUIRE(computed == Approx(tc.expected_delta).epsilon(1e-9));

        // Direct log_delta evaluation vs std::log
        const double log_comp = RootHermite::log_delta(tc.beta);
        REQUIRE(log_comp == Approx(std::log(tc.expected_delta)).epsilon(1e-9));
        REQUIRE(std::exp(log_comp) == Approx(computed).epsilon(1e-12));
    }
}

TEST_CASE("RootHermite - Monotonicity Across Full Range", "[security][root_hermite]") {
    SECTION("Dense integer check from beta = 2 to 100") {
        double prev_delta = RootHermite::delta(2.0);
        REQUIRE(prev_delta > 1.0);

        for (int b = 3; b <= 100; ++b) {
            const double curr_delta = RootHermite::delta(static_cast<double>(b));
            REQUIRE(curr_delta > 1.0);
            REQUIRE(curr_delta < prev_delta); // strictly decreasing
            REQUIRE(RootHermite::log_delta(static_cast<double>(b)) > 0.0);
            prev_delta = curr_delta;
        }
    }

    SECTION("Coarse check from beta = 100 to 1000") {
        double prev_delta = RootHermite::delta(100.0);
        for (int b = 150; b <= 1000; b += 50) {
            const double curr_delta = RootHermite::delta(static_cast<double>(b));
            REQUIRE(curr_delta > 1.0);
            REQUIRE(curr_delta < prev_delta);
            prev_delta = curr_delta;
        }
    }
}

TEST_CASE("RootHermite - Asymptotic Behavior (beta -> infinity)", "[security][root_hermite]") {
    const double d_1000 = RootHermite::delta(1000.0);
    const double d_10000 = RootHermite::delta(10000.0);
    const double d_100000 = RootHermite::delta(100000.0);

    REQUIRE(d_1000 > 1.0);
    REQUIRE(d_10000 > 1.0);
    REQUIRE(d_100000 > 1.0);

    REQUIRE(d_10000 < d_1000);
    REQUIRE(d_100000 < d_10000);

    // Delta approaches 1.0
    REQUIRE(d_10000 == Approx(1.0).margin(5e-4));
    REQUIRE(d_100000 == Approx(1.0).margin(5e-5));
}

TEST_CASE("RootHermite - Multi-Precision MPFR Consistency", "[security][root_hermite]") {
    const prec_t precisions[] = {128, 256, 512};

    for (prec_t prec : precisions) {
        ScopedPrecision sp(prec);

        for (double b : {50.0, 100.0, 200.0, 300.0, 500.0}) {
            CAPTURE(prec, b);
            BigReal beta_rr(b, prec);
            BigReal delta_rr = RootHermite::delta_rr(beta_rr, prec);
            BigReal log_delta_rr = RootHermite::log_delta_rr(beta_rr, prec);

            // Double wrapper consistency
            double delta_d = RootHermite::delta(b);
            REQUIRE(delta_rr.to_double() == Approx(delta_d).epsilon(1e-12));

            // Log consistency: exp(log(delta)) == delta
            BigReal exp_log = log_delta_rr.exp();
            BigReal diff = (exp_log - delta_rr).abs();
            REQUIRE(diff.to_double() < 1e-14);
        }
    }
}

TEST_CASE("RootHermite - Low Beta Interpolation and Smoothness", "[security][root_hermite]") {
    SECTION("Exact match at empirical table knots") {
        REQUIRE(RootHermite::delta(2.0) == Approx(1.02190));
        REQUIRE(RootHermite::delta(5.0) == Approx(1.01862));
        REQUIRE(RootHermite::delta(10.0) == Approx(1.01616));
        REQUIRE(RootHermite::delta(20.0) == Approx(1.01420));
        REQUIRE(RootHermite::delta(40.0) == Approx(1.01295));
        REQUIRE(RootHermite::delta(50.0) == Approx(1.012064863555).epsilon(1e-9));
    }

    SECTION("Continuity at boundary beta = 50.0") {
        const double d_below = RootHermite::delta(50.0 - 1e-6);
        const double d_knot  = RootHermite::delta(50.0);
        const double d_above = RootHermite::delta(50.0 + 1e-6);

        REQUIRE(d_below == Approx(d_knot).epsilon(1e-6));
        REQUIRE(d_above == Approx(d_knot).epsilon(1e-6));
    }
}

TEST_CASE("RootHermite - Error Handling for Out-of-Bounds Beta", "[security][root_hermite]") {
    REQUIRE_THROWS_AS(RootHermite::delta(1.99), InvalidParameterException);
    REQUIRE_THROWS_AS(RootHermite::delta(0.0), InvalidParameterException);
    REQUIRE_THROWS_AS(RootHermite::delta(-5.0), InvalidParameterException);

    REQUIRE_THROWS_AS(RootHermite::log_delta(1.0), InvalidParameterException);
    REQUIRE_THROWS_AS(RootHermite::delta_gh(1.0), InvalidParameterException);
    REQUIRE_THROWS_AS(RootHermite::log_delta_gh(0.5), InvalidParameterException);

    BigReal beta_bad(1.5, 128);
    REQUIRE_THROWS_AS(RootHermite::delta_rr(beta_bad, 128), InvalidParameterException);
    REQUIRE_THROWS_AS(RootHermite::log_delta_rr(beta_bad, 128), InvalidParameterException);
}

