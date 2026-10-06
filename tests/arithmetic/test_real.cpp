#include <catch2/catch_test_macros.hpp>
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/precision.hpp"
#include "dbdd/core/exceptions.hpp"

using namespace dbdd;

TEST_CASE("BigReal precision management and ScopedPrecision RAII", "[arithmetic][real]") {
    REQUIRE(PrecisionManager::get_default_precision() == 256);

    {
        ScopedPrecision p128(128);
        REQUIRE(PrecisionManager::get_default_precision() == 128);
        BigReal x(2.0);
        REQUIRE(x.precision() == 128);

        {
            ScopedPrecision p512(512);
            REQUIRE(PrecisionManager::get_default_precision() == 512);
            BigReal y(2.0);
            REQUIRE(y.precision() == 512);
        }

        REQUIRE(PrecisionManager::get_default_precision() == 128);
    }

    REQUIRE(PrecisionManager::get_default_precision() == 256);

    REQUIRE_THROWS_AS(PrecisionManager::set_default_precision(16), NumericalPrecisionException);
}

TEST_CASE("BigReal transcendental roundtrip identities across 128, 256, 512 bits", "[arithmetic][real]") {
    for (prec_t prec : {128, 256, 512}) {
        ScopedPrecision scope(prec);

        BigReal x(123.456789);
        BigReal log_exp_x = exp(log(x));
        BigReal diff = abs(log_exp_x - x);
        REQUIRE(diff < BigReal("1e-35", prec));

        BigReal sqrt_sq_x = sqrt(x) * sqrt(x);
        BigReal sqrt_diff = abs(sqrt_sq_x - x);
        REQUIRE(sqrt_diff < BigReal("1e-35", prec));

        // Trigonometric identity: sin^2(x) + cos^2(x) == 1
        BigReal s = sin(x);
        BigReal c = cos(x);
        BigReal trig_sum = s * s + c * c;
        REQUIRE(abs(trig_sum - BigReal(1.0, prec)) < BigReal("1e-35", prec));
    }
}

TEST_CASE("BigReal high precision constants", "[arithmetic][real]") {
    ScopedPrecision scope(256);

    BigReal pi = BigReal::pi();
    BigReal e = BigReal::e();
    BigReal ln2 = BigReal::ln2();

    REQUIRE(pi > BigReal(3.14159));
    REQUIRE(pi < BigReal(3.14160));

    REQUIRE(e > BigReal(2.71828));
    REQUIRE(e < BigReal(2.71829));

    // exp(1) == e
    REQUIRE(abs(exp(BigReal(1.0)) - e) < BigReal("1e-70"));

    // log(2) == ln2
    REQUIRE(abs(log(BigReal(2.0)) - ln2) < BigReal("1e-70"));
}

TEST_CASE("BigReal log-domain stability (large power comparison)", "[arithmetic][real]") {
    ScopedPrecision scope(256);

    // In DBDD, powers like delta^(2*beta - d - 1) are huge and must be handled via log
    // Evaluate log(x^k) = k * log(x)
    BigReal delta("1.005");
    long exponent = 10000;

    BigReal log_power = static_cast<double>(exponent) * log(delta);
    REQUIRE(log_power > BigReal(0.0));
    REQUIRE(log_power.is_finite());
}

TEST_CASE("BigReal conversions and predicates", "[arithmetic][real]") {
    BigReal val("12.75");
    REQUIRE(val.to_double() == 12.75);
    REQUIRE(val.to_bigint() == BigInt(12));

    BigRational rat = val.to_bigrational();
    REQUIRE(rat == BigRational(51, 4));

    REQUIRE(val.is_finite());
    REQUIRE(!val.is_nan());
    REQUIRE(!val.is_inf());
    REQUIRE(!val.is_zero());
    REQUIRE(val.sign() > 0);
}
