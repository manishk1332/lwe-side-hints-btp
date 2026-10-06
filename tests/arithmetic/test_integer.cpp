#include <catch2/catch_test_macros.hpp>
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/core/exceptions.hpp"

using namespace dbdd;

TEST_CASE("BigInt basic arithmetic operations", "[arithmetic][integer]") {
    BigInt a(42);
    BigInt b(13);

    REQUIRE(a + b == BigInt(55));
    REQUIRE(a - b == BigInt(29));
    REQUIRE(a * b == BigInt(546));
    REQUIRE(a / b == BigInt(3));
    REQUIRE(a % b == BigInt(3));
    REQUIRE(-a == BigInt(-42));
    REQUIRE(+a == BigInt(42));

    BigInt c = a;
    REQUIRE(++c == BigInt(43));
    REQUIRE(c++ == BigInt(43));
    REQUIRE(c == BigInt(44));
    REQUIRE(--c == BigInt(43));
    REQUIRE(c-- == BigInt(43));
    REQUIRE(c == BigInt(42));
}

TEST_CASE("BigInt compound assignment operators", "[arithmetic][integer]") {
    BigInt x(10);
    x += 5;
    REQUIRE(x == BigInt(15));
    x -= 3;
    REQUIRE(x == BigInt(12));
    x *= 2;
    REQUIRE(x == BigInt(24));
    x /= 4;
    REQUIRE(x == BigInt(6));
    x %= 4;
    REQUIRE(x == BigInt(2));
}

TEST_CASE("BigInt division and modulo by zero", "[arithmetic][integer]") {
    BigInt a(100);
    BigInt zero(0);

    REQUIRE_THROWS_AS(a / zero, DivisionByZeroException);
    REQUIRE_THROWS_AS(a % zero, DivisionByZeroException);
    REQUIRE_THROWS_AS(BigInt::div_floor(a, zero), DivisionByZeroException);
    REQUIRE_THROWS_AS(BigInt::mod_floor(a, zero), DivisionByZeroException);
}

TEST_CASE("BigInt floor division and modulo (lattice coset invariant)", "[arithmetic][integer]") {
    BigInt a(-7);
    BigInt b(3);

    BigInt q = BigInt::div_floor(a, b);
    BigInt r = BigInt::mod_floor(a, b);

    REQUIRE(q == BigInt(-3));
    REQUIRE(r == BigInt(2));
    REQUIRE(q * b + r == a);
    REQUIRE(r >= BigInt(0));
    REQUIRE(r < b);
}

TEST_CASE("BigInt arbitrarily large precision (no overflow)", "[arithmetic][integer]") {
    BigInt two(2);
    BigInt pow100 = BigInt::pow(two, 100);
    BigInt pow200 = BigInt::pow(two, 200);

    REQUIRE(pow100 * pow100 == pow200);
    REQUIRE(pow200 / pow100 == pow100);
    REQUIRE(pow200 > pow100);
}

TEST_CASE("BigInt GCD, LCM, and Extended GCD", "[arithmetic][integer]") {
    BigInt a(240);
    BigInt b(46);

    BigInt g = BigInt::gcd(a, b);
    REQUIRE(g == BigInt(2));

    BigInt l = BigInt::lcm(a, b);
    REQUIRE(l == (a * b) / g);

    BigInt s, t;
    BigInt xg = BigInt::xgcd(a, b, s, t);
    REQUIRE(xg == g);
    REQUIRE(s * a + t * b == g);
}

TEST_CASE("BigInt integer square root", "[arithmetic][integer]") {
    REQUIRE(BigInt(0).sqrt() == BigInt(0));
    REQUIRE(BigInt(1).sqrt() == BigInt(1));
    REQUIRE(BigInt(49).sqrt() == BigInt(7));
    REQUIRE(BigInt(50).sqrt() == BigInt(7));
    REQUIRE(BigInt(1000000).sqrt() == BigInt(1000));

    REQUIRE_THROWS_AS(BigInt(-4).sqrt(), NumericalPrecisionException);
}

TEST_CASE("BigInt string parsing and formatting roundtrips", "[arithmetic][integer]") {
    std::string dec_str = "123456789012345678901234567890";
    BigInt val(dec_str);
    REQUIRE(val.to_string() == dec_str);

    std::string hex_str = "1a2b3c4d5e6f";
    BigInt hex_val(hex_str, 16);
    REQUIRE(hex_val.to_string(16) == hex_str);
}

TEST_CASE("BigInt explicit conversions and bounds checking", "[arithmetic][integer]") {
    BigInt small(12345);
    REQUIRE(small.to_double() == 12345.0);
    REQUIRE(small.to_int64() == 12345);

    BigInt huge = BigInt::pow(BigInt(10), 30);
    REQUIRE_THROWS_AS(huge.to_int64(), NumericalPrecisionException);
}

TEST_CASE("BigInt comparisons and orderings", "[arithmetic][integer]") {
    BigInt neg(-10);
    BigInt zero(0);
    BigInt pos(10);

    REQUIRE(neg < zero);
    REQUIRE(zero < pos);
    REQUIRE(neg <= zero);
    REQUIRE(pos >= zero);
    REQUIRE(pos > neg);
    REQUIRE(neg != pos);
    REQUIRE(neg == BigInt(-10));
}
