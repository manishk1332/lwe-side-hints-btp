#include <catch2/catch_test_macros.hpp>
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/core/exceptions.hpp"

using namespace dbdd;

TEST_CASE("BigRational canonical form invariants", "[arithmetic][rational]") {
    BigRational r1(4, 8);
    REQUIRE(r1.numerator() == BigInt(1));
    REQUIRE(r1.denominator() == BigInt(2));

    BigRational r2(-3, -6);
    REQUIRE(r2.numerator() == BigInt(1));
    REQUIRE(r2.denominator() == BigInt(2));

    BigRational r3(3, -6);
    REQUIRE(r3.numerator() == BigInt(-1));
    REQUIRE(r3.denominator() == BigInt(2));

    BigRational r4(-3, 6);
    REQUIRE(r4.numerator() == BigInt(-1));
    REQUIRE(r4.denominator() == BigInt(2));
}

TEST_CASE("BigRational exact arithmetic", "[arithmetic][rational]") {
    BigRational a(1, 3);
    BigRational b(1, 6);

    REQUIRE(a + b == BigRational(1, 2));
    REQUIRE(a - b == BigRational(1, 6));

    BigRational c(7, 11);
    BigRational d(11, 7);
    REQUIRE(c * d == BigRational(1));

    BigRational e(2, 5);
    BigRational f(3, 7);
    REQUIRE(e / f == BigRational(14, 15));
}

TEST_CASE("BigRational division by zero exceptions", "[arithmetic][rational]") {
    REQUIRE_THROWS_AS(BigRational(5, 0), DivisionByZeroException);
    REQUIRE_THROWS_AS(BigRational(BigInt(5), BigInt(0)), DivisionByZeroException);

    BigRational r(3, 4);
    BigRational zero(0);
    REQUIRE_THROWS_AS(r / zero, DivisionByZeroException);
    REQUIRE_THROWS_AS(r /= zero, DivisionByZeroException);
}

TEST_CASE("BigRational floor, ceil, and round", "[arithmetic][rational]") {
    // 7/3 = 2.333...
    BigRational pos1(7, 3);
    REQUIRE(pos1.floor() == BigInt(2));
    REQUIRE(pos1.ceil() == BigInt(3));
    REQUIRE(pos1.round() == BigInt(2));

    // 8/3 = 2.666...
    BigRational pos2(8, 3);
    REQUIRE(pos2.floor() == BigInt(2));
    REQUIRE(pos2.ceil() == BigInt(3));
    REQUIRE(pos2.round() == BigInt(3));

    // 5/2 = 2.5
    BigRational half_pos(5, 2);
    REQUIRE(half_pos.floor() == BigInt(2));
    REQUIRE(half_pos.ceil() == BigInt(3));
    REQUIRE(half_pos.round() == BigInt(3));

    // -7/3 = -2.333...
    BigRational neg1(-7, 3);
    REQUIRE(neg1.floor() == BigInt(-3));
    REQUIRE(neg1.ceil() == BigInt(-2));
    REQUIRE(neg1.round() == BigInt(-2));

    // -8/3 = -2.666...
    BigRational neg2(-8, 3);
    REQUIRE(neg2.floor() == BigInt(-3));
    REQUIRE(neg2.ceil() == BigInt(-2));
    REQUIRE(neg2.round() == BigInt(-3));

    // -5/2 = -2.5
    BigRational half_neg(-5, 2);
    REQUIRE(half_neg.floor() == BigInt(-3));
    REQUIRE(half_neg.ceil() == BigInt(-2));
    REQUIRE(half_neg.round() == BigInt(-3));
}

TEST_CASE("BigRational comparisons", "[arithmetic][rational]") {
    BigRational a(1, 3);
    BigRational b(1, 2);
    BigRational c(2, 6);

    REQUIRE(a < b);
    REQUIRE(b > a);
    REQUIRE(a <= b);
    REQUIRE(b >= a);
    REQUIRE(a == c);
    REQUIRE(a != b);

    BigRational neg_a(-1, 3);
    BigRational neg_b(-1, 2);
    REQUIRE(neg_b < neg_a);
}

TEST_CASE("BigRational string and double conversions", "[arithmetic][rational]") {
    BigRational r("355/113");
    REQUIRE(r.numerator() == BigInt(355));
    REQUIRE(r.denominator() == BigInt(113));
    REQUIRE(r.to_string() == "355/113");

    double d = BigRational(1, 4).to_double();
    REQUIRE(d == 0.25);
}
