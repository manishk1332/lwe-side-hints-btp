#include <catch2/catch_test_macros.hpp>
#include "dbdd/linalg/vector.hpp"
#include "dbdd/core/exceptions.hpp"

using namespace dbdd;

TEST_CASE("Vector basic operations and arithmetic", "[linalg][vector]") {
    Vector<BigInt> u = {BigInt(1), BigInt(2), BigInt(3)};
    Vector<BigInt> v = {BigInt(4), BigInt(5), BigInt(6)};

    Vector<BigInt> sum = u + v;
    REQUIRE(sum[0] == BigInt(5));
    REQUIRE(sum[1] == BigInt(7));
    REQUIRE(sum[2] == BigInt(9));

    Vector<BigInt> diff = v - u;
    REQUIRE(diff[0] == BigInt(3));
    REQUIRE(diff[1] == BigInt(3));
    REQUIRE(diff[2] == BigInt(3));

    Vector<BigInt> scaled = u * BigInt(3);
    REQUIRE(scaled[0] == BigInt(3));
    REQUIRE(scaled[1] == BigInt(6));
    REQUIRE(scaled[2] == BigInt(9));
}

TEST_CASE("Vector inner product and norm invariant", "[linalg][vector]") {
    // Invariant: ||v||^2 = <v, v>
    Vector<BigInt> v = {BigInt(2), BigInt(-3), BigInt(6)};
    // 2^2 + (-3)^2 + 6^2 = 4 + 9 + 36 = 49
    REQUIRE(v.dot(v) == BigInt(49));
    REQUIRE(v.squared_norm() == BigInt(49));
    REQUIRE(v.norm() == BigReal(7.0));

    // Bilinearity & Symmetry
    Vector<BigInt> u = {BigInt(1), BigInt(2), BigInt(-1)};
    REQUIRE(u.dot(v) == v.dot(u));

    BigInt a(2), b(3);
    Vector<BigInt> w = {BigInt(3), BigInt(0), BigInt(1)};
    BigInt lhs = (a * u + b * v).dot(w);
    BigInt rhs = a * u.dot(w) + b * v.dot(w);
    REQUIRE(lhs == rhs);
}

TEST_CASE("Vector slicing and concatenation", "[linalg][vector]") {
    Vector<BigInt> v = {BigInt(10), BigInt(20), BigInt(30), BigInt(40), BigInt(50)};
    Vector<BigInt> sub = v.slice(1, 3);
    REQUIRE(sub.size() == 3);
    REQUIRE(sub[0] == BigInt(20));
    REQUIRE(sub[1] == BigInt(30));
    REQUIRE(sub[2] == BigInt(40));

    Vector<BigInt> v1 = {BigInt(1), BigInt(2)};
    Vector<BigInt> v2 = {BigInt(3), BigInt(4)};
    Vector<BigInt> cat = v1.concat(v2);
    REQUIRE(cat.size() == 4);
    REQUIRE(cat[0] == BigInt(1));
    REQUIRE(cat[3] == BigInt(4));
}

TEST_CASE("Vector dimension mismatch exceptions", "[linalg][vector]") {
    Vector<BigInt> u = {BigInt(1), BigInt(2)};
    Vector<BigInt> v = {BigInt(1), BigInt(2), BigInt(3)};

    REQUIRE_THROWS_AS(u + v, DimensionMismatchException);
    REQUIRE_THROWS_AS(u - v, DimensionMismatchException);
    REQUIRE_THROWS_AS(u.dot(v), DimensionMismatchException);
}

TEST_CASE("Vector type casting across domains", "[linalg][vector]") {
    Vector<BigInt> vi = {BigInt(1), BigInt(2), BigInt(3)};
    Vector<BigRational> vr = vi.cast<BigRational>();
    REQUIRE(vr[0] == BigRational(1));
    REQUIRE(vr[2] == BigRational(3));

    Vector<BigReal> vf = vr.cast<BigReal>();
    REQUIRE(vf[1] == BigReal(2.0));
}

TEST_CASE("Vector Eigen conversion roundtrip", "[linalg][vector]") {
    Vector<double> vd = {1.5, 2.5, -3.5};
    Eigen::Matrix<double, 1, Eigen::Dynamic> ev = vd.to_eigen();
    REQUIRE(ev.cols() == 3);
    REQUIRE(ev(0, 0) == 1.5);
    REQUIRE(ev(0, 2) == -3.5);

    Vector<double> back = Vector<double>::from_eigen(ev);
    REQUIRE(back == vd);
}
