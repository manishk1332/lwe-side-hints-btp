#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include <iostream>

namespace dbdd {

BigRational::BigRational() : value_(0, 1) {}

BigRational::BigRational(const BigInt& num) : value_(num.raw(), 1) {}

BigRational::BigRational(int num) : value_(num, 1) {}

BigRational::BigRational(long num) : value_(num, 1) {}

BigRational::BigRational(long long num)
    : value_(mpz_class(std::to_string(num)), 1) {}

BigRational::BigRational(const BigInt& num, const BigInt& den) {
    if (den.sign() == 0) {
        throw DivisionByZeroException("BigRational denominator cannot be zero");
    }
    value_ = mpq_class(num.raw(), den.raw());
    canonicalize();
}

BigRational::BigRational(int64_t num, int64_t den) {
    if (den == 0) {
        throw DivisionByZeroException("BigRational denominator cannot be zero");
    }
    value_ = mpq_class(mpz_class(std::to_string(num)), mpz_class(std::to_string(den)));
    canonicalize();
}

BigRational::BigRational(const std::string& str, int base) {
    if (mpq_set_str(value_.get_mpq_t(), str.c_str(), base) != 0) {
        throw DBDDException("Failed to parse BigRational string: " + str);
    }
    canonicalize();
}

BigRational::BigRational(const char* str, int base) {
    if (str == nullptr || mpq_set_str(value_.get_mpq_t(), str, base) != 0) {
        throw DBDDException("Failed to parse BigRational string");
    }
    canonicalize();
}

BigRational::BigRational(mpq_class val) : value_(std::move(val)) {
    canonicalize();
}

void BigRational::canonicalize() {
    value_.canonicalize();
}

BigInt BigRational::numerator() const {
    return BigInt(value_.get_num());
}

BigInt BigRational::denominator() const {
    return BigInt(value_.get_den());
}

double BigRational::to_double() const {
    return value_.get_d();
}

long double BigRational::to_long_double() const {
    return static_cast<long double>(value_.get_d());
}

std::string BigRational::to_string(int base) const {
    return value_.get_str(base);
}

BigReal BigRational::to_mpfr(prec_t prec) const {
    return BigReal(*this, prec);
}

BigRational BigRational::operator+() const {
    return *this;
}

BigRational BigRational::operator-() const {
    BigRational res;
    mpq_neg(res.value_.get_mpq_t(), value_.get_mpq_t());
    return res;
}

BigRational& BigRational::operator+=(const BigRational& rhs) {
    value_ += rhs.value_;
    return *this;
}

BigRational& BigRational::operator-=(const BigRational& rhs) {
    value_ -= rhs.value_;
    return *this;
}

BigRational& BigRational::operator*=(const BigRational& rhs) {
    value_ *= rhs.value_;
    return *this;
}

BigRational& BigRational::operator/=(const BigRational& rhs) {
    if (rhs.sign() == 0) {
        throw DivisionByZeroException("BigRational division by zero");
    }
    value_ /= rhs.value_;
    return *this;
}

BigRational operator+(BigRational lhs, const BigRational& rhs) {
    lhs += rhs;
    return lhs;
}

BigRational operator-(BigRational lhs, const BigRational& rhs) {
    lhs -= rhs;
    return lhs;
}

BigRational operator*(BigRational lhs, const BigRational& rhs) {
    lhs *= rhs;
    return lhs;
}

BigRational operator/(BigRational lhs, const BigRational& rhs) {
    lhs /= rhs;
    return lhs;
}

std::strong_ordering BigRational::operator<=>(const BigRational& rhs) const noexcept {
    int cmp = mpq_cmp(value_.get_mpq_t(), rhs.value_.get_mpq_t());
    if (cmp < 0) return std::strong_ordering::less;
    if (cmp > 0) return std::strong_ordering::greater;
    return std::strong_ordering::equal;
}

bool BigRational::operator==(const BigRational& rhs) const noexcept {
    return mpq_equal(value_.get_mpq_t(), rhs.value_.get_mpq_t()) != 0;
}

BigRational BigRational::abs() const {
    BigRational res;
    mpq_abs(res.value_.get_mpq_t(), value_.get_mpq_t());
    return res;
}

int BigRational::sign() const noexcept {
    return mpq_sgn(value_.get_mpq_t());
}

BigInt BigRational::floor() const {
    BigInt res;
    mpz_fdiv_q(res.raw().get_mpz_t(), value_.get_num_mpz_t(), value_.get_den_mpz_t());
    return res;
}

BigInt BigRational::ceil() const {
    BigInt res;
    mpz_cdiv_q(res.raw().get_mpz_t(), value_.get_num_mpz_t(), value_.get_den_mpz_t());
    return res;
}

BigInt BigRational::round() const {
    if (sign() >= 0) {
        // floor((2*num + den) / (2*den))
        BigInt num = numerator();
        BigInt den = denominator();
        BigInt shifted = BigInt(2) * num + den;
        BigInt divisor = BigInt(2) * den;
        return BigInt::div_floor(shifted, divisor);
    } else {
        // -round(|q|)
        BigRational positive_q = abs();
        return -positive_q.round();
    }
}

std::ostream& operator<<(std::ostream& os, const BigRational& val) {
    return os << val.to_string();
}

std::istream& operator>>(std::istream& is, BigRational& val) {
    std::string str;
    if (is >> str) {
        val = BigRational(str);
    }
    return is;
}

} // namespace dbdd
