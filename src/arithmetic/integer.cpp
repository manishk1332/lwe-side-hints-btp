#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/real.hpp"
#include <iostream>
#include <limits>

namespace dbdd {

BigInt::BigInt() : value_(0) {}
BigInt::BigInt(int val) : value_(val) {}
BigInt::BigInt(long val) : value_(val) {}
BigInt::BigInt(long long val) : value_(std::to_string(val)) {}
BigInt::BigInt(unsigned int val) : value_(val) {}
BigInt::BigInt(unsigned long val) : value_(val) {}
BigInt::BigInt(unsigned long long val) : value_(std::to_string(val)) {}
BigInt::BigInt(mpz_class val) : value_(std::move(val)) {}

BigInt::BigInt(const std::string& str, int base) {
    if (mpz_set_str(value_.get_mpz_t(), str.c_str(), base) != 0) {
        throw DBDDException("Failed to parse BigInt string: " + str);
    }
}

BigInt::BigInt(const char* str, int base) {
    if (str == nullptr || mpz_set_str(value_.get_mpz_t(), str, base) != 0) {
        throw DBDDException("Failed to parse BigInt string");
    }
}

double BigInt::to_double() const {
    return value_.get_d();
}

long double BigInt::to_long_double() const {
    try {
        return std::stold(to_string());
    } catch (...) {
        return static_cast<long double>(to_double());
    }
}

int64_t BigInt::to_int64() const {
    static const BigInt min_i64(std::numeric_limits<int64_t>::min());
    static const BigInt max_i64(std::numeric_limits<int64_t>::max());

    if (*this < min_i64 || *this > max_i64) {
        throw NumericalPrecisionException("BigInt out of 64-bit signed integer range: " + to_string());
    }
    return std::stoll(to_string());
}

std::string BigInt::to_string(int base) const {
    return value_.get_str(base);
}

BigReal BigInt::to_mpfr(prec_t prec) const {
    return BigReal(*this, prec);
}

BigInt BigInt::operator+() const {
    return *this;
}

BigInt BigInt::operator-() const {
    return BigInt(-value_);
}

BigInt& BigInt::operator++() {
    ++value_;
    return *this;
}

BigInt BigInt::operator++(int) {
    BigInt tmp(*this);
    ++value_;
    return tmp;
}

BigInt& BigInt::operator--() {
    --value_;
    return *this;
}

BigInt BigInt::operator--(int) {
    BigInt tmp(*this);
    --value_;
    return tmp;
}

BigInt& BigInt::operator+=(const BigInt& rhs) {
    value_ += rhs.value_;
    return *this;
}

BigInt& BigInt::operator-=(const BigInt& rhs) {
    value_ -= rhs.value_;
    return *this;
}

BigInt& BigInt::operator*=(const BigInt& rhs) {
    value_ *= rhs.value_;
    return *this;
}

BigInt& BigInt::operator/=(const BigInt& rhs) {
    if (rhs.value_ == 0) {
        throw DivisionByZeroException("BigInt division by zero");
    }
    value_ /= rhs.value_;
    return *this;
}

BigInt& BigInt::operator%=(const BigInt& rhs) {
    if (rhs.value_ == 0) {
        throw DivisionByZeroException("BigInt modulo by zero");
    }
    value_ %= rhs.value_;
    return *this;
}

BigInt operator+(BigInt lhs, const BigInt& rhs) {
    lhs += rhs;
    return lhs;
}

BigInt operator-(BigInt lhs, const BigInt& rhs) {
    lhs -= rhs;
    return lhs;
}

BigInt operator*(BigInt lhs, const BigInt& rhs) {
    lhs *= rhs;
    return lhs;
}

BigInt operator/(BigInt lhs, const BigInt& rhs) {
    lhs /= rhs;
    return lhs;
}

BigInt operator%(BigInt lhs, const BigInt& rhs) {
    lhs %= rhs;
    return lhs;
}

BigInt& BigInt::operator&=(const BigInt& rhs) {
    value_ &= rhs.value_;
    return *this;
}

BigInt& BigInt::operator|=(const BigInt& rhs) {
    value_ |= rhs.value_;
    return *this;
}

BigInt& BigInt::operator^=(const BigInt& rhs) {
    value_ ^= rhs.value_;
    return *this;
}

BigInt& BigInt::operator<<=(unsigned long shift) {
    value_ <<= shift;
    return *this;
}

BigInt& BigInt::operator>>=(unsigned long shift) {
    value_ >>= shift;
    return *this;
}

BigInt BigInt::operator~() const {
    return BigInt(~value_);
}

BigInt operator&(BigInt lhs, const BigInt& rhs) {
    lhs &= rhs;
    return lhs;
}

BigInt operator|(BigInt lhs, const BigInt& rhs) {
    lhs |= rhs;
    return lhs;
}

BigInt operator^(BigInt lhs, const BigInt& rhs) {
    lhs ^= rhs;
    return lhs;
}

BigInt operator<<(BigInt lhs, unsigned long shift) {
    lhs <<= shift;
    return lhs;
}

BigInt operator>>(BigInt lhs, unsigned long shift) {
    lhs >>= shift;
    return lhs;
}

std::strong_ordering BigInt::operator<=>(const BigInt& rhs) const noexcept {
    int cmp = mpz_cmp(value_.get_mpz_t(), rhs.value_.get_mpz_t());
    if (cmp < 0) return std::strong_ordering::less;
    if (cmp > 0) return std::strong_ordering::greater;
    return std::strong_ordering::equal;
}

bool BigInt::operator==(const BigInt& rhs) const noexcept {
    return mpz_cmp(value_.get_mpz_t(), rhs.value_.get_mpz_t()) == 0;
}

BigInt BigInt::abs() const {
    BigInt res;
    mpz_abs(res.value_.get_mpz_t(), value_.get_mpz_t());
    return res;
}

int BigInt::sign() const noexcept {
    return mpz_sgn(value_.get_mpz_t());
}

BigInt BigInt::sqrt() const {
    if (value_ < 0) {
        throw NumericalPrecisionException("Cannot take square root of negative integer");
    }
    BigInt res;
    mpz_sqrt(res.value_.get_mpz_t(), value_.get_mpz_t());
    return res;
}

BigInt BigInt::div_floor(const BigInt& a, const BigInt& b) {
    if (b.value_ == 0) {
        throw DivisionByZeroException("BigInt div_floor division by zero");
    }
    BigInt res;
    mpz_fdiv_q(res.value_.get_mpz_t(), a.value_.get_mpz_t(), b.value_.get_mpz_t());
    return res;
}

BigInt BigInt::mod_floor(const BigInt& a, const BigInt& b) {
    if (b.value_ == 0) {
        throw DivisionByZeroException("BigInt mod_floor division by zero");
    }
    BigInt res;
    mpz_fdiv_r(res.value_.get_mpz_t(), a.value_.get_mpz_t(), b.value_.get_mpz_t());
    return res;
}

BigInt BigInt::gcd(const BigInt& a, const BigInt& b) {
    BigInt res;
    mpz_gcd(res.value_.get_mpz_t(), a.value_.get_mpz_t(), b.value_.get_mpz_t());
    return res;
}

BigInt BigInt::lcm(const BigInt& a, const BigInt& b) {
    BigInt res;
    mpz_lcm(res.value_.get_mpz_t(), a.value_.get_mpz_t(), b.value_.get_mpz_t());
    return res;
}

BigInt BigInt::xgcd(const BigInt& a, const BigInt& b, BigInt& s, BigInt& t) {
    BigInt g;
    mpz_gcdext(g.value_.get_mpz_t(), s.value_.get_mpz_t(), t.value_.get_mpz_t(),
               a.value_.get_mpz_t(), b.value_.get_mpz_t());
    return g;
}

BigInt BigInt::pow(const BigInt& base, unsigned long exp) {
    BigInt res;
    mpz_pow_ui(res.value_.get_mpz_t(), base.value_.get_mpz_t(), exp);
    return res;
}

std::ostream& operator<<(std::ostream& os, const BigInt& val) {
    return os << val.to_string();
}

std::istream& operator>>(std::istream& is, BigInt& val) {
    std::string str;
    if (is >> str) {
        val = BigInt(str);
    }
    return is;
}

} // namespace dbdd
