#include "dbdd/arithmetic/real.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include <cstring>
#include <iostream>
#include <vector>

namespace dbdd {

namespace {
prec_t resolve_prec(prec_t prec) {
    return (prec == 0) ? PrecisionManager::get_default_precision() : prec;
}
} // namespace

BigReal::BigReal() {
    prec_t p = PrecisionManager::get_default_precision();
    mpfr_init2(value_, p);
    mpfr_set_zero(value_, 1);
    initialized_ = true;
}

BigReal BigReal::with_precision(prec_t prec) {
    prec_t p = resolve_prec(prec);
    BigReal res;
    if (res.precision() != p) {
        mpfr_set_prec(res.value_, p);
    }
    mpfr_set_zero(res.value_, 1);
    return res;
}

BigReal BigReal::zero(prec_t prec) {
    return with_precision(prec);
}

BigReal::BigReal(double val, prec_t prec) {
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    mpfr_set_d(value_, val, MPFR_RNDN);
    initialized_ = true;
}

BigReal::BigReal(long double val, prec_t prec) {
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    mpfr_set_ld(value_, val, MPFR_RNDN);
    initialized_ = true;
}

BigReal::BigReal(int val, prec_t prec) : BigReal(static_cast<int64_t>(val), prec) {}

BigReal::BigReal(int64_t val, prec_t prec) {
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    mpfr_set_sj(value_, val, MPFR_RNDN);
    initialized_ = true;
}

BigReal::BigReal(const BigInt& val, prec_t prec) {
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    mpfr_set_z(value_, val.raw().get_mpz_t(), MPFR_RNDN);
    initialized_ = true;
}

BigReal::BigReal(const BigRational& val, prec_t prec) {
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    mpfr_set_q(value_, val.raw().get_mpq_t(), MPFR_RNDN);
    initialized_ = true;
}

BigReal::BigReal(const std::string& str, prec_t prec, int base) {
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    if (mpfr_set_str(value_, str.c_str(), base, MPFR_RNDN) != 0) {
        mpfr_clear(value_);
        throw DBDDException("Failed to parse BigReal string: " + str);
    }
    initialized_ = true;
}

BigReal::BigReal(const char* str, prec_t prec, int base) {
    if (str == nullptr) {
        throw DBDDException("Null string passed to BigReal constructor");
    }
    prec_t p = resolve_prec(prec);
    mpfr_init2(value_, p);
    if (mpfr_set_str(value_, str, base, MPFR_RNDN) != 0) {
        mpfr_clear(value_);
        throw DBDDException("Failed to parse BigReal string");
    }
    initialized_ = true;
}

BigReal::BigReal(const BigReal& other) {
    if (other.initialized_) {
        mpfr_init2(value_, other.precision());
        mpfr_set(value_, other.value_, MPFR_RNDN);
        initialized_ = true;
    }
}

BigReal& BigReal::operator=(const BigReal& other) {
    if (this != &other) {
        if (!initialized_) {
            mpfr_init2(value_, other.precision());
            initialized_ = true;
        } else if (precision() != other.precision()) {
            mpfr_set_prec(value_, other.precision());
        }
        mpfr_set(value_, other.value_, MPFR_RNDN);
    }
    return *this;
}

BigReal::BigReal(BigReal&& other) noexcept {
    if (other.initialized_) {
        std::memcpy(value_, other.value_, sizeof(mpfr_t));
        other.initialized_ = false;
        initialized_ = true;
    }
}

BigReal& BigReal::operator=(BigReal&& other) noexcept {
    if (this != &other) {
        if (initialized_) {
            mpfr_clear(value_);
        }
        if (other.initialized_) {
            std::memcpy(value_, other.value_, sizeof(mpfr_t));
            other.initialized_ = false;
            initialized_ = true;
        } else {
            initialized_ = false;
        }
    }
    return *this;
}

BigReal::~BigReal() {
    if (initialized_) {
        mpfr_clear(value_);
        initialized_ = false;
    }
}

prec_t BigReal::precision() const noexcept {
    return mpfr_get_prec(value_);
}

void BigReal::set_precision(prec_t new_prec, mpfr_rnd_t rnd) {
    mpfr_prec_round(value_, new_prec, rnd);
}

double BigReal::to_double() const {
    return mpfr_get_d(value_, MPFR_RNDN);
}

long double BigReal::to_long_double() const {
    return mpfr_get_ld(value_, MPFR_RNDN);
}

std::string BigReal::to_string(size_t digits, int base) const {
    char* buf = nullptr;
    int ret = 0;
    if (digits == 0) {
        ret = mpfr_asprintf(&buf, "%.*Rg", static_cast<int>(precision() / 3), value_);
    } else {
        ret = mpfr_asprintf(&buf, "%.*Rg", static_cast<int>(digits), value_);
    }
    if (ret < 0 || buf == nullptr) {
        // Fallback using mpfr_get_str
        mpfr_exp_t exp = 0;
        char* str_buf = mpfr_get_str(nullptr, &exp, base, digits, value_, MPFR_RNDN);
        if (str_buf == nullptr) {
            return "0";
        }
        std::string res(str_buf);
        mpfr_free_str(str_buf);
        return res;
    }
    std::string res(buf);
    mpfr_free_str(buf);
    return res;
}

BigInt BigReal::to_bigint(mpfr_rnd_t rnd) const {
    BigInt res;
    mpfr_get_z(res.raw().get_mpz_t(), value_, rnd);
    return res;
}

BigRational BigReal::to_bigrational() const {
    BigInt z;
    mpfr_exp_t exp = mpfr_get_z_2exp(z.raw().get_mpz_t(), value_);
    if (exp >= 0) {
        BigInt factor = BigInt(1) << static_cast<unsigned long>(exp);
        return BigRational(z * factor);
    } else {
        BigInt den = BigInt(1) << static_cast<unsigned long>(-exp);
        return BigRational(z, den);
    }
}

BigReal BigReal::operator+() const {
    return *this;
}

BigReal BigReal::operator-() const {
    BigReal res = with_precision(precision());
    mpfr_neg(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal& BigReal::operator+=(const BigReal& rhs) {
    prec_t p = std::max(precision(), rhs.precision());
    if (precision() < p) {
        set_precision(p);
    }
    mpfr_add(value_, value_, rhs.value_, MPFR_RNDN);
    return *this;
}

BigReal& BigReal::operator-=(const BigReal& rhs) {
    prec_t p = std::max(precision(), rhs.precision());
    if (precision() < p) {
        set_precision(p);
    }
    mpfr_sub(value_, value_, rhs.value_, MPFR_RNDN);
    return *this;
}

BigReal& BigReal::operator*=(const BigReal& rhs) {
    prec_t p = std::max(precision(), rhs.precision());
    if (precision() < p) {
        set_precision(p);
    }
    mpfr_mul(value_, value_, rhs.value_, MPFR_RNDN);
    return *this;
}

BigReal& BigReal::operator/=(const BigReal& rhs) {
    prec_t p = std::max(precision(), rhs.precision());
    if (precision() < p) {
        set_precision(p);
    }
    mpfr_div(value_, value_, rhs.value_, MPFR_RNDN);
    return *this;
}

BigReal operator+(const BigReal& lhs, const BigReal& rhs) {
    prec_t p = std::max(lhs.precision(), rhs.precision());
    BigReal res = BigReal::with_precision(p);
    mpfr_add(res.value_, lhs.value_, rhs.value_, MPFR_RNDN);
    return res;
}

BigReal operator-(const BigReal& lhs, const BigReal& rhs) {
    prec_t p = std::max(lhs.precision(), rhs.precision());
    BigReal res = BigReal::with_precision(p);
    mpfr_sub(res.value_, lhs.value_, rhs.value_, MPFR_RNDN);
    return res;
}

BigReal operator*(const BigReal& lhs, const BigReal& rhs) {
    prec_t p = std::max(lhs.precision(), rhs.precision());
    BigReal res = BigReal::with_precision(p);
    mpfr_mul(res.value_, lhs.value_, rhs.value_, MPFR_RNDN);
    return res;
}

BigReal operator/(const BigReal& lhs, const BigReal& rhs) {
    prec_t p = std::max(lhs.precision(), rhs.precision());
    BigReal res = BigReal::with_precision(p);
    mpfr_div(res.value_, lhs.value_, rhs.value_, MPFR_RNDN);
    return res;
}

std::partial_ordering BigReal::operator<=>(const BigReal& rhs) const noexcept {
    if (is_nan() || rhs.is_nan()) {
        return std::partial_ordering::unordered;
    }
    int cmp = mpfr_cmp(value_, rhs.value_);
    if (cmp < 0) return std::partial_ordering::less;
    if (cmp > 0) return std::partial_ordering::greater;
    return std::partial_ordering::equivalent;
}

bool BigReal::operator==(const BigReal& rhs) const noexcept {
    if (is_nan() || rhs.is_nan()) {
        return false;
    }
    return mpfr_equal_p(value_, rhs.value_) != 0;
}

bool BigReal::operator!=(const BigReal& rhs) const noexcept {
    return !(*this == rhs);
}

bool BigReal::operator<(const BigReal& rhs) const noexcept {
    return mpfr_less_p(value_, rhs.value_) != 0;
}

bool BigReal::operator<=(const BigReal& rhs) const noexcept {
    return mpfr_lessequal_p(value_, rhs.value_) != 0;
}

bool BigReal::operator>(const BigReal& rhs) const noexcept {
    return mpfr_greater_p(value_, rhs.value_) != 0;
}

bool BigReal::operator>=(const BigReal& rhs) const noexcept {
    return mpfr_greaterequal_p(value_, rhs.value_) != 0;
}

bool BigReal::is_nan() const noexcept {
    return mpfr_nan_p(value_) != 0;
}

bool BigReal::is_inf() const noexcept {
    return mpfr_inf_p(value_) != 0;
}

bool BigReal::is_zero() const noexcept {
    return mpfr_zero_p(value_) != 0;
}

bool BigReal::is_finite() const noexcept {
    return mpfr_number_p(value_) != 0;
}

int BigReal::sign() const noexcept {
    return mpfr_sgn(value_);
}

BigReal BigReal::abs() const {
    BigReal res = with_precision(precision());
    mpfr_abs(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::sqrt() const {
    if (*this < 0) {
        throw NumericalPrecisionException("Square root of negative BigReal");
    }
    BigReal res = with_precision(precision());
    mpfr_sqrt(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::exp() const {
    BigReal res = with_precision(precision());
    mpfr_exp(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::log() const {
    if (*this <= 0) {
        throw NumericalPrecisionException("Logarithm of non-positive BigReal");
    }
    BigReal res = with_precision(precision());
    mpfr_log(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::log2() const {
    if (*this <= 0) {
        throw NumericalPrecisionException("Log2 of non-positive BigReal");
    }
    BigReal res = with_precision(precision());
    mpfr_log2(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::log10() const {
    if (*this <= 0) {
        throw NumericalPrecisionException("Log10 of non-positive BigReal");
    }
    BigReal res = with_precision(precision());
    mpfr_log10(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::pow(const BigReal& exp_val) const {
    prec_t p = std::max(precision(), exp_val.precision());
    BigReal res = with_precision(p);
    mpfr_pow(res.value_, value_, exp_val.value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::pow(long exp_val) const {
    BigReal res = with_precision(precision());
    mpfr_pow_si(res.value_, value_, exp_val, MPFR_RNDN);
    return res;
}

BigReal BigReal::floor() const {
    BigReal res = with_precision(precision());
    mpfr_floor(res.value_, value_);
    return res;
}

BigReal BigReal::ceil() const {
    BigReal res = with_precision(precision());
    mpfr_ceil(res.value_, value_);
    return res;
}

BigReal BigReal::round() const {
    BigReal res = with_precision(precision());
    mpfr_round(res.value_, value_);
    return res;
}

BigReal BigReal::sin() const {
    BigReal res = with_precision(precision());
    mpfr_sin(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::cos() const {
    BigReal res = with_precision(precision());
    mpfr_cos(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::tan() const {
    BigReal res = with_precision(precision());
    mpfr_tan(res.value_, value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::pi(prec_t prec) {
    prec_t p = resolve_prec(prec);
    BigReal res = with_precision(p);
    mpfr_const_pi(res.value_, MPFR_RNDN);
    return res;
}

BigReal BigReal::e(prec_t prec) {
    prec_t p = resolve_prec(prec);
    BigReal one(1.0, p);
    return one.exp();
}

BigReal BigReal::ln2(prec_t prec) {
    prec_t p = resolve_prec(prec);
    BigReal res = with_precision(p);
    mpfr_const_log2(res.value_, MPFR_RNDN);
    return res;
}

std::ostream& operator<<(std::ostream& os, const BigReal& val) {
    return os << val.to_string();
}

std::istream& operator>>(std::istream& is, BigReal& val) {
    std::string str;
    if (is >> str) {
        val = BigReal(str);
    }
    return is;
}

} // namespace dbdd
