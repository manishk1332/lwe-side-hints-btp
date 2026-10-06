#pragma once

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/precision.hpp"
#include <mpfr.h>
#include <string>
#include <iosfwd>
#include <compare>

namespace dbdd {

class BigInt;
class BigRational;

/**
 * @brief Arbitrary-precision floating-point wrapper around MPFR.
 * 
 * Provides configurable precision, transcendental mathematical functions,
 * and exact round-trip conversion utilities.
 */
class BigReal {
public:
    // Constructors
    BigReal();
    static BigReal with_precision(prec_t prec);
    static BigReal zero(prec_t prec = 0);
    BigReal(double val, prec_t prec = 0);
    BigReal(long double val, prec_t prec = 0);
    BigReal(int val, prec_t prec = 0);
    BigReal(int64_t val, prec_t prec = 0);
    explicit BigReal(const BigInt& val, prec_t prec = 0);
    explicit BigReal(const BigRational& val, prec_t prec = 0);
    explicit BigReal(const std::string& str, prec_t prec = 0, int base = 10);
    explicit BigReal(const char* str, prec_t prec = 0, int base = 10);

    // Copy and Move
    BigReal(const BigReal& other);
    BigReal& operator=(const BigReal& other);
    BigReal(BigReal&& other) noexcept;
    BigReal& operator=(BigReal&& other) noexcept;

    ~BigReal();

    // Precision management
    [[nodiscard]] prec_t precision() const noexcept;
    void set_precision(prec_t new_prec, mpfr_rnd_t rnd = MPFR_RNDN);

    // Explicit conversions
    [[nodiscard]] explicit operator double() const { return to_double(); }
    [[nodiscard]] explicit operator long double() const { return to_long_double(); }

    [[nodiscard]] double to_double() const;
    [[nodiscard]] long double to_long_double() const;
    [[nodiscard]] std::string to_string(size_t digits = 0, int base = 10) const;
    [[nodiscard]] BigInt to_bigint(mpfr_rnd_t rnd = MPFR_RNDZ) const;
    [[nodiscard]] BigRational to_bigrational() const;

    // Raw underlying access for MPFR routines
    [[nodiscard]] mpfr_srcptr raw() const noexcept { return value_; }
    [[nodiscard]] mpfr_ptr raw() noexcept { return value_; }

    // Unary operators
    BigReal operator+() const;
    BigReal operator-() const;

    // Arithmetic operators
    BigReal& operator+=(const BigReal& rhs);
    BigReal& operator-=(const BigReal& rhs);
    BigReal& operator*=(const BigReal& rhs);
    BigReal& operator/=(const BigReal& rhs);

    friend BigReal operator+(const BigReal& lhs, const BigReal& rhs);
    friend BigReal operator-(const BigReal& lhs, const BigReal& rhs);
    friend BigReal operator*(const BigReal& lhs, const BigReal& rhs);
    friend BigReal operator/(const BigReal& lhs, const BigReal& rhs);

    // Comparisons
    std::partial_ordering operator<=>(const BigReal& rhs) const noexcept;
    bool operator==(const BigReal& rhs) const noexcept;
    bool operator!=(const BigReal& rhs) const noexcept;
    bool operator<(const BigReal& rhs) const noexcept;
    bool operator<=(const BigReal& rhs) const noexcept;
    bool operator>(const BigReal& rhs) const noexcept;
    bool operator>=(const BigReal& rhs) const noexcept;

    // Status / Predicates
    [[nodiscard]] bool is_nan() const noexcept;
    [[nodiscard]] bool is_inf() const noexcept;
    [[nodiscard]] bool is_zero() const noexcept;
    [[nodiscard]] bool is_finite() const noexcept;
    [[nodiscard]] int sign() const noexcept;

    // Mathematical functions
    [[nodiscard]] BigReal abs() const;
    [[nodiscard]] BigReal sqrt() const;
    [[nodiscard]] BigReal exp() const;
    [[nodiscard]] BigReal log() const;
    [[nodiscard]] BigReal log2() const;
    [[nodiscard]] BigReal log10() const;
    [[nodiscard]] BigReal pow(const BigReal& exp) const;
    [[nodiscard]] BigReal pow(long exp) const;
    [[nodiscard]] BigReal floor() const;
    [[nodiscard]] BigReal ceil() const;
    [[nodiscard]] BigReal round() const;
    [[nodiscard]] BigReal sin() const;
    [[nodiscard]] BigReal cos() const;
    [[nodiscard]] BigReal tan() const;

    // Mathematical constants
    static BigReal pi(prec_t prec = 0);
    static BigReal e(prec_t prec = 0);
    static BigReal ln2(prec_t prec = 0);

    // Stream I/O
    friend std::ostream& operator<<(std::ostream& os, const BigReal& val);
    friend std::istream& operator>>(std::istream& is, BigReal& val);

private:
    mpfr_t value_;
    bool initialized_{false};
};

// Free functions
inline BigReal abs(const BigReal& x) { return x.abs(); }
inline BigReal sqrt(const BigReal& x) { return x.sqrt(); }
inline BigReal exp(const BigReal& x) { return x.exp(); }
inline BigReal log(const BigReal& x) { return x.log(); }
inline BigReal log2(const BigReal& x) { return x.log2(); }
inline BigReal log10(const BigReal& x) { return x.log10(); }
inline BigReal pow(const BigReal& base, const BigReal& exp) { return base.pow(exp); }
inline BigReal pow(const BigReal& base, long exp) { return base.pow(exp); }
inline BigReal floor(const BigReal& x) { return x.floor(); }
inline BigReal ceil(const BigReal& x) { return x.ceil(); }
inline BigReal round(const BigReal& x) { return x.round(); }
inline BigReal sin(const BigReal& x) { return x.sin(); }
inline BigReal cos(const BigReal& x) { return x.cos(); }
inline BigReal tan(const BigReal& x) { return x.tan(); }

} // namespace dbdd
