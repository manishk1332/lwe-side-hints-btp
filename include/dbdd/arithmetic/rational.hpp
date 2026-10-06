#pragma once

#include "dbdd/arithmetic/integer.hpp"
#include <gmpxx.h>
#include <string>
#include <iosfwd>
#include <compare>

namespace dbdd {

class BigReal;

/**
 * @brief Exact rational number wrapper around GMP mpq_class.
 * 
 * Guarantees canonical irreducible form with positive denominator.
 */
class BigRational {
public:
    // Constructors
    BigRational();
    BigRational(const BigInt& num);
    BigRational(int num);
    BigRational(long num);
    BigRational(long long num);
    BigRational(const BigInt& num, const BigInt& den);
    BigRational(int64_t num, int64_t den);
    explicit BigRational(const std::string& str, int base = 10);
    explicit BigRational(const char* str, int base = 10);
    explicit BigRational(mpq_class val);

    // Copy and Move
    BigRational(const BigRational&) = default;
    BigRational& operator=(const BigRational&) = default;
    BigRational(BigRational&&) noexcept = default;
    BigRational& operator=(BigRational&&) noexcept = default;

    ~BigRational() = default;

    // Component access
    [[nodiscard]] BigInt numerator() const;
    [[nodiscard]] BigInt denominator() const;

    // Explicit conversions
    [[nodiscard]] explicit operator double() const { return to_double(); }
    [[nodiscard]] explicit operator long double() const { return to_long_double(); }

    [[nodiscard]] double to_double() const;
    [[nodiscard]] long double to_long_double() const;
    [[nodiscard]] std::string to_string(int base = 10) const;
    [[nodiscard]] BigReal to_mpfr(prec_t prec = 0) const;

    // Raw underlying access for library-internal routines
    [[nodiscard]] const mpq_class& raw() const noexcept { return value_; }
    [[nodiscard]] mpq_class& raw() noexcept { return value_; }

    // Unary operators
    BigRational operator+() const;
    BigRational operator-() const;

    // Arithmetic operators
    BigRational& operator+=(const BigRational& rhs);
    BigRational& operator-=(const BigRational& rhs);
    BigRational& operator*=(const BigRational& rhs);
    BigRational& operator/=(const BigRational& rhs);

    friend BigRational operator+(BigRational lhs, const BigRational& rhs);
    friend BigRational operator-(BigRational lhs, const BigRational& rhs);
    friend BigRational operator*(BigRational lhs, const BigRational& rhs);
    friend BigRational operator/(BigRational lhs, const BigRational& rhs);

    // Comparisons
    std::strong_ordering operator<=>(const BigRational& rhs) const noexcept;
    bool operator==(const BigRational& rhs) const noexcept;

    // Mathematical functions
    [[nodiscard]] BigRational abs() const;
    [[nodiscard]] int sign() const noexcept;
    [[nodiscard]] BigInt floor() const;
    [[nodiscard]] BigInt ceil() const;
    [[nodiscard]] BigInt round() const;

    // Stream I/O
    friend std::ostream& operator<<(std::ostream& os, const BigRational& val);
    friend std::istream& operator>>(std::istream& is, BigRational& val);

private:
    void canonicalize();

    mpq_class value_;
};

inline BigRational abs(const BigRational& x) { return x.abs(); }

} // namespace dbdd
