#pragma once

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include <gmpxx.h>
#include <string>
#include <iosfwd>
#include <compare>

namespace dbdd {

class BigReal;

/**
 * @brief Arbitrary-precision integer wrapper around GMP mpz_class.
 * 
 * Enforces explicit conversions to floating-point types to prevent silent precision loss.
 */
class BigInt {
public:
    // Constructors
    BigInt();
    BigInt(int val);
    BigInt(long val);
    BigInt(long long val);
    BigInt(unsigned int val);
    BigInt(unsigned long val);
    BigInt(unsigned long long val);
    explicit BigInt(const std::string& str, int base = 10);
    explicit BigInt(const char* str, int base = 10);
    explicit BigInt(mpz_class val);

    // Copy and Move
    BigInt(const BigInt&) = default;
    BigInt& operator=(const BigInt&) = default;
    BigInt(BigInt&&) noexcept = default;
    BigInt& operator=(BigInt&&) noexcept = default;

    ~BigInt() = default;

    // Explicit conversions
    [[nodiscard]] explicit operator double() const { return to_double(); }
    [[nodiscard]] explicit operator long double() const { return to_long_double(); }
    [[nodiscard]] explicit operator int64_t() const { return to_int64(); }

    [[nodiscard]] double to_double() const;
    [[nodiscard]] long double to_long_double() const;
    [[nodiscard]] int64_t to_int64() const;
    [[nodiscard]] std::string to_string(int base = 10) const;
    [[nodiscard]] BigReal to_mpfr(prec_t prec = 0) const;

    // Raw underlying access for library-internal routines
    [[nodiscard]] const mpz_class& raw() const noexcept { return value_; }
    [[nodiscard]] mpz_class& raw() noexcept { return value_; }

    // Unary operators
    BigInt operator+() const;
    BigInt operator-() const;
    BigInt& operator++();
    BigInt operator++(int);
    BigInt& operator--();
    BigInt operator--(int);

    // Arithmetic operators
    BigInt& operator+=(const BigInt& rhs);
    BigInt& operator-=(const BigInt& rhs);
    BigInt& operator*=(const BigInt& rhs);
    BigInt& operator/=(const BigInt& rhs);
    BigInt& operator%=(const BigInt& rhs);

    friend BigInt operator+(BigInt lhs, const BigInt& rhs);
    friend BigInt operator-(BigInt lhs, const BigInt& rhs);
    friend BigInt operator*(BigInt lhs, const BigInt& rhs);
    friend BigInt operator/(BigInt lhs, const BigInt& rhs);
    friend BigInt operator%(BigInt lhs, const BigInt& rhs);

    // Bitwise operators
    BigInt& operator&=(const BigInt& rhs);
    BigInt& operator|=(const BigInt& rhs);
    BigInt& operator^=(const BigInt& rhs);
    BigInt& operator<<=(unsigned long shift);
    BigInt& operator>>=(unsigned long shift);

    BigInt operator~() const;
    friend BigInt operator&(BigInt lhs, const BigInt& rhs);
    friend BigInt operator|(BigInt lhs, const BigInt& rhs);
    friend BigInt operator^(BigInt lhs, const BigInt& rhs);
    friend BigInt operator<<(BigInt lhs, unsigned long shift);
    friend BigInt operator>>(BigInt lhs, unsigned long shift);

    // Comparisons
    std::strong_ordering operator<=>(const BigInt& rhs) const noexcept;
    bool operator==(const BigInt& rhs) const noexcept;

    // Mathematical functions
    [[nodiscard]] BigInt abs() const;
    [[nodiscard]] int sign() const noexcept;
    [[nodiscard]] BigInt sqrt() const;

    static BigInt div_floor(const BigInt& a, const BigInt& b);
    static BigInt mod_floor(const BigInt& a, const BigInt& b);
    static BigInt gcd(const BigInt& a, const BigInt& b);
    static BigInt lcm(const BigInt& a, const BigInt& b);
    static BigInt xgcd(const BigInt& a, const BigInt& b, BigInt& s, BigInt& t);
    static BigInt pow(const BigInt& base, unsigned long exp);

    // Stream I/O
    friend std::ostream& operator<<(std::ostream& os, const BigInt& val);
    friend std::istream& operator>>(std::istream& is, BigInt& val);

private:
    mpz_class value_;
};

inline BigInt abs(const BigInt& x) { return x.abs(); }
inline BigInt sqrt(const BigInt& x) { return x.sqrt(); }

} // namespace dbdd
