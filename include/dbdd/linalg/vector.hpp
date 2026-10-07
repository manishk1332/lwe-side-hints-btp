#pragma once

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include <Eigen/Dense>
#include <vector>
#include <initializer_list>
#include <iostream>
#include <cmath>
#include <type_traits>

namespace dbdd {

/**
 * @brief Dense row vector conforming to the paper's row-vector conventions.
 * 
 * Supports exact (BigInt, BigRational) and arbitrary-precision floating-point (BigReal, double) arithmetic.
 */
template <typename T>
class Vector {
public:
    // Constructors
    Vector() = default;

    explicit Vector(size_t size, const T& default_val = T{})
        : data_(size, default_val) {}

    Vector(std::initializer_list<T> list)
        : data_(list) {}

    explicit Vector(std::vector<T> data)
        : data_(std::move(data)) {}

    template <typename InputIt>
    Vector(InputIt first, InputIt last)
        : data_(first, last) {}

    // Factory methods
    static Vector zero(size_t size) {
        return Vector(size, T{});
    }

    static Vector unit(size_t size, size_t index, const T& val = T{1}) {
        if (index >= size) {
            throw IndexOutOfBoundsException("Unit vector index " + std::to_string(index) +
                                            " out of bounds for size " + std::to_string(size));
        }
        Vector res(size, T{});
        res[index] = val;
        return res;
    }

    // Size and capacity
    [[nodiscard]] size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    // Element access
    T& operator[](size_t i) noexcept { return data_[i]; }
    const T& operator[](size_t i) const noexcept { return data_[i]; }

    T& at(size_t i) {
        if (i >= data_.size()) {
            throw IndexOutOfBoundsException("Vector index " + std::to_string(i) +
                                            " out of range (size: " + std::to_string(data_.size()) + ")");
        }
        return data_.at(i);
    }

    const T& at(size_t i) const {
        if (i >= data_.size()) {
            throw IndexOutOfBoundsException("Vector index " + std::to_string(i) +
                                            " out of range (size: " + std::to_string(data_.size()) + ")");
        }
        return data_.at(i);
    }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

    [[nodiscard]] const std::vector<T>& raw_data() const noexcept { return data_; }
    [[nodiscard]] std::vector<T>& raw_data() noexcept { return data_; }

    // Unary operators
    Vector operator+() const { return *this; }

    Vector operator-() const {
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) {
            res[i] = -data_[i];
        }
        return res;
    }

    // Arithmetic operators
    Vector& operator+=(const Vector& rhs) {
        if (size() != rhs.size()) {
            throw DimensionMismatchException("Vector addition size mismatch: " +
                                            std::to_string(size()) + " vs " + std::to_string(rhs.size()));
        }
        for (size_t i = 0; i < size(); ++i) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    Vector& operator-=(const Vector& rhs) {
        if (size() != rhs.size()) {
            throw DimensionMismatchException("Vector subtraction size mismatch: " +
                                            std::to_string(size()) + " vs " + std::to_string(rhs.size()));
        }
        for (size_t i = 0; i < size(); ++i) {
            data_[i] -= rhs.data_[i];
        }
        return *this;
    }

    Vector& operator*=(const T& scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    Vector& operator/=(const T& scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }

    friend Vector operator+(Vector lhs, const Vector& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Vector operator-(Vector lhs, const Vector& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Vector operator*(Vector lhs, const T& scalar) {
        lhs *= scalar;
        return lhs;
    }

    friend Vector operator*(const T& scalar, Vector rhs) {
        rhs *= scalar;
        return rhs;
    }

    friend Vector operator/(Vector lhs, const T& scalar) {
        lhs /= scalar;
        return lhs;
    }

    // Dot product & Norms
    // Paper Reference: Section 2 / docs/mathematical_conventions.md
    // Mathematical Purpose: Standard inner product <u, v> = sum(u_i * v_i)
    [[nodiscard]] T dot(const Vector& other) const {
        if (size() != other.size()) {
            throw DimensionMismatchException("Vector dot product size mismatch: " +
                                            std::to_string(size()) + " vs " + std::to_string(other.size()));
        }
        T sum{};
        for (size_t i = 0; i < size(); ++i) {
            sum += data_[i] * other.data_[i];
        }
        return sum;
    }

    // Paper Reference: Section 2
    // Mathematical Purpose: Squared Euclidean norm ||v||^2 = <v, v>
    [[nodiscard]] T squared_norm() const {
        return dot(*this);
    }

    [[nodiscard]] BigReal norm() const {
        T sq = squared_norm();
        if constexpr (std::is_same_v<T, BigReal>) {
            return sq.sqrt();
        } else if constexpr (std::is_same_v<T, BigInt> || std::is_same_v<T, BigRational>) {
            return sq.to_mpfr().sqrt();
        } else {
            return BigReal(std::sqrt(static_cast<double>(sq)));
        }
    }

    // Structural operations
    [[nodiscard]] Vector slice(size_t start, size_t length) const {
        if (start + length > size()) {
            throw IndexOutOfBoundsException("Vector slice range out of bounds");
        }
        Vector res(length);
        for (size_t i = 0; i < length; ++i) {
            res[i] = data_[start + i];
        }
        return res;
    }

    [[nodiscard]] Vector concat(const Vector& other) const {
        Vector res(size() + other.size());
        for (size_t i = 0; i < size(); ++i) {
            res[i] = data_[i];
        }
        for (size_t i = 0; i < other.size(); ++i) {
            res[size() + i] = other.data_[i];
        }
        return res;
    }

    // Conversions
    template <typename U>
    [[nodiscard]] Vector<U> cast() const {
        Vector<U> res(size());
        for (size_t i = 0; i < size(); ++i) {
            if constexpr (std::is_same_v<T, U>) {
                res[i] = data_[i];
            } else if constexpr (std::is_same_v<U, BigReal>) {
                if constexpr (std::is_same_v<T, BigInt> || std::is_same_v<T, BigRational>) {
                    res[i] = data_[i].to_mpfr();
                } else {
                    res[i] = BigReal(data_[i]);
                }
            } else if constexpr (std::is_same_v<U, BigRational>) {
                if constexpr (std::is_same_v<T, BigInt>) {
                    res[i] = BigRational(data_[i]);
                } else if constexpr (std::is_same_v<T, BigReal>) {
                    res[i] = data_[i].to_bigrational();
                } else if constexpr (std::is_same_v<T, double>) {
                    res[i] = BigReal(data_[i]).to_bigrational();
                } else {
                    res[i] = BigRational(data_[i]);
                }
            } else if constexpr (std::is_same_v<U, BigInt>) {
                if constexpr (std::is_same_v<T, BigRational>) {
                    res[i] = data_[i].round();
                } else if constexpr (std::is_same_v<T, BigReal>) {
                    res[i] = data_[i].to_bigint();
                } else {
                    res[i] = BigInt(data_[i]);
                }
            } else if constexpr (std::is_same_v<U, double>) {
                if constexpr (requires { data_[i].to_double(); }) {
                    res[i] = data_[i].to_double();
                } else {
                    res[i] = static_cast<double>(data_[i]);
                }
            } else {
                res[i] = static_cast<U>(data_[i]);
            }
        }
        return res;
    }

    [[nodiscard]] Eigen::Matrix<double, 1, Eigen::Dynamic> to_eigen() const {
        Eigen::Matrix<double, 1, Eigen::Dynamic> ev(1, static_cast<Eigen::Index>(size()));
        for (size_t i = 0; i < size(); ++i) {
            if constexpr (requires { data_[i].to_double(); }) {
                ev(0, static_cast<Eigen::Index>(i)) = data_[i].to_double();
            } else {
                ev(0, static_cast<Eigen::Index>(i)) = static_cast<double>(data_[i]);
            }
        }
        return ev;
    }

    template <typename Derived>
    static Vector<T> from_eigen(const Eigen::MatrixBase<Derived>& ev) {
        size_t sz = static_cast<size_t>(ev.size());
        Vector<T> res(sz);
        for (size_t i = 0; i < sz; ++i) {
            double val = ev(static_cast<Eigen::Index>(i));
            if constexpr (std::is_same_v<T, BigReal>) {
                res[i] = BigReal(val);
            } else if constexpr (std::is_same_v<T, BigRational>) {
                res[i] = BigReal(val).to_bigrational();
            } else if constexpr (std::is_same_v<T, BigInt>) {
                res[i] = BigInt(static_cast<long long>(std::round(val)));
            } else {
                res[i] = static_cast<T>(val);
            }
        }
        return res;
    }

    // Equality
    bool operator==(const Vector& rhs) const noexcept {
        if (size() != rhs.size()) return false;
        for (size_t i = 0; i < size(); ++i) {
            if (data_[i] != rhs.data_[i]) return false;
        }
        return true;
    }

    bool operator!=(const Vector& rhs) const noexcept {
        return !(*this == rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "(";
        for (size_t i = 0; i < v.size(); ++i) {
            os << v[i];
            if (i + 1 < v.size()) os << ", ";
        }
        os << ")";
        return os;
    }

private:
    std::vector<T> data_;
};

template <typename T>
T dot(const Vector<T>& u, const Vector<T>& v) {
    return u.dot(v);
}

} // namespace dbdd
