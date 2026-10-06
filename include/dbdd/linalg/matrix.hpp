#pragma once

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/linalg/vector.hpp"
#include <Eigen/Dense>
#include <vector>
#include <initializer_list>
#include <iostream>
#include <type_traits>

namespace dbdd {

/**
 * @brief Dense matrix conforming to the paper's row-basis conventions.
 * 
 * Row $i$ represents basis vector $b_i$.
 * Multiplication by vector on the left satisfies $v M$ (row-vector convention).
 */
template <typename T>
class Matrix {
public:
    // Constructors
    Matrix() : rows_(0), cols_(0) {}

    Matrix(size_t rows, size_t cols, const T& default_val = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, default_val) {}

    Matrix(size_t rows, size_t cols, std::initializer_list<T> list)
        : rows_(rows), cols_(cols), data_(list) {
        if (data_.size() != rows * cols) {
            throw DimensionMismatchException("Initializer list size does not match matrix dimensions");
        }
    }

    Matrix(size_t rows, size_t cols, std::vector<T> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows * cols) {
            throw DimensionMismatchException("Vector data size does not match matrix dimensions");
        }
    }

    // Factory methods
    static Matrix zero(size_t rows, size_t cols) {
        return Matrix(rows, cols, T{});
    }

    static Matrix identity(size_t n) {
        Matrix res(n, n, T{});
        for (size_t i = 0; i < n; ++i) {
            res(i, i) = T{1};
        }
        return res;
    }

    static Matrix diagonal(const Vector<T>& diag) {
        size_t n = diag.size();
        Matrix res(n, n, T{});
        for (size_t i = 0; i < n; ++i) {
            res(i, i) = diag[i];
        }
        return res;
    }

    static Matrix from_rows(const std::vector<Vector<T>>& rows) {
        if (rows.empty()) {
            return Matrix(0, 0);
        }
        size_t num_r = rows.size();
        size_t num_c = rows[0].size();
        Matrix res(num_r, num_c);
        for (size_t i = 0; i < num_r; ++i) {
            if (rows[i].size() != num_c) {
                throw DimensionMismatchException("Row size mismatch when constructing Matrix from rows");
            }
            for (size_t j = 0; j < num_c; ++j) {
                res(i, j) = rows[i][j];
            }
        }
        return res;
    }

    // Dimensions
    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }
    [[nodiscard]] bool is_square() const noexcept { return rows_ == cols_; }

    // Element access
    T& operator()(size_t r, size_t c) noexcept {
        return data_[r * cols_ + c];
    }

    const T& operator()(size_t r, size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    T& at(size_t r, size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw IndexOutOfBoundsException("Matrix index (" + std::to_string(r) + ", " +
                                            std::to_string(c) + ") out of bounds (" +
                                            std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return data_.at(r * cols_ + c);
    }

    const T& at(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw IndexOutOfBoundsException("Matrix index (" + std::to_string(r) + ", " +
                                            std::to_string(c) + ") out of bounds (" +
                                            std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
        return data_.at(r * cols_ + c);
    }

    [[nodiscard]] const std::vector<T>& raw_data() const noexcept { return data_; }
    [[nodiscard]] std::vector<T>& raw_data() noexcept { return data_; }

    // Row & Column access (Row-basis convention)
    [[nodiscard]] Vector<T> row(size_t r) const {
        if (r >= rows_) {
            throw IndexOutOfBoundsException("Row index " + std::to_string(r) + " out of bounds");
        }
        Vector<T> res(cols_);
        for (size_t c = 0; c < cols_; ++c) {
            res[c] = (*this)(r, c);
        }
        return res;
    }

    void set_row(size_t r, const Vector<T>& v) {
        if (r >= rows_) {
            throw IndexOutOfBoundsException("Row index " + std::to_string(r) + " out of bounds");
        }
        if (v.size() != cols_) {
            throw DimensionMismatchException("Row vector size mismatch");
        }
        for (size_t c = 0; c < cols_; ++c) {
            (*this)(r, c) = v[c];
        }
    }

    [[nodiscard]] Vector<T> col(size_t c) const {
        if (c >= cols_) {
            throw IndexOutOfBoundsException("Column index " + std::to_string(c) + " out of bounds");
        }
        Vector<T> res(rows_);
        for (size_t r = 0; r < rows_; ++r) {
            res[r] = (*this)(r, c);
        }
        return res;
    }

    void set_col(size_t c, const Vector<T>& v) {
        if (c >= cols_) {
            throw IndexOutOfBoundsException("Column index " + std::to_string(c) + " out of bounds");
        }
        if (v.size() != rows_) {
            throw DimensionMismatchException("Column vector size mismatch");
        }
        for (size_t r = 0; r < rows_; ++r) {
            (*this)(r, c) = v[r];
        }
    }

    void swap_rows(size_t r1, size_t r2) {
        if (r1 >= rows_ || r2 >= rows_) {
            throw IndexOutOfBoundsException("Row index out of bounds in swap_rows");
        }
        if (r1 == r2) return;
        for (size_t c = 0; c < cols_; ++c) {
            std::swap((*this)(r1, c), (*this)(r2, c));
        }
    }

    void row_add_multiple(size_t target_r, size_t source_r, const T& factor) {
        if (target_r >= rows_ || source_r >= rows_) {
            throw IndexOutOfBoundsException("Row index out of bounds in row_add_multiple");
        }
        for (size_t c = 0; c < cols_; ++c) {
            (*this)(target_r, c) += factor * (*this)(source_r, c);
        }
    }

    // Arithmetic operators
    Matrix operator+() const { return *this; }

    Matrix operator-() const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            res.data_[i] = -data_[i];
        }
        return res;
    }

    Matrix& operator+=(const Matrix& rhs) {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
            throw DimensionMismatchException("Matrix addition dimension mismatch");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& rhs) {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
            throw DimensionMismatchException("Matrix subtraction dimension mismatch");
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= rhs.data_[i];
        }
        return *this;
    }

    Matrix& operator*=(const T& scalar) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    Matrix& operator/=(const T& scalar) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }

    friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Matrix operator-(Matrix lhs, const Matrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Matrix operator*(Matrix lhs, const T& scalar) {
        lhs *= scalar;
        return lhs;
    }

    friend Matrix operator*(const T& scalar, Matrix rhs) {
        rhs *= scalar;
        return rhs;
    }

    friend Matrix operator/(Matrix lhs, const T& scalar) {
        lhs /= scalar;
        return lhs;
    }

    // Matrix Multiplication
    [[nodiscard]] Matrix operator*(const Matrix& rhs) const {
        if (cols_ != rhs.rows_) {
            throw DimensionMismatchException("Matrix multiplication dimension mismatch: (" +
                                            std::to_string(rows_) + "x" + std::to_string(cols_) + ") * (" +
                                            std::to_string(rhs.rows_) + "x" + std::to_string(rhs.cols_) + ")");
        }
        Matrix res(rows_, rhs.cols_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                const T& r_ik = (*this)(i, k);
                for (size_t j = 0; j < rhs.cols_; ++j) {
                    res(i, j) += r_ik * rhs(k, j);
                }
            }
        }
        return res;
    }

    // Vector Multiplications (Row convention: v * M)
    // Paper Reference: Section 1 / docs/mathematical_conventions.md
    // Mathematical Purpose: Row-vector product v * M
    friend Vector<T> operator*(const Vector<T>& v, const Matrix& M) {
        if (v.size() != M.rows_) {
            throw DimensionMismatchException("Vector-Matrix multiplication dimension mismatch: vector size " +
                                            std::to_string(v.size()) + " vs matrix rows " + std::to_string(M.rows_));
        }
        Vector<T> res(M.cols_, T{});
        for (size_t r = 0; r < M.rows_; ++r) {
            const T& vr = v[r];
            for (size_t c = 0; c < M.cols_; ++c) {
                res[c] += vr * M(r, c);
            }
        }
        return res;
    }

    // Column-vector product M * v_col
    [[nodiscard]] Vector<T> operator*(const Vector<T>& col_v) const {
        if (cols_ != col_v.size()) {
            throw DimensionMismatchException("Matrix-Vector multiplication dimension mismatch: matrix cols " +
                                            std::to_string(cols_) + " vs vector size " + std::to_string(col_v.size()));
        }
        Vector<T> res(rows_, T{});
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                res[r] += (*this)(r, c) * col_v[c];
            }
        }
        return res;
    }

    // Transpose
    // Paper Reference: Section 3 / docs/mathematical_conventions.md
    [[nodiscard]] Matrix transpose() const {
        Matrix res(cols_, rows_);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                res(c, r) = (*this)(r, c);
            }
        }
        return res;
    }

    // Gram matrix G = B * B^T
    [[nodiscard]] Matrix gram() const {
        return (*this) * transpose();
    }

    // Submatrix extraction
    [[nodiscard]] Matrix submatrix(size_t start_r, size_t start_c, size_t num_r, size_t num_c) const {
        if (start_r + num_r > rows_ || start_c + num_c > cols_) {
            throw IndexOutOfBoundsException("Submatrix range exceeds matrix dimensions");
        }
        Matrix res(num_r, num_c);
        for (size_t r = 0; r < num_r; ++r) {
            for (size_t c = 0; c < num_c; ++c) {
                res(r, c) = (*this)(start_r + r, start_c + c);
            }
        }
        return res;
    }

    // Block constructions
    static Matrix block_horizontal(const Matrix& A, const Matrix& B) {
        if (A.rows() != B.rows()) {
            throw DimensionMismatchException("block_horizontal rows mismatch");
        }
        Matrix res(A.rows(), A.cols() + B.cols());
        for (size_t r = 0; r < A.rows(); ++r) {
            for (size_t c = 0; c < A.cols(); ++c) {
                res(r, c) = A(r, c);
            }
            for (size_t c = 0; c < B.cols(); ++c) {
                res(r, A.cols() + c) = B(r, c);
            }
        }
        return res;
    }

    static Matrix block_vertical(const Matrix& A, const Matrix& B) {
        if (A.cols() != B.cols()) {
            throw DimensionMismatchException("block_vertical cols mismatch");
        }
        Matrix res(A.rows() + B.rows(), A.cols());
        for (size_t c = 0; c < A.cols(); ++c) {
            for (size_t r = 0; r < A.rows(); ++r) {
                res(r, c) = A(r, c);
            }
            for (size_t r = 0; r < B.rows(); ++r) {
                res(A.rows() + r, c) = B(r, c);
            }
        }
        return res;
    }

    static Matrix block_diagonal(const Matrix& A, const Matrix& B) {
        Matrix res(A.rows() + B.rows(), A.cols() + B.cols(), T{});
        for (size_t r = 0; r < A.rows(); ++r) {
            for (size_t c = 0; c < A.cols(); ++c) {
                res(r, c) = A(r, c);
            }
        }
        for (size_t r = 0; r < B.rows(); ++r) {
            for (size_t c = 0; c < B.cols(); ++c) {
                res(A.rows() + r, A.cols() + c) = B(r, c);
            }
        }
        return res;
    }

    // Determinant
    // Paper Reference: Section 1 / docs/algorithm_notes.md
    // Mathematical Purpose: Exact fraction-free Bareiss algorithm for integral/rational types,
    //                       Gaussian elimination with partial pivoting for floating types.
    [[nodiscard]] T determinant() const {
        if (!is_square()) {
            throw DimensionMismatchException("Determinant requires a square matrix");
        }
        size_t n = rows_;
        if (n == 0) return T{1};
        if (n == 1) return (*this)(0, 0);
        if (n == 2) {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }

        if constexpr (std::is_same_v<T, BigInt> || std::is_same_v<T, BigRational>) {
            return bareiss_determinant();
        } else {
            return float_determinant();
        }
    }

    // Rank via row echelon form
    [[nodiscard]] size_t rank() const {
        if (empty()) return 0;
        Matrix<BigRational> work;
        if constexpr (std::is_same_v<T, BigRational>) {
            work = *this;
        } else if constexpr (std::is_same_v<T, BigInt>) {
            work = this->template cast<BigRational>();
        } else {
            // Floating point rank with threshold
            return float_rank();
        }

        size_t r_count = 0;
        size_t lead = 0;
        size_t m_rows = work.rows();
        size_t m_cols = work.cols();

        for (size_t r = 0; r < m_rows && lead < m_cols; ++lead) {
            size_t pivot_r = r;
            while (pivot_r < m_rows && work(pivot_r, lead).sign() == 0) {
                ++pivot_r;
            }
            if (pivot_r == m_rows) {
                continue;
            }
            work.swap_rows(r, pivot_r);
            BigRational pivot = work(r, lead);
            for (size_t i = r + 1; i < m_rows; ++i) {
                if (work(i, lead).sign() != 0) {
                    BigRational factor = work(i, lead) / pivot;
                    for (size_t j = lead; j < m_cols; ++j) {
                        work(i, j) -= factor * work(r, j);
                    }
                }
            }
            ++r_count;
            ++r;
        }
        return r_count;
    }

    // Inversion
    [[nodiscard]] Matrix inverse() const {
        if (!is_square()) {
            throw DimensionMismatchException("Matrix inverse requires a square matrix");
        }
        if constexpr (std::is_same_v<T, BigRational>) {
            return rational_inverse();
        } else if constexpr (std::is_same_v<T, BigInt>) {
            throw DBDDException("Inversion of MatrixZZ is not generally in ZZ; cast to MatrixQQ first");
        } else {
            return float_inverse();
        }
    }

    // Linear Solver: Solves x * A = b (row-vector convention)
    // Paper Reference: Section 1 / Section 3
    [[nodiscard]] Vector<T> solve_row(const Vector<T>& b) const {
        if (!is_square()) {
            throw DimensionMismatchException("Matrix linear solver requires a square matrix");
        }
        if (b.size() != cols_) {
            throw DimensionMismatchException("Vector size mismatch in solve_row: " +
                                            std::to_string(b.size()) + " vs " + std::to_string(cols_));
        }
        return b * inverse();
    }

    // Linear Solver: Solves A * x = b (column-vector convention)
    [[nodiscard]] Vector<T> solve_col(const Vector<T>& b) const {
        if (!is_square()) {
            throw DimensionMismatchException("Matrix linear solver requires a square matrix");
        }
        if (b.size() != rows_) {
            throw DimensionMismatchException("Vector size mismatch in solve_col: " +
                                            std::to_string(b.size()) + " vs " + std::to_string(rows_));
        }
        return inverse() * b;
    }

    [[nodiscard]] Vector<T> solve(const Vector<T>& b) const {
        return solve_row(b);
    }

    // Type casting
    template <typename U>
    [[nodiscard]] Matrix<U> cast() const {
        Matrix<U> res(rows_, cols_);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                const T& val = (*this)(r, c);
                if constexpr (std::is_same_v<T, U>) {
                    res(r, c) = val;
                } else if constexpr (std::is_same_v<U, BigReal>) {
                    if constexpr (std::is_same_v<T, BigInt> || std::is_same_v<T, BigRational>) {
                        res(r, c) = val.to_mpfr();
                    } else {
                        res(r, c) = BigReal(val);
                    }
                } else if constexpr (std::is_same_v<U, BigRational>) {
                    if constexpr (std::is_same_v<T, BigInt>) {
                        res(r, c) = BigRational(val);
                    } else if constexpr (std::is_same_v<T, BigReal>) {
                        res(r, c) = val.to_bigrational();
                    } else {
                        res(r, c) = BigRational(val);
                    }
                } else if constexpr (std::is_same_v<U, BigInt>) {
                    if constexpr (std::is_same_v<T, BigRational>) {
                        res(r, c) = val.round();
                    } else if constexpr (std::is_same_v<T, BigReal>) {
                        res(r, c) = val.to_bigint();
                    } else {
                        res(r, c) = BigInt(val);
                    }
                } else if constexpr (std::is_same_v<U, double>) {
                    if constexpr (requires { val.to_double(); }) {
                        res(r, c) = val.to_double();
                    } else {
                        res(r, c) = static_cast<double>(val);
                    }
                } else {
                    res(r, c) = static_cast<U>(val);
                }
            }
        }
        return res;
    }

    // Eigen bridge
    [[nodiscard]] Eigen::MatrixXd to_eigen() const {
        Eigen::MatrixXd em(static_cast<Eigen::Index>(rows_), static_cast<Eigen::Index>(cols_));
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                const T& val = (*this)(r, c);
                if constexpr (requires { val.to_double(); }) {
                    em(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = val.to_double();
                } else {
                    em(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = static_cast<double>(val);
                }
            }
        }
        return em;
    }

    template <typename Derived>
    static Matrix<T> from_eigen(const Eigen::MatrixBase<Derived>& em) {
        size_t r_count = static_cast<size_t>(em.rows());
        size_t c_count = static_cast<size_t>(em.cols());
        Matrix<T> res(r_count, c_count);
        for (size_t r = 0; r < r_count; ++r) {
            for (size_t c = 0; c < c_count; ++c) {
                double val = em(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
                if constexpr (std::is_same_v<T, BigReal>) {
                    res(r, c) = BigReal(val);
                } else if constexpr (std::is_same_v<T, BigRational>) {
                    res(r, c) = BigRational(val);
                } else if constexpr (std::is_same_v<T, BigInt>) {
                    res(r, c) = BigInt(static_cast<long long>(std::round(val)));
                } else {
                    res(r, c) = static_cast<T>(val);
                }
            }
        }
        return res;
    }

    // Equality
    bool operator==(const Matrix& rhs) const noexcept {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) return false;
        return data_ == rhs.data_;
    }

    bool operator!=(const Matrix& rhs) const noexcept {
        return !(*this == rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "[\n";
        for (size_t r = 0; r < m.rows(); ++r) {
            os << "  " << m.row(r) << "\n";
        }
        os << "]";
        return os;
    }

private:
    T bareiss_determinant() const {
        size_t n = rows_;
        Matrix<T> M = *this;
        T prev_pivot = T{1};
        int sign = 1;

        for (size_t k = 0; k < n - 1; ++k) {
            size_t pivot_row = k;
            while (pivot_row < n && M(pivot_row, k) == T{}) {
                ++pivot_row;
            }
            if (pivot_row == n) {
                return T{};
            }
            if (pivot_row != k) {
                M.swap_rows(k, pivot_row);
                sign = -sign;
            }

            T cur_pivot = M(k, k);
            for (size_t i = k + 1; i < n; ++i) {
                for (size_t j = k + 1; j < n; ++j) {
                    T num = cur_pivot * M(i, j) - M(i, k) * M(k, j);
                    M(i, j) = num / prev_pivot;
                }
            }
            prev_pivot = cur_pivot;
        }

        T det = M(n - 1, n - 1);
        return (sign > 0) ? det : -det;
    }

    T float_determinant() const {
        size_t n = rows_;
        Matrix<T> M = *this;
        T det = T{1};

        for (size_t k = 0; k < n; ++k) {
            size_t pivot_row = k;
            T max_val = (M(k, k) < T{}) ? -M(k, k) : M(k, k);
            for (size_t i = k + 1; i < n; ++i) {
                T abs_val = (M(i, k) < T{}) ? -M(i, k) : M(i, k);
                if (abs_val > max_val) {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }
            if (max_val == T{}) {
                return T{};
            }
            if (pivot_row != k) {
                M.swap_rows(k, pivot_row);
                det = -det;
            }
            T pivot = M(k, k);
            det = det * pivot;
            for (size_t i = k + 1; i < n; ++i) {
                T factor = M(i, k) / pivot;
                for (size_t j = k + 1; j < n; ++j) {
                    M(i, j) -= factor * M(k, j);
                }
            }
        }
        return det;
    }

    size_t float_rank() const {
        Eigen::MatrixXd em = to_eigen();
        Eigen::FullPivLU<Eigen::MatrixXd> lu(em);
        return static_cast<size_t>(lu.rank());
    }

    Matrix<BigRational> rational_inverse() const {
        size_t n = rows_;
        Matrix<BigRational> aug = block_horizontal(*this, Matrix<BigRational>::identity(n));

        for (size_t k = 0; k < n; ++k) {
            size_t pivot_row = k;
            while (pivot_row < n && aug(pivot_row, k).sign() == 0) {
                ++pivot_row;
            }
            if (pivot_row == n) {
                throw SingularMatrixException("Matrix is singular and cannot be inverted");
            }
            aug.swap_rows(k, pivot_row);
            BigRational pivot = aug(k, k);
            for (size_t j = 0; j < 2 * n; ++j) {
                aug(k, j) /= pivot;
            }
            for (size_t i = 0; i < n; ++i) {
                if (i != k && aug(i, k).sign() != 0) {
                    BigRational factor = aug(i, k);
                    for (size_t j = 0; j < 2 * n; ++j) {
                        aug(i, j) -= factor * aug(k, j);
                    }
                }
            }
        }
        return aug.submatrix(0, n, n, n);
    }

    Matrix<T> float_inverse() const {
        size_t n = rows_;
        Matrix<T> aug = block_horizontal(*this, Matrix<T>::identity(n));

        for (size_t k = 0; k < n; ++k) {
            size_t pivot_row = k;
            T max_val = (aug(k, k) < T{}) ? -aug(k, k) : aug(k, k);
            for (size_t i = k + 1; i < n; ++i) {
                T abs_val = (aug(i, k) < T{}) ? -aug(i, k) : aug(i, k);
                if (abs_val > max_val) {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }
            if (max_val == T{}) {
                throw SingularMatrixException("Matrix is singular and cannot be inverted");
            }
            aug.swap_rows(k, pivot_row);
            T pivot = aug(k, k);
            for (size_t j = 0; j < 2 * n; ++j) {
                aug(k, j) /= pivot;
            }
            for (size_t i = 0; i < n; ++i) {
                if (i != k) {
                    T factor = aug(i, k);
                    for (size_t j = 0; j < 2 * n; ++j) {
                        aug(i, j) -= factor * aug(k, j);
                    }
                }
            }
        }
        return aug.submatrix(0, n, n, n);
    }

    size_t rows_;
    size_t cols_;
    std::vector<T> data_;
};

} // namespace dbdd
