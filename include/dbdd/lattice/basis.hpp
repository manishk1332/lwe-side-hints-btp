#pragma once

// Paper Reference: Albrecht et al. Section 2 / docs/mathematical_conventions.md Section 1
// Mathematical Purpose: Lattice basis matrix conforming strictly to the row-basis convention:
//     row i of matrix B is basis vector b_i.
//     v = x * B represents integer linear combination of basis vectors.
// Preconditions:
//     Rows are linearly independent (rank == d, with d <= m).

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <vector>

namespace dbdd {

template <typename T>
class Basis {
public:
    // Constructors
    Basis() : B_(0, 0) {}

    explicit Basis(Matrix<T> matrix, bool validate_rank = true)
        : B_(std::move(matrix)) {
        if (B_.rows() > B_.cols()) {
            throw DimensionMismatchException("Basis rank cannot exceed ambient dimension (rows > cols)");
        }
        if (validate_rank && !B_.empty()) {
            if (B_.rank() != B_.rows()) {
                throw SingularMatrixException("Basis vectors are linearly dependent (rank < dimension)");
            }
        }
    }

    explicit Basis(const std::vector<Vector<T>>& rows, bool validate_rank = true)
        : Basis(Matrix<T>::from_rows(rows), validate_rank) {}

    // Factory methods
    static Basis identity(size_t n) {
        return Basis(Matrix<T>::identity(n), false);
    }

    static Basis diagonal(const Vector<T>& diag) {
        return Basis(Matrix<T>::diagonal(diag), true);
    }

    static Basis from_rows(const std::vector<Vector<T>>& rows, bool validate_rank = true) {
        return Basis(Matrix<T>::from_rows(rows), validate_rank);
    }

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return B_.rows(); }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return B_.cols(); }
    [[nodiscard]] size_t rank() const { return B_.rank(); }
    [[nodiscard]] bool empty() const noexcept { return B_.empty(); }
    [[nodiscard]] bool is_full_rank() const noexcept { return dimension() == ambient_dimension(); }

    // Matrix and row access (Row-basis convention)
    [[nodiscard]] const Matrix<T>& matrix() const noexcept { return B_; }
    [[nodiscard]] Matrix<T>& raw_matrix() noexcept { return B_; }

    [[nodiscard]] Vector<T> row(size_t i) const {
        return B_.row(i);
    }

    [[nodiscard]] Vector<T> operator[](size_t i) const {
        return B_.row(i);
    }

    void set_row(size_t i, const Vector<T>& v) {
        B_.set_row(i, v);
    }

    // Elementary unimodular row operations
    void swap_rows(size_t i, size_t j) {
        B_.swap_rows(i, j);
    }

    void row_add_multiple(size_t target, size_t source, const T& factor) {
        B_.row_add_multiple(target, source, factor);
    }

    // Linear combinations & coordinate evaluation: v = x * B
    // Paper Reference: Section 2
    // Mathematical Purpose: Row-vector product x * B
    [[nodiscard]] Vector<T> linear_combination(const Vector<T>& coords) const {
        if (coords.size() != dimension()) {
            throw DimensionMismatchException("Coordinate vector size mismatch: " +
                                            std::to_string(coords.size()) + " vs dimension " +
                                            std::to_string(dimension()));
        }
        return coords * B_;
    }

    /**
     * @brief Computes coordinates x such that x * B = v in the ambient space.
     * For full rank (d == m): x = v * B^{-1}.
     * For subspace lattice (d < m): x = v * B^T * (B * B^T)^{-1}.
     */
    template <typename OutT = BigRational>
    [[nodiscard]] Vector<OutT> coordinates_of(const Vector<T>& v) const {
        if (v.size() != ambient_dimension()) {
            throw DimensionMismatchException("Vector ambient dimension mismatch");
        }
        size_t d = dimension();
        size_t m = ambient_dimension();

        Matrix<OutT> B_cast = B_.template cast<OutT>();
        Vector<OutT> v_cast = v.template cast<OutT>();

        if (d == m) {
            Matrix<OutT> B_inv = B_cast.inverse();
            return v_cast * B_inv;
        } else {
            // General subspace pseudoinverse in row convention:
            // x = v * B^T * (B * B^T)^{-1}
            Matrix<OutT> G = B_cast * B_cast.transpose(); // (d x d)
            Matrix<OutT> G_inv = G.inverse();             // (d x d)
            Matrix<OutT> pinv = B_cast.transpose() * G_inv; // (m x d)
            return v_cast * pinv;                         // 1 x d
        }
    }

    // Type casting
    template <typename U>
    [[nodiscard]] Basis<U> cast() const {
        return Basis<U>(B_.template cast<U>(), false);
    }

    bool operator==(const Basis& rhs) const noexcept {
        return B_ == rhs.B_;
    }

    bool operator!=(const Basis& rhs) const noexcept {
        return B_ != rhs.B_;
    }

private:
    Matrix<T> B_;
};

using BasisZZ = Basis<BigInt>;
using BasisQQ = Basis<BigRational>;
using BasisRR = Basis<BigReal>;
using BasisDouble = Basis<double>;

} // namespace dbdd
