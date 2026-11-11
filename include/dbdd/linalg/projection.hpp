#pragma once

// Paper Reference: Dachman-Soled et al. Section 2, Section 3.5 / docs/mathematical_conventions.md Section 7
// Mathematical Purpose: Orthogonal projection operators conforming to the row-vector convention.
// For a non-zero row vector v:
//     P_v^perp = I - v^T v / ||v||^2
// Invariants guaranteed:
//     P^2 = P (idempotent), P^T = P (symmetric), v P = 0 (annihilates normal vector).

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <type_traits>

namespace dbdd {

/**
 * @brief Computes the outer product v^T * w as an (n x m) matrix,
 * where v is size n and w is size m.
 */
template <typename T>
Matrix<T> outer_product(const Vector<T>& v, const Vector<T>& w) {
    size_t n = v.size();
    size_t m = w.size();
    Matrix<T> res(n, m);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            res(i, j) = v[i] * w[j];
        }
    }
    return res;
}

/**
 * @brief Computes the projection matrix P_v^parallel onto the 1D subspace spanned by row vector v.
 * 
 * Formula: P_v^parallel = v^T v / ||v||^2
 * 
 * @tparam T Field type (e.g. BigRational, BigReal, double)
 * @param v Non-zero row vector
 * @return Matrix<T> Projection matrix of dimension (m x m)
 */
template <typename T>
Matrix<T> parallel_projection_matrix(const Vector<T>& v) {
    if (v.empty()) {
        throw DimensionMismatchException("Cannot compute projection matrix for an empty vector");
    }
    T sq_norm = v.squared_norm();
    if (sq_norm == T{}) {
        throw DivisionByZeroException("Cannot compute projection matrix for the zero vector");
    }

    size_t m = v.size();
    Matrix<T> outer = outer_product(v, v);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            outer(i, j) /= sq_norm;
        }
    }
    return outer;
}

/**
 * @brief Computes the projection matrix P_v^perp onto the orthogonal complement of row vector v.
 * 
 * Formula: P_v^perp = I_m - v^T v / ||v||^2
 * Invariants: P^2 = P, P^T = P, v * P = 0.
 * 
 * @tparam T Field type (e.g. BigRational, BigReal, double)
 * @param v Non-zero row vector
 * @return Matrix<T> Projection matrix of dimension (m x m)
 */
template <typename T>
Matrix<T> orthogonal_projection_matrix(const Vector<T>& v) {
    size_t m = v.size();
    Matrix<T> P_par = parallel_projection_matrix(v);
    Matrix<T> I = Matrix<T>::identity(m);
    return I - P_par;
}

/**
 * @brief Overload for integer vectors VectorZZ, returning exact rational projection MatrixQQ.
 */
inline MatrixQQ orthogonal_projection_matrix(const VectorZZ& v) {
    return orthogonal_projection_matrix(v.cast<BigRational>());
}

inline MatrixQQ parallel_projection_matrix(const VectorZZ& v) {
    return parallel_projection_matrix(v.cast<BigRational>());
}

/**
 * @brief Computes the projection matrix P_V^parallel onto the subspace spanned by rows of V.
 * 
 * Formula: P_V^parallel = V^T * (V * V^T)^{-1} * V
 * Preconditions: rows of V must be linearly independent.
 * 
 * @tparam T Field type (BigRational, BigReal, double)
 * @param V Matrix whose rows form a basis of the subspace (k x m, with k <= m)
 * @return Matrix<T> Projection matrix of dimension (m x m)
 */
template <typename T>
Matrix<T> parallel_projection_matrix(const Matrix<T>& V) {
    if (V.empty()) {
        throw DimensionMismatchException("Cannot compute projection for an empty matrix");
    }
    size_t k = V.rows();
    size_t m = V.cols();
    if (k > m) {
        throw DimensionMismatchException("Subspace dimension cannot exceed ambient dimension");
    }

    // Gram matrix G = V * V^T (k x k)
    Matrix<T> G = V * V.transpose();
    Matrix<T> G_inv = G.inverse();

    // P_parallel = V^T * G_inv * V (m x m)
    Matrix<T> VT_Ginv = V.transpose() * G_inv; // (m x k)
    return VT_Ginv * V;                         // (m x m)
}

/**
 * @brief Computes the projection matrix P_V^perp onto the orthogonal complement of the row span of V.
 * 
 * Formula: P_V^perp = I_m - V^T * (V * V^T)^{-1} * V
 * Invariants: P^2 = P, P^T = P, V * P = 0.
 * 
 * @tparam T Field type (BigRational, BigReal, double)
 * @param V Matrix whose rows span the subspace (k x m)
 * @return Matrix<T> Projection matrix of dimension (m x m)
 */
template <typename T>
Matrix<T> orthogonal_complement_projection_matrix(const Matrix<T>& V) {
    size_t m = V.cols();
    Matrix<T> P_par = parallel_projection_matrix(V);
    Matrix<T> I = Matrix<T>::identity(m);
    return I - P_par;
}

/**
 * @brief Overloads for integer matrices MatrixZZ, returning exact rational projections MatrixQQ.
 */
inline MatrixQQ parallel_projection_matrix(const MatrixZZ& V) {
    return parallel_projection_matrix(V.cast<BigRational>());
}

inline MatrixQQ orthogonal_complement_projection_matrix(const MatrixZZ& V) {
    return orthogonal_complement_projection_matrix(V.cast<BigRational>());
}

/**
 * @brief Projects row vector x using projection matrix P.
 * In row-vector convention, this computes x * P.
 */
template <typename T>
Vector<T> project_vector(const Vector<T>& x, const Matrix<T>& P) {
    return x * P;
}

/**
 * @brief Projects rows of matrix M using projection matrix P.
 * In row-vector convention, this computes M * P.
 */
template <typename T>
Matrix<T> project_matrix(const Matrix<T>& M, const Matrix<T>& P) {
    return M * P;
}

} // namespace dbdd
