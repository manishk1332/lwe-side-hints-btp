#pragma once

// Paper Reference: Albrecht et al. Section 3.1, Section 3.2, Section 3.5 / main_plan.txt Section 10
// Mathematical Purpose: Exact primitivity verification for:
//   1. Integer vectors: gcd(v_0, ..., v_{m-1}) == 1
//   2. Lattice vectors: v in Lambda is primitive iff in basis coordinates v = x * B, gcd(x) == 1
//   3. Dual lattice vectors: w in Lambda* is primitive iff in dual basis w = y * D, gcd(y) == 1
//   4. Sets of vectors / Sublattices: V in Lambda is primitive iff all invariant factors in
//      Smith Normal Form of coordinate matrix X (where V = X * B) equal 1.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lattice/basis.hpp"
#include <vector>

namespace dbdd {

/**
 * @brief Computes the greatest common divisor of all coordinates in an integer vector.
 * Returns 0 if all coordinates are 0.
 */
inline BigInt vector_gcd(const VectorZZ& v) {
    BigInt g(0);
    for (size_t i = 0; i < v.size(); ++i) {
        g = BigInt::gcd(g, v[i].abs());
        if (g == BigInt(1)) {
            break;
        }
    }
    return g;
}

/**
 * @brief Tests if an integer vector v is primitive in Z^m (gcd(v) == 1).
 */
inline bool is_primitive_vector(const VectorZZ& v) {
    if (v.empty()) return false;
    return vector_gcd(v) == BigInt(1);
}

/**
 * @brief Divides an integer vector by its gcd, making it primitive.
 * Throws DBDDException if v is the zero vector.
 */
inline VectorZZ make_primitive(const VectorZZ& v) {
    BigInt g = vector_gcd(v);
    if (g == BigInt(0)) {
        throw DBDDException("Cannot make zero vector primitive");
    }
    VectorZZ res(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        res[i] = v[i] / g;
    }
    return res;
}

/**
 * @brief Tests if a vector v is in the lattice Lambda and is primitive in Lambda.
 * Expresses v = x * B. Returns true iff x has all integer coordinates and gcd(x) == 1.
 */
template <typename T>
bool is_primitive_in_lattice(const Vector<T>& v, const Basis<T>& basis) {
    if (basis.empty() || v.empty()) return false;

    Vector<BigRational> coords = basis.template coordinates_of<BigRational>(v);
    VectorZZ x_int(coords.size());

    for (size_t i = 0; i < coords.size(); ++i) {
        if (coords[i].denominator() != BigInt(1)) {
            return false; // Vector is not in the integer lattice
        }
        x_int[i] = coords[i].numerator();
    }

    return is_primitive_vector(x_int);
}

/**
 * @brief Tests if a dual vector w is primitive in the dual lattice Lambda*.
 */
inline bool is_primitive_dual_vector(const VectorQQ& w, const BasisQQ& dual_basis) {
    return is_primitive_in_lattice(w, dual_basis);
}

/**
 * @brief Normalizes a vector v in Lambda to be primitive in Lambda:
 * computes x such that v = x * B, divides x by gcd(x), and returns (x / gcd(x)) * B.
 */
template <typename T>
Vector<T> make_primitive_in_lattice(const Vector<T>& v, const Basis<T>& basis) {
    Vector<BigRational> coords = basis.template coordinates_of<BigRational>(v);
    VectorZZ x_int(coords.size());

    for (size_t i = 0; i < coords.size(); ++i) {
        if (coords[i].denominator() != BigInt(1)) {
            throw PrimitivityViolationException("Vector is not in the lattice");
        }
        x_int[i] = coords[i].numerator();
    }

    BigInt g = vector_gcd(x_int);
    if (g == BigInt(0)) {
        throw PrimitivityViolationException("Zero vector cannot be made primitive");
    }

    Vector<T> x_prim(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        BigInt prim_coord = x_int[i] / g;
        if constexpr (std::is_same_v<T, BigInt>) {
            x_prim[i] = prim_coord;
        } else if constexpr (std::is_same_v<T, BigRational>) {
            x_prim[i] = BigRational(prim_coord);
        } else {
            x_prim[i] = static_cast<T>(prim_coord.to_double());
        }
    }

    return basis.linear_combination(x_prim);
}

/**
 * @brief Computes the invariant factors (elementary divisors) of an integer matrix X in Z^{k x d}.
 * Uses elementary integer row and column operations (Smith Normal Form algorithm).
 */
std::vector<BigInt> compute_smith_normal_form_invariant_factors(MatrixZZ X);

/**
 * @brief Tests if a set of vectors V in Lambda is primitive in Lambda.
 * 
 * V (represented as rows of matrix V) is primitive in Lambda iff
 * span_R(V) cap Lambda = span_Z(V), which is equivalent to all invariant
 * factors of coordinate matrix X (where V = X * B) equal 1.
 */
bool is_primitive_set_in_lattice(const MatrixZZ& V, const BasisZZ& basis);

} // namespace dbdd
