#pragma once

#include <cstddef>
#include <cstdint>
#include <mpfr.h>

namespace dbdd {

// Basic dimension and indexing types
using dim_t = std::size_t;
using index_t = std::size_t;
using prec_t = mpfr_prec_t;

// Forward declarations of arithmetic wrappers
class BigInt;
class BigRational;
class BigReal;

// Forward declarations of linear algebra wrappers
template <typename T>
class Vector;

template <typename T>
class Matrix;

// Concrete type aliases for standard mathematical domains
using VectorZZ = Vector<BigInt>;
using VectorQQ = Vector<BigRational>;
using VectorRR = Vector<BigReal>;
using VectorDouble = Vector<double>;

using MatrixZZ = Matrix<BigInt>;
using MatrixQQ = Matrix<BigRational>;
using MatrixRR = Matrix<BigReal>;
using MatrixDouble = Matrix<double>;

} // namespace dbdd
