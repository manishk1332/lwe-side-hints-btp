#include "dbdd/linalg/projection.hpp"

namespace dbdd {

// Explicit template instantiations for standard numeric domains
template Matrix<BigRational> outer_product(const Vector<BigRational>&, const Vector<BigRational>&);
template Matrix<BigReal> outer_product(const Vector<BigReal>&, const Vector<BigReal>&);
template Matrix<double> outer_product(const Vector<double>&, const Vector<double>&);

template Matrix<BigRational> parallel_projection_matrix(const Vector<BigRational>&);
template Matrix<BigReal> parallel_projection_matrix(const Vector<BigReal>&);
template Matrix<double> parallel_projection_matrix(const Vector<double>&);

template Matrix<BigRational> orthogonal_projection_matrix(const Vector<BigRational>&);
template Matrix<BigReal> orthogonal_projection_matrix(const Vector<BigReal>&);
template Matrix<double> orthogonal_projection_matrix(const Vector<double>&);

template Matrix<BigRational> parallel_projection_matrix(const Matrix<BigRational>&);
template Matrix<BigReal> parallel_projection_matrix(const Matrix<BigReal>&);
template Matrix<double> parallel_projection_matrix(const Matrix<double>&);

template Matrix<BigRational> orthogonal_complement_projection_matrix(const Matrix<BigRational>&);
template Matrix<BigReal> orthogonal_complement_projection_matrix(const Matrix<BigReal>&);
template Matrix<double> orthogonal_complement_projection_matrix(const Matrix<double>&);

} // namespace dbdd
