#include "dbdd/linalg/gram_schmidt.hpp"

namespace dbdd {

template GramSchmidtState<BigRational> compute_gram_schmidt_cgs(const Matrix<BigRational>&);
template GramSchmidtState<BigReal> compute_gram_schmidt_cgs(const Matrix<BigReal>&);
template GramSchmidtState<double> compute_gram_schmidt_cgs(const Matrix<double>&);

template GramSchmidtState<BigRational> compute_gram_schmidt_mgs(const Matrix<BigRational>&);
template GramSchmidtState<BigReal> compute_gram_schmidt_mgs(const Matrix<BigReal>&);
template GramSchmidtState<double> compute_gram_schmidt_mgs(const Matrix<double>&);

template void update_gram_schmidt_swap(GramSchmidtState<BigRational>&, size_t);
template void update_gram_schmidt_swap(GramSchmidtState<BigReal>&, size_t);
template void update_gram_schmidt_swap(GramSchmidtState<double>&, size_t);

template void update_gram_schmidt_row_add(GramSchmidtState<BigRational>&, size_t, size_t, const BigRational&);
template void update_gram_schmidt_row_add(GramSchmidtState<BigReal>&, size_t, size_t, const BigReal&);
template void update_gram_schmidt_row_add(GramSchmidtState<double>&, size_t, size_t, const double&);

} // namespace dbdd
