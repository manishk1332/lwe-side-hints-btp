#include "dbdd/lattice/volume.hpp"
#include "dbdd/linalg/eigendecomposition.hpp"

namespace dbdd {

template BigReal compute_log_volume_from_gs(const GramSchmidtState<BigRational>&);
template BigReal compute_log_volume_from_gs(const GramSchmidtState<BigReal>&);
template BigReal compute_log_volume_from_gs(const GramSchmidtState<double>&);

template BigReal compute_volume_from_gs(const GramSchmidtState<BigRational>&);
template BigReal compute_volume_from_gs(const GramSchmidtState<BigReal>&);
template BigReal compute_volume_from_gs(const GramSchmidtState<double>&);

template BigReal compute_log_volume(const Basis<BigInt>&);
template BigReal compute_log_volume(const Basis<BigRational>&);
template BigReal compute_log_volume(const Basis<BigReal>&);
template BigReal compute_log_volume(const Basis<double>&);

template BigReal compute_volume(const Basis<BigInt>&);
template BigReal compute_volume(const Basis<BigRational>&);
template BigReal compute_volume(const Basis<BigReal>&);
template BigReal compute_volume(const Basis<double>&);

BigReal compute_restricted_determinant(const MatrixDouble& cov, double rank_tol) {
    auto eigen = SymmetricEigendecomposition::decompose(cov, rank_tol);
    return eigen.restricted_determinant();
}

BigReal compute_log_restricted_determinant(const MatrixDouble& cov, double rank_tol) {
    auto eigen = SymmetricEigendecomposition::decompose(cov, rank_tol);
    return eigen.log_restricted_determinant();
}

BigReal compute_isotropized_volume(
    const BigReal& original_volume,
    const MatrixDouble& cov,
    double rank_tol) {
    BigReal log_orig = original_volume.log();
    BigReal log_iso = compute_isotropized_log_volume(log_orig, cov, rank_tol);
    return log_iso.exp();
}

BigReal compute_isotropized_log_volume(
    const BigReal& original_log_volume,
    const MatrixDouble& cov,
    double rank_tol) {
    BigReal log_rdet = compute_log_restricted_determinant(cov, rank_tol);
    return original_log_volume - BigReal(0.5) * log_rdet;
}

} // namespace dbdd
