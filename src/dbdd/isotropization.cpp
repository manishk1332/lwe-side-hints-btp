#include "dbdd/dbdd/isotropization.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/linalg/eigendecomposition.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/core/exceptions.hpp"

namespace dbdd {

MatrixDouble Isotropization::compute_transform(
    const MatrixDouble& cov_h,
    double rank_tol) {
    auto eigen = SymmetricEigendecomposition::decompose(cov_h, rank_tol);
    return eigen.matrix_inv_sqrt();
}

MatrixDouble Isotropization::compute_matrix_sqrt(
    const MatrixDouble& cov_h,
    double rank_tol) {
    auto eigen = SymmetricEigendecomposition::decompose(cov_h, rank_tol);
    return eigen.matrix_sqrt();
}

MatrixDouble Isotropization::compute_isotropized_covariance(
    const MatrixDouble& cov_h,
    const MatrixDouble& M) {
    if (cov_h.rows() != cov_h.cols() || M.rows() != M.cols() || cov_h.rows() != M.rows()) {
        throw DimensionMismatchException("Isotropization::compute_isotropized_covariance: Matrix dimension mismatch");
    }

    // In row-vector convention, covariance transforms as M^T * cov_h * M = M * cov_h * M (since M is symmetric)
    MatrixDouble cov_iso = M * cov_h * M;

    // Enforce strict symmetry
    size_t d = cov_iso.rows();
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (cov_iso(i, j) + cov_iso(j, i));
            cov_iso(i, j) = avg;
            cov_iso(j, i) = avg;
        }
    }

    return cov_iso;
}

UsvpInstance Isotropization::isotropize(
    const DBDDInstance& instance,
    double rank_tol) {
    // If instance is not origin-centered, homogenize first
    if (!instance.is_homogenized()) {
        DBDDInstance homog_inst = instance;
        homog_inst.homogenize();
        return isotropize(homog_inst, rank_tol);
    }

    const MatrixDouble& cov_h = instance.covariance();
    size_t amb_dim = instance.ambient_dimension();

    // Compute transformation matrix M = (Sigma_h^{1/2})^\dagger
    MatrixDouble M = compute_transform(cov_h, rank_tol);

    // Transform basis: B_iso = B * M
    MatrixDouble B_double(instance.dimension(), amb_dim);
    const auto& B_zz = instance.matrix();
    for (size_t i = 0; i < instance.dimension(); ++i) {
        for (size_t j = 0; j < amb_dim; ++j) {
            B_double(i, j) = B_zz(i, j).to_double();
        }
    }

    MatrixDouble B_iso = B_double * M;

    // Compute volume shortcut: Vol(Lambda_iso) = Vol(Lambda) / sqrt(rdet(Sigma_h))
    BigReal vol_iso = compute_isotropized_volume(instance.volume(), cov_h, rank_tol);
    BigReal log_vol_iso = compute_isotropized_log_volume(instance.log_volume(), cov_h, rank_tol);

    // Transform target vector: s_iso = s_bar * M
    std::optional<VectorDouble> s_iso = std::nullopt;
    if (instance.has_target()) {
        VectorDouble s_double = instance.target_double();
        s_iso = s_double * M;
    }

    return UsvpInstance(
        std::move(B_iso),
        std::move(vol_iso),
        std::move(log_vol_iso),
        std::move(M),
        std::move(s_iso));
}

void Isotropization::apply(
    DBDDInstance& instance,
    double rank_tol) {
    if (!instance.is_homogenized()) {
        instance.homogenize();
    }

    const MatrixDouble& cov_h = instance.covariance();
    MatrixDouble M = compute_transform(cov_h, rank_tol);
    MatrixDouble cov_iso = compute_isotropized_covariance(cov_h, M);

    VectorDouble zero_mean(instance.ambient_dimension(), 0.0);
    instance.set_distribution(std::move(zero_mean), std::move(cov_iso));
}

} // namespace dbdd
