#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/core/exceptions.hpp"
#include <cmath>

namespace dbdd {

UsvpInstance::UsvpInstance(
    MatrixDouble basis_matrix,
    BigReal volume,
    BigReal log_volume,
    MatrixDouble transformation_matrix,
    std::optional<VectorDouble> target)
    : dim_(basis_matrix.rows()),
      amb_dim_(basis_matrix.cols()),
      basis_matrix_(std::move(basis_matrix)),
      volume_(std::move(volume)),
      log_volume_(std::move(log_volume)),
      M_(std::move(transformation_matrix)),
      target_(std::move(target)) {
    if (target_.has_value() && target_->size() != amb_dim_) {
        throw DimensionMismatchException(
            "UsvpInstance: Target size (" + std::to_string(target_->size()) +
            ") does not match ambient dimension (" + std::to_string(amb_dim_) + ")");
    }
}

BasisDouble UsvpInstance::basis() const {
    return BasisDouble(basis_matrix_, /*validate_rank=*/false);
}

BasisRR UsvpInstance::basis_rr() const {
    MatrixRR m_rr(dim_, amb_dim_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < amb_dim_; ++j) {
            m_rr(i, j) = BigReal(basis_matrix_(i, j));
        }
    }
    return BasisRR(std::move(m_rr), /*validate_rank=*/false);
}

LatticeDouble UsvpInstance::lattice() const {
    return LatticeDouble(basis(), volume_);
}

const VectorDouble& UsvpInstance::target() const {
    if (!target_.has_value()) {
        throw DBDDException("UsvpInstance: Instance has no ground-truth target vector");
    }
    return target_.value();
}

double UsvpInstance::target_norm() const {
    return std::sqrt(target_squared_norm());
}

double UsvpInstance::target_squared_norm() const {
    return target().squared_norm();
}

BasisZZ UsvpInstance::to_integer_basis(double scale_factor) const {
    if (scale_factor <= 0.0) {
        throw InvalidParameterException("UsvpInstance::to_integer_basis: scale_factor must be positive");
    }

    MatrixZZ int_matrix(dim_, amb_dim_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < amb_dim_; ++j) {
            double scaled = basis_matrix_(i, j) * scale_factor;
            int64_t rounded = std::llround(scaled);
            int_matrix(i, j) = BigInt(rounded);
        }
    }

    return BasisZZ(std::move(int_matrix), /*validate_rank=*/false);
}

VectorDouble UsvpInstance::reconstruct_vector(
    const VectorDouble& w_iso,
    const MatrixDouble& sqrt_cov) const {
    if (w_iso.size() != dim_) {
        throw DimensionMismatchException(
            "UsvpInstance::reconstruct_vector: Vector dimension (" + std::to_string(w_iso.size()) +
            ") does not match instance dimension (" + std::to_string(dim_) + ")");
    }
    if (sqrt_cov.rows() != amb_dim_ || sqrt_cov.cols() != amb_dim_) {
        throw DimensionMismatchException(
            "UsvpInstance::reconstruct_vector: Square root covariance dimension mismatch");
    }

    // In row-vector convention: w = w_iso * sqrt_cov
    return w_iso * sqrt_cov;
}

SecurityEstimate UsvpInstance::estimate_attack_gsa(double beta_min, double beta_max) const {
    return GSAIntersect::estimate(*this, beta_min, beta_max);
}

size_t UsvpInstance::estimate_beta_gsa() const {
    return estimate_attack_gsa().beta;
}

} // namespace dbdd
