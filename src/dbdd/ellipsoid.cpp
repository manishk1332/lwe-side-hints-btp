// Paper Reference: Albrecht et al. Section 2, Section 3 / docs/mathematical_conventions.md Section 3
// Mathematical Purpose: Implementation of Ellipsoid E(mu, Sigma), PSD validation,
// Moore-Penrose pseudoinverse, and Mahalanobis containment with degenerate subspace support.

#include "dbdd/dbdd/ellipsoid.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <algorithm>

namespace dbdd {

Ellipsoid::Ellipsoid()
    : dim_(0), rank_(0), mean_(), covariance_(), pinv_(), eigenvalues_() {}

Ellipsoid::Ellipsoid(
    VectorDouble mean,
    MatrixDouble covariance,
    double psd_tolerance,
    double rank_tolerance)
    : dim_(mean.size()),
      rank_(0),
      mean_(std::move(mean)),
      covariance_(std::move(covariance)),
      psd_tol_(psd_tolerance),
      rank_tol_(rank_tolerance) {
    if (!covariance_.is_square()) {
        throw DimensionMismatchException("Ellipsoid covariance matrix must be square, got " +
            std::to_string(covariance_.rows()) + "x" + std::to_string(covariance_.cols()));
    }
    if (covariance_.rows() != dim_) {
        throw DimensionMismatchException("Ellipsoid mean dimension (" + std::to_string(dim_) +
            ") does not match covariance dimension (" + std::to_string(covariance_.rows()) + ")");
    }
    decompose_and_validate();
}

Ellipsoid Ellipsoid::from_covariance(
    MatrixDouble covariance,
    double psd_tolerance,
    double rank_tolerance) {
    size_t d = covariance.rows();
    VectorDouble zero_mean(d, 0.0);
    return Ellipsoid(std::move(zero_mean), std::move(covariance), psd_tolerance, rank_tolerance);
}

Ellipsoid Ellipsoid::standard(size_t dim) {
    VectorDouble mean(dim, 0.0);
    MatrixDouble cov = MatrixDouble::identity(dim);
    return Ellipsoid(std::move(mean), std::move(cov));
}

Ellipsoid Ellipsoid::diagonal(
    VectorDouble mean,
    VectorDouble variances,
    double psd_tolerance,
    double rank_tolerance) {
    if (mean.size() != variances.size()) {
        throw DimensionMismatchException("Ellipsoid mean size (" + std::to_string(mean.size()) +
            ") does not match variances size (" + std::to_string(variances.size()) + ")");
    }
    MatrixDouble cov = MatrixDouble::diagonal(variances);
    return Ellipsoid(std::move(mean), std::move(cov), psd_tolerance, rank_tolerance);
}

void Ellipsoid::decompose_and_validate() {
    if (dim_ == 0) {
        rank_ = 0;
        pinv_ = MatrixDouble(0, 0);
        eigenvalues_ = VectorDouble();
        return;
    }

    // 1. Enforce strict mathematical symmetry: Sigma <- 1/2 (Sigma + Sigma^T)
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = i + 1; j < dim_; ++j) {
            double avg = 0.5 * (covariance_(i, j) + covariance_(j, i));
            covariance_(i, j) = avg;
            covariance_(j, i) = avg;
        }
    }

    // 2. Perform spectral decomposition via Eigen
    Eigen::MatrixXd em = covariance_.to_eigen();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(em);
    if (solver.info() != Eigen::Success) {
        throw NumericalPrecisionException("SelfAdjointEigenSolver failed to compute eigendecomposition of covariance");
    }

    const auto& raw_eigs = solver.eigenvalues();
    double min_ev = raw_eigs(0);
    if (min_ev < -psd_tol_) {
        throw NonPositiveSemidefiniteException(
            "Covariance matrix is not positive semidefinite: minimum eigenvalue is " +
            std::to_string(min_ev) + " (tolerance is " + std::to_string(psd_tol_) + ")");
    }

    // 3. Clamp small negative eigenvalues in [-psd_tol_, 0) to 0.0 and construct pseudoinverse
    rank_ = 0;
    eigenvalues_ = VectorDouble(dim_);
    Eigen::VectorXd clamped_eigs(dim_);
    Eigen::VectorXd inv_eigs = Eigen::VectorXd::Zero(dim_);

    for (size_t i = 0; i < dim_; ++i) {
        double ev = raw_eigs(static_cast<Eigen::Index>(i));
        if (ev < 0.0) {
            ev = 0.0;
        }
        eigenvalues_[i] = ev;
        clamped_eigs(static_cast<Eigen::Index>(i)) = ev;

        if (ev > rank_tol_) {
            inv_eigs(static_cast<Eigen::Index>(i)) = 1.0 / ev;
            rank_++;
        }
    }

    const Eigen::MatrixXd& Q = solver.eigenvectors();

    // Reconstruct sanitized symmetric covariance matrix
    Eigen::MatrixXd clean_cov = Q * clamped_eigs.asDiagonal() * Q.transpose();
    clean_cov = 0.5 * (clean_cov + clean_cov.transpose());
    covariance_ = MatrixDouble::from_eigen(clean_cov);

    // Compute Moore-Penrose pseudoinverse: Sigma^dagger = Q * Lambda^dagger * Q^T
    Eigen::MatrixXd em_pinv = Q * inv_eigs.asDiagonal() * Q.transpose();
    em_pinv = 0.5 * (em_pinv + em_pinv.transpose());
    pinv_ = MatrixDouble::from_eigen(em_pinv);
}

double Ellipsoid::mahalanobis_distance_squared(const VectorDouble& x) const {
    if (x.size() != dim_) {
        throw DimensionMismatchException("Vector dimension (" + std::to_string(x.size()) +
            ") does not match ellipsoid dimension (" + std::to_string(dim_) + ")");
    }
    if (dim_ == 0) return 0.0;

    VectorDouble delta = x - mean_;
    VectorDouble pinv_delta = delta * pinv_; // row vector * matrix
    double dist_sq = delta.dot(pinv_delta);
    return std::max(0.0, dist_sq);
}

double Ellipsoid::mahalanobis_distance(const VectorDouble& x) const {
    return std::sqrt(mahalanobis_distance_squared(x));
}

bool Ellipsoid::is_in_support(const VectorDouble& x, double tol) const {
    if (x.size() != dim_) {
        throw DimensionMismatchException("Vector dimension (" + std::to_string(x.size()) +
            ") does not match ellipsoid dimension (" + std::to_string(dim_) + ")");
    }
    if (dim_ == 0) return true;
    if (is_full_rank()) return true;

    VectorDouble delta = x - mean_;
    // Projection onto support: delta_proj = delta * (Sigma * Sigma^dagger)
    MatrixDouble P = support_projector();
    VectorDouble delta_proj = delta * P;
    VectorDouble delta_ker = delta - delta_proj;

    return delta_ker.norm() <= tol;
}

VectorDouble Ellipsoid::project_onto_support(const VectorDouble& x) const {
    if (x.size() != dim_) {
        throw DimensionMismatchException("Vector dimension (" + std::to_string(x.size()) +
            ") does not match ellipsoid dimension (" + std::to_string(dim_) + ")");
    }
    if (dim_ == 0 || is_full_rank()) return x;

    VectorDouble delta = x - mean_;
    MatrixDouble P = support_projector();
    VectorDouble delta_proj = delta * P;
    return mean_ + delta_proj;
}

MatrixDouble Ellipsoid::support_projector() const {
    if (dim_ == 0) return MatrixDouble(0, 0);
    if (is_full_rank()) return MatrixDouble::identity(dim_);

    // P = Sigma * Sigma^dagger
    MatrixDouble P = covariance_ * pinv_;
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = i + 1; j < dim_; ++j) {
            double avg = 0.5 * (P(i, j) + P(j, i));
            P(i, j) = avg;
            P(j, i) = avg;
        }
    }
    return P;
}

bool Ellipsoid::contains(
    const VectorDouble& x,
    double threshold,
    bool check_support,
    double support_tol) const {
    if (check_support && !is_in_support(x, support_tol)) {
        return false;
    }
    return mahalanobis_distance_squared(x) <= (threshold + 1e-9);
}

double Ellipsoid::volume() const {
    if (dim_ == 0) return 0.0;
    if (!is_full_rank()) return 0.0; // Volume in ambient space is zero for degenerate ellipsoids

    return std::exp(log_volume());
}

double Ellipsoid::log_volume() const {
    if (dim_ == 0 || rank_ == 0) return -std::numeric_limits<double>::infinity();

    double k = static_cast<double>(rank_);
    double pi_term = (k / 2.0) * std::log(std::numbers::pi);
    double gamma_term = std::lgamma(k / 2.0 + 1.0);

    double log_det_sqrt = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
        if (eigenvalues_[i] > rank_tol_) {
            log_det_sqrt += 0.5 * std::log(eigenvalues_[i]);
        }
    }

    return pi_term - gamma_term + log_det_sqrt;
}

bool Ellipsoid::operator==(const Ellipsoid& rhs) const noexcept {
    if (dim_ != rhs.dim_ || rank_ != rhs.rank_) return false;
    for (size_t i = 0; i < dim_; ++i) {
        if (std::abs(mean_[i] - rhs.mean_[i]) > 1e-11) return false;
        for (size_t j = 0; j < dim_; ++j) {
            if (std::abs(covariance_(i, j) - rhs.covariance_(i, j)) > 1e-11) return false;
        }
    }
    return true;
}

bool Ellipsoid::operator!=(const Ellipsoid& rhs) const noexcept {
    return !(*this == rhs);
}

} // namespace dbdd
