#include "dbdd/dbdd/homogenization.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/linalg/projection.hpp"
#include "dbdd/core/exceptions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

namespace dbdd {

std::pair<VectorDouble, MatrixDouble> Homogenization::homogenize(
    const VectorDouble& mean,
    const MatrixDouble& cov) {
    if (cov.rows() != cov.cols()) {
        throw DimensionMismatchException(
            "Homogenization::homogenize: Covariance matrix must be square (got " +
            std::to_string(cov.rows()) + "x" + std::to_string(cov.cols()) + ")");
    }
    if (mean.size() != cov.rows()) {
        throw DimensionMismatchException(
            "Homogenization::homogenize: Mean dimension (" + std::to_string(mean.size()) +
            ") does not match covariance dimension (" + std::to_string(cov.rows()) + ")");
    }

    if (mean.empty()) {
        return {VectorDouble(), MatrixDouble(0, 0)};
    }

    size_t d = mean.size();

    // Compute outer product mu^T * mu (in row-vector convention, outer_product(mean, mean) produces d x d matrix)
    MatrixDouble outer = outer_product(mean, mean);

    // Compute homogenized covariance: Sigma_h = Sigma + mu^T * mu
    MatrixDouble cov_h = cov + outer;

    // Enforce strict numerical symmetry: Sigma_h <- 1/2 (Sigma_h + Sigma_h^T)
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (cov_h(i, j) + cov_h(j, i));
            cov_h(i, j) = avg;
            cov_h(j, i) = avg;
        }
    }

    // Centered mean: mu_h = 0 in R^d
    VectorDouble mu_h(d, 0.0);

    return {std::move(mu_h), std::move(cov_h)};
}

Ellipsoid Homogenization::homogenize(const Ellipsoid& ell) {
    if (ell.empty()) {
        return Ellipsoid();
    }

    auto [mu_h, cov_h] = homogenize(ell.mean(), ell.covariance());
    return Ellipsoid(std::move(mu_h), std::move(cov_h), ell.psd_tolerance(), ell.rank_tolerance());
}

void Homogenization::apply(DBDDInstance& instance) {
    auto [mu_h, cov_h] = homogenize(instance.mean(), instance.covariance());
    // Updates distribution while preserving lattice basis, volume, dimension, and target vector.
    instance.set_distribution(std::move(mu_h), std::move(cov_h));
}

bool Homogenization::verify_inclusion(
    const Ellipsoid& orig_ell,
    const Ellipsoid& homog_ell,
    double threshold,
    size_t num_samples) {
    if (orig_ell.empty()) {
        return true;
    }
    if (orig_ell.dimension() != homog_ell.dimension()) {
        throw DimensionMismatchException(
            "Homogenization::verify_inclusion: Dimension mismatch between original (" +
            std::to_string(orig_ell.dimension()) + ") and homogenized (" +
            std::to_string(homog_ell.dimension()) + ")");
    }

    size_t d = orig_ell.dimension();
    constexpr double inclusion_tol = 1e-7;

    // Handle degenerate zero-rank ellipsoid (single point mu)
    if (orig_ell.rank() == 0) {
        return homog_ell.mahalanobis_distance_squared(orig_ell.mean()) <= threshold + inclusion_tol;
    }

    // Spectral decomposition of original covariance
    Eigen::MatrixXd em = orig_ell.covariance().to_eigen();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(em);
    if (solver.info() != Eigen::Success) {
        throw NumericalPrecisionException(
            "Homogenization::verify_inclusion: Eigendecomposition of covariance failed");
    }

    const auto& eigenvalues = solver.eigenvalues();
    const auto& eigenvectors = solver.eigenvectors();

    // Extract active principal axes
    struct ActiveMode {
        double std_dev;
        VectorDouble vec;
    };
    std::vector<ActiveMode> active_modes;
    active_modes.reserve(d);

    for (size_t col = 0; col < d; ++col) {
        double ev = eigenvalues(col);
        if (ev > orig_ell.rank_tolerance()) {
            VectorDouble v(d);
            for (size_t r = 0; r < d; ++r) {
                v[r] = eigenvectors(r, col);
            }
            active_modes.push_back({std::sqrt(ev), std::move(v)});
        }
    }

    const VectorDouble& mu = orig_ell.mean();

    // 1. Verify principal axis extreme boundary points: mu +/- sqrt(lambda_k) * q_k
    for (const auto& mode : active_modes) {
        VectorDouble x_plus(d);
        VectorDouble x_minus(d);
        for (size_t i = 0; i < d; ++i) {
            double delta = mode.std_dev * mode.vec[i];
            x_plus[i] = mu[i] + delta;
            x_minus[i] = mu[i] - delta;
        }

        if (homog_ell.mahalanobis_distance_squared(x_plus) > threshold + inclusion_tol) {
            return false;
        }
        if (homog_ell.mahalanobis_distance_squared(x_minus) > threshold + inclusion_tol) {
            return false;
        }
    }

    // 2. Verify randomly sampled spherical boundary points
    if (num_samples > 0 && active_modes.size() > 1) {
        std::mt19937_64 rng(42);
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        size_t rank = active_modes.size();

        for (size_t sample = 0; sample < num_samples; ++sample) {
            std::vector<double> u(rank);
            double sq_norm = 0.0;
            for (size_t k = 0; k < rank; ++k) {
                u[k] = normal_dist(rng);
                sq_norm += u[k] * u[k];
            }

            if (sq_norm < 1e-14) {
                continue;
            }
            double inv_norm = 1.0 / std::sqrt(sq_norm);
            for (size_t k = 0; k < rank; ++k) {
                u[k] *= inv_norm;
            }

            // Construct boundary point: x = mu + sum_k (u_k * std_dev_k) * vec_k
            VectorDouble x = mu;
            for (size_t k = 0; k < rank; ++k) {
                double coeff = u[k] * active_modes[k].std_dev;
                for (size_t i = 0; i < d; ++i) {
                    x[i] += coeff * active_modes[k].vec[i];
                }
            }

            if (homog_ell.mahalanobis_distance_squared(x) > threshold + inclusion_tol) {
                return false;
            }
        }
    }

    return true;
}

} // namespace dbdd
