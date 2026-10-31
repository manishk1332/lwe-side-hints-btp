#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/aposteriori_hint.hpp"
#include "dbdd/hints/short_vector_hint.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <algorithm>

namespace dbdd {

// =============================================================================
// DBDDStateStatistics Implementation
// =============================================================================

DBDDStateStatistics::DBDDStateStatistics(
    size_t dim,
    size_t amb_dim,
    BigReal volume,
    VectorDouble mean,
    MatrixDouble covariance)
    : dim_(dim),
      amb_dim_(amb_dim),
      volume_(std::move(volume)),
      mean_(std::move(mean)),
      covariance_(std::move(covariance)) {
    if (mean_.size() != amb_dim_) {
        throw DimensionMismatchException(
            "DBDDStateStatistics: Mean size (" + std::to_string(mean_.size()) +
            ") does not match ambient dimension (" + std::to_string(amb_dim_) + ")");
    }
    if (covariance_.rows() != amb_dim_ || covariance_.cols() != amb_dim_) {
        throw DimensionMismatchException(
            "DBDDStateStatistics: Covariance dimensions (" + std::to_string(covariance_.rows()) +
            "x" + std::to_string(covariance_.cols()) + ") do not match ambient dimension (" +
            std::to_string(amb_dim_) + ")");
    }
    log_volume_ = volume_.log();
}

DBDDStateStatistics::DBDDStateStatistics(
    size_t dim,
    BigReal volume,
    VectorDouble mean,
    MatrixDouble covariance)
    : DBDDStateStatistics(dim, mean.size(), std::move(volume), std::move(mean), std::move(covariance)) {}

DBDDStateStatistics DBDDStateStatistics::from_dbdd(const DBDDInstance& dbdd) {
    return DBDDStateStatistics(
        dbdd.dimension(),
        dbdd.ambient_dimension(),
        dbdd.volume(),
        dbdd.mean(),
        dbdd.covariance());
}

void DBDDStateStatistics::set_volume(BigReal vol) {
    volume_ = std::move(vol);
    log_volume_ = volume_.log();
}

void DBDDStateStatistics::set_log_volume(BigReal log_vol) {
    log_volume_ = std::move(log_vol);
    volume_ = log_volume_.exp();
}

void DBDDStateStatistics::set_distribution(VectorDouble mean, MatrixDouble cov) {
    if (mean.size() != amb_dim_) {
        throw DimensionMismatchException("DBDDStateStatistics::set_distribution: Mean dimension mismatch");
    }
    if (cov.rows() != amb_dim_ || cov.cols() != amb_dim_) {
        throw DimensionMismatchException("DBDDStateStatistics::set_distribution: Covariance dimension mismatch");
    }
    mean_ = std::move(mean);
    covariance_ = std::move(cov);
}

void DBDDStateStatistics::integrate_perfect_hint(
    const VectorDouble& v,
    double l,
    double dual_norm) {
    if (v.size() == amb_dim_) {
        if (l == 0.0) {
            PerfectHint hint = PerfectHint::from_homogenized(v);
            integrate_perfect_hint(hint, dual_norm);
        } else {
            PerfectHint hint = PerfectHint::from_secret(v, l);
            integrate_perfect_hint(hint, dual_norm);
        }
    } else if (v.size() == amb_dim_ - 1) {
        PerfectHint hint = PerfectHint::from_secret(v, l);
        integrate_perfect_hint(hint, dual_norm);
    } else {
        throw DimensionMismatchException(
            "DBDDStateStatistics::integrate_perfect_hint: Hint vector size (" +
            std::to_string(v.size()) + ") incompatible with ambient dimension (" +
            std::to_string(amb_dim_) + ")");
    }
}

void DBDDStateStatistics::integrate_perfect_hint(const PerfectHint& hint, double dual_norm) {
    if (dual_norm <= 0.0) {
        throw InvalidParameterException("DBDDStateStatistics::integrate_perfect_hint: dual_norm must be > 0");
    }

    VectorDouble v_bar = hint.homogenize_double(amb_dim_);
    auto [new_mean, new_cov] = PerfectHint::update_distribution(mean_, covariance_, v_bar);
    mean_ = std::move(new_mean);
    covariance_ = std::move(new_cov);

    if (dim_ > 0) {
        dim_ -= 1;
    }

    if (std::abs(dual_norm - 1.0) > 1e-12) {
        BigReal norm_rr(dual_norm);
        volume_ = volume_ * norm_rr;
        log_volume_ = volume_.log();
    }
}

void DBDDStateStatistics::integrate_modular_hint(
    const VectorDouble& v,
    double l,
    double k) {
    if (k <= 0.0) {
        throw InvalidParameterException("DBDDStateStatistics::integrate_modular_hint: modulus k must be > 0");
    }
    if (v.size() == amb_dim_ && l == 0.0) {
        ModularHint hint = ModularHint::from_homogenized(v, k);
        integrate_modular_hint(hint);
    } else {
        ModularHint hint = ModularHint::from_secret(v, l, k);
        integrate_modular_hint(hint);
    }
}

void DBDDStateStatistics::integrate_modular_hint(const ModularHint& hint) {
    VectorDouble v_bar = hint.homogenize_double(amb_dim_);
    double var = 0.0;
    for (size_t i = 0; i < amb_dim_; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < amb_dim_; ++j) {
            row_sum += v_bar[j] * covariance_(j, i);
        }
        var += row_sum * v_bar[i];
    }

    bool smooth = ModularHint::check_smoothness(var, hint.modulus_double(), hint.smoothness_threshold());
    if (!smooth) {
        auto [new_mean, new_cov] = ModularHint::update_distribution(
            mean_, covariance_, v_bar, hint.modulus_double());
        mean_ = std::move(new_mean);
        covariance_ = std::move(new_cov);
    }

    BigReal k_rr(hint.modulus_double());
    volume_ = volume_ * k_rr;
    log_volume_ = volume_.log();
}

void DBDDStateStatistics::integrate_approximate_hint(
    const VectorDouble& v,
    double l,
    double sigma) {
    if (sigma <= 0.0) {
        throw InvalidParameterException("DBDDStateStatistics::integrate_approximate_hint: sigma must be > 0");
    }

    if (v.size() == amb_dim_) {
        if (l == 0.0) {
            ApproximateHint hint = ApproximateHint::from_homogenized(v, sigma * sigma);
            integrate_approximate_hint(hint);
        } else {
            ApproximateHint hint = ApproximateHint::from_secret(v, l, sigma * sigma);
            integrate_approximate_hint(hint);
        }
    } else if (v.size() == amb_dim_ - 1) {
        ApproximateHint hint = ApproximateHint::from_secret(v, l, sigma * sigma);
        integrate_approximate_hint(hint);
    } else {
        throw DimensionMismatchException(
            "DBDDStateStatistics::integrate_approximate_hint: Vector dimension mismatch");
    }
}

void DBDDStateStatistics::integrate_approximate_hint(const ApproximateHint& hint) {
    VectorDouble v_bar = hint.homogenize_double(amb_dim_);
    auto [new_mean, new_cov] = ApproximateHint::update_distribution(
        mean_, covariance_, v_bar, hint.noise_variance());
    mean_ = std::move(new_mean);
    covariance_ = std::move(new_cov);
}

void DBDDStateStatistics::integrate_aposteriori_hint(const AposterioriHint& hint) {
    VectorDouble v_bar = hint.homogenize_double(amb_dim_);
    auto [new_mean, new_cov] = AposterioriApproximateHint::update_distribution(
        mean_, covariance_, v_bar, hint.posterior_mean(), hint.posterior_variance());
    mean_ = std::move(new_mean);
    covariance_ = std::move(new_cov);
}

void DBDDStateStatistics::integrate_short_vector_hint(const VectorDouble& v) {
    if (v.size() != amb_dim_) {
        throw DimensionMismatchException("DBDDStateStatistics::integrate_short_vector_hint: Vector size mismatch");
    }

    double v_sq_norm = 0.0;
    for (size_t i = 0; i < amb_dim_; ++i) {
        v_sq_norm += v[i] * v[i];
    }
    if (v_sq_norm <= 1e-15) {
        throw InvalidParameterException("DBDDStateStatistics::integrate_short_vector_hint: Vector norm cannot be zero");
    }
    double v_norm = std::sqrt(v_sq_norm);

    // Dimension drops by 1
    if (dim_ > 0) {
        dim_ -= 1;
    }

    // Volume divides by ||v||
    BigReal norm_rr(v_norm);
    volume_ = volume_ / norm_rr;
    log_volume_ = volume_.log();

    // Project orthogonal to v: P = I - (v^T * v) / ||v||^2
    MatrixDouble P(amb_dim_, amb_dim_, 0.0);
    for (size_t i = 0; i < amb_dim_; ++i) {
        P(i, i) = 1.0;
        for (size_t j = 0; j < amb_dim_; ++j) {
            P(i, j) -= (v[i] * v[j]) / v_sq_norm;
        }
    }

    // mu' = mu * P
    VectorDouble new_mean(amb_dim_, 0.0);
    for (size_t j = 0; j < amb_dim_; ++j) {
        for (size_t i = 0; i < amb_dim_; ++i) {
            new_mean[j] += mean_[i] * P(i, j);
        }
    }
    mean_ = std::move(new_mean);

    // Sigma' = P * Sigma * P
    MatrixDouble temp(amb_dim_, amb_dim_, 0.0);
    for (size_t i = 0; i < amb_dim_; ++i) {
        for (size_t j = 0; j < amb_dim_; ++j) {
            for (size_t k = 0; k < amb_dim_; ++k) {
                temp(i, j) += P(i, k) * covariance_(k, j);
            }
        }
    }

    MatrixDouble new_cov(amb_dim_, amb_dim_, 0.0);
    for (size_t i = 0; i < amb_dim_; ++i) {
        for (size_t j = 0; j < amb_dim_; ++j) {
            for (size_t k = 0; k < amb_dim_; ++k) {
                new_cov(i, j) += temp(i, k) * P(k, j);
            }
        }
    }

    // Enforce symmetry
    for (size_t i = 0; i < amb_dim_; ++i) {
        for (size_t j = i + 1; j < amb_dim_; ++j) {
            double avg = 0.5 * (new_cov(i, j) + new_cov(j, i));
            new_cov(i, j) = avg;
            new_cov(j, i) = avg;
        }
    }
    covariance_ = std::move(new_cov);
}

void DBDDStateStatistics::integrate_short_vector_hint(const ShortVectorHint& hint) {
    if (hint.is_multi_vector()) {
        const auto& V_int = hint.hint_matrix();
        for (size_t row = 0; row < V_int.rows(); ++row) {
            VectorDouble v_double(amb_dim_);
            for (size_t j = 0; j < amb_dim_; ++j) {
                v_double[j] = V_int(row, j).to_double();
            }
            integrate_short_vector_hint(v_double);
        }
    } else {
        VectorDouble v_double(amb_dim_);
        const auto& v_int = hint.hint_vector();
        for (size_t i = 0; i < amb_dim_; ++i) {
            v_double[i] = v_int[i].to_double();
        }
        integrate_short_vector_hint(v_double);
    }
}

void DBDDStateStatistics::homogenize() {
    covariance_ = homogenized_covariance();
    for (size_t i = 0; i < amb_dim_; ++i) {
        mean_[i] = 0.0;
    }
}

bool DBDDStateStatistics::is_homogenized(double tol) const {
    double norm_sq = 0.0;
    for (size_t i = 0; i < amb_dim_; ++i) {
        norm_sq += mean_[i] * mean_[i];
    }
    return std::sqrt(norm_sq) <= tol;
}

MatrixDouble DBDDStateStatistics::homogenized_covariance() const {
    MatrixDouble cov_h = covariance_;
    for (size_t i = 0; i < amb_dim_; ++i) {
        for (size_t j = 0; j < amb_dim_; ++j) {
            cov_h(i, j) += mean_[i] * mean_[j];
        }
    }
    return cov_h;
}

double DBDDStateStatistics::isotropized_log_volume(double rank_tol) const {
    MatrixDouble cov_h = homogenized_covariance();
    return compute_isotropized_log_volume(log_volume_, cov_h, rank_tol).to_double();
}

BigReal DBDDStateStatistics::isotropized_volume(double rank_tol) const {
    MatrixDouble cov_h = homogenized_covariance();
    return compute_isotropized_volume(volume_, cov_h, rank_tol);
}

SecurityEstimate DBDDStateStatistics::estimate_attack_gsa(
    double beta_min,
    double beta_max,
    double rank_tol) const {
    double ln_vol_iso = isotropized_log_volume(rank_tol);
    return GSAIntersect::estimate(dim_, ln_vol_iso, beta_min, beta_max);
}

size_t DBDDStateStatistics::estimate_beta_gsa(double rank_tol) const {
    return estimate_attack_gsa(2.0, 2000.0, rank_tol).beta;
}

// =============================================================================
// DiagonalCovariance Implementation
// =============================================================================

DiagonalCovariance::DiagonalCovariance(
    size_t dim,
    size_t amb_dim,
    BigReal log_volume,
    std::vector<double> variances,
    std::vector<double> mean)
    : dim_(dim),
      amb_dim_(amb_dim),
      log_volume_(std::move(log_volume)),
      variances_(std::move(variances)),
      mean_(std::move(mean)) {
    if (variances_.size() != amb_dim_) {
        throw DimensionMismatchException(
            "DiagonalCovariance: Variances size (" + std::to_string(variances_.size()) +
            ") does not match ambient dimension (" + std::to_string(amb_dim_) + ")");
    }
    if (mean_.empty()) {
        mean_ = std::vector<double>(amb_dim_, 0.0);
    } else if (mean_.size() != amb_dim_) {
        throw DimensionMismatchException(
            "DiagonalCovariance: Mean size (" + std::to_string(mean_.size()) +
            ") does not match ambient dimension (" + std::to_string(amb_dim_) + ")");
    }
}

DiagonalCovariance DiagonalCovariance::from_state_statistics(
    const DBDDStateStatistics& stats,
    double off_diag_tol) {
    size_t m = stats.ambient_dimension();
    const auto& cov = stats.covariance();

    // Check that off-diagonal entries are effectively zero
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (i != j && std::abs(cov(i, j)) > off_diag_tol) {
                throw InvalidParameterException(
                    "DiagonalCovariance::from_state_statistics: Covariance is not diagonal at (" +
                    std::to_string(i) + ", " + std::to_string(j) + "): " + std::to_string(cov(i, j)));
            }
        }
    }

    std::vector<double> vars(m);
    for (size_t i = 0; i < m; ++i) {
        vars[i] = cov(i, i);
    }

    std::vector<double> mu(m);
    const auto& stats_mu = stats.mean();
    for (size_t i = 0; i < m; ++i) {
        mu[i] = stats_mu[i];
    }

    return DiagonalCovariance(stats.dimension(), m, stats.log_volume(), std::move(vars), std::move(mu));
}

DiagonalCovariance DiagonalCovariance::from_dbdd(
    const DBDDInstance& dbdd,
    double off_diag_tol) {
    return from_state_statistics(DBDDStateStatistics::from_dbdd(dbdd), off_diag_tol);
}

DiagonalCovariance DiagonalCovariance::from_lwe_parameters(
    size_t n,
    size_t m,
    double q,
    double sigma_e,
    double sigma_s) {
    size_t d = m + n + 1;
    double ln_vol = static_cast<double>(m) * std::log(q);

    std::vector<double> vars(d);
    for (size_t i = 0; i < m; ++i) {
        vars[i] = sigma_e * sigma_e;
    }
    for (size_t i = 0; i < n; ++i) {
        vars[m + i] = sigma_s * sigma_s;
    }
    vars[d - 1] = 0.0; // Homogeneous coordinate deterministic

    std::vector<double> mu(d, 0.0);
    mu[d - 1] = 1.0;

    return DiagonalCovariance(d, d, BigReal(ln_vol), std::move(vars), std::move(mu));
}

double DiagonalCovariance::variance(size_t i) const {
    if (i >= amb_dim_) {
        throw IndexOutOfBoundsException("DiagonalCovariance::variance: index out of bounds");
    }
    return variances_[i];
}

double DiagonalCovariance::mean(size_t i) const {
    if (i >= amb_dim_) {
        throw IndexOutOfBoundsException("DiagonalCovariance::mean: index out of bounds");
    }
    return mean_[i];
}

void DiagonalCovariance::integrate_coordinate_perfect_hint(size_t coord_idx, double l) {
    if (coord_idx >= amb_dim_) {
        throw IndexOutOfBoundsException("DiagonalCovariance::integrate_coordinate_perfect_hint: index out of bounds");
    }
    if (dim_ > 0) {
        dim_ -= 1;
    }
    variances_[coord_idx] = 0.0;
    mean_[coord_idx] = l;
}

void DiagonalCovariance::integrate_coordinate_modular_hint(
    size_t coord_idx,
    double l,
    double k) {
    if (coord_idx >= amb_dim_) {
        throw IndexOutOfBoundsException("DiagonalCovariance::integrate_coordinate_modular_hint: index out of bounds");
    }
    if (k <= 0.0) {
        throw InvalidParameterException("DiagonalCovariance::integrate_coordinate_modular_hint: k must be > 0");
    }
    log_volume_ = log_volume_ + BigReal(std::log(k));

    if (std::sqrt(variances_[coord_idx]) < k) {
        auto [new_mu, new_var] = ModularHint::discrete_gaussian_1d_coset(
            mean_[coord_idx], variances_[coord_idx], k, l);
        mean_[coord_idx] = new_mu;
        variances_[coord_idx] = new_var;
    }
}

void DiagonalCovariance::integrate_coordinate_approximate_hint(
    size_t coord_idx,
    double l,
    double sigma) {
    if (coord_idx >= amb_dim_) {
        throw IndexOutOfBoundsException("DiagonalCovariance::integrate_coordinate_approximate_hint: index out of bounds");
    }
    if (sigma <= 0.0) {
        throw InvalidParameterException("DiagonalCovariance::integrate_coordinate_approximate_hint: sigma must be > 0");
    }

    double prior_var = variances_[coord_idx];
    double noise_var = sigma * sigma;
    double denom = prior_var + noise_var;

    variances_[coord_idx] = (prior_var * noise_var) / denom;
    mean_[coord_idx] += (prior_var / denom) * (l - mean_[coord_idx]);
}

void DiagonalCovariance::homogenize() {
    for (size_t i = 0; i < amb_dim_; ++i) {
        variances_[i] += mean_[i] * mean_[i];
        mean_[i] = 0.0;
    }
}

double DiagonalCovariance::isotropized_log_volume(double tol) const {
    double ln_prod_active = 0.0;
    double mu_z_sq_norm = 0.0;
    double sum_mu_sq_over_var = 0.0;

    for (size_t i = 0; i < amb_dim_; ++i) {
        if (variances_[i] > tol) {
            ln_prod_active += std::log(variances_[i]);
            sum_mu_sq_over_var += (mean_[i] * mean_[i]) / variances_[i];
        } else {
            mu_z_sq_norm += mean_[i] * mean_[i];
        }
    }

    double ln_rdet = 0.0;
    if (mu_z_sq_norm > tol) {
        ln_rdet = std::log(mu_z_sq_norm) + ln_prod_active;
    } else {
        ln_rdet = std::log(1.0 + sum_mu_sq_over_var) + ln_prod_active;
    }

    return log_volume_.to_double() - 0.5 * ln_rdet;
}

SecurityEstimate DiagonalCovariance::estimate_attack_gsa(
    double beta_min,
    double beta_max) const {
    double ln_vol_iso = isotropized_log_volume();
    return GSAIntersect::estimate(dim_, ln_vol_iso, beta_min, beta_max);
}

size_t DiagonalCovariance::estimate_beta_gsa() const {
    return estimate_attack_gsa().beta;
}

} // namespace dbdd
