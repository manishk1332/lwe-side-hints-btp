#include "dbdd/hints/aposteriori_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include <cmath>

namespace dbdd {

// Paper Reference: Dachman-Soled et al. Section 3.3 / Equations (24)-(26)
// Mathematical Purpose: Implementation of AposterioriApproximateHint for DBDD.
// Directly conditions target prior distribution (mu, Sigma) along directional projection v_bar
// to match posterior parameters (mu_post, var_post), leaving the lattice basis Lambda strictly unchanged.

AposterioriApproximateHint::AposterioriApproximateHint(VectorZZ v, double mu_post, double var_post)
    : v_int_(std::move(v)),
      mu_post_(mu_post),
      var_post_(var_post),
      is_integer_(true),
      is_already_homogenized_(false) {
    if (var_post_ < 0.0) {
        throw InvalidParameterException(
            "AposterioriApproximateHint: posterior variance cannot be negative, got " +
            std::to_string(var_post_));
    }
    if (v_int_.empty()) {
        throw InvalidParameterException("AposterioriApproximateHint: hint vector cannot be empty");
    }
    v_double_ = VectorDouble(v_int_.size());
    for (size_t i = 0; i < v_int_.size(); ++i) {
        v_double_[i] = v_int_[i].to_double();
    }
}

AposterioriApproximateHint::AposterioriApproximateHint(VectorDouble v, double mu_post, double var_post)
    : v_double_(std::move(v)),
      mu_post_(mu_post),
      var_post_(var_post),
      is_integer_(false),
      is_already_homogenized_(false) {
    if (var_post_ < 0.0) {
        throw InvalidParameterException(
            "AposterioriApproximateHint: posterior variance cannot be negative, got " +
            std::to_string(var_post_));
    }
    if (v_double_.empty()) {
        throw InvalidParameterException("AposterioriApproximateHint: hint vector cannot be empty");
    }
    v_int_ = VectorZZ(v_double_.size());
    for (size_t i = 0; i < v_double_.size(); ++i) {
        v_int_[i] = BigInt(static_cast<long long>(std::round(v_double_[i])));
    }
}

AposterioriApproximateHint AposterioriApproximateHint::from_secret(VectorZZ v, double mu_post, double var_post) {
    return AposterioriApproximateHint(std::move(v), mu_post, var_post);
}

AposterioriApproximateHint AposterioriApproximateHint::from_secret(VectorDouble v, double mu_post, double var_post) {
    return AposterioriApproximateHint(std::move(v), mu_post, var_post);
}

AposterioriApproximateHint AposterioriApproximateHint::from_homogenized(VectorZZ v_bar, double mu_post, double var_post) {
    AposterioriApproximateHint hint(std::move(v_bar), mu_post, var_post);
    hint.is_already_homogenized_ = true;
    return hint;
}

AposterioriApproximateHint AposterioriApproximateHint::from_homogenized(VectorDouble v_bar, double mu_post, double var_post) {
    AposterioriApproximateHint hint(std::move(v_bar), mu_post, var_post);
    hint.is_already_homogenized_ = true;
    return hint;
}

const VectorZZ& AposterioriApproximateHint::hint_vector_int() const { return v_int_; }
const VectorDouble& AposterioriApproximateHint::hint_vector_double() const { return v_double_; }

std::string AposterioriApproximateHint::description() const {
    return "AposterioriApproximateHint(<s, v>, mu_post = " + std::to_string(mu_post_) +
           ", var_post = " + std::to_string(var_post_) + ")";
}

VectorZZ AposterioriApproximateHint::homogenize_int(size_t ambient_dim) const {
    if (is_already_homogenized_) {
        if (v_int_.size() != ambient_dim) {
            throw DimensionMismatchException(
                "AposterioriApproximateHint::homogenize_int: Already homogenized vector size (" +
                std::to_string(v_int_.size()) + ") does not match ambient dimension (" +
                std::to_string(ambient_dim) + ")");
        }
        return v_int_;
    }

    if (v_int_.size() == ambient_dim) {
        return v_int_;
    }

    if (v_int_.size() == ambient_dim - 1) {
        VectorZZ res(ambient_dim);
        for (size_t i = 0; i < ambient_dim - 1; ++i) {
            res[i] = v_int_[i];
        }
        res[ambient_dim - 1] = BigInt(0);
        return res;
    }

    if (v_int_.size() < ambient_dim - 1) {
        size_t n = v_int_.size();
        size_t m = ambient_dim - 1 - n;
        VectorZZ res(ambient_dim);
        for (size_t i = 0; i < m; ++i) {
            res[i] = BigInt(0);
        }
        for (size_t j = 0; j < n; ++j) {
            res[m + j] = v_int_[j];
        }
        res[ambient_dim - 1] = BigInt(0);
        return res;
    }

    throw DimensionMismatchException(
        "AposterioriApproximateHint::homogenize_int: Hint vector size (" + std::to_string(v_int_.size()) +
        ") exceeds ambient dimension (" + std::to_string(ambient_dim) + ")");
}

VectorDouble AposterioriApproximateHint::homogenize_double(size_t ambient_dim) const {
    if (is_already_homogenized_) {
        if (v_double_.size() != ambient_dim) {
            throw DimensionMismatchException(
                "AposterioriApproximateHint::homogenize_double: Already homogenized vector size (" +
                std::to_string(v_double_.size()) + ") does not match ambient dimension (" +
                std::to_string(ambient_dim) + ")");
        }
        return v_double_;
    }

    if (v_double_.size() == ambient_dim) {
        return v_double_;
    }

    if (v_double_.size() == ambient_dim - 1) {
        VectorDouble res(ambient_dim);
        for (size_t i = 0; i < ambient_dim - 1; ++i) {
            res[i] = v_double_[i];
        }
        res[ambient_dim - 1] = 0.0;
        return res;
    }

    if (v_double_.size() < ambient_dim - 1) {
        size_t n = v_double_.size();
        size_t m = ambient_dim - 1 - n;
        VectorDouble res(ambient_dim);
        for (size_t i = 0; i < m; ++i) {
            res[i] = 0.0;
        }
        for (size_t j = 0; j < n; ++j) {
            res[m + j] = v_double_[j];
        }
        res[ambient_dim - 1] = 0.0;
        return res;
    }

    throw DimensionMismatchException(
        "AposterioriApproximateHint::homogenize_double: Hint vector size (" +
        std::to_string(v_double_.size()) + ") exceeds ambient dimension (" +
        std::to_string(ambient_dim) + ")");
}

std::pair<VectorDouble, MatrixDouble> AposterioriApproximateHint::update_distribution(
    const VectorDouble& mean,
    const MatrixDouble& cov,
    const VectorDouble& v_bar,
    double mu_post,
    double var_post) {

    size_t d = mean.size();
    if (cov.rows() != d || cov.cols() != d) {
        throw DimensionMismatchException(
            "AposterioriApproximateHint::update_distribution: Covariance dimensions (" +
            std::to_string(cov.rows()) + "x" + std::to_string(cov.cols()) +
            ") do not match mean dimension (" + std::to_string(d) + ")");
    }
    if (v_bar.size() != d) {
        throw DimensionMismatchException(
            "AposterioriApproximateHint::update_distribution: Hint vector dimension (" +
            std::to_string(v_bar.size()) + ") does not match mean dimension (" +
            std::to_string(d) + ")");
    }
    if (var_post < 0.0) {
        throw InvalidParameterException(
            "AposterioriApproximateHint::update_distribution: Posterior variance cannot be negative, got " +
            std::to_string(var_post));
    }

    // w = v_bar * cov (row vector 1 x d)
    VectorDouble w(d);
    for (size_t j = 0; j < d; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < d; ++i) {
            sum += v_bar[i] * cov(i, j);
        }
        w[j] = sum;
    }

    // Prior directional variance along v_bar: sigma_prior^2 = v_bar * cov * v_bar^T = w * v_bar^T
    double sigma_prior_sq = 0.0;
    for (size_t j = 0; j < d; ++j) {
        sigma_prior_sq += w[j] * v_bar[j];
    }

    if (sigma_prior_sq <= 1e-15) {
        throw SingularMatrixException(
            "AposterioriApproximateHint::update_distribution: Prior variance along direction is degenerate or non-positive: " +
            std::to_string(sigma_prior_sq));
    }

    // Prior mean projection: mu_prior = <mean, v_bar>
    double mu_prior = 0.0;
    for (size_t j = 0; j < d; ++j) {
        mu_prior += mean[j] * v_bar[j];
    }

    // mu' = mu + ((mu_post - mu_prior) / sigma_prior^2) * w
    double scale_mu = (mu_post - mu_prior) / sigma_prior_sq;
    VectorDouble new_mean(d);
    for (size_t i = 0; i < d; ++i) {
        new_mean[i] = mean[i] + scale_mu * w[i];
    }

    // Delta_sigma^2 = sigma_prior^2 - var_post
    double delta_var = sigma_prior_sq - var_post;
    double scale_cov = delta_var / (sigma_prior_sq * sigma_prior_sq);

    // Sigma' = Sigma - (Delta_sigma^2 / (sigma_prior^2)^2) * (w^T * w)
    MatrixDouble new_cov(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            new_cov(i, j) = cov(i, j) - scale_cov * (w[i] * w[j]);
        }
    }

    // Enforce exact numerical symmetry on new_cov
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (new_cov(i, j) + new_cov(j, i));
            new_cov(i, j) = avg;
            new_cov(j, i) = avg;
        }
    }

    return {new_mean, new_cov};
}

void AposterioriApproximateHint::apply(DBDDInstance& instance) const {
    size_t amb_dim = instance.ambient_dimension();
    VectorDouble v_bar = homogenize_double(amb_dim);

    auto [new_mean, new_cov] = update_distribution(
        instance.mean(), instance.covariance(), v_bar, mu_post_, var_post_);

    // Zero lattice mutations: update only the target distribution
    instance.set_distribution(std::move(new_mean), std::move(new_cov));
}

} // namespace dbdd
