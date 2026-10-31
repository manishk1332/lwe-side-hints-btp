#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include <cmath>
#include <algorithm>

namespace dbdd {

// Paper Reference: Albrecht et al. Section 3.3 / Equations (20)-(23)
// Mathematical Purpose: Implementation of ApproximateHint for DBDD.
// Updates target prior distribution (mu, Sigma) via Gaussian conditioning on noisy linear observation,
// leaving the lattice basis Lambda strictly unchanged.

ApproximateHint::ApproximateHint(VectorZZ v, BigInt l, double variance)
    : v_int_(std::move(v)),
      l_int_(std::move(l)),
      variance_(variance),
      is_integer_(true),
      is_already_homogenized_(false) {
    if (variance_ <= 0.0) {
        throw InvalidParameterException(
            "ApproximateHint: noise variance must be positive, got " + std::to_string(variance_));
    }
    if (v_int_.empty()) {
        throw InvalidParameterException("ApproximateHint: hint vector cannot be empty");
    }
    v_double_ = VectorDouble(v_int_.size());
    for (size_t i = 0; i < v_int_.size(); ++i) {
        v_double_[i] = v_int_[i].to_double();
    }
    l_double_ = l_int_.to_double();
}

ApproximateHint::ApproximateHint(VectorDouble v, double l, double variance)
    : v_double_(std::move(v)),
      l_double_(l),
      variance_(variance),
      is_integer_(false),
      is_already_homogenized_(false) {
    if (variance_ <= 0.0) {
        throw InvalidParameterException(
            "ApproximateHint: noise variance must be positive, got " + std::to_string(variance_));
    }
    if (v_double_.empty()) {
        throw InvalidParameterException("ApproximateHint: hint vector cannot be empty");
    }
    v_int_ = VectorZZ(v_double_.size());
    for (size_t i = 0; i < v_double_.size(); ++i) {
        v_int_[i] = BigInt(static_cast<long long>(std::round(v_double_[i])));
    }
    l_int_ = BigInt(static_cast<long long>(std::round(l_double_)));
}

ApproximateHint ApproximateHint::from_secret(VectorZZ v, BigInt l, double variance) {
    return ApproximateHint(std::move(v), std::move(l), variance);
}

ApproximateHint ApproximateHint::from_secret(VectorDouble v, double l, double variance) {
    return ApproximateHint(std::move(v), l, variance);
}

ApproximateHint ApproximateHint::from_homogenized(VectorZZ v_bar, double variance) {
    ApproximateHint hint(std::move(v_bar), BigInt(0), variance);
    hint.is_already_homogenized_ = true;
    return hint;
}

ApproximateHint ApproximateHint::from_homogenized(VectorDouble v_bar, double variance) {
    ApproximateHint hint(std::move(v_bar), 0.0, variance);
    hint.is_already_homogenized_ = true;
    return hint;
}

const VectorZZ& ApproximateHint::hint_vector_int() const { return v_int_; }
const BigInt& ApproximateHint::hint_scalar_int() const { return l_int_; }
const VectorDouble& ApproximateHint::hint_vector_double() const { return v_double_; }
double ApproximateHint::hint_scalar_double() const { return l_double_; }

std::string ApproximateHint::description() const {
    return "ApproximateHint(<s, v> + e = " +
           (is_integer_ ? l_int_.to_string() : std::to_string(l_double_)) +
           ", var = " + std::to_string(variance_) + ")";
}

VectorZZ ApproximateHint::homogenize_int(size_t ambient_dim) const {
    if (is_already_homogenized_) {
        if (v_int_.size() != ambient_dim) {
            throw DimensionMismatchException(
                "ApproximateHint::homogenize_int: Already homogenized vector size (" +
                std::to_string(v_int_.size()) + ") does not match ambient dimension (" +
                std::to_string(ambient_dim) + ")");
        }
        return v_int_;
    }

    if (v_int_.size() == ambient_dim) {
        VectorZZ res = v_int_;
        res[ambient_dim - 1] -= l_int_;
        return res;
    }

    if (v_int_.size() == ambient_dim - 1) {
        VectorZZ res(ambient_dim);
        for (size_t i = 0; i < ambient_dim - 1; ++i) {
            res[i] = v_int_[i];
        }
        res[ambient_dim - 1] = -l_int_;
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
        res[ambient_dim - 1] = -l_int_;
        return res;
    }

    throw DimensionMismatchException(
        "ApproximateHint::homogenize_int: Hint vector size (" + std::to_string(v_int_.size()) +
        ") exceeds ambient dimension (" + std::to_string(ambient_dim) + ")");
}

VectorDouble ApproximateHint::homogenize_double(size_t ambient_dim) const {
    VectorZZ v_zz = homogenize_int(ambient_dim);
    VectorDouble res(ambient_dim);
    for (size_t i = 0; i < ambient_dim; ++i) {
        res[i] = v_zz[i].to_double();
    }
    return res;
}

std::pair<VectorDouble, MatrixDouble> ApproximateHint::update_distribution(
    const VectorDouble& mean,
    const MatrixDouble& cov,
    const VectorDouble& v_bar,
    double variance) {

    size_t d = mean.size();
    if (cov.rows() != d || cov.cols() != d) {
        throw DimensionMismatchException(
            "ApproximateHint::update_distribution: Covariance dimensions (" +
            std::to_string(cov.rows()) + "x" + std::to_string(cov.cols()) +
            ") do not match mean dimension (" + std::to_string(d) + ")");
    }
    if (v_bar.size() != d) {
        throw DimensionMismatchException(
            "ApproximateHint::update_distribution: Hint vector dimension (" +
            std::to_string(v_bar.size()) + ") does not match mean dimension (" +
            std::to_string(d) + ")");
    }
    if (variance <= 0.0) {
        throw InvalidParameterException(
            "ApproximateHint::update_distribution: Noise variance must be positive, got " +
            std::to_string(variance));
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

    // Directional variance along v_bar: v_bar * cov * v_bar^T = w * v_bar^T
    double var_v = 0.0;
    for (size_t j = 0; j < d; ++j) {
        var_v += w[j] * v_bar[j];
    }

    // denom = v_bar * cov * v_bar^T + sigma_e^2
    double denom = var_v + variance;

    // Inner product <mean, v_bar>
    double mean_dot_v = 0.0;
    for (size_t j = 0; j < d; ++j) {
        mean_dot_v += mean[j] * v_bar[j];
    }

    // mu' = mu - (mean_dot_v / denom) * w
    double alpha = mean_dot_v / denom;
    VectorDouble new_mean(d);
    for (size_t i = 0; i < d; ++i) {
        new_mean[i] = mean[i] - alpha * w[i];
    }

    // Sigma' = Sigma - (w^T * w) / denom
    MatrixDouble new_cov(d, d);
    double inv_denom = 1.0 / denom;
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            new_cov(i, j) = cov(i, j) - (w[i] * w[j]) * inv_denom;
        }
    }

    // Enforce exact symmetry on new_cov
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (new_cov(i, j) + new_cov(j, i));
            new_cov(i, j) = avg;
            new_cov(j, i) = avg;
        }
    }

    return {new_mean, new_cov};
}

void ApproximateHint::apply(DBDDInstance& instance) const {
    size_t amb_dim = instance.ambient_dimension();
    VectorDouble v_bar = homogenize_double(amb_dim);

    auto [new_mean, new_cov] = update_distribution(
        instance.mean(), instance.covariance(), v_bar, variance_);

    // Zero lattice mutations: update only the distribution
    instance.set_distribution(std::move(new_mean), std::move(new_cov));
}

} // namespace dbdd
