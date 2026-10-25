// Paper Reference: Albrecht et al. Section 2, Section 3 / docs/mathematical_conventions.md
// Mathematical Purpose: Implementation of DBDDInstance, Kannan embedding integration,
// target vector containment and chi-squared validation, and future hint/transformation stubs.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/aposteriori_hint.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace dbdd {

namespace {

// Peter Acklam's algorithm for standard normal quantile function Phi^{-1}(p)
double normal_quantile(double p) {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    const double a1 = -3.969683028665376e+01;
    const double a2 =  2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 =  1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 =  2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 =  1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 =  6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 =  4.374664141464968e+00;
    const double c6 =  2.938163982698783e+00;

    const double d1 =  7.784695709041462e-03;
    const double d2 =  3.224671290700398e-01;
    const double d3 =  2.445134137142996e+00;
    const double d4 =  3.754408661907416e+00;

    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    if (p < p_low) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p > p_high) {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    double q = p - 0.5;
    double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

} // namespace

DBDDInstance DBDDInstance::from_kannan(const KannanEmbedding& kannan) {
    std::optional<VectorZZ> target = std::nullopt;
    std::optional<VectorZZ> coords = std::nullopt;

    if (kannan.has_target()) {
        target = kannan.target();
        coords = kannan.target_coordinates();
    }

    return DBDDInstance(
        kannan.lattice(),
        kannan.mean(),
        kannan.covariance(),
        std::move(target),
        std::move(coords));
}

DBDDInstance::DBDDInstance(
    LatticeZZ lattice,
    VectorDouble mean,
    MatrixDouble covariance,
    std::optional<VectorZZ> target,
    std::optional<VectorZZ> target_coords)
    : lattice_(std::move(lattice)),
      ellipsoid_(std::move(mean), std::move(covariance)),
      target_(std::move(target)),
      target_coords_(std::move(target_coords)) {
    size_t m = lattice_.ambient_dimension();
    if (ellipsoid_.dimension() != m) {
        throw DimensionMismatchException(
            "DBDDInstance: Ellipsoid dimension (" + std::to_string(ellipsoid_.dimension()) +
            ") does not match lattice ambient dimension (" + std::to_string(m) + ")");
    }
    if (target_.has_value() && target_->size() != m) {
        throw DimensionMismatchException(
            "DBDDInstance: Target vector dimension (" + std::to_string(target_->size()) +
            ") does not match lattice ambient dimension (" + std::to_string(m) + ")");
    }
    if (target_coords_.has_value() && target_coords_->size() != lattice_.dimension()) {
        throw DimensionMismatchException(
            "DBDDInstance: Target coordinates dimension (" + std::to_string(target_coords_->size()) +
            ") does not match lattice dimension (" + std::to_string(lattice_.dimension()) + ")");
    }
}

DBDDInstance::DBDDInstance(
    BasisZZ basis,
    VectorDouble mean,
    MatrixDouble covariance,
    std::optional<VectorZZ> target,
    std::optional<VectorZZ> target_coords)
    : DBDDInstance(
          LatticeZZ(std::move(basis)),
          std::move(mean),
          std::move(covariance),
          std::move(target),
          std::move(target_coords)) {}

DBDDInstance::DBDDInstance(
    MatrixZZ basis_matrix,
    VectorDouble mean,
    MatrixDouble covariance,
    std::optional<VectorZZ> target,
    std::optional<VectorZZ> target_coords)
    : DBDDInstance(
          LatticeZZ(std::move(basis_matrix)),
          std::move(mean),
          std::move(covariance),
          std::move(target),
          std::move(target_coords)) {}

const VectorZZ& DBDDInstance::target() const {
    if (!target_.has_value()) {
        throw InvalidParameterException("DBDDInstance does not contain a ground-truth target vector");
    }
    return *target_;
}

VectorDouble DBDDInstance::target_double() const {
    if (!target_.has_value()) {
        throw InvalidParameterException("DBDDInstance does not contain a ground-truth target vector");
    }
    const auto& s = *target_;
    VectorDouble s_d(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        s_d[i] = s[i].to_double();
    }
    return s_d;
}

const VectorZZ& DBDDInstance::target_coordinates() const {
    if (!target_coords_.has_value()) {
        throw InvalidParameterException("DBDDInstance does not contain ground-truth target coordinates");
    }
    return *target_coords_;
}

void DBDDInstance::set_target(VectorZZ target, std::optional<VectorZZ> target_coords) {
    if (target.size() != ambient_dimension()) {
        throw DimensionMismatchException(
            "DBDDInstance::set_target: Target dimension (" + std::to_string(target.size()) +
            ") does not match ambient dimension (" + std::to_string(ambient_dimension()) + ")");
    }
    if (target_coords.has_value() && target_coords->size() != dimension()) {
        throw DimensionMismatchException(
            "DBDDInstance::set_target: Target coordinates dimension (" + std::to_string(target_coords->size()) +
            ") does not match lattice dimension (" + std::to_string(dimension()) + ")");
    }
    target_ = std::move(target);
    target_coords_ = std::move(target_coords);
}

bool DBDDInstance::target_in_lattice() const {
    if (!target_.has_value()) {
        throw InvalidParameterException("DBDDInstance does not contain a ground-truth target vector");
    }
    if (target_coords_.has_value()) {
        VectorZZ actual = (*target_coords_) * lattice_.basis().matrix();
        return actual == (*target_);
    }
    return lattice_.contains(*target_);
}

double DBDDInstance::target_mahalanobis_distance_squared() const {
    return ellipsoid_.mahalanobis_distance_squared(target_double());
}

bool DBDDInstance::target_in_ellipsoid(double threshold) const {
    return ellipsoid_.contains(target_double(), threshold, /*check_support=*/true);
}

double DBDDInstance::chi2_threshold(double confidence) const {
    size_t k = ellipsoid_.rank();
    if (k == 0) return 0.0;

    double kd = static_cast<double>(k);
    double z = normal_quantile(confidence);

    // Wilson-Hilferty transformation: chi^2_p(k) ~ k * (1 - 2/(9k) + z * sqrt(2/(9k)))^3
    double factor = 1.0 - (2.0 / (9.0 * kd)) + z * std::sqrt(2.0 / (9.0 * kd));
    if (factor <= 0.0) {
        return kd + 4.0 * std::sqrt(2.0 * kd);
    }
    double quant = kd * factor * factor * factor;
    return std::max(quant, kd);
}

bool DBDDInstance::verify_target(double confidence) const {
    if (!has_target()) {
        return false;
    }
    if (!target_in_lattice()) {
        return false;
    }
    double thresh = chi2_threshold(confidence);
    return target_in_ellipsoid(thresh);
}

void DBDDInstance::set_lattice(LatticeZZ new_lattice) {
    if (new_lattice.ambient_dimension() != ellipsoid_.dimension()) {
        throw DimensionMismatchException(
            "DBDDInstance::set_lattice: New lattice ambient dimension (" +
            std::to_string(new_lattice.ambient_dimension()) +
            ") does not match ellipsoid dimension (" + std::to_string(ellipsoid_.dimension()) + ")");
    }
    lattice_ = std::move(new_lattice);
}

void DBDDInstance::set_distribution(VectorDouble new_mean, MatrixDouble new_covariance) {
    if (new_mean.size() != ambient_dimension()) {
        throw DimensionMismatchException(
            "DBDDInstance::set_distribution: Mean dimension (" + std::to_string(new_mean.size()) +
            ") does not match ambient dimension (" + std::to_string(ambient_dimension()) + ")");
    }
    ellipsoid_ = Ellipsoid(std::move(new_mean), std::move(new_covariance));
}

void DBDDInstance::integrate_hint(const Hint& hint) {
    hint.apply(*this);
}

// =============================================================================
// Structured Extension Points for Future Phases
// =============================================================================

void DBDDInstance::integrate_perfect_hint(const VectorZZ& v, const BigInt& l) {
    PerfectHint hint(v, l);
    integrate_hint(hint);
}

void DBDDInstance::integrate_perfect_hint(const VectorDouble& v, double l) {
    PerfectHint hint(v, l);
    integrate_hint(hint);
}

void DBDDInstance::integrate_modular_hint(const VectorZZ& v, const BigInt& l, const BigInt& k) {
    ModularHint hint(v, l, k);
    integrate_hint(hint);
}

void DBDDInstance::integrate_modular_hint(const VectorDouble& v, double l, double k) {
    ModularHint hint(v, l, k);
    integrate_hint(hint);
}

void DBDDInstance::integrate_approximate_hint(const VectorDouble& v, double l, double variance) {
    ApproximateHint hint(v, l, variance);
    integrate_hint(hint);
}

void DBDDInstance::integrate_approximate_hint(const VectorZZ& v, const BigInt& l, double variance) {
    ApproximateHint hint(v, l, variance);
    integrate_hint(hint);
}

void DBDDInstance::integrate_aposteriori_hint(const VectorDouble& v, double mu_post, double var_post) {
    AposterioriApproximateHint hint(v, mu_post, var_post);
    integrate_hint(hint);
}

void DBDDInstance::integrate_aposteriori_hint(const VectorZZ& v, double mu_post, double var_post) {
    AposterioriApproximateHint hint(v, mu_post, var_post);
    integrate_hint(hint);
}

void DBDDInstance::integrate_short_vector_hint(const VectorZZ& /*v*/) {
    throw DBDDException("integrate_short_vector_hint is scheduled for implementation in Phase 10");
}

void DBDDInstance::homogenize() {
    throw DBDDException("homogenize is scheduled for implementation in Phase 11");
}

void DBDDInstance::isotropize() {
    throw DBDDException("isotropize is scheduled for implementation in Phase 12");
}

BasisZZ DBDDInstance::to_usvp() const {
    throw DBDDException("to_usvp is scheduled for implementation in Phase 12");
}

} // namespace dbdd
