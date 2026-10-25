#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lattice/dual.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include <cmath>
#include <algorithm>

namespace dbdd {

// Paper Reference: Albrecht et al. Section 3.2 / Equations (15)-(19)
// Mathematical Purpose: Implementation of ModularHint for DBDD.
// Coordinates dual densification, exact dependency elimination, primal sparsification,
// and 1D discrete Gaussian distribution updates on periodic cosets.

ModularHint::ModularHint(VectorZZ v, BigInt l, BigInt k, double smoothness_threshold)
    : v_int_(std::move(v)),
      l_int_(std::move(l)),
      k_int_(std::move(k)),
      smoothness_threshold_(smoothness_threshold),
      is_integer_(true),
      is_already_homogenized_(false) {
    if (k_int_ < BigInt(2)) {
        throw InvalidParameterException("ModularHint: modulus k must be >= 2, got " + k_int_.to_string());
    }
    if (v_int_.empty()) {
        throw InvalidParameterException("ModularHint: hint vector cannot be empty");
    }
    v_double_ = VectorDouble(v_int_.size());
    for (size_t i = 0; i < v_int_.size(); ++i) {
        v_double_[i] = v_int_[i].to_double();
    }
    l_double_ = l_int_.to_double();
    k_double_ = k_int_.to_double();
}

ModularHint::ModularHint(VectorDouble v, double l, double k, double smoothness_threshold)
    : v_double_(std::move(v)),
      l_double_(l),
      k_double_(k),
      smoothness_threshold_(smoothness_threshold),
      is_integer_(false),
      is_already_homogenized_(false) {
    if (k_double_ < 1.999) {
        throw InvalidParameterException("ModularHint: modulus k must be >= 2, got " + std::to_string(k_double_));
    }
    if (v_double_.empty()) {
        throw InvalidParameterException("ModularHint: hint vector cannot be empty");
    }
    v_int_ = VectorZZ(v_double_.size());
    for (size_t i = 0; i < v_double_.size(); ++i) {
        v_int_[i] = BigInt(static_cast<long long>(std::round(v_double_[i])));
    }
    l_int_ = BigInt(static_cast<long long>(std::round(l_double_)));
    k_int_ = BigInt(static_cast<long long>(std::round(k_double_)));
}

ModularHint ModularHint::from_secret(VectorZZ v, BigInt l, BigInt k, double smoothness_threshold) {
    return ModularHint(std::move(v), std::move(l), std::move(k), smoothness_threshold);
}

ModularHint ModularHint::from_secret(VectorDouble v, double l, double k, double smoothness_threshold) {
    return ModularHint(std::move(v), l, k, smoothness_threshold);
}

ModularHint ModularHint::from_homogenized(VectorZZ v_bar, BigInt k, double smoothness_threshold) {
    ModularHint hint(std::move(v_bar), BigInt(0), std::move(k), smoothness_threshold);
    hint.is_already_homogenized_ = true;
    return hint;
}

ModularHint ModularHint::from_homogenized(VectorDouble v_bar, double k, double smoothness_threshold) {
    ModularHint hint(std::move(v_bar), 0.0, k, smoothness_threshold);
    hint.is_already_homogenized_ = true;
    return hint;
}

const VectorZZ& ModularHint::hint_vector_int() const { return v_int_; }
const BigInt& ModularHint::hint_scalar_int() const { return l_int_; }
const BigInt& ModularHint::modulus_int() const { return k_int_; }
const VectorDouble& ModularHint::hint_vector_double() const { return v_double_; }
double ModularHint::hint_scalar_double() const { return l_double_; }
double ModularHint::modulus_double() const { return k_double_; }

std::string ModularHint::description() const {
    return "ModularHint(<s, v> = " + (is_integer_ ? l_int_.to_string() : std::to_string(l_double_)) +
           " (mod " + (is_integer_ ? k_int_.to_string() : std::to_string(k_double_)) + "))";
}

VectorZZ ModularHint::homogenize_int(size_t ambient_dim) const {
    if (is_already_homogenized_) {
        if (v_int_.size() != ambient_dim) {
            throw DimensionMismatchException(
                "ModularHint::homogenize_int: Already homogenized vector size (" +
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
        "ModularHint::homogenize_int: Hint vector size (" + std::to_string(v_int_.size()) +
        ") exceeds ambient dimension (" + std::to_string(ambient_dim) + ")");
}

VectorDouble ModularHint::homogenize_double(size_t ambient_dim) const {
    VectorZZ v_zz = homogenize_int(ambient_dim);
    VectorDouble res(ambient_dim);
    for (size_t i = 0; i < ambient_dim; ++i) {
        res[i] = v_zz[i].to_double();
    }
    return res;
}

bool ModularHint::check_smoothness(double directional_variance, double k, double threshold) {
    if (directional_variance <= 0.0 || k <= 0.0) {
        return false;
    }
    double sigma_v = std::sqrt(directional_variance);
    return sigma_v >= threshold * k;
}

bool ModularHint::is_smooth(const DBDDInstance& instance) const {
    VectorDouble v_bar = homogenize_double(instance.ambient_dimension());
    const MatrixDouble& cov = instance.covariance();
    size_t d = v_bar.size();

    double var = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < d; ++j) {
            row_sum += v_bar[j] * cov(j, i);
        }
        var += row_sum * v_bar[i];
    }

    return check_smoothness(var, k_double_, smoothness_threshold_);
}

std::pair<double, double> ModularHint::discrete_gaussian_1d_coset(
    double mu,
    double sigma_sq,
    double k,
    double l) {

    if (k <= 0.0) {
        throw InvalidParameterException("ModularHint::discrete_gaussian_1d_coset: k must be positive");
    }

    // Deterministic prior limit
    if (sigma_sq <= 1e-12) {
        double j_star_d = std::round((mu - l) / k);
        double y_star = l + j_star_d * k;
        if (std::abs(y_star - mu) > 1e-6) {
            throw InvalidParameterException(
                "ModularHint::discrete_gaussian_1d_coset: Hint contradicts deterministic prior (<mu, v_bar> = " +
                std::to_string(mu) + ", coset distance = " + std::to_string(std::abs(y_star - mu)) + ")");
        }
        return {y_star, 0.0};
    }

    double sigma = std::sqrt(sigma_sq);

    // Deep smooth limit shortcut (variance unchanged, mean = mu)
    if (sigma >= 4.0 * k) {
        return {mu, sigma_sq};
    }

    // Coset summation centered at closest integer index j_0
    long long j_0 = static_cast<long long>(std::round((mu - l) / k));
    double R = 8.0 * sigma;
    long long j_min = static_cast<long long>(std::floor((mu - l - R) / k));
    long long j_max = static_cast<long long>(std::ceil((mu - l + R) / k));

    // Ensure adequate window around j_0
    j_min = std::min(j_min, j_0 - 5);
    j_max = std::max(j_max, j_0 + 5);

    double delta_0 = (l + static_cast<double>(j_0) * k) - mu;
    double delta_0_sq = delta_0 * delta_0;

    double z0 = 0.0;
    double z1 = 0.0;
    double z2 = 0.0;

    for (long long j = j_min; j <= j_max; ++j) {
        double y_j = l + static_cast<double>(j) * k;
        double diff = y_j - mu;
        double log_w = -(diff * diff - delta_0_sq) / (2.0 * sigma_sq);
        if (log_w < -50.0) {
            continue; // Numerically negligible
        }
        double w = std::exp(log_w);
        z0 += w;
        z1 += y_j * w;
        z2 += y_j * y_j * w;
    }

    if (z0 <= 0.0) {
        return {mu, sigma_sq};
    }

    double mu_scalar = z1 / z0;
    double second_moment = z2 / z0;
    double var_scalar = second_moment - mu_scalar * mu_scalar;

    // Variance monotonicity: conditional variance along direction cannot exceed prior variance
    var_scalar = std::max(0.0, std::min(var_scalar, sigma_sq));

    return {mu_scalar, var_scalar};
}

std::pair<VectorDouble, MatrixDouble> ModularHint::update_distribution(
    const VectorDouble& mean,
    const MatrixDouble& cov,
    const VectorDouble& v_bar,
    double k) {

    size_t d = mean.size();
    if (cov.rows() != d || cov.cols() != d) {
        throw DimensionMismatchException(
            "ModularHint::update_distribution: Covariance dimensions (" +
            std::to_string(cov.rows()) + "x" + std::to_string(cov.cols()) +
            ") do not match mean dimension (" + std::to_string(d) + ")");
    }
    if (v_bar.size() != d) {
        throw DimensionMismatchException(
            "ModularHint::update_distribution: Hint vector dimension (" +
            std::to_string(v_bar.size()) + ") does not match mean dimension (" +
            std::to_string(d) + ")");
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

    // sigma_Y^2 = v_bar * cov * v_bar^T = w * v_bar^T
    double sigma_Y_sq = 0.0;
    for (size_t j = 0; j < d; ++j) {
        sigma_Y_sq += w[j] * v_bar[j];
    }

    // Inner product <mean, v_bar>
    double mu_Y = 0.0;
    for (size_t j = 0; j < d; ++j) {
        mu_Y += mean[j] * v_bar[j];
    }

    if (sigma_Y_sq <= 1e-12) {
        double j_star_d = std::round(mu_Y / k);
        if (std::abs(mu_Y - j_star_d * k) > 1e-6) {
            throw InvalidParameterException(
                "ModularHint::update_distribution: Hint contradicts deterministic prior (<mu, v_bar> = " +
                std::to_string(mu_Y) + " is not a multiple of k = " + std::to_string(k) + ")");
        }
        return {mean, cov};
    }

    auto [mu_scalar, sigma_scalar_sq] = discrete_gaussian_1d_coset(mu_Y, sigma_Y_sq, k, 0.0);

    double delta_mu = mu_scalar - mu_Y;
    double delta_var = sigma_Y_sq - sigma_scalar_sq;
    if (delta_var < 0.0) {
        delta_var = 0.0;
    }

    // mu' = mu + (delta_mu / sigma_Y^2) * w
    VectorDouble new_mean(d);
    double alpha = delta_mu / sigma_Y_sq;
    for (size_t i = 0; i < d; ++i) {
        new_mean[i] = mean[i] + alpha * w[i];
    }

    // Sigma' = Sigma - (delta_var / (sigma_Y^2)^2) * (w^T * w)
    MatrixDouble new_cov(d, d);
    double beta = delta_var / (sigma_Y_sq * sigma_Y_sq);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            new_cov(i, j) = cov(i, j) - beta * (w[i] * w[j]);
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

BasisZZ ModularHint::sparsify_lattice(
    const BasisZZ& basis,
    const VectorZZ& v_bar,
    const BigInt& k) {

    const MatrixZZ& B = basis.matrix();
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();

    if (d == 0) {
        return basis;
    }

    // u_i = <b_i, v_bar>
    VectorZZ u = v_bar * B.transpose();

    // g = gcd(|u_1|, ..., |u_d|, k)
    BigInt g = k;
    for (size_t i = 0; i < d; ++i) {
        g = BigInt::gcd(g, u[i].abs());
        if (g == BigInt(1)) {
            break;
        }
    }

    // If every basis vector satisfies <b_i, v_bar> == 0 (mod k), lattice is already sparsified
    if (g == k) {
        return basis;
    }

    // Dual basis D in Q^{d x m} satisfying B * D^T = I_d
    MatrixQQ D = compute_dual_basis(B);

    // v_bar_V / k = (u / k) * D in Q^m
    VectorQQ u_q = u.cast<BigRational>();
    BigRational k_q(k);
    VectorQQ u_over_k = u_q / k_q;
    VectorQQ w = u_over_k * D; // (1 x m)

    // Form integer dependency vector c = (-u_1, ..., -u_d, k) in Z^{d+1}
    VectorZZ c(d + 1);
    for (size_t i = 0; i < d; ++i) {
        c[i] = -u[i];
    }
    c[d] = k;

    // Primitive dependency vector a = c / g in Z^{d+1}
    VectorZZ a(d + 1);
    for (size_t i = 0; i < d + 1; ++i) {
        a[i] = c[i] / g;
    }

    // Unimodular completion of a to U in GL_{d+1}(Z) with last row equal to a
    MatrixZZ U = PerfectHint::complete_primitive_vector_unimodular(a);

    // Augmented dual matrix D_aug of size (d+1) x m
    MatrixQQ D_aug(d + 1, m);
    for (size_t i = 0; i < d; ++i) {
        D_aug.set_row(i, D.row(i));
    }
    D_aug.set_row(d, w);

    // D_unimod = U * D_aug
    MatrixQQ D_unimod = U.cast<BigRational>() * D_aug;

    // Extract first d rows as full-rank dual basis D'_raw
    MatrixQQ D_raw(d, m);
    for (size_t i = 0; i < d; ++i) {
        D_raw.set_row(i, D_unimod.row(i));
    }

    // Dual LLL reduction
    auto dual_res = lll_exact(BasisQQ(D_raw, false));

    // Primal inversion: B'_raw = primal_from_dual(D'_red)
    BasisQQ primal_q = primal_from_dual(dual_res.basis);

    // Convert exact rationals to integers
    MatrixZZ primal_z(d, m);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < m; ++j) {
            const BigRational& val = primal_q.matrix()(i, j);
            if (val.denominator() != BigInt(1)) {
                throw NumericalPrecisionException(
                    "ModularHint::sparsify_lattice: Primal entry (" + std::to_string(i) + ", " +
                    std::to_string(j) + ") is not an integer: " + val.to_string());
            }
            primal_z(i, j) = val.numerator();
        }
    }

    // Primal LLL reduction
    auto primal_res = lll_exact(BasisZZ(std::move(primal_z), false));
    return primal_res.basis;
}

LatticeZZ ModularHint::update_lattice(
    const LatticeZZ& lattice,
    const VectorZZ& v_bar,
    const BigInt& k) {

    const MatrixZZ& B = lattice.basis().matrix();
    size_t d = lattice.dimension();

    VectorZZ u = v_bar * B.transpose();
    BigInt g = k;
    for (size_t i = 0; i < d; ++i) {
        g = BigInt::gcd(g, u[i].abs());
        if (g == BigInt(1)) {
            break;
        }
    }

    if (g == k) {
        return lattice;
    }

    BasisZZ new_basis = sparsify_lattice(lattice.basis(), v_bar, k);
    BigInt index = k / g;
    BigReal new_vol = lattice.volume() * BigReal(index);
    return LatticeZZ(std::move(new_basis), new_vol);
}

void ModularHint::apply(DBDDInstance& instance) const {
    size_t amb_dim = instance.ambient_dimension();
    VectorZZ v_bar_int = homogenize_int(amb_dim);
    VectorDouble v_bar_double = homogenize_double(amb_dim);

    // Ground-truth target verification
    if (instance.has_target()) {
        const auto& target = instance.target();
        BigInt inner = target.dot(v_bar_int);
        BigInt rem = inner % k_int_;
        if (rem < BigInt(0)) {
            rem += k_int_;
        }
        if (rem != BigInt(0)) {
            throw InvalidParameterException(
                "ModularHint::apply: Ground truth target vector does not satisfy modular hint! <s_bar, v_bar> = " +
                inner.to_string() + " != 0 (mod " + k_int_.to_string() + ")");
        }
    }

    // Determine smooth vs non-smooth regime
    bool smooth = is_smooth(instance);
    VectorDouble new_mean;
    MatrixDouble new_cov;

    if (smooth) {
        new_mean = instance.mean();
        new_cov = instance.covariance();
    } else {
        auto [m_up, c_up] = update_distribution(
            instance.mean(), instance.covariance(), v_bar_double, k_double_);
        new_mean = std::move(m_up);
        new_cov = std::move(c_up);
    }

    // Sparsify lattice
    LatticeZZ new_lattice = update_lattice(instance.lattice(), v_bar_int, k_int_);

    // Reconstruct target coordinates in new basis
    std::optional<VectorZZ> new_target_coords = std::nullopt;
    if (instance.has_target()) {
        const auto& target = instance.target();
        MatrixQQ D_prime = compute_dual_basis(new_lattice.basis().matrix());
        VectorQQ target_q = target.cast<BigRational>();
        VectorQQ x_q = target_q * D_prime.transpose();
        VectorZZ x_new(x_q.size());
        for (size_t i = 0; i < x_q.size(); ++i) {
            if (x_q[i].denominator() != BigInt(1)) {
                throw NumericalPrecisionException(
                    "ModularHint::apply: Target coordinates in sparsified basis are not integers: " +
                    x_q[i].to_string());
            }
            x_new[i] = x_q[i].numerator();
        }
        new_target_coords = std::move(x_new);
    }

    VectorZZ saved_target;
    bool had_target = instance.has_target();
    if (had_target) {
        saved_target = instance.target();
    }

    instance.set_lattice(std::move(new_lattice));
    instance.set_distribution(std::move(new_mean), std::move(new_cov));
    if (had_target) {
        instance.set_target(std::move(saved_target), std::move(new_target_coords));
    }
}

} // namespace dbdd
