#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lattice/dual.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include <cmath>
#include <algorithm>

namespace dbdd {

// Paper Reference: Albrecht et al. Section 3.1 / Equations (12), (13)
// Mathematical Purpose: Implementation of PerfectHint for DBDD.
// Coordinates both exact lattice hyperplane restriction and conditional Gaussian distribution updates.

PerfectHint::PerfectHint(VectorZZ v, BigInt l)
    : v_int_(std::move(v)), l_int_(std::move(l)), is_integer_(true), is_already_homogenized_(false) {
    if (v_int_.empty()) {
        throw InvalidParameterException("PerfectHint: hint vector cannot be empty");
    }
    v_double_ = VectorDouble(v_int_.size());
    for (size_t i = 0; i < v_int_.size(); ++i) {
        v_double_[i] = v_int_[i].to_double();
    }
    l_double_ = l_int_.to_double();
}

PerfectHint::PerfectHint(VectorDouble v, double l)
    : v_double_(std::move(v)), l_double_(l), is_integer_(false), is_already_homogenized_(false) {
    if (v_double_.empty()) {
        throw InvalidParameterException("PerfectHint: hint vector cannot be empty");
    }
    v_int_ = VectorZZ(v_double_.size());
    for (size_t i = 0; i < v_double_.size(); ++i) {
        v_int_[i] = BigInt(static_cast<long long>(std::round(v_double_[i])));
    }
    l_int_ = BigInt(static_cast<long long>(std::round(l_double_)));
}

PerfectHint PerfectHint::from_secret(VectorZZ v, BigInt l) {
    return PerfectHint(std::move(v), std::move(l));
}

PerfectHint PerfectHint::from_secret(VectorDouble v, double l) {
    return PerfectHint(std::move(v), l);
}

PerfectHint PerfectHint::from_homogenized(VectorZZ v_bar) {
    PerfectHint hint(std::move(v_bar), BigInt(0));
    hint.is_already_homogenized_ = true;
    return hint;
}

PerfectHint PerfectHint::from_homogenized(VectorDouble v_bar) {
    PerfectHint hint(std::move(v_bar), 0.0);
    hint.is_already_homogenized_ = true;
    return hint;
}

const VectorZZ& PerfectHint::hint_vector_int() const { return v_int_; }
const BigInt& PerfectHint::hint_scalar_int() const { return l_int_; }
const VectorDouble& PerfectHint::hint_vector_double() const { return v_double_; }
double PerfectHint::hint_scalar_double() const { return l_double_; }

std::string PerfectHint::description() const {
    return "PerfectHint(<s, v> = " + (is_integer_ ? l_int_.to_string() : std::to_string(l_double_)) + ")";
}

VectorZZ PerfectHint::homogenize_int(size_t ambient_dim) const {
    if (is_already_homogenized_) {
        if (v_int_.size() != ambient_dim) {
            throw DimensionMismatchException(
                "PerfectHint::homogenize_int: Already homogenized vector size (" +
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
        "PerfectHint::homogenize_int: Hint vector size (" + std::to_string(v_int_.size()) +
        ") exceeds ambient dimension (" + std::to_string(ambient_dim) + ")");
}

VectorDouble PerfectHint::homogenize_double(size_t ambient_dim) const {
    VectorZZ v_zz = homogenize_int(ambient_dim);
    VectorDouble res(ambient_dim);
    for (size_t i = 0; i < ambient_dim; ++i) {
        res[i] = v_zz[i].to_double();
    }
    return res;
}

std::pair<VectorDouble, MatrixDouble> PerfectHint::update_distribution(
    const VectorDouble& mean,
    const MatrixDouble& cov,
    const VectorDouble& v_bar) {

    size_t d = mean.size();
    if (cov.rows() != d || cov.cols() != d) {
        throw DimensionMismatchException(
            "PerfectHint::update_distribution: Covariance dimensions (" +
            std::to_string(cov.rows()) + "x" + std::to_string(cov.cols()) +
            ") do not match mean dimension (" + std::to_string(d) + ")");
    }
    if (v_bar.size() != d) {
        throw DimensionMismatchException(
            "PerfectHint::update_distribution: Hint vector dimension (" +
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

    // denom = v_bar * cov * v_bar^T = w * v_bar^T
    double denom = 0.0;
    for (size_t j = 0; j < d; ++j) {
        denom += w[j] * v_bar[j];
    }

    // Inner product <mean, v_bar>
    double mean_dot_v = 0.0;
    for (size_t j = 0; j < d; ++j) {
        mean_dot_v += mean[j] * v_bar[j];
    }

    if (denom <= 1e-11) {
        if (std::abs(mean_dot_v) > 1e-7) {
            throw InvalidParameterException(
                "PerfectHint::update_distribution: Hint contradicts deterministic prior (<mu, v_bar> = " +
                std::to_string(mean_dot_v) + " != 0 with variance 0)");
        }
        return {mean, cov};
    }

    // Sigma' = Sigma - (w^T * w) / denom
    MatrixDouble new_cov(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            new_cov(i, j) = cov(i, j) - (w[i] * w[j]) / denom;
        }
    }

    // Enforce symmetry on new_cov
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (new_cov(i, j) + new_cov(j, i));
            new_cov(i, j) = avg;
            new_cov(j, i) = avg;
        }
    }

    // mu' = mu - (mean_dot_v / denom) * w
    double alpha = mean_dot_v / denom;
    VectorDouble new_mean(d);
    for (size_t i = 0; i < d; ++i) {
        new_mean[i] = mean[i] - alpha * w[i];
    }

    return {new_mean, new_cov};
}

MatrixZZ PerfectHint::complete_primitive_vector_unimodular(const VectorZZ& a) {
    size_t d = a.size();
    if (d < 2) {
        throw DimensionMismatchException("complete_primitive_vector_unimodular requires dimension >= 2");
    }

    VectorZZ v = a;
    MatrixZZ U = MatrixZZ::identity(d);

    if (v[d - 1] == BigInt(0)) {
        size_t swap_idx = d;
        for (size_t k = 0; k < d - 1; ++k) {
            if (v[k] != BigInt(0)) {
                swap_idx = k;
                break;
            }
        }
        if (swap_idx == d) {
            throw InvalidParameterException("Cannot complete the all-zero vector to a unimodular matrix");
        }
        std::swap(v[swap_idx], v[d - 1]);
        U.swap_rows(swap_idx, d - 1);
    }

    for (size_t i = 0; i < d - 1; ++i) {
        if (v[i] == BigInt(0)) {
            continue;
        }

        BigInt x = v[i];
        BigInt y = v[d - 1];

        BigInt s, t;
        BigInt g = BigInt::xgcd(x, y, s, t);

        BigInt x_over_g = x / g;
        BigInt y_over_g = y / g;

        VectorZZ row_i = U.row(i);
        VectorZZ row_last = U.row(d - 1);

        for (size_t c = 0; c < d; ++c) {
            BigInt new_i = -t * row_i[c] + s * row_last[c];
            BigInt new_last = x_over_g * row_i[c] + y_over_g * row_last[c];
            U(i, c) = new_i;
            U(d - 1, c) = new_last;
        }

        v[i] = BigInt(0);
        v[d - 1] = g;
    }

    if (v[d - 1] != BigInt(1)) {
        if (v[d - 1] == BigInt(-1)) {
            for (size_t c = 0; c < d; ++c) {
                U(d - 1, c) = -U(d - 1, c);
            }
            v[d - 1] = BigInt(1);
        } else {
            throw InvalidParameterException("Vector is not primitive: gcd = " + v[d - 1].to_string());
        }
    }

    return U;
}

BigReal PerfectHint::primitive_dual_norm(const BasisZZ& primal_basis, const VectorZZ& v_bar) {
    const MatrixZZ& B = primal_basis.matrix();
    size_t d = primal_basis.dimension();

    VectorZZ u = v_bar * B.transpose();

    BigInt g(0);
    for (size_t i = 0; i < d; ++i) {
        g = BigInt::gcd(g, u[i].abs());
    }
    if (g == BigInt(0)) {
        throw InvalidParameterException("v_bar is orthogonal to primal basis span");
    }

    MatrixQQ D = compute_dual_basis(B);
    VectorQQ v_bar_q = v_bar.cast<BigRational>();
    VectorQQ z = v_bar_q * D.transpose(); // size d: z_i = <v_bar, d_i>

    VectorQQ u_q = u.cast<BigRational>();
    BigRational denom = u_q.dot(z); // ||v_bar_V||^2

    BigReal denom_real = denom.to_mpfr();
    BigReal g_real = BigReal(g);
    return denom_real.sqrt() / g_real;
}

BasisZZ PerfectHint::intersect_hyperplane(const BasisZZ& basis, const VectorZZ& v_bar) {
    const MatrixZZ& B = basis.matrix();
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();

    if (d < 2) {
        throw InvalidParameterException("Cannot intersect hyperplane on lattice with dimension < 2");
    }

    VectorZZ u = v_bar * B.transpose();

    BigInt g(0);
    for (size_t i = 0; i < d; ++i) {
        g = BigInt::gcd(g, u[i].abs());
    }
    if (g == BigInt(0)) {
        return basis;
    }

    VectorZZ a(d);
    for (size_t i = 0; i < d; ++i) {
        a[i] = u[i] / g;
    }

    MatrixQQ D = compute_dual_basis(B);
    VectorQQ v_bar_q = v_bar.cast<BigRational>();
    VectorQQ z = v_bar_q * D.transpose(); // size d: z_i = <v_bar, d_i>

    VectorQQ u_q = u.cast<BigRational>();
    BigRational denom = u_q.dot(z); // ||v_bar_V||^2

    // v_bar_V = u_q * D (1 x m)
    VectorQQ v_bar_V = u_q * D;

    // D_proj = D - (z^T * v_bar_V) / denom (d x m)
    MatrixQQ D_proj(d, m);
    for (size_t i = 0; i < d; ++i) {
        BigRational factor = z[i] / denom;
        for (size_t j = 0; j < m; ++j) {
            D_proj(i, j) = D(i, j) - factor * v_bar_V[j];
        }
    }

    MatrixZZ U = complete_primitive_vector_unimodular(a);
    MatrixQQ D_unimod = U.cast<BigRational>() * D_proj;

    MatrixQQ D_sub(d - 1, m);
    for (size_t i = 0; i < d - 1; ++i) {
        D_sub.set_row(i, D_unimod.row(i));
    }

    auto dual_res = lll_exact(BasisQQ(D_sub, false));
    BasisQQ primal_q = primal_from_dual(dual_res.basis);

    MatrixZZ primal_z(d - 1, m);
    for (size_t i = 0; i < d - 1; ++i) {
        for (size_t j = 0; j < m; ++j) {
            const BigRational& val = primal_q.matrix()(i, j);
            if (val.denominator() != BigInt(1)) {
                throw NumericalPrecisionException(
                    "Primal basis entry (" + std::to_string(i) + ", " + std::to_string(j) +
                    ") is not an integer: " + val.to_string());
            }
            primal_z(i, j) = val.numerator();
        }
    }

    auto primal_res = lll_exact(BasisZZ(std::move(primal_z), false));
    return primal_res.basis;
}

LatticeZZ PerfectHint::update_lattice(const LatticeZZ& lattice, const VectorZZ& v_bar) {
    BasisZZ new_basis = intersect_hyperplane(lattice.basis(), v_bar);
    if (new_basis.dimension() == lattice.dimension()) {
        return lattice;
    }

    BigReal w_norm = primitive_dual_norm(lattice.basis(), v_bar);
    BigReal new_vol = lattice.volume() * w_norm;
    return LatticeZZ(std::move(new_basis), new_vol);
}

void PerfectHint::apply(DBDDInstance& instance) const {
    size_t amb_dim = instance.ambient_dimension();
    VectorZZ v_bar_int = homogenize_int(amb_dim);
    VectorDouble v_bar_double = homogenize_double(amb_dim);

    if (instance.has_target()) {
        const auto& target = instance.target();
        BigInt inner = target.dot(v_bar_int);
        if (inner != BigInt(0)) {
            throw InvalidParameterException(
                "PerfectHint::apply: Ground truth target vector does not satisfy the hint! <s_bar, v_bar> = " +
                inner.to_string() + " != 0");
        }
    }

    auto [new_mean, new_cov] = update_distribution(
        instance.mean(), instance.covariance(), v_bar_double);

    LatticeZZ new_lattice = update_lattice(instance.lattice(), v_bar_int);

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
                    "Target coordinates in new basis are not integers: " + x_q[i].to_string());
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
