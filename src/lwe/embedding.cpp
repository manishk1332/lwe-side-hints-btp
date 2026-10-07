#include "dbdd/lwe/embedding.hpp"
#include <cmath>

namespace dbdd {

VectorZZ KannanEmbedding::compute_k_vector(
    const MatrixZZ& A, const VectorZZ& b, const BigInt& q,
    const VectorZZ& s, const VectorZZ& e) {
    size_t m = A.rows();
    size_t n = A.cols();

    if (s.size() != n) {
        throw DimensionMismatchException("compute_k_vector: secret size mismatch");
    }
    if (e.size() != m) {
        throw DimensionMismatchException("compute_k_vector: error size mismatch");
    }
    if (b.size() != m) {
        throw DimensionMismatchException("compute_k_vector: target b size mismatch");
    }

    VectorZZ s_AT = s * A.transpose();
    VectorZZ k(m);

    for (size_t j = 0; j < m; ++j) {
        BigInt num = e[j] + s_AT[j] - b[j];
        if (BigInt::mod_floor(num, q) != BigInt(0)) {
            throw InvalidParameterException("Inconsistent LWE instance: coordinate " +
                                            std::to_string(j) + " does not satisfy e + s * A^T - b = 0 mod q");
        }
        k[j] = num / q;
    }

    return k;
}

KannanEmbedding::KannanEmbedding(
    const MatrixZZ& A, const VectorZZ& b, const BigInt& q,
    double sigma_e, double sigma_s,
    double mu_e, double mu_s,
    std::optional<VectorZZ> secret,
    std::optional<VectorZZ> error)
    : m_(A.rows()),
      n_(A.cols()),
      d_(m_ + n_ + 1),
      q_(q),
      target_(std::nullopt),
      coords_(std::nullopt) {
    if (m_ == 0 || n_ == 0) {
        throw InvalidParameterException("KannanEmbedding dimensions m and n must be strictly positive");
    }
    if (q_ < BigInt(2)) {
        throw InvalidParameterException("KannanEmbedding modulus q must be >= 2");
    }
    if (b.size() != m_) {
        throw DimensionMismatchException("KannanEmbedding: vector b dimension mismatch");
    }
    if (sigma_e <= 0.0 || sigma_s <= 0.0) {
        throw InvalidParameterException("Standard deviations sigma_e and sigma_s must be strictly positive");
    }

    // Construct the (m + n + 1) x (m + n + 1) basis matrix B:
    // [ q * I_m    0     0 ]
    // [   A^T    -I_n    0 ]
    // [    b       0     1 ]
    MatrixZZ B(d_, d_, BigInt(0));

    // Block 0: q * I_m
    for (size_t i = 0; i < m_; ++i) {
        B(i, i) = q_;
    }

    // Block 1: A^T of size n x m, and -I_n of size n x n
    for (size_t i = 0; i < n_; ++i) {
        size_t row = m_ + i;
        for (size_t j = 0; j < m_; ++j) {
            B(row, j) = A(j, i); // (A^T)(i, j) = A(j, i)
        }
        B(row, m_ + i) = BigInt(-1);
    }

    // Block 2: b of size 1 x m, and 1
    size_t row_b = m_ + n_;
    for (size_t j = 0; j < m_; ++j) {
        B(row_b, j) = b[j];
    }
    B(row_b, row_b) = BigInt(1);

    basis_ = BasisZZ(std::move(B), false); // full rank analytically

    BigInt q_pow_m = BigInt::pow(q_, static_cast<unsigned long>(m_));
    BigReal known_vol = q_pow_m.to_mpfr();
    lattice_ = LatticeZZ(basis_, known_vol);

    // If ground truth is provided, compute target s_bar = (e, s, 1) and coords x = (k, -s, 1)
    if (secret.has_value() && error.has_value()) {
        const auto& s = *secret;
        const auto& e = *error;

        VectorZZ target(d_);
        for (size_t j = 0; j < m_; ++j) {
            target[j] = e[j];
        }
        for (size_t i = 0; i < n_; ++i) {
            target[m_ + i] = s[i];
        }
        target[d_ - 1] = BigInt(1);
        target_ = target;

        VectorZZ k = compute_k_vector(A, b, q_, s, e);
        VectorZZ coords(d_);
        for (size_t j = 0; j < m_; ++j) {
            coords[j] = k[j];
        }
        for (size_t i = 0; i < n_; ++i) {
            coords[m_ + i] = -s[i];
        }
        coords[d_ - 1] = BigInt(1);
        coords_ = coords;
    }

    // Initialize mean vector: mu = (mu_e ... mu_e, mu_s ... mu_s, 1)
    VectorDouble mean(d_, 0.0);
    for (size_t j = 0; j < m_; ++j) {
        mean[j] = mu_e;
    }
    for (size_t i = 0; i < n_; ++i) {
        mean[m_ + i] = mu_s;
    }
    mean[d_ - 1] = 1.0;
    mean_ = std::move(mean);

    // Initialize covariance matrix: Sigma = diag(sigma_e^2 * I_m, sigma_s^2 * I_n, 0)
    MatrixDouble Sigma = MatrixDouble::zero(d_, d_);
    double var_e = sigma_e * sigma_e;
    double var_s = sigma_s * sigma_s;

    for (size_t j = 0; j < m_; ++j) {
        Sigma(j, j) = var_e;
    }
    for (size_t i = 0; i < n_; ++i) {
        Sigma(m_ + i, m_ + i) = var_s;
    }
    Sigma(d_ - 1, d_ - 1) = 0.0;
    covariance_ = std::move(Sigma);
}

KannanEmbedding KannanEmbedding::from_lwe(
    const LWEInstance& instance,
    double sigma_e, double sigma_s,
    double mu_e, double mu_s) {
    std::optional<VectorZZ> secret = std::nullopt;
    std::optional<VectorZZ> error = std::nullopt;

    if (instance.has_ground_truth()) {
        secret = instance.test_only_secret();
        error = instance.test_only_error();
    }

    return KannanEmbedding(
        instance.A(), instance.b(), instance.q(),
        sigma_e, sigma_s,
        mu_e, mu_s,
        std::move(secret), std::move(error));
}

KannanEmbedding KannanEmbedding::from_lwe(
    const LWEInstance& instance,
    const Distribution& error_dist,
    const Distribution& secret_dist) {
    return from_lwe(
        instance,
        error_dist.standard_deviation(),
        secret_dist.standard_deviation(),
        error_dist.mean(),
        secret_dist.mean());
}

const VectorZZ& KannanEmbedding::target() const {
    if (!target_.has_value()) {
        throw InvalidParameterException("KannanEmbedding does not contain ground-truth target vector");
    }
    return *target_;
}

const VectorZZ& KannanEmbedding::target_coordinates() const {
    if (!coords_.has_value()) {
        throw InvalidParameterException("KannanEmbedding does not contain ground-truth target coordinates");
    }
    return *coords_;
}

bool KannanEmbedding::verify_target_invariant() const {
    if (!target_.has_value() || !coords_.has_value()) {
        return false;
    }
    VectorZZ actual_target = (*coords_) * basis_.matrix();
    return actual_target == (*target_);
}

bool KannanEmbedding::verify_covariance_invariants() const {
    // 1. Homogeneous coordinate variance must be 0
    if (std::abs(covariance_(d_ - 1, d_ - 1)) > 1e-12) {
        return false;
    }

    // 2. Symmetry check
    for (size_t i = 0; i < d_; ++i) {
        for (size_t j = i + 1; j < d_; ++j) {
            if (std::abs(covariance_(i, j) - covariance_(j, i)) > 1e-12) {
                return false;
            }
        }
    }

    // 3. Positive diagonal elements for the first d - 1 coordinates
    for (size_t i = 0; i < d_ - 1; ++i) {
        if (covariance_(i, i) <= 0.0) {
            return false;
        }
    }

    // 4. Rank check: must be exactly d - 1
    return covariance_.rank() == (d_ - 1);
}

} // namespace dbdd
