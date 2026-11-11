#include "dbdd/lwe/secret_recovery.hpp"
#include <cmath>

namespace dbdd {

// Paper Reference: Dachman-Soled et al. Section 3.5 / main_plan.txt Section 22
// Mathematical Purpose: Implementation of SecretReconstructor for recovering the original LWE secret
// after projected lattice solving on DBDD instances with short-vector hints.

SecretReconstructor::SecretReconstructor(LWEInstance lwe)
    : lwe_(std::move(lwe)) {}

void SecretReconstructor::record_projection(VectorZZ v) {
    projections_.push_back(std::move(v));
}

void SecretReconstructor::record_projections(const std::vector<VectorZZ>& V) {
    for (const auto& v : V) {
        projections_.push_back(v);
    }
}

VectorZZ SecretReconstructor::recover_secret(const VectorDouble& candidate_w) const {
    size_t n = lwe_.n();
    size_t m = lwe_.m();
    size_t ambient_dim = m + n + 1;

    if (candidate_w.size() != ambient_dim) {
        throw DimensionMismatchException(
            "SecretReconstructor::recover_secret: candidate dimension (" +
            std::to_string(candidate_w.size()) + ") does not match ambient dimension (" +
            std::to_string(ambient_dim) + ")");
    }

    // Target vector s_bar = (e, s, 1) has last coordinate 1.
    // Lattice reduction can output either +w' or -w'.
    // If the last coordinate is negative, flip the sign.
    double sign = 1.0;
    if (candidate_w[ambient_dim - 1] < -0.2) {
        sign = -1.0;
    }

    // Secret coordinates s_0, ..., s_{n-1} are located at indices m, ..., m + n - 1
    VectorZZ secret(n);
    for (size_t j = 0; j < n; ++j) {
        double val = sign * candidate_w[m + j];
        secret[j] = BigInt(static_cast<long long>(std::round(val)));
    }

    return secret;
}

VectorZZ SecretReconstructor::recover_secret(const VectorZZ& candidate_w) const {
    size_t n = lwe_.n();
    size_t m = lwe_.m();
    size_t ambient_dim = m + n + 1;

    if (candidate_w.size() != ambient_dim) {
        throw DimensionMismatchException(
            "SecretReconstructor::recover_secret: candidate dimension (" +
            std::to_string(candidate_w.size()) + ") does not match ambient dimension (" +
            std::to_string(ambient_dim) + ")");
    }

    BigInt sign(1);
    if (candidate_w[ambient_dim - 1] < BigInt(0)) {
        sign = BigInt(-1);
    }

    VectorZZ secret(n);
    for (size_t j = 0; j < n; ++j) {
        secret[j] = sign * candidate_w[m + j];
    }

    return secret;
}

VectorZZ SecretReconstructor::recover_target(const VectorDouble& candidate_w) const {
    VectorZZ secret = recover_secret(candidate_w);
    size_t n = lwe_.n();
    size_t m = lwe_.m();
    size_t ambient_dim = m + n + 1;

    VectorZZ target(ambient_dim);

    // Reconstruct error vector e = b - s * A^T (mod q) centered in [-q/2, q/2)
    const BigInt& q = lwe_.q();
    BigInt half_q = q / 2;

    for (size_t i = 0; i < m; ++i) {
        BigInt pred(0);
        for (size_t j = 0; j < n; ++j) {
            pred += secret[j] * lwe_.A()(i, j);
        }
        BigInt diff = lwe_.b()[i] - pred;
        BigInt e_i = diff % q;
        if (e_i > half_q) {
            e_i -= q;
        } else if (e_i < -half_q) {
            e_i += q;
        }
        target[i] = e_i;
    }

    for (size_t j = 0; j < n; ++j) {
        target[m + j] = secret[j];
    }
    target[ambient_dim - 1] = BigInt(1);

    return target;
}

VectorZZ SecretReconstructor::recover_target(const VectorZZ& candidate_w) const {
    VectorDouble w_double(candidate_w.size());
    for (size_t i = 0; i < candidate_w.size(); ++i) {
        w_double[i] = candidate_w[i].to_double();
    }
    return recover_target(w_double);
}

bool SecretReconstructor::verify_recovery(const VectorZZ& secret) const {
    if (secret.size() != lwe_.n()) {
        return false;
    }

    const BigInt& q = lwe_.q();
    BigInt half_q = q / 2;

    for (size_t i = 0; i < lwe_.m(); ++i) {
        BigInt pred(0);
        for (size_t j = 0; j < lwe_.n(); ++j) {
            pred += secret[j] * lwe_.A()(i, j);
        }
        BigInt diff = lwe_.b()[i] - pred;
        BigInt e_i = diff % q;
        if (e_i > half_q) {
            e_i -= q;
        } else if (e_i < -half_q) {
            e_i += q;
        }

        // Noise in LWE must be strictly bounded well within q/2
        if (e_i.abs() >= half_q) {
            return false;
        }
    }

    return true;
}

} // namespace dbdd
