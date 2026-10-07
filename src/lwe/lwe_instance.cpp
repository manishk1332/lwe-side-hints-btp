#include "dbdd/lwe/lwe_instance.hpp"
#include <limits>

namespace dbdd {

LWEInstance::LWEInstance(size_t n, size_t m, BigInt q, MatrixZZ A, VectorZZ b)
    : n_(n), m_(m), q_(std::move(q)), A_(std::move(A)), b_(std::move(b)),
      secret_(std::nullopt), error_(std::nullopt) {
    validate_dimensions();
}

LWEInstance::LWEInstance(size_t n, size_t m, BigInt q, MatrixZZ A, VectorZZ b,
                         VectorZZ secret, VectorZZ error)
    : n_(n), m_(m), q_(std::move(q)), A_(std::move(A)), b_(std::move(b)),
      secret_(std::move(secret)), error_(std::move(error)) {
    validate_dimensions();

    if (secret_->size() != n_) {
        throw DimensionMismatchException("LWE ground-truth secret dimension mismatch: expected " +
                                        std::to_string(n_) + ", got " + std::to_string(secret_->size()));
    }
    if (error_->size() != m_) {
        throw DimensionMismatchException("LWE ground-truth error dimension mismatch: expected " +
                                        std::to_string(m_) + ", got " + std::to_string(error_->size()));
    }

    // Verify mathematical invariant: b = s * A^T + e mod q
    VectorZZ s_AT = (*secret_) * A_.transpose();
    for (size_t j = 0; j < m_; ++j) {
        BigInt expected_bj = BigInt::mod_floor(s_AT[j] + (*error_)[j], q_);
        BigInt actual_bj = BigInt::mod_floor(b_[j], q_);
        if (expected_bj != actual_bj) {
            throw InvalidParameterException("Inconsistent LWE instance: b[" + std::to_string(j) +
                                            "] != (s * A^T + e)[" + std::to_string(j) + "] mod q");
        }
    }
}

void LWEInstance::validate_dimensions() const {
    if (n_ == 0 || m_ == 0) {
        throw InvalidParameterException("LWE dimensions n and m must be strictly positive");
    }
    if (q_ < BigInt(2)) {
        throw InvalidParameterException("LWE modulus q must be >= 2");
    }
    if (A_.rows() != m_ || A_.cols() != n_) {
        throw DimensionMismatchException("Matrix A dimensions (" + std::to_string(A_.rows()) +
                                        "x" + std::to_string(A_.cols()) + ") do not match (m=" +
                                        std::to_string(m_) + ", n=" + std::to_string(n_) + ")");
    }
    if (b_.size() != m_) {
        throw DimensionMismatchException("Vector b dimension (" + std::to_string(b_.size()) +
                                        ") does not match m=" + std::to_string(m_));
    }
}

const VectorZZ& LWEInstance::test_only_secret() const {
    if (!secret_.has_value()) {
        throw InvalidParameterException("LWEInstance does not contain ground-truth secret");
    }
    return *secret_;
}

const VectorZZ& LWEInstance::test_only_error() const {
    if (!error_.has_value()) {
        throw InvalidParameterException("LWEInstance does not contain ground-truth error");
    }
    return *error_;
}

VectorZZ LWEInstance::sample_secret(size_t n, Distribution& dist, std::mt19937_64& rng) {
    return dist.sample_vector(n, rng);
}

VectorZZ LWEInstance::sample_error(size_t m, Distribution& dist, std::mt19937_64& rng) {
    return dist.sample_vector(m, rng);
}

MatrixZZ LWEInstance::sample_uniform_matrix(size_t rows, size_t cols, const BigInt& q, std::mt19937_64& rng) {
    MatrixZZ M(rows, cols);
    if (q <= BigInt(std::numeric_limits<int64_t>::max())) {
        int64_t q_i64 = q.to_int64();
        std::uniform_int_distribution<int64_t> dist(0, q_i64 - 1);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                M(i, j) = BigInt(dist(rng));
            }
        }
    } else {
        // Multi-limb sampling for very large moduli
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                BigInt val = (BigInt(dist(rng)) << 64) | BigInt(dist(rng));
                M(i, j) = BigInt::mod_floor(val, q);
            }
        }
    }
    return M;
}

LWEInstance LWEInstance::generate_random(
    size_t n, size_t m, const BigInt& q,
    Distribution& secret_dist, Distribution& error_dist,
    std::mt19937_64& rng) {
    VectorZZ s = sample_secret(n, secret_dist, rng);
    VectorZZ e = sample_error(m, error_dist, rng);
    MatrixZZ A = sample_uniform_matrix(m, n, q, rng);

    // Compute b = s * A^T + e mod q
    VectorZZ s_AT = s * A.transpose();
    VectorZZ b(m);
    for (size_t j = 0; j < m; ++j) {
        b[j] = BigInt::mod_floor(s_AT[j] + e[j], q);
    }

    return LWEInstance(n, m, q, std::move(A), std::move(b), std::move(s), std::move(e));
}

LWEInstance LWEInstance::generate_random(
    size_t n, size_t m, const BigInt& q,
    Distribution& secret_dist, Distribution& error_dist,
    uint64_t seed) {
    std::mt19937_64 rng(seed);
    return generate_random(n, m, q, secret_dist, error_dist, rng);
}

bool LWEInstance::verify_secret(const VectorZZ& candidate_s) const {
    if (candidate_s.size() != n_) {
        throw DimensionMismatchException("Candidate secret size mismatch: expected " +
                                        std::to_string(n_) + ", got " + std::to_string(candidate_s.size()));
    }
    if (has_ground_truth()) {
        return candidate_s == *secret_;
    }

    // Default heuristic check: centered residual norm <= q / 4
    double threshold = (q_ / BigInt(4)).to_double();
    return verify_secret(candidate_s, threshold);
}

bool LWEInstance::verify_secret(const VectorZZ& candidate_s, double max_noise_norm) const {
    if (candidate_s.size() != n_) {
        throw DimensionMismatchException("Candidate secret size mismatch: expected " +
                                        std::to_string(n_) + ", got " + std::to_string(candidate_s.size()));
    }

    VectorZZ cand_s_AT = candidate_s * A_.transpose();
    BigInt half_q = q_ / BigInt(2);

    for (size_t j = 0; j < m_; ++j) {
        BigInt diff = b_[j] - cand_s_AT[j];
        BigInt rem = BigInt::mod_floor(diff, q_);
        if (rem > half_q) {
            rem -= q_;
        }
        double noise_coord = std::abs(rem.to_double());
        if (noise_coord > max_noise_norm) {
            return false;
        }
    }
    return true;
}

} // namespace dbdd
