#include "dbdd/lwe/distribution.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

namespace dbdd {

// ============================================================================
// DiscreteGaussian Implementation
// ============================================================================

DiscreteGaussian::DiscreteGaussian(double sigma, uint64_t seed)
    : sigma_(sigma),
      s_(sigma * std::sqrt(2.0 * std::numbers::pi)),
      tail_bound_k_(0),
      norm_factor_S_(0.0),
      rng_(seed) {
    if (sigma <= 0.0) {
        throw InvalidParameterException("DiscreteGaussian standard deviation sigma must be strictly positive");
    }
    init_tables();
}

DiscreteGaussian DiscreteGaussian::from_gaussian_parameter(double s, uint64_t seed) {
    if (s <= 0.0) {
        throw InvalidParameterException("DiscreteGaussian parameter s must be strictly positive");
    }
    double sigma = s / std::sqrt(2.0 * std::numbers::pi);
    return DiscreteGaussian(sigma, seed);
}

void DiscreteGaussian::init_tables() {
    tail_bound_k_ = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::ceil(10.0 * sigma_)));

    // S = sum_{x = -K}^K exp(-x^2 / (2 * sigma^2))
    double sum = 1.0;
    double two_sigma_sq = 2.0 * sigma_ * sigma_;
    for (int64_t k = 1; k <= tail_bound_k_; ++k) {
        sum += 2.0 * std::exp(-static_cast<double>(k * k) / two_sigma_sq);
    }
    norm_factor_S_ = sum;

    // Build cumulative CDF table
    size_t total_points = static_cast<size_t>(2 * tail_bound_k_ + 1);
    cdf_.resize(total_points);
    double accum = 0.0;
    for (size_t i = 0; i < total_points; ++i) {
        int64_t x = -tail_bound_k_ + static_cast<int64_t>(i);
        double p = std::exp(-static_cast<double>(x * x) / two_sigma_sq) / norm_factor_S_;
        accum += p;
        cdf_[i] = accum;
    }
    cdf_.back() = 1.0; // Ensure exact 1.0 at upper bound
}

double DiscreteGaussian::probability(int64_t x) const {
    if (std::abs(x) > tail_bound_k_) {
        return 0.0;
    }
    double two_sigma_sq = 2.0 * sigma_ * sigma_;
    return std::exp(-static_cast<double>(x * x) / two_sigma_sq) / norm_factor_S_;
}

int64_t DiscreteGaussian::sample() {
    return sample(rng_);
}

int64_t DiscreteGaussian::sample(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    size_t idx = static_cast<size_t>(std::distance(cdf_.begin(), it));
    if (idx >= cdf_.size()) {
        idx = cdf_.size() - 1;
    }
    return -tail_bound_k_ + static_cast<int64_t>(idx);
}

int64_t DiscreteGaussian::sample_rejection(std::mt19937_64& rng) {
    std::uniform_int_distribution<int64_t> x_dist(-tail_bound_k_, tail_bound_k_);
    std::uniform_real_distribution<double> u_dist(0.0, 1.0);
    double two_sigma_sq = 2.0 * sigma_ * sigma_;

    while (true) {
        int64_t x = x_dist(rng);
        double u = u_dist(rng);
        if (u <= std::exp(-static_cast<double>(x * x) / two_sigma_sq)) {
            return x;
        }
    }
}

// ============================================================================
// CenteredBinomial Implementation
// ============================================================================

double CenteredBinomial::compute_combinations(size_t n, size_t k) {
    if (k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n / 2) k = n - k;

    double res = 1.0;
    for (size_t i = 1; i <= k; ++i) {
        res = res * static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return res;
}

CenteredBinomial::CenteredBinomial(size_t eta, uint64_t seed)
    : eta_(eta), rng_(seed) {
    if (eta == 0) {
        throw InvalidParameterException("CenteredBinomial parameter eta must be >= 1");
    }

    size_t n = 2 * eta_;
    double denom = std::pow(4.0, static_cast<double>(eta_));
    size_t num_values = 2 * eta_ + 1;
    pmf_.resize(num_values);

    for (size_t i = 0; i < num_values; ++i) {
        pmf_[i] = compute_combinations(n, i) / denom;
    }
}

double CenteredBinomial::probability(int64_t x) const {
    if (x < -static_cast<int64_t>(eta_) || x > static_cast<int64_t>(eta_)) {
        return 0.0;
    }
    size_t idx = static_cast<size_t>(x + static_cast<int64_t>(eta_));
    return pmf_[idx];
}

int64_t CenteredBinomial::sample() {
    return sample(rng_);
}

int64_t CenteredBinomial::sample(std::mt19937_64& rng) {
    if (eta_ <= 32) {
        uint64_t r = rng();
        uint32_t mask = (eta_ == 32) ? 0xFFFFFFFFU : ((1U << eta_) - 1U);
        int a = std::popcount(static_cast<uint32_t>(r) & mask);
        int b = std::popcount(static_cast<uint32_t>(r >> 32) & mask);
        return static_cast<int64_t>(a - b);
    }

    int sum_a = 0;
    int sum_b = 0;
    size_t remaining = eta_;
    while (remaining > 0) {
        size_t take = std::min(remaining, size_t{32});
        uint64_t r = rng();
        uint32_t mask = (take == 32) ? 0xFFFFFFFFU : ((1U << take) - 1U);
        sum_a += std::popcount(static_cast<uint32_t>(r) & mask);
        sum_b += std::popcount(static_cast<uint32_t>(r >> 32) & mask);
        remaining -= take;
    }
    return static_cast<int64_t>(sum_a - sum_b);
}

// ============================================================================
// UniformTernary Implementation
// ============================================================================

UniformTernary::UniformTernary(double p, uint64_t seed)
    : p_(p), rng_(seed) {
    if (p < 0.0 || p > 1.0) {
        throw InvalidParameterException("UniformTernary parameter p must be in [0, 1]");
    }
}

double UniformTernary::probability(int64_t x) const {
    if (x == 1 || x == -1) {
        return p_ / 2.0;
    }
    if (x == 0) {
        return 1.0 - p_;
    }
    return 0.0;
}

int64_t UniformTernary::sample() {
    return sample(rng_);
}

int64_t UniformTernary::sample(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    if (u < p_ / 2.0) {
        return -1;
    }
    if (u < p_) {
        return 1;
    }
    return 0;
}

// ============================================================================
// ConfigurableDiscrete Implementation
// ============================================================================

ConfigurableDiscrete::ConfigurableDiscrete(const std::vector<int64_t>& values,
                                           const std::vector<double>& probabilities,
                                           uint64_t seed)
    : rng_(seed) {
    init_from_raw(values, probabilities);
}

ConfigurableDiscrete::ConfigurableDiscrete(const std::vector<std::pair<int64_t, double>>& pmf,
                                           uint64_t seed)
    : rng_(seed) {
    std::vector<int64_t> values;
    std::vector<double> probs;
    values.reserve(pmf.size());
    probs.reserve(pmf.size());
    for (const auto& [val, prob] : pmf) {
        values.push_back(val);
        probs.push_back(prob);
    }
    init_from_raw(values, probs);
}

void ConfigurableDiscrete::init_from_raw(const std::vector<int64_t>& values,
                                         const std::vector<double>& probs) {
    if (values.empty()) {
        throw InvalidParameterException("ConfigurableDiscrete requires non-empty values");
    }
    if (values.size() != probs.size()) {
        throw DimensionMismatchException("ConfigurableDiscrete values and probabilities size mismatch");
    }

    double total_prob = 0.0;
    for (double p : probs) {
        if (p < 0.0) {
            throw InvalidParameterException("ConfigurableDiscrete probabilities must be non-negative");
        }
        total_prob += p;
    }
    if (total_prob <= 0.0) {
        throw InvalidParameterException("ConfigurableDiscrete total probability must be positive");
    }

    values_ = values;
    cdf_.resize(values_.size());
    prob_map_.clear();

    double accum = 0.0;
    mean_ = 0.0;
    for (size_t i = 0; i < values_.size(); ++i) {
        double norm_p = probs[i] / total_prob;
        accum += norm_p;
        cdf_[i] = accum;
        prob_map_[values_[i]] += norm_p;
        mean_ += static_cast<double>(values_[i]) * norm_p;
    }
    cdf_.back() = 1.0;

    variance_ = 0.0;
    for (size_t i = 0; i < values_.size(); ++i) {
        double norm_p = probs[i] / total_prob;
        double diff = static_cast<double>(values_[i]) - mean_;
        variance_ += diff * diff * norm_p;
    }
}

double ConfigurableDiscrete::probability(int64_t x) const {
    auto it = prob_map_.find(x);
    if (it != prob_map_.end()) {
        return it->second;
    }
    return 0.0;
}

int64_t ConfigurableDiscrete::sample() {
    return sample(rng_);
}

int64_t ConfigurableDiscrete::sample(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    size_t idx = static_cast<size_t>(std::distance(cdf_.begin(), it));
    if (idx >= values_.size()) {
        idx = values_.size() - 1;
    }
    return values_[idx];
}

} // namespace dbdd
