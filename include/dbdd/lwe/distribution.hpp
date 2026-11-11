#pragma once

// Paper Reference: Dachman-Soled et al. Section 2, Section 3
// Mathematical Purpose: Concrete discrete distributions for LWE secrets and noise:
//     - Discrete Gaussian D_{Z, sigma}
//     - Centered Binomial CBD(eta)
//     - Uniform Ternary {-1, 0, 1}
//     - Arbitrary Configurable Discrete PMF
// Invariants:
//     - mean() and variance() return exact theoretical moments
//     - probability(x) returns P(X = x)
//     - sample(rng) samples from the distribution

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/linalg/vector.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include <map>

namespace dbdd {

/**
 * @brief Abstract base class for discrete probability distributions on integers.
 */
class Distribution {
public:
    virtual ~Distribution() = default;

    [[nodiscard]] virtual double mean() const = 0;
    [[nodiscard]] virtual double variance() const = 0;

    [[nodiscard]] virtual double standard_deviation() const {
        return std::sqrt(variance());
    }

    [[nodiscard]] virtual double probability(int64_t x) const = 0;

    virtual int64_t sample() = 0;
    virtual int64_t sample(std::mt19937_64& rng) = 0;

    virtual void set_seed(uint64_t seed) = 0;

    [[nodiscard]] virtual std::unique_ptr<Distribution> clone() const = 0;

    /**
     * @brief Samples a row vector of independent samples from this distribution.
     */
    virtual VectorZZ sample_vector(size_t dim, std::mt19937_64& rng) {
        VectorZZ res(dim);
        for (size_t i = 0; i < dim; ++i) {
            res[i] = BigInt(sample(rng));
        }
        return res;
    }

    virtual VectorZZ sample_vector(size_t dim) {
        VectorZZ res(dim);
        for (size_t i = 0; i < dim; ++i) {
            res[i] = BigInt(sample());
        }
        return res;
    }
};

/**
 * @brief Discrete Gaussian distribution D_{Z, sigma} centered at 0.
 *
 * Parameterized either by standard deviation sigma or Gaussian parameter s = sigma * sqrt(2*pi).
 * Uses precomputed CDF table inversion for O(log sigma) sampling without rejection overhead.
 */
class DiscreteGaussian : public Distribution {
public:
    explicit DiscreteGaussian(double sigma, uint64_t seed = 42);

    static DiscreteGaussian from_standard_deviation(double sigma, uint64_t seed = 42) {
        return DiscreteGaussian(sigma, seed);
    }

    static DiscreteGaussian from_gaussian_parameter(double s, uint64_t seed = 42);

    [[nodiscard]] double sigma() const noexcept { return sigma_; }
    [[nodiscard]] double gaussian_parameter_s() const noexcept { return s_; }

    [[nodiscard]] double mean() const override { return 0.0; }
    [[nodiscard]] double variance() const override { return sigma_ * sigma_; }
    [[nodiscard]] double probability(int64_t x) const override;

    int64_t sample() override;
    int64_t sample(std::mt19937_64& rng) override;
    int64_t sample_rejection(std::mt19937_64& rng);

    void set_seed(uint64_t seed) override { rng_.seed(seed); }

    [[nodiscard]] std::unique_ptr<Distribution> clone() const override {
        return std::make_unique<DiscreteGaussian>(*this);
    }

private:
    void init_tables();

    double sigma_;
    double s_;
    int64_t tail_bound_k_;
    double norm_factor_S_;
    std::vector<double> cdf_;
    std::mt19937_64 rng_;
};

/**
 * @brief Centered Binomial Distribution CBD(eta) with support [-eta, eta].
 *
 * Defined as the difference of two independent Binomial(eta, 0.5) variables.
 * Variance is eta / 2.
 */
class CenteredBinomial : public Distribution {
public:
    explicit CenteredBinomial(size_t eta, uint64_t seed = 42);

    [[nodiscard]] size_t eta() const noexcept { return eta_; }

    [[nodiscard]] double mean() const override { return 0.0; }
    [[nodiscard]] double variance() const override { return 0.5 * static_cast<double>(eta_); }
    [[nodiscard]] double probability(int64_t x) const override;

    int64_t sample() override;
    int64_t sample(std::mt19937_64& rng) override;

    void set_seed(uint64_t seed) override { rng_.seed(seed); }

    [[nodiscard]] std::unique_ptr<Distribution> clone() const override {
        return std::make_unique<CenteredBinomial>(*this);
    }

private:
    static double compute_combinations(size_t n, size_t k);

    size_t eta_;
    std::vector<double> pmf_;
    std::mt19937_64 rng_;
};

/**
 * @brief Uniform Ternary distribution with support in {-1, 0, 1}.
 *
 * Parameterized by p in [0, 1]:
 *   P(X = 1) = p / 2, P(X = -1) = p / 2, P(X = 0) = 1 - p.
 * Default p = 2/3 yields uniform {-1, 0, 1}.
 */
class UniformTernary : public Distribution {
public:
    explicit UniformTernary(double p = 2.0 / 3.0, uint64_t seed = 42);

    [[nodiscard]] double p() const noexcept { return p_; }

    [[nodiscard]] double mean() const override { return 0.0; }
    [[nodiscard]] double variance() const override { return p_; }
    [[nodiscard]] double probability(int64_t x) const override;

    int64_t sample() override;
    int64_t sample(std::mt19937_64& rng) override;

    void set_seed(uint64_t seed) override { rng_.seed(seed); }

    [[nodiscard]] std::unique_ptr<Distribution> clone() const override {
        return std::make_unique<UniformTernary>(*this);
    }

private:
    double p_;
    std::mt19937_64 rng_;
};

/**
 * @brief Arbitrary configurable discrete distribution defined by a discrete PMF.
 */
class ConfigurableDiscrete : public Distribution {
public:
    ConfigurableDiscrete(const std::vector<int64_t>& values,
                         const std::vector<double>& probabilities,
                         uint64_t seed = 42);

    explicit ConfigurableDiscrete(const std::vector<std::pair<int64_t, double>>& pmf,
                                 uint64_t seed = 42);

    [[nodiscard]] double mean() const override { return mean_; }
    [[nodiscard]] double variance() const override { return variance_; }
    [[nodiscard]] double probability(int64_t x) const override;

    int64_t sample() override;
    int64_t sample(std::mt19937_64& rng) override;

    void set_seed(uint64_t seed) override { rng_.seed(seed); }

    [[nodiscard]] std::unique_ptr<Distribution> clone() const override {
        return std::make_unique<ConfigurableDiscrete>(*this);
    }

private:
    void init_from_raw(const std::vector<int64_t>& values,
                       const std::vector<double>& probs);

    double mean_{0.0};
    double variance_{0.0};
    std::vector<int64_t> values_;
    std::vector<double> cdf_;
    std::map<int64_t, double> prob_map_;
    std::mt19937_64 rng_;
};

} // namespace dbdd
