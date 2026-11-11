// Paper Reference: thesis extension "Budgeted Hint Selection as D-Optimal Design", Section 6.
// Mathematical Purpose: Synthetic candidate pools A-D. See candidate_pools.hpp.

#include "dbdd/hints/candidate_pools.hpp"
#include "dbdd/core/exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>

namespace dbdd {

namespace {

class PoolBuilder {
public:
    explicit PoolBuilder(const CandidatePoolOptions& options)
        : opts_(options), rng_(options.seed), coords_(options.n + options.m), amb_dim_(options.n + options.m + 1) {
        if (opts_.n == 0 || opts_.m == 0) {
            throw InvalidParameterException("CandidatePools: n and m must be >= 1");
        }
        if (!(opts_.sigma_min > 0.0) || !(opts_.sigma_max >= opts_.sigma_min)) {
            throw InvalidParameterException("CandidatePools: require 0 < sigma_min <= sigma_max");
        }
    }

    [[nodiscard]] size_t coords() const noexcept { return coords_; }
    [[nodiscard]] std::mt19937_64& rng() noexcept { return rng_; }
    [[nodiscard]] std::vector<HintCandidate> take() { return std::move(pool_); }
    [[nodiscard]] size_t size() const noexcept { return pool_.size(); }

    // Coordinate c in [0, n + m): error coordinate if c < m, secret coordinate otherwise.
    [[nodiscard]] std::string coord_name(size_t c) const {
        return c < opts_.m ? "e[" + std::to_string(c) + "]" : "s[" + std::to_string(c - opts_.m) + "]";
    }

    [[nodiscard]] double draw_variance() {
        std::uniform_real_distribution<double> log_sigma(std::log(opts_.sigma_min), std::log(opts_.sigma_max));
        const double sigma = std::exp(log_sigma(rng_));
        return sigma * sigma;
    }

    [[nodiscard]] size_t uniform(size_t lo, size_t hi) { // inclusive
        return std::uniform_int_distribution<size_t>(lo, hi)(rng_);
    }

    void add_coordinate(size_t c) {
        HintCandidate h;
        h.v_bar = VectorDouble(amb_dim_, 0.0);
        h.v_bar[c] = 1.0;
        h.variance = draw_variance();
        h.label = coord_name(c);
        pool_.push_back(std::move(h));
    }

    void add_combination(const std::vector<size_t>& support, const std::vector<double>& coeffs, std::string label) {
        HintCandidate h;
        h.v_bar = VectorDouble(amb_dim_, 0.0);
        for (size_t t = 0; t < support.size(); ++t) {
            h.v_bar[support[t]] = coeffs[t];
        }
        h.variance = draw_variance();
        h.label = std::move(label);
        pool_.push_back(std::move(h));
    }

    // Random distinct coordinates forming the "base" of pools B and C. With the default size this is every
    // coordinate; for a small target it is a random subset of about half the target.
    [[nodiscard]] std::vector<size_t> base_coordinates() {
        std::vector<size_t> all(coords_);
        std::iota(all.begin(), all.end(), size_t{0});
        if (opts_.target_size == 0) {
            return all;
        }
        std::shuffle(all.begin(), all.end(), rng_);
        const size_t base = std::min(coords_, std::max<size_t>(1, (opts_.target_size + 1) / 2));
        all.resize(std::min(base, opts_.target_size));
        std::sort(all.begin(), all.end());
        return all;
    }

    [[nodiscard]] const CandidatePoolOptions& options() const noexcept { return opts_; }

private:
    CandidatePoolOptions opts_;
    std::mt19937_64 rng_;
    size_t coords_;
    size_t amb_dim_;
    std::vector<HintCandidate> pool_;
};

} // namespace

std::vector<HintCandidate> CandidatePools::pool_a(const CandidatePoolOptions& options) {
    PoolBuilder b(options);
    for (size_t c = 0; c < b.coords(); ++c) {
        b.add_coordinate(c);
    }
    return b.take();
}

std::vector<HintCandidate> CandidatePools::pool_b(const CandidatePoolOptions& options) {
    PoolBuilder b(options);
    const std::vector<size_t> base = b.base_coordinates();
    for (size_t c : base) {
        b.add_coordinate(c);
    }

    // The coordinates that receive repeated measurements: a random half of the base.
    std::vector<size_t> repeated = base;
    std::shuffle(repeated.begin(), repeated.end(), b.rng());
    repeated.resize(std::max<size_t>(1, repeated.size() / 2));

    if (options.target_size == 0) {
        for (size_t c : repeated) {
            const size_t extra = b.uniform(2, 4);
            for (size_t t = 0; t < extra; ++t) {
                b.add_coordinate(c);
            }
        }
    } else {
        while (b.size() < options.target_size) {
            b.add_coordinate(repeated[b.uniform(0, repeated.size() - 1)]);
        }
    }
    return b.take();
}

std::vector<HintCandidate> CandidatePools::pool_c(const CandidatePoolOptions& options) {
    PoolBuilder b(options);
    const std::vector<size_t> base = b.base_coordinates();
    for (size_t c : base) {
        b.add_coordinate(c);
    }

    const size_t n = options.n;
    const size_t m = options.m;
    const size_t total = options.target_size == 0 ? 2 * b.coords() : options.target_size;
    bool pair_next = true;
    while (b.size() < total) {
        if (pair_next && n >= 2) {
            // Pair sum s_i + s_j.
            const size_t i = b.uniform(0, n - 1);
            size_t j = b.uniform(0, n - 2);
            if (j >= i) {
                ++j;
            }
            b.add_combination({m + i, m + j}, {1.0, 1.0},
                              "s[" + std::to_string(i) + "]+s[" + std::to_string(j) + "]");
        } else {
            // Window sum s_i + ... + s_{i+len-1}.
            const size_t len = std::min(n, b.uniform(4, 8));
            const size_t start = b.uniform(0, n - len);
            std::vector<size_t> support(len);
            std::iota(support.begin(), support.end(), m + start);
            b.add_combination(support, std::vector<double>(len, 1.0),
                              "sum(s[" + std::to_string(start) + ".." + std::to_string(start + len - 1) + "])");
        }
        pair_next = !pair_next;
    }
    return b.take();
}

std::vector<HintCandidate> CandidatePools::pool_d(const CandidatePoolOptions& options) {
    PoolBuilder b(options);
    const size_t total = options.target_size == 0 ? 2 * b.coords() : options.target_size;
    const size_t max_weight = std::min<size_t>(32, b.coords());
    const size_t min_weight = std::min<size_t>(8, max_weight);

    std::vector<size_t> all(b.coords());
    std::iota(all.begin(), all.end(), size_t{0});
    for (size_t t = 0; t < total; ++t) {
        const size_t weight = b.uniform(min_weight, max_weight);
        std::shuffle(all.begin(), all.end(), b.rng());
        std::vector<size_t> support(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(weight));
        std::vector<double> coeffs(weight);
        for (double& c : coeffs) {
            c = b.uniform(0, 1) == 0 ? 1.0 : -1.0;
        }
        b.add_combination(support, coeffs, "dense#" + std::to_string(t));
    }
    return b.take();
}

std::vector<HintCandidate> CandidatePools::make(CandidatePoolKind kind, const CandidatePoolOptions& options) {
    switch (kind) {
        case CandidatePoolKind::A: return pool_a(options);
        case CandidatePoolKind::B: return pool_b(options);
        case CandidatePoolKind::C: return pool_c(options);
        case CandidatePoolKind::D: return pool_d(options);
    }
    throw InvalidParameterException("CandidatePools::make: unknown pool kind");
}

const char* CandidatePools::name(CandidatePoolKind kind) noexcept {
    switch (kind) {
        case CandidatePoolKind::A: return "A";
        case CandidatePoolKind::B: return "B";
        case CandidatePoolKind::C: return "C";
        case CandidatePoolKind::D: return "D";
    }
    return "?";
}

} // namespace dbdd
