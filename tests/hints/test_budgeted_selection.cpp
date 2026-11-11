// tests/hints/test_budgeted_selection.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 4.3, Eq. (22);
//                  thesis extension "Budgeted Hint Selection as D-Optimal Design".
// Mathematical Purpose: Invariant tests for the budgeted hint selector: the marginal-gain formula, submodularity,
// the greedy guarantee, the certified upper bound, and agreement with the existing estimator pipeline.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dbdd/hints/budgeted_selection.hpp"
#include "dbdd/hints/candidate_pools.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/core/exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

namespace {

const double kOneMinusInvE = 1.0 - std::exp(-1.0);

// Prior on r free coordinates plus the homogeneous coordinate: mean (0, ..., 0, 1), covariance diag(C, 0).
DBDDStateStatistics make_prior(size_t r, uint64_t seed, bool diagonal) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> var(0.5, 6.0);

    const size_t amb = r + 1;
    MatrixDouble cov(amb, amb, 0.0);
    if (diagonal) {
        for (size_t i = 0; i < r; ++i) {
            cov(i, i) = var(rng);
        }
    } else {
        MatrixDouble a(r, r, 0.0);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < r; ++j) {
                a(i, j) = gauss(rng);
            }
        }
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < r; ++j) {
                double s = (i == j) ? 1.0 : 0.0;
                for (size_t t = 0; t < r; ++t) {
                    s += a(i, t) * a(j, t);
                }
                cov(i, j) = s;
            }
        }
    }
    VectorDouble mean(amb, 0.0);
    mean[amb - 1] = 1.0;
    return DBDDStateStatistics(amb, amb, BigReal(1.0e6), mean, cov);
}

// Dense random candidates with continuous coefficients and variances (no exact ties).
std::vector<HintCandidate> make_dense_pool(size_t count, size_t r, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> var(0.5, 4.0);
    std::vector<HintCandidate> pool(count);
    for (size_t i = 0; i < count; ++i) {
        pool[i].v_bar = VectorDouble(r + 1, 0.0);
        for (size_t j = 0; j < r; ++j) {
            pool[i].v_bar[j] = gauss(rng);
        }
        pool[i].variance = var(rng);
        pool[i].label = "dense#" + std::to_string(i);
    }
    return pool;
}

double quadratic_form(const VectorDouble& v, const MatrixDouble& cov) {
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        for (size_t j = 0; j < v.size(); ++j) {
            s += v[i] * cov(i, j) * v[j];
        }
    }
    return s;
}

double log_rdet(const MatrixDouble& cov) {
    return compute_log_restricted_determinant(cov).to_double();
}

// Integrates the listed candidates one by one with the library's scalar approximate-hint update.
MatrixDouble integrate_sequentially(
    const DBDDStateStatistics& prior,
    const std::vector<HintCandidate>& pool,
    const std::vector<size_t>& order) {
    VectorDouble mean = prior.mean();
    MatrixDouble cov = prior.covariance();
    for (size_t i : order) {
        auto updated = ApproximateHint::update_distribution(mean, cov, pool[i].v_bar, pool[i].variance);
        mean = std::move(updated.first);
        cov = std::move(updated.second);
    }
    return cov;
}

std::set<size_t> as_set(const std::vector<size_t>& v) {
    return std::set<size_t>(v.begin(), v.end());
}

} // namespace

TEST_CASE("Budgeted selection - marginal gain equals the log-determinant drop (Lemma 2)", "[hints][budgeted]") {
    const DBDDStateStatistics prior = make_prior(8, 11, false);
    const std::vector<HintCandidate> pool = make_dense_pool(6, 8, 12);
    const BudgetedHintSelector selector(prior);

    SECTION("Single hint") {
        for (size_t i = 0; i < pool.size(); ++i) {
            const double q = quadratic_form(pool[i].v_bar, prior.covariance());
            const double formula = BudgetedHintSelector::marginal_gain(q, pool[i].variance);
            const MatrixDouble after = integrate_sequentially(prior, pool, {i});
            const double drop = 0.5 * (log_rdet(prior.covariance()) - log_rdet(after));
            REQUIRE_THAT(formula, WithinAbs(drop, 1e-9));
            REQUIRE_THAT(selector.gain_of_set(pool, {i}), WithinAbs(drop, 1e-9));
        }
    }

    SECTION("Set of hints: Gram-space f(S), incremental engine and determinant drop agree") {
        const std::vector<size_t> subset = {4, 1, 3};
        const MatrixDouble after = integrate_sequentially(prior, pool, subset);
        const double drop = 0.5 * (log_rdet(prior.covariance()) - log_rdet(after));
        REQUIRE_THAT(selector.gain_of_set(pool, subset), WithinAbs(drop, 1e-9));

        const BudgetedSelectionResult all = selector.select_static_top_k(pool, pool.size());
        const double f_all = selector.gain_of_set(pool, {0, 1, 2, 3, 4, 5});
        REQUIRE_THAT(all.total_gain, WithinAbs(f_all, 1e-9));
        REQUIRE_THAT(std::accumulate(all.step_gains.begin(), all.step_gains.end(), 0.0),
                     WithinAbs(all.total_gain, 1e-12));
    }

    SECTION("Gain is zero for zero directional variance and negative rounding is clamped") {
        REQUIRE(BudgetedHintSelector::marginal_gain(0.0, 2.0) == 0.0);
        REQUIRE(BudgetedHintSelector::marginal_gain(-1e-15, 2.0) == 0.0);
    }
}

TEST_CASE("Budgeted selection - f is monotone and submodular (Theorem 1)", "[hints][budgeted]") {
    std::mt19937_64 rng(2024);
    for (uint64_t trial = 0; trial < 20; ++trial) {
        const size_t r = 7;
        const size_t count = 9;
        const DBDDStateStatistics prior = make_prior(r, 100 + trial, trial % 2 == 0);
        const std::vector<HintCandidate> pool = make_dense_pool(count, r, 200 + trial);
        const BudgetedHintSelector selector(prior);

        std::vector<size_t> perm(count);
        std::iota(perm.begin(), perm.end(), size_t{0});
        std::shuffle(perm.begin(), perm.end(), rng);

        const size_t extra = perm[0];                                        // i, not in T
        const std::vector<size_t> t_set(perm.begin() + 1, perm.begin() + 6); // |T| = 5
        const std::vector<size_t> s_set(perm.begin() + 1, perm.begin() + 3); // S subset of T, |S| = 2

        std::vector<size_t> s_plus = s_set;
        s_plus.push_back(extra);
        std::vector<size_t> t_plus = t_set;
        t_plus.push_back(extra);

        const double f_s = selector.gain_of_set(pool, s_set);
        const double f_t = selector.gain_of_set(pool, t_set);
        const double gain_s = selector.gain_of_set(pool, s_plus) - f_s;
        const double gain_t = selector.gain_of_set(pool, t_plus) - f_t;

        REQUIRE(f_t >= f_s - 1e-10);         // monotone
        REQUIRE(gain_s >= -1e-10);
        REQUIRE(gain_t >= -1e-10);
        REQUIRE(gain_t <= gain_s + 1e-10);   // diminishing returns
    }
}

TEST_CASE("Budgeted selection - greedy matches a from-scratch reference (incremental update)", "[hints][budgeted]") {
    for (uint64_t trial = 0; trial < 5; ++trial) {
        const size_t r = 10;
        const size_t count = 25;
        const size_t k = 7;
        const DBDDStateStatistics prior = make_prior(r, 300 + trial, false);
        const std::vector<HintCandidate> pool = make_dense_pool(count, r, 400 + trial);
        const BudgetedHintSelector selector(prior);
        const BudgetedSelectionResult greedy = selector.select_greedy(pool, k);

        // Reference: recompute every quadratic form from the updated covariance at each step.
        VectorDouble mean = prior.mean();
        MatrixDouble cov = prior.covariance();
        std::vector<bool> used(count, false);
        std::vector<size_t> ref_order;
        double ref_total = 0.0;
        double ref_bound = 0.0;
        const auto top_k_remaining = [&]() {
            std::vector<double> gains;
            for (size_t i = 0; i < count; ++i) {
                if (!used[i]) {
                    gains.push_back(BudgetedHintSelector::marginal_gain(
                        quadratic_form(pool[i].v_bar, cov), pool[i].variance));
                }
            }
            std::sort(gains.begin(), gains.end(), std::greater<double>());
            double s = 0.0;
            for (size_t t = 0; t < std::min(k, gains.size()); ++t) {
                s += gains[t];
            }
            return s;
        };
        ref_bound = top_k_remaining();
        for (size_t step = 0; step < k; ++step) {
            size_t best = count;
            double best_gain = -1.0;
            for (size_t i = 0; i < count; ++i) {
                if (!used[i]) {
                    const double g = BudgetedHintSelector::marginal_gain(
                        quadratic_form(pool[i].v_bar, cov), pool[i].variance);
                    if (g > best_gain) {
                        best_gain = g;
                        best = i;
                    }
                }
            }
            used[best] = true;
            ref_order.push_back(best);
            ref_total += best_gain;
            auto updated = ApproximateHint::update_distribution(mean, cov, pool[best].v_bar, pool[best].variance);
            mean = std::move(updated.first);
            cov = std::move(updated.second);
            ref_bound = std::min(ref_bound, ref_total + top_k_remaining());
        }
        const double kd = static_cast<double>(k);
        ref_bound = std::min(ref_bound, ref_total / (1.0 - std::pow(1.0 - 1.0 / kd, kd)));

        REQUIRE(greedy.selected_indices == ref_order);
        REQUIRE_THAT(greedy.total_gain, WithinAbs(ref_total, 1e-10));
        REQUIRE_THAT(greedy.upper_bound_gain, WithinAbs(ref_bound, 1e-9));
        REQUIRE(greedy.step_gains.size() == k);
        // Greedy step gains are non-increasing (a consequence of submodularity).
        for (size_t t = 1; t < k; ++t) {
            REQUIRE(greedy.step_gains[t] <= greedy.step_gains[t - 1] + 1e-10);
        }
    }
}

TEST_CASE("Budgeted selection - lazy greedy returns the greedy set (Theorem 4)", "[hints][budgeted]") {
    for (uint64_t trial = 0; trial < 8; ++trial) {
        const size_t r = 12;
        const size_t count = 40;
        const size_t k = 9;
        const DBDDStateStatistics prior = make_prior(r, 500 + trial, trial % 2 == 1);
        const std::vector<HintCandidate> pool = make_dense_pool(count, r, 600 + trial);
        const BudgetedHintSelector selector(prior);

        const BudgetedSelectionResult greedy = selector.select_greedy(pool, k);
        const BudgetedSelectionResult lazy = selector.select_lazy_greedy(pool, k);

        REQUIRE(lazy.selected_indices == greedy.selected_indices);
        REQUIRE_THAT(lazy.total_gain, WithinAbs(greedy.total_gain, 1e-9));
        REQUIRE(lazy.gain_evaluations <= greedy.gain_evaluations);
        REQUIRE(lazy.upper_bound_gain >= lazy.total_gain);
    }
}

TEST_CASE("Budgeted selection - greedy guarantee and certificate against brute force (Theorems 2, 3)",
          "[hints][budgeted]") {
    for (uint64_t trial = 0; trial < 10; ++trial) {
        const size_t r = 6;
        const size_t count = 12;
        const size_t k = 3;
        const DBDDStateStatistics prior = make_prior(r, 700 + trial, false);
        const std::vector<HintCandidate> pool = make_dense_pool(count, r, 800 + trial);
        const BudgetedHintSelector selector(prior);

        const BudgetedSelectionResult exact = selector.select_exact(pool, k);
        const BudgetedSelectionResult greedy = selector.select_greedy(pool, k);
        const BudgetedSelectionResult lazy = selector.select_lazy_greedy(pool, k);
        const BudgetedSelectionResult fixed = selector.select_static_top_k(pool, k);
        const BudgetedSelectionResult random = selector.select_random(pool, k, 99 + trial);
        const BudgetedSelectionResult look = selector.select_lookahead2(pool, k, 6);

        REQUIRE(exact.gain_evaluations == 220); // C(12, 3)
        REQUIRE_THAT(exact.total_gain, WithinAbs(selector.gain_of_set(pool, exact.selected_indices), 1e-9));
        REQUIRE_THAT(exact.upper_bound_gain, WithinAbs(exact.total_gain, 1e-12));

        // No strategy beats the optimum; greedy is within the Nemhauser-Wolsey-Fisher factor.
        for (const BudgetedSelectionResult* res : {&greedy, &lazy, &fixed, &random, &look}) {
            REQUIRE(res->total_gain <= exact.total_gain + 1e-9);
            REQUIRE(res->upper_bound_gain >= exact.total_gain - 1e-9); // every certificate is valid
            REQUIRE(res->upper_bound_gain >= res->total_gain);
            REQUIRE(res->selected_indices.size() == k);
            REQUIRE(as_set(res->selected_indices).size() == k);
        }
        REQUIRE(greedy.total_gain >= kOneMinusInvE * exact.total_gain);

        // Ordering of the implied block sizes.
        REQUIRE(greedy.beta_opt_lower <= exact.beta_after + 1e-9);
        REQUIRE(exact.beta_after <= greedy.beta_after + 1e-9);
        REQUIRE(greedy.beta_after <= greedy.beta_before + 1e-9);
    }
}

TEST_CASE("Budgeted selection - additive case: static, greedy and exact coincide", "[hints][budgeted]") {
    const size_t r = 10;
    const DBDDStateStatistics prior = make_prior(r, 900, true); // diagonal prior
    std::mt19937_64 rng(901);
    std::uniform_real_distribution<double> var(0.2, 9.0);
    std::vector<HintCandidate> pool(r);
    for (size_t i = 0; i < r; ++i) {
        pool[i].v_bar = VectorDouble(r + 1, 0.0);
        pool[i].v_bar[i] = 1.0; // one hint per distinct coordinate
        pool[i].variance = var(rng);
    }
    const BudgetedHintSelector selector(prior);
    const size_t k = 4;

    const BudgetedSelectionResult fixed = selector.select_static_top_k(pool, k);
    const BudgetedSelectionResult greedy = selector.select_greedy(pool, k);
    const BudgetedSelectionResult lazy = selector.select_lazy_greedy(pool, k);
    const BudgetedSelectionResult exact = selector.select_exact(pool, k);

    REQUIRE(as_set(fixed.selected_indices) == as_set(exact.selected_indices));
    REQUIRE(as_set(greedy.selected_indices) == as_set(exact.selected_indices));
    REQUIRE(as_set(lazy.selected_indices) == as_set(exact.selected_indices));
    REQUIRE_THAT(greedy.total_gain, WithinAbs(exact.total_gain, 1e-10));
    // In the additive case U(empty) is the sum of the k best gains, i.e. the optimum itself.
    REQUIRE_THAT(greedy.upper_bound_gain, WithinAbs(exact.total_gain, 1e-10));
}

TEST_CASE("Budgeted selection - posterior is independent of the integration order", "[hints][budgeted]") {
    const DBDDStateStatistics prior = make_prior(9, 1000, false);
    const std::vector<HintCandidate> pool = make_dense_pool(20, 9, 1001);
    const BudgetedHintSelector selector(prior);
    const BudgetedSelectionResult greedy = selector.select_greedy(pool, 6);

    std::vector<size_t> reversed(greedy.selected_indices.rbegin(), greedy.selected_indices.rend());
    const MatrixDouble forward = integrate_sequentially(prior, pool, greedy.selected_indices);
    const MatrixDouble backward = integrate_sequentially(prior, pool, reversed);
    for (size_t i = 0; i < forward.rows(); ++i) {
        for (size_t j = 0; j < forward.cols(); ++j) {
            REQUIRE_THAT(forward(i, j), WithinAbs(backward(i, j), 1e-9));
        }
    }
    REQUIRE_THAT(selector.gain_of_set(pool, reversed), WithinAbs(greedy.total_gain, 1e-9));
}

TEST_CASE("Budgeted selection - lookahead-2 returns a valid selection", "[hints][budgeted]") {
    const DBDDStateStatistics prior = make_prior(10, 1100, false);
    const std::vector<HintCandidate> pool = make_dense_pool(30, 10, 1101);
    const BudgetedHintSelector selector(prior);

    for (size_t k : {size_t{1}, size_t{4}, size_t{7}}) { // odd budgets exercise the final single slot
        const BudgetedSelectionResult look = selector.select_lookahead2(pool, k, 8);
        REQUIRE(look.selected_indices.size() == k);
        REQUIRE(as_set(look.selected_indices).size() == k);
        REQUIRE_THAT(look.total_gain, WithinAbs(selector.gain_of_set(pool, look.selected_indices), 1e-9));
        REQUIRE(look.upper_bound_gain >= look.total_gain);
        REQUIRE(look.beta_after <= look.beta_before + 1e-9);
    }
    REQUIRE_THROWS_AS(selector.select_lookahead2(pool, 4, 1), InvalidParameterException);
}

TEST_CASE("Budgeted selection - beta shortcut agrees with the estimator pipeline", "[hints][budgeted][integration]") {
    // LWE-shaped prior: n = m = 20, q = 3301, variance 20, d = 41.
    const size_t n = 20;
    const size_t m = 20;
    const size_t amb = n + m + 1;
    MatrixDouble cov(amb, amb, 0.0);
    for (size_t i = 0; i + 1 < amb; ++i) {
        cov(i, i) = 20.0;
    }
    VectorDouble mean(amb, 0.0);
    mean[amb - 1] = 1.0;
    DBDDStateStatistics prior(amb, amb, BigReal(1.0), mean, cov);
    prior.set_log_volume(BigReal(static_cast<double>(m) * std::log(3301.0)));

    CandidatePoolOptions options;
    options.n = n;
    options.m = m;
    options.seed = 7;
    const std::vector<HintCandidate> pool = CandidatePools::pool_c(options);

    const BudgetedHintSelector selector(prior);
    const BudgetedSelectionResult greedy = selector.select_greedy(pool, 12);
    REQUIRE(greedy.dimension == amb);
    REQUIRE_THAT(greedy.ln_vol_iso_before, WithinAbs(prior.isotropized_log_volume(), 1e-9));
    REQUIRE(greedy.beta_before_int == prior.estimate_attack_gsa().beta);

    DBDDStateStatistics posterior = prior;
    for (size_t i : greedy.selected_indices) {
        // NOTE: this API takes the standard deviation, the candidate stores the variance.
        posterior.integrate_approximate_hint(pool[i].v_bar, 0.0, std::sqrt(pool[i].variance));
    }
    REQUIRE(posterior.dimension() == amb);
    REQUIRE_THAT(posterior.isotropized_log_volume(),
                 WithinAbs(greedy.ln_vol_iso_before + greedy.total_gain, 1e-7));
    const SecurityEstimate estimate = posterior.estimate_attack_gsa();
    REQUIRE(estimate.beta == greedy.beta_after_int);
    REQUIRE_THAT(estimate.beta_continuous, WithinAbs(greedy.beta_after, 1e-5));
}

TEST_CASE("Budgeted selection - input validation", "[hints][budgeted]") {
    const DBDDStateStatistics prior = make_prior(5, 1200, false);
    const BudgetedHintSelector selector(prior);
    std::vector<HintCandidate> pool = make_dense_pool(4, 5, 1201);

    SECTION("Budget larger than the pool") {
        REQUIRE_THROWS_AS(selector.select_greedy(pool, 5), InvalidParameterException);
        REQUIRE_THROWS_AS(selector.select_lazy_greedy(pool, 5), InvalidParameterException);
        REQUIRE_THROWS_AS(selector.select_exact(pool, 5), InvalidParameterException);
    }
    SECTION("Non-positive variance") {
        pool[2].variance = 0.0;
        REQUIRE_THROWS_AS(selector.select_greedy(pool, 2), InvalidParameterException);
        REQUIRE_THROWS_AS(BudgetedHintSelector::marginal_gain(1.0, 0.0), InvalidParameterException);
    }
    SECTION("Wrong vector length") {
        pool[1].v_bar = VectorDouble(3, 1.0);
        REQUIRE_THROWS_AS(selector.select_static_top_k(pool, 2), DimensionMismatchException);
    }
    SECTION("Exact search refuses large pools") {
        const std::vector<HintCandidate> big = make_dense_pool(17, 5, 1202);
        REQUIRE_THROWS_AS(selector.select_exact(big, 2), InvalidParameterException);
    }
    SECTION("gain_of_set rejects duplicates and out-of-range indices") {
        REQUIRE_THROWS_AS(selector.gain_of_set(pool, {1, 1}), InvalidParameterException);
        REQUIRE_THROWS_AS(selector.gain_of_set(pool, {9}), IndexOutOfBoundsException);
    }
    SECTION("Zero budget is an empty selection") {
        const BudgetedSelectionResult none = selector.select_greedy(pool, 0);
        REQUIRE(none.selected_indices.empty());
        REQUIRE(none.total_gain == 0.0);
        REQUIRE_THAT(none.beta_after, WithinAbs(none.beta_before, 1e-12));
    }
}

TEST_CASE("Candidate pools - sizes, determinism and noise range", "[hints][budgeted][pools]") {
    CandidatePoolOptions options;
    options.n = 12;
    options.m = 10;
    options.seed = 5;
    const size_t amb = options.n + options.m + 1;

    SECTION("Pool A has one hint per coordinate") {
        const auto pool = CandidatePools::pool_a(options);
        REQUIRE(pool.size() == options.n + options.m);
        for (size_t i = 0; i < pool.size(); ++i) {
            REQUIRE(pool[i].v_bar.size() == amb);
            REQUIRE(pool[i].v_bar[i] == 1.0);
            REQUIRE(pool[i].v_bar[amb - 1] == 0.0);
        }
    }
    SECTION("Target size is honoured and the noise stays in range") {
        options.target_size = 57;
        for (CandidatePoolKind kind : {CandidatePoolKind::B, CandidatePoolKind::C, CandidatePoolKind::D}) {
            const auto pool = CandidatePools::make(kind, options);
            REQUIRE(pool.size() == 57);
            for (const HintCandidate& h : pool) {
                REQUIRE(h.v_bar.size() == amb);
                REQUIRE(h.v_bar[amb - 1] == 0.0);
                REQUIRE(h.variance >= options.sigma_min * options.sigma_min - 1e-12);
                REQUIRE(h.variance <= options.sigma_max * options.sigma_max + 1e-9);
                REQUIRE_FALSE(h.label.empty());
            }
        }
    }
    SECTION("Default sizes exceed pool A and generation is deterministic") {
        const auto b1 = CandidatePools::pool_b(options);
        const auto b2 = CandidatePools::pool_b(options);
        REQUIRE(b1.size() > options.n + options.m);
        REQUIRE(b1.size() == b2.size());
        for (size_t i = 0; i < b1.size(); ++i) {
            REQUIRE(b1[i].label == b2[i].label);
            REQUIRE(b1[i].variance == b2[i].variance);
        }
        REQUIRE(CandidatePools::pool_c(options).size() == 2 * (options.n + options.m));
        REQUIRE(CandidatePools::pool_d(options).size() == 2 * (options.n + options.m));
    }
    SECTION("Invalid options are rejected") {
        options.sigma_min = 0.0;
        REQUIRE_THROWS_AS(CandidatePools::pool_a(options), InvalidParameterException);
    }
}
