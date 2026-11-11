// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 4.3, Equations (21)-(23);
//                  thesis extension "Budgeted Hint Selection as D-Optimal Design".
// Mathematical Purpose: Selection strategies for a budget of k approximate hints. See budgeted_selection.hpp.

#include "dbdd/hints/budgeted_selection.hpp"
#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/core/exceptions.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <utility>

namespace dbdd {

namespace {

// A marginal gain at or below this threshold (in nats) is treated as "no information left".
constexpr double kGainEps = 1e-12;

class Stopwatch {
public:
    void start() { begin_ = std::chrono::steady_clock::now(); }
    void stop() {
        total_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin_).count();
    }
    [[nodiscard]] double elapsed_ms() const noexcept { return total_ms_; }

private:
    std::chrono::steady_clock::time_point begin_{};
    double total_ms_{0.0};
};

void check_budget(size_t k, size_t pool_size, const char* where) {
    if (k > pool_size) {
        throw InvalidParameterException(
            std::string(where) + ": budget k (" + std::to_string(k) +
            ") exceeds the candidate pool size (" + std::to_string(pool_size) + ")");
    }
}

// Sum of the `count` largest values (all of them if there are fewer).
double sum_of_largest(std::vector<double> values, size_t count) {
    if (values.size() > count) {
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count), values.end(),
                         std::greater<double>());
        values.resize(count);
    }
    return std::accumulate(values.begin(), values.end(), 0.0);
}

// Worst-case greedy bound of Nemhauser-Wolsey-Fisher: f(optimal) <= f(greedy_k) / (1 - (1 - 1/k)^k).
double greedy_worst_case_bound(double greedy_gain, size_t k) {
    const double kd = static_cast<double>(k);
    return greedy_gain / (1.0 - std::pow(1.0 - 1.0 / kd, kd));
}

// Incremental engine shared by all strategies. Holds the running covariance Sigma_S, the candidate rows in
// sparse form, and (optionally) the directional variances q_i = w_i Sigma_S w_i^T of all candidates.
class Engine {
public:
    Engine(const MatrixDouble& covariance, size_t amb_dim, const std::vector<HintCandidate>& candidates)
        : sigma_(covariance.to_eigen()),
          rows_(candidates.size()),
          variances_(candidates.size()),
          selected_(candidates.size(), false) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            const HintCandidate& c = candidates[i];
            if (c.v_bar.size() != amb_dim) {
                throw DimensionMismatchException(
                    "BudgetedHintSelector: candidate " + std::to_string(i) + " has length " +
                    std::to_string(c.v_bar.size()) + ", expected ambient dimension " + std::to_string(amb_dim));
            }
            if (!(c.variance > 0.0) || !std::isfinite(c.variance)) {
                throw InvalidParameterException(
                    "BudgetedHintSelector: candidate " + std::to_string(i) +
                    " must have a finite noise variance > 0");
            }
            variances_[i] = c.variance;
            for (size_t j = 0; j < amb_dim; ++j) {
                if (c.v_bar[j] != 0.0) {
                    rows_[i].emplace_back(static_cast<Eigen::Index>(j), c.v_bar[j]);
                }
            }
        }
    }

    [[nodiscard]] size_t size() const noexcept { return rows_.size(); }
    [[nodiscard]] bool is_selected(size_t i) const noexcept { return selected_[i]; }
    [[nodiscard]] double variance(size_t i) const noexcept { return variances_[i]; }
    [[nodiscard]] double total_gain() const noexcept { return total_gain_; }
    [[nodiscard]] const std::vector<size_t>& order() const noexcept { return order_; }
    [[nodiscard]] const std::vector<double>& step_gains() const noexcept { return step_gains_; }

    // u = Sigma_S * w_i^T
    [[nodiscard]] Eigen::VectorXd apply_sigma(size_t i) const {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(sigma_.rows());
        for (const auto& [idx, val] : rows_[i]) {
            u.noalias() += val * sigma_.col(idx);
        }
        return u;
    }

    // <w_i, u>
    [[nodiscard]] double dot(size_t i, const Eigen::VectorXd& u) const {
        double s = 0.0;
        for (const auto& [idx, val] : rows_[i]) {
            s += val * u(idx);
        }
        return s;
    }

    // q_i = w_i Sigma_S w_i^T computed from scratch.
    [[nodiscard]] double fresh_q(size_t i) const {
        double s = 0.0;
        for (const auto& [ia, va] : rows_[i]) {
            for (const auto& [ib, vb] : rows_[i]) {
                s += va * vb * sigma_(ia, ib);
            }
        }
        return std::max(s, 0.0);
    }

    void compute_all_q() {
        q_.assign(size(), 0.0);
        for (size_t i = 0; i < size(); ++i) {
            if (!selected_[i]) {
                q_[i] = fresh_q(i);
            }
        }
    }

    [[nodiscard]] double q(size_t i) const noexcept { return q_[i]; }
    [[nodiscard]] double gain(size_t i) const { return BudgetedHintSelector::marginal_gain(q_[i], variances_[i]); }

    // Integrates candidate j: rank-1 covariance update, and (if requested) the incremental update of every
    // remaining q_i. Returns the realised marginal gain.
    double select(size_t j, bool update_all_q) {
        const Eigen::VectorXd u = apply_sigma(j);
        const double qj = std::max(dot(j, u), 0.0);
        const double denom = qj + variances_[j];
        const double g = BudgetedHintSelector::marginal_gain(qj, variances_[j]);

        selected_[j] = true;
        if (update_all_q) {
            for (size_t i = 0; i < size(); ++i) {
                if (!selected_[i]) {
                    const double p = dot(i, u);
                    q_[i] = std::max(q_[i] - (p * p) / denom, 0.0);
                }
            }
            q_[j] = 0.0;
        }
        sigma_.noalias() -= (u / denom) * u.transpose();

        total_gain_ += g;
        order_.push_back(j);
        step_gains_.push_back(g);
        return g;
    }

    // Sum of the `count` largest marginal gains among the candidates not yet selected. Requires current q.
    [[nodiscard]] double top_remaining_gain_sum(size_t count) const {
        std::vector<double> gains;
        gains.reserve(size());
        for (size_t i = 0; i < size(); ++i) {
            if (!selected_[i]) {
                gains.push_back(gain(i));
            }
        }
        return sum_of_largest(std::move(gains), count);
    }

    // Indices of the unselected candidates, best current gain first (ties towards the smaller index).
    [[nodiscard]] std::vector<size_t> ranked_unselected() const {
        std::vector<size_t> idx;
        idx.reserve(size());
        for (size_t i = 0; i < size(); ++i) {
            if (!selected_[i]) {
                idx.push_back(i);
            }
        }
        std::stable_sort(idx.begin(), idx.end(), [this](size_t a, size_t b) { return gain(a) > gain(b); });
        return idx;
    }

private:
    Eigen::MatrixXd sigma_;
    std::vector<std::vector<std::pair<Eigen::Index, double>>> rows_;
    std::vector<double> variances_;
    std::vector<bool> selected_;
    std::vector<double> q_;
    double total_gain_{0.0};
    std::vector<size_t> order_;
    std::vector<double> step_gains_;
};

// M = I + D^{-1/2} (W Sigma W^T) D^{-1/2}; then f(S) = 0.5 * ln det(M_SS).
Eigen::MatrixXd scaled_gram_matrix(const Engine& engine) {
    const size_t n = engine.size();
    Eigen::MatrixXd m = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    for (size_t i = 0; i < n; ++i) {
        const Eigen::VectorXd u = engine.apply_sigma(i);
        for (size_t j = 0; j < n; ++j) {
            m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) +=
                engine.dot(j, u) / std::sqrt(engine.variance(i) * engine.variance(j));
        }
    }
    return m;
}

double gram_gain(const Eigen::MatrixXd& m, const std::vector<size_t>& subset) {
    const auto k = static_cast<Eigen::Index>(subset.size());
    if (k == 0) {
        return 0.0;
    }
    Eigen::MatrixXd sub(k, k);
    for (Eigen::Index a = 0; a < k; ++a) {
        for (Eigen::Index b = 0; b < k; ++b) {
            sub(a, b) = m(static_cast<Eigen::Index>(subset[static_cast<size_t>(a)]),
                          static_cast<Eigen::Index>(subset[static_cast<size_t>(b)]));
        }
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(sub);
    if (llt.info() != Eigen::Success) {
        throw NumericalPrecisionException("BudgetedHintSelector: Gram matrix is not numerically positive definite");
    }
    double log_det = 0.0;
    const auto& l = llt.matrixLLT();
    for (Eigen::Index a = 0; a < k; ++a) {
        log_det += 2.0 * std::log(l(a, a));
    }
    return 0.5 * log_det;
}

BudgetedSelectionResult make_result(
    const Engine& engine,
    double upper_bound_gain,
    size_t gain_evaluations,
    double elapsed_ms,
    size_t dim,
    double ln_vol_iso) {
    BudgetedSelectionResult r;
    r.selected_indices = engine.order();
    r.step_gains = engine.step_gains();
    r.total_gain = engine.total_gain();
    // A valid upper bound on f(optimal) can never be below a gain that was actually achieved.
    r.upper_bound_gain = std::max(upper_bound_gain, r.total_gain);
    r.dimension = dim;
    r.ln_vol_iso_before = ln_vol_iso;
    r.gain_evaluations = gain_evaluations;
    r.elapsed_ms = elapsed_ms;

    const SecurityEstimate before = GSAIntersect::estimate(dim, ln_vol_iso);
    const SecurityEstimate after = GSAIntersect::estimate(dim, ln_vol_iso + r.total_gain);
    const SecurityEstimate bound = GSAIntersect::estimate(dim, ln_vol_iso + r.upper_bound_gain);
    r.beta_before = before.beta_continuous;
    r.beta_before_int = before.beta;
    r.beta_after = after.beta_continuous;
    r.beta_after_int = after.beta;
    r.beta_opt_lower = bound.beta_continuous;
    r.beta_opt_lower_int = bound.beta;
    return r;
}

} // namespace

BudgetedHintSelector::BudgetedHintSelector(const DBDDStateStatistics& prior)
    : dim_(prior.dimension()),
      amb_dim_(prior.ambient_dimension()),
      ln_vol_iso_(prior.isotropized_log_volume()),
      covariance_(prior.covariance()) {}

double BudgetedHintSelector::marginal_gain(double q, double variance) {
    if (!(variance > 0.0)) {
        throw InvalidParameterException("BudgetedHintSelector::marginal_gain: variance must be > 0");
    }
    return 0.5 * std::log1p(std::max(q, 0.0) / variance);
}

BudgetedSelectionResult BudgetedHintSelector::select_random(
    const std::vector<HintCandidate>& candidates, size_t k, uint64_t seed) const {
    Engine engine(covariance_, amb_dim_, candidates);
    check_budget(k, engine.size(), "BudgetedHintSelector::select_random");

    std::vector<size_t> perm(engine.size());
    std::iota(perm.begin(), perm.end(), size_t{0});
    std::mt19937_64 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    engine.compute_all_q(); // for the certificate only
    double bound = engine.top_remaining_gain_sum(k);

    Stopwatch sw;
    sw.start();
    for (size_t t = 0; t < k; ++t) {
        engine.select(perm[t], true);
    }
    sw.stop();

    bound = std::min(bound, engine.total_gain() + engine.top_remaining_gain_sum(k));
    return make_result(engine, bound, 0, sw.elapsed_ms(), dim_, ln_vol_iso_);
}

BudgetedSelectionResult BudgetedHintSelector::select_static_top_k(
    const std::vector<HintCandidate>& candidates, size_t k) const {
    Engine engine(covariance_, amb_dim_, candidates);
    check_budget(k, engine.size(), "BudgetedHintSelector::select_static_top_k");

    Stopwatch sw;
    sw.start();
    engine.compute_all_q();
    const std::vector<size_t> ranking = engine.ranked_unselected();
    sw.stop();
    const size_t evaluations = engine.size();
    double bound = engine.top_remaining_gain_sum(k);

    sw.start();
    for (size_t t = 0; t < k; ++t) {
        engine.select(ranking[t], true);
    }
    sw.stop();

    bound = std::min(bound, engine.total_gain() + engine.top_remaining_gain_sum(k));
    return make_result(engine, bound, evaluations, sw.elapsed_ms(), dim_, ln_vol_iso_);
}

BudgetedSelectionResult BudgetedHintSelector::select_greedy(
    const std::vector<HintCandidate>& candidates, size_t k) const {
    Engine engine(covariance_, amb_dim_, candidates);
    check_budget(k, engine.size(), "BudgetedHintSelector::select_greedy");
    const size_t n = engine.size();

    Stopwatch sw;
    sw.start();
    engine.compute_all_q();
    sw.stop();
    size_t evaluations = n;
    double bound = engine.top_remaining_gain_sum(k); // U(empty)

    while (engine.order().size() < k) {
        sw.start();
        size_t best = n;
        double best_gain = -1.0;
        for (size_t i = 0; i < n; ++i) {
            if (!engine.is_selected(i)) {
                const double g = engine.gain(i);
                if (g > best_gain) {
                    best_gain = g;
                    best = i;
                }
            }
        }
        sw.stop();
        if (best == n || best_gain <= kGainEps) {
            break;
        }
        sw.start();
        engine.select(best, true);
        sw.stop();
        evaluations += n - engine.order().size();
        bound = std::min(bound, engine.total_gain() + engine.top_remaining_gain_sum(k)); // U(S_j)
    }

    if (k > 0 && engine.order().size() == k) {
        bound = std::min(bound, greedy_worst_case_bound(engine.total_gain(), k));
    }
    return make_result(engine, bound, evaluations, sw.elapsed_ms(), dim_, ln_vol_iso_);
}

BudgetedSelectionResult BudgetedHintSelector::select_lazy_greedy(
    const std::vector<HintCandidate>& candidates, size_t k) const {
    Engine engine(covariance_, amb_dim_, candidates);
    check_budget(k, engine.size(), "BudgetedHintSelector::select_lazy_greedy");
    const size_t n = engine.size();

    struct Entry {
        double gain;
        size_t index;
        size_t stamp; // |S| at the time the gain was computed
    };
    // Max-heap on the gain; ties towards the smaller index, as in select_greedy.
    const auto worse = [](const Entry& a, const Entry& b) {
        return a.gain < b.gain || (a.gain == b.gain && a.index > b.index);
    };
    std::priority_queue<Entry, std::vector<Entry>, decltype(worse)> heap(worse);

    Stopwatch sw;
    sw.start();
    std::vector<double> initial_gains(n);
    for (size_t i = 0; i < n; ++i) {
        initial_gains[i] = marginal_gain(engine.fresh_q(i), engine.variance(i));
        heap.push(Entry{initial_gains[i], i, 0});
    }
    size_t evaluations = n;

    while (engine.order().size() < k && !heap.empty()) {
        const Entry top = heap.top();
        heap.pop();
        const size_t current = engine.order().size();
        if (top.stamp == current) {
            // The gain is fresh and on top; every other key is an upper bound on its candidate's current gain.
            if (top.gain <= kGainEps) {
                break;
            }
            engine.select(top.index, false);
        } else {
            const double g = marginal_gain(engine.fresh_q(top.index), engine.variance(top.index));
            ++evaluations;
            heap.push(Entry{g, top.index, current});
        }
    }
    sw.stop();

    // Certificate (not part of the timed selection).
    double bound = sum_of_largest(initial_gains, k); // U(empty)
    engine.compute_all_q();
    bound = std::min(bound, engine.total_gain() + engine.top_remaining_gain_sum(k));
    if (k > 0 && engine.order().size() == k) {
        bound = std::min(bound, greedy_worst_case_bound(engine.total_gain(), k));
    }
    return make_result(engine, bound, evaluations, sw.elapsed_ms(), dim_, ln_vol_iso_);
}

BudgetedSelectionResult BudgetedHintSelector::select_lookahead2(
    const std::vector<HintCandidate>& candidates, size_t k, size_t m) const {
    if (m < 2) {
        throw InvalidParameterException("BudgetedHintSelector::select_lookahead2: m must be >= 2");
    }
    Engine engine(covariance_, amb_dim_, candidates);
    check_budget(k, engine.size(), "BudgetedHintSelector::select_lookahead2");
    const size_t n = engine.size();

    Stopwatch sw;
    sw.start();
    engine.compute_all_q();
    sw.stop();
    size_t evaluations = n;
    double bound = engine.top_remaining_gain_sum(k); // U(empty)

    sw.start();
    while (engine.order().size() < k) {
        std::vector<size_t> top = engine.ranked_unselected();
        if (top.empty()) {
            break;
        }
        if (top.size() > m) {
            top.resize(m);
        }
        const size_t remaining = k - engine.order().size();

        if (remaining == 1 || top.size() == 1) {
            if (engine.gain(top[0]) <= kGainEps) {
                break;
            }
            engine.select(top[0], true);
            evaluations += n - engine.order().size();
            continue;
        }

        size_t best_a = top[0];
        size_t best_b = top[1];
        double best_joint = -1.0;
        for (size_t ai = 0; ai < top.size(); ++ai) {
            const size_t a = top[ai];
            const Eigen::VectorXd u = engine.apply_sigma(a);
            const double gain_a = engine.gain(a);
            const double denom = engine.q(a) + engine.variance(a);
            for (size_t bi = ai + 1; bi < top.size(); ++bi) {
                const size_t b = top[bi];
                const double cross = engine.dot(b, u);
                const double q_b_given_a = engine.q(b) - (cross * cross) / denom;
                const double joint = gain_a + marginal_gain(q_b_given_a, engine.variance(b));
                ++evaluations;
                if (joint > best_joint) {
                    best_joint = joint;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        if (best_joint <= kGainEps) {
            break;
        }
        engine.select(best_a, true);
        engine.select(best_b, true);
        evaluations += 2 * (n - engine.order().size()) + 1;
    }
    sw.stop();

    bound = std::min(bound, engine.total_gain() + engine.top_remaining_gain_sum(k));
    return make_result(engine, bound, evaluations, sw.elapsed_ms(), dim_, ln_vol_iso_);
}

BudgetedSelectionResult BudgetedHintSelector::select_exact(
    const std::vector<HintCandidate>& candidates, size_t k) const {
    if (candidates.size() > max_exact_pool_size()) {
        throw InvalidParameterException(
            "BudgetedHintSelector::select_exact: pool size " + std::to_string(candidates.size()) +
            " exceeds the limit of " + std::to_string(max_exact_pool_size()));
    }
    Engine engine(covariance_, amb_dim_, candidates);
    check_budget(k, engine.size(), "BudgetedHintSelector::select_exact");
    const size_t n = engine.size();

    Stopwatch sw;
    sw.start();
    const Eigen::MatrixXd gram = scaled_gram_matrix(engine);

    // Enumerate all k-subsets in lexicographic order.
    std::vector<size_t> subset(k);
    std::iota(subset.begin(), subset.end(), size_t{0});
    std::vector<size_t> best_subset = subset;
    double best_gain = -1.0;
    size_t evaluations = 0;
    bool more = true;
    while (more) {
        const double g = gram_gain(gram, subset);
        ++evaluations;
        if (g > best_gain) {
            best_gain = g;
            best_subset = subset;
        }
        // Advance to the next combination.
        more = false;
        for (size_t pos = k; pos-- > 0;) {
            if (subset[pos] != pos + n - k) {
                ++subset[pos];
                for (size_t t = pos + 1; t < k; ++t) {
                    subset[t] = subset[t - 1] + 1;
                }
                more = true;
                break;
            }
        }
    }
    sw.stop();

    for (size_t j : best_subset) {
        engine.select(j, false);
    }
    // The enumerated set is optimal, so its own gain is the exact bound.
    return make_result(engine, engine.total_gain(), evaluations, sw.elapsed_ms(), dim_, ln_vol_iso_);
}

double BudgetedHintSelector::gain_of_set(
    const std::vector<HintCandidate>& candidates, const std::vector<size_t>& indices) const {
    std::vector<bool> seen(candidates.size(), false);
    for (size_t i : indices) {
        if (i >= candidates.size()) {
            throw IndexOutOfBoundsException("BudgetedHintSelector::gain_of_set: index out of range");
        }
        if (seen[i]) {
            throw InvalidParameterException("BudgetedHintSelector::gain_of_set: duplicate index");
        }
        seen[i] = true;
    }
    std::vector<HintCandidate> chosen;
    chosen.reserve(indices.size());
    for (size_t i : indices) {
        chosen.push_back(candidates[i]);
    }
    const Engine engine(covariance_, amb_dim_, chosen);
    const Eigen::MatrixXd gram = scaled_gram_matrix(engine);
    std::vector<size_t> all(chosen.size());
    std::iota(all.begin(), all.end(), size_t{0});
    return gram_gain(gram, all);
}

} // namespace dbdd
