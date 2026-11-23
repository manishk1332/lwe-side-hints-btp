// examples/07_attack_validation.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 3.4 (GSA-Intersect, probabilistic simulation) and
//                  Section 5.2 / Figure 5 (predictions versus attacks).
// Mathematical Purpose: Validate the block-size predictions against real attacks. For each instance we run
// progressive BKZ (beta = 2, 3, ...) with our own implementation and record the first block size at which the
// secret is found, then compare it with the GSA-Intersect and probabilistic predictions.
//
//   V1  perfect hints on secret coordinates (lattice intersected with the hint hyperplanes, isotropic target).
//   V2  approximate hints on coordinates of (e, s): the posterior covariance stays diagonal, so the lattice is
//       centred on the posterior mean and whitened coordinate-wise, scaled by 2^10 and rounded to an integer basis.
//   V3  budgeted selection (thesis extension): greedy versus static top-k on a pool of repeated coordinate
//       measurements, attacked as in V2.
//
// Usage:
//   example_07_attack_validation --pilot [--verbose]           one timed baseline attack
//   example_07_attack_validation [--n N] [--q Q] [--eta E] [--trials T] [--tours R] [--only V1|V2|V3]
//                                [--k K] [--trial-begin T0] [--out file.csv]
// The report uses --n 30 --q 31 --eta 2 --tours 8 --beta-max 30 --timeout 900, 10 trials per point
// (20 for V3), each trial run as a separate process in parallel.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/reduction/bkz.hpp"
#include "dbdd/security/security_estimator.hpp"
#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/hints/budgeted_selection.hpp"
#include "dbdd/hints/candidate_pools.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace dbdd;

namespace {

struct Config {
    bool pilot = false;
    size_t n = 30;
    long q = 31;
    size_t eta = 2;
    size_t trials = 10;
    size_t beta_max = 30;
    double bkz_timeout = 900.0;
    std::string out = "data/attack_validation.csv";
    std::string only;   // run a single experiment: V1, V2 or V3
    long k = -1;        // run a single hint count (V1, V2)
    size_t trial_begin = 0;
    size_t tours = 8;   // BKZ tours per block size in the progressive schedule
};
size_t g_tours = 8;

struct Prediction {
    double gsa_cont = 0.0;
    size_t gsa_int = 0;
    double prob_avg = 0.0;
};

struct AttackOutcome {
    bool success = false;
    size_t beta = 0;
    double seconds = 0.0;
};

bool g_verbose = false;

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Progressive BKZ: BKZ-2, BKZ-3, ... on the same basis until `found` reports the target among the rows.
template <class Found>
AttackOutcome progressive_bkz(BasisZZ basis, const Found& found, size_t beta_max, double timeout) {
    AttackOutcome out;
    const double t0 = now_seconds();
    const size_t d = basis.dimension();
    for (size_t beta = 2; beta <= std::min(beta_max, d); ++beta) {
        BKZParameters params(beta, g_tours, 0.99);
        params.timeout_seconds = std::max(1.0, timeout - (now_seconds() - t0));
        BKZResult res = bkz_reduce(basis, params);
        basis = std::move(res.basis);
        if (g_verbose) {
            double best = 1e300;
            for (size_t r = 0; r < basis.dimension(); ++r) {
                double nn = 0.0;
                for (size_t j = 0; j < basis.row(r).size(); ++j) {
                    const double x = basis.row(r)[j].to_double();
                    nn += x * x;
                }
                best = std::min(best, nn);
            }
            std::cerr << "    beta " << beta << ": " << res.tours_completed << " tours, " << res.total_svp_nodes
                      << " nodes, min ||b||^2 = " << best << ", t = " << (now_seconds() - t0) << " s\n";
        }
        if (found(basis)) {
            out.success = true;
            out.beta = beta;
            break;
        }
        if (now_seconds() - t0 > timeout) {
            break;
        }
    }
    out.seconds = now_seconds() - t0;
    return out;
}

bool contains_pm(const BasisZZ& basis, const VectorZZ& target) {
    for (size_t r = 0; r < basis.dimension(); ++r) {
        const VectorZZ& row = basis.row(r);
        if (row == target) {
            return true;
        }
        bool neg = true;
        for (size_t j = 0; j < row.size() && neg; ++j) {
            neg = (row[j] == -target[j]);
        }
        if (neg) {
            return true;
        }
    }
    return false;
}

Prediction predict(const DBDDStateStatistics& stats) {
    Prediction p;
    const SecurityEstimate gsa = stats.estimate_attack_gsa();
    p.gsa_cont = gsa.beta_continuous;
    p.gsa_int = gsa.beta;
    p.prob_avg = SecurityEstimator::probabilistic(stats).beta_continuous;
    return p;
}

class Instance {
public:
    Instance(const Config& cfg, uint64_t seed)
        : q_(cfg.q), dist_s_(cfg.eta, seed * 3 + 1), dist_e_(cfg.eta, seed * 3 + 2),
          lwe_(LWEInstance::generate_random(cfg.n, cfg.n, q_, dist_s_, dist_e_, seed)),
          kannan_(KannanEmbedding::from_lwe(lwe_, dist_e_, dist_s_)) {}

    [[nodiscard]] const LWEInstance& lwe() const { return lwe_; }
    [[nodiscard]] const KannanEmbedding& kannan() const { return kannan_; }
    [[nodiscard]] double prior_variance() const { return dist_s_.variance(); }
    [[nodiscard]] size_t free_coords() const { return lwe_.m() + lwe_.n(); }
    [[nodiscard]] DBDDStateStatistics prior_stats() const {
        return DBDDInstance::from_kannan(kannan_).to_state_statistics();
    }

private:
    BigInt q_;
    CenteredBinomial dist_s_;
    CenteredBinomial dist_e_;
    LWEInstance lwe_;
    KannanEmbedding kannan_;
};

// V1: perfect hints revealing the first k secret coordinates. A revealed coordinate s_h is substituted into the
// samples (b <- b - A_h s_h mod q) and its column removed. The Kannan lattice of the reduced instance is the
// original lattice intersected with the hint hyperplanes (dimension d - k, same volume, since each hint is a unit
// dual vector), which avoids the costly exact intersection on the full DBDD instance.
std::pair<Prediction, AttackOutcome> attack_perfect(const Instance& inst, size_t k, const Config& cfg, size_t& dim) {
    const LWEInstance& lwe = inst.lwe();
    const size_t n = lwe.n();
    const size_t m = lwe.m();
    const size_t kk = std::min(k, n);
    const BigInt& q = lwe.q();
    MatrixZZ A(m, n - kk, BigInt(0));
    VectorZZ b = lwe.b();
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < kk; ++j) {
            b[i] -= lwe.A()(i, j) * lwe.test_only_secret()[j];
        }
        b[i] = ((b[i] % q) + q) % q;
        for (size_t j = kk; j < n; ++j) {
            A(i, j - kk) = lwe.A()(i, j);
        }
    }
    VectorZZ s(n - kk, BigInt(0));
    for (size_t j = kk; j < n; ++j) {
        s[j - kk] = lwe.test_only_secret()[j];
    }
    const LWEInstance reduced(n - kk, m, q, A, b, s, lwe.test_only_error());
    const double sigma = std::sqrt(inst.prior_variance());
    const KannanEmbedding kan = KannanEmbedding::from_lwe(reduced, sigma, sigma);
    dim = kan.dimension();
    const Prediction pred = predict(DBDDInstance::from_kannan(kan).to_state_statistics());
    const VectorZZ target = kan.target();
    if (g_verbose) {
        double tn = 0.0;
        for (size_t j = 0; j < target.size(); ++j) tn += target[j].to_double() * target[j].to_double();
        std::cerr << "  target ||t||^2 = " << tn << "\n";
    }
    const AttackOutcome out = progressive_bkz(
        kan.basis(), [&](const BasisZZ& basis) { return contains_pm(basis, target); }, cfg.beta_max, cfg.bkz_timeout);
    return {pred, out};
}

// V2 / V3: approximate hints on coordinates. `precision[j]` is the summed precision (1/sigma^2) of the hints on
// ambient coordinate j and `weighted_obs[j]` the precision-weighted sum of the observations.
std::pair<Prediction, AttackOutcome> attack_diagonal(
    const Instance& inst,
    const std::vector<double>& precision,
    const std::vector<double>& weighted_obs,
    const Config& cfg) {
    const KannanEmbedding& kan = inst.kannan();
    const size_t d = kan.dimension();
    const size_t last = d - 1;
    const double prior_var = inst.prior_variance();

    // Posterior mean and variance per coordinate (prior mean 0).
    std::vector<double> mu(d, 0.0);
    std::vector<double> var(d, prior_var);
    DBDDStateStatistics stats = inst.prior_stats();
    for (size_t j = 0; j < last; ++j) {
        if (precision[j] > 0.0) {
            const double post_prec = 1.0 / prior_var + precision[j];
            var[j] = 1.0 / post_prec;
            mu[j] = var[j] * weighted_obs[j];
            VectorDouble v(d, 0.0);
            v[j] = 1.0;
            stats.integrate_approximate_hint(v, 0.0, std::sqrt(1.0 / precision[j]));
        }
    }
    const Prediction pred = predict(stats);

    // Centre on the posterior mean (x_j -> x_j - mu_j x_last), whiten (divide by sqrt(var_j)), scale and round.
    constexpr double kScale = 1024.0;
    const MatrixZZ& B = kan.matrix();
    MatrixZZ scaled(d, d, BigInt(0));
    for (size_t r = 0; r < d; ++r) {
        const double x_last = B(r, last).to_double();
        for (size_t j = 0; j < last; ++j) {
            const double x = (B(r, j).to_double() - mu[j] * x_last) / std::sqrt(var[j]);
            scaled(r, j) = BigInt(static_cast<long long>(std::llround(kScale * x)));
        }
        scaled(r, last) = BigInt(static_cast<long long>(std::llround(kScale * x_last)));
    }
    // The target's coefficients in the Kannan basis are known in the experiment, so the transformed target is
    // exactly c * round(S B') and can be recognised among the reduced rows.
    const VectorZZ& c = kan.target_coordinates();
    VectorZZ target(d, BigInt(0));
    for (size_t r = 0; r < d; ++r) {
        if (c[r] != BigInt(0)) {
            for (size_t j = 0; j < d; ++j) {
                target[j] += c[r] * scaled(r, j);
            }
        }
    }
    const AttackOutcome out = progressive_bkz(
        BasisZZ(scaled), [&](const BasisZZ& b) { return contains_pm(b, target); }, cfg.beta_max, cfg.bkz_timeout);
    return {pred, out};
}

// Approximate hints with noise variance noise_var on k random coordinates of (e, s).
std::pair<Prediction, AttackOutcome> attack_approximate(
    const Instance& inst, size_t k, double noise_var, uint64_t seed, const Config& cfg) {
    const size_t d = inst.kannan().dimension();
    const VectorZZ& t = inst.kannan().target();
    std::vector<size_t> coords(inst.free_coords());
    std::iota(coords.begin(), coords.end(), size_t{0});
    std::mt19937_64 rng(seed);
    std::shuffle(coords.begin(), coords.end(), rng);
    std::normal_distribution<double> noise(0.0, std::sqrt(noise_var));
    std::vector<double> precision(d, 0.0), weighted(d, 0.0);
    for (size_t i = 0; i < k && i < coords.size(); ++i) {
        const size_t j = coords[i];
        const double y = t[j].to_double() + noise(rng);
        precision[j] += 1.0 / noise_var;
        weighted[j] += y / noise_var;
    }
    return attack_diagonal(inst, precision, weighted, cfg);
}

// Selected candidates (unit vectors) turned into per-coordinate precisions with simulated observations.
std::pair<Prediction, AttackOutcome> attack_selection(
    const Instance& inst, const std::vector<HintCandidate>& pool, const std::vector<size_t>& selected,
    uint64_t seed, const Config& cfg) {
    const size_t d = inst.kannan().dimension();
    const VectorZZ& t = inst.kannan().target();
    std::mt19937_64 rng(seed);
    std::vector<double> precision(d, 0.0), weighted(d, 0.0);
    for (size_t idx : selected) {
        const HintCandidate& h = pool[idx];
        size_t j = d;
        for (size_t c = 0; c < d; ++c) {
            if (h.v_bar[c] != 0.0) {
                j = c;
                break;
            }
        }
        std::normal_distribution<double> noise(0.0, std::sqrt(h.variance));
        const double y = t[j].to_double() + noise(rng);
        precision[j] += 1.0 / h.variance;
        weighted[j] += y / h.variance;
    }
    return attack_diagonal(inst, precision, weighted, cfg);
}

Config parse(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--pilot") cfg.pilot = true;
        else if (a == "--verbose") g_verbose = true;
        else if (a == "--n" && i + 1 < argc) cfg.n = std::stoul(next());
        else if (a == "--q" && i + 1 < argc) cfg.q = std::stol(next());
        else if (a == "--eta" && i + 1 < argc) cfg.eta = std::stoul(next());
        else if (a == "--trials" && i + 1 < argc) cfg.trials = std::stoul(next());
        else if (a == "--beta-max" && i + 1 < argc) cfg.beta_max = std::stoul(next());
        else if (a == "--timeout" && i + 1 < argc) cfg.bkz_timeout = std::stod(next());
        else if (a == "--out" && i + 1 < argc) cfg.out = next();
        else if (a == "--only" && i + 1 < argc) cfg.only = next();
        else if (a == "--k" && i + 1 < argc) cfg.k = std::stol(next());
        else if (a == "--trial-begin" && i + 1 < argc) cfg.trial_begin = std::stoul(next());
        else if (a == "--tours" && i + 1 < argc) cfg.tours = std::stoul(next());
        else {
            std::cerr << "unknown argument " << a << "\n";
            std::exit(2);
        }
    }
    return cfg;
}

void pilot(Config cfg) {
    std::cout << "Timed baseline attack for n=" << cfg.n << " q=" << cfg.q << " eta=" << cfg.eta << ":\n";
    for (uint64_t s = 0; s < 1; ++s) {
        Instance inst(cfg, 9000 + s);
        size_t dim = 0;
        const auto [pred, out] = attack_perfect(inst, 0, cfg, dim);
        std::cout << "  d=" << dim << " GSA " << pred.gsa_cont << " prob " << pred.prob_avg << "  observed "
                  << (out.success ? std::to_string(out.beta) : std::string("fail")) << " in " << out.seconds
                  << " s\n";
    }
}

struct Summary {
    std::vector<double> observed;
    size_t successes = 0;
    size_t runs = 0;
    double gsa = 0.0, prob = 0.0;
    size_t dim = 0;
};

void print_summary(const std::string& label, size_t k, const Summary& s) {
    double mean = 0.0, sd = 0.0;
    if (!s.observed.empty()) {
        mean = std::accumulate(s.observed.begin(), s.observed.end(), 0.0) / static_cast<double>(s.observed.size());
        for (double x : s.observed) sd += (x - mean) * (x - mean);
        sd = std::sqrt(sd / static_cast<double>(s.observed.size()));
    }
    std::cout << std::left << std::setw(10) << label << std::right << std::setw(5) << k << std::setw(6) << s.dim
              << std::fixed << std::setprecision(2) << std::setw(10) << s.gsa / static_cast<double>(s.runs)
              << std::setw(10) << s.prob / static_cast<double>(s.runs) << std::setw(10) << mean << std::setw(8) << sd
              << std::setw(6) << s.successes << "/" << s.runs << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const Config cfg = parse(argc, argv);
    g_tours = cfg.tours;
    std::cout << "DBDD Framework: 07_attack_validation (predicted versus observed BKZ block size)\n";
    if (cfg.pilot) {
        pilot(cfg);
        return 0;
    }

    const std::filesystem::path out_path(cfg.out);
    if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
    std::ofstream csv(cfg.out);
    csv << "experiment,k,trial,dim,beta_gsa_cont,beta_gsa_int,beta_prob_avg,beta_observed,success,seconds\n";
    csv << std::setprecision(10);
    const auto row = [&](const std::string& exp, size_t k, size_t trial, size_t dim, const Prediction& p,
                         const AttackOutcome& o) {
        csv << exp << "," << k << "," << trial << "," << dim << "," << p.gsa_cont << "," << p.gsa_int << ","
            << p.prob_avg << "," << (o.success ? std::to_string(o.beta) : std::string("")) << ","
            << (o.success ? 1 : 0) << "," << o.seconds << "\n";
        csv.flush();
    };
    const auto add = [](Summary& s, size_t dim, const Prediction& p, const AttackOutcome& o) {
        s.dim = dim;
        s.gsa += p.gsa_cont;
        s.prob += p.prob_avg;
        ++s.runs;
        if (o.success) {
            ++s.successes;
            s.observed.push_back(static_cast<double>(o.beta));
        }
    };

    std::cout << "n = m = " << cfg.n << ", q = " << cfg.q << ", CBD(" << cfg.eta << "), d = " << (2 * cfg.n + 1)
              << ", " << cfg.trials << " trials per point\n";
    std::cout << std::left << std::setw(10) << "exp" << std::right << std::setw(5) << "k" << std::setw(6) << "dim"
              << std::setw(10) << "GSA" << std::setw(10) << "prob" << std::setw(10) << "observed" << std::setw(8)
              << "sd" << std::setw(8) << "solved\n";

    // V1: perfect coordinate hints.
    if (cfg.only.empty() || cfg.only == "V1") {
        for (size_t k : {0, 2, 4, 6, 8, 12}) {
            if (k > cfg.n) break;
            if (cfg.k >= 0 && static_cast<size_t>(cfg.k) != k) continue;
            Summary s;
            for (size_t t = cfg.trial_begin; t < cfg.trial_begin + cfg.trials; ++t) {
                Instance inst(cfg, 1000 + t);
                size_t dim = 0;
                const auto [p, o] = attack_perfect(inst, k, cfg, dim);
                row("V1", k, t, dim, p, o);
                add(s, dim, p, o);
            }
            print_summary("V1", k, s);
        }
    }

    // V2: approximate coordinate hints, sigma_h^2 = 3.
    if (cfg.only.empty() || cfg.only == "V2") {
        // Approximate hints on 0, 1/4, 1/2, 3/4 and all of the 2n coordinates of (e, s).
        for (size_t quarter = 0; quarter <= 4; ++quarter) {
            const size_t k = quarter * 2 * cfg.n / 4;
            if (cfg.k >= 0 && static_cast<size_t>(cfg.k) != k) continue;
            Summary s;
            for (size_t t = cfg.trial_begin; t < cfg.trial_begin + cfg.trials; ++t) {
                Instance inst(cfg, 1000 + t);
                const auto [p, o] = attack_approximate(inst, k, 3.0, 5000 + 100 * k + t, cfg);
                row("V2", k, t, 2 * cfg.n + 1, p, o);
                add(s, 2 * cfg.n + 1, p, o);
            }
            print_summary("V2", k, s);
        }
    }

    // V3: greedy versus static top-k on a pool of repeated coordinate measurements (pool B).
    if (cfg.only.empty() || cfg.only == "V3") {
        // Pick the budget with the largest predicted greedy/static gap on the first instance. GSA-Intersect is
        // degenerate at this dimension, so the gap is measured with the probabilistic simulation.
        size_t best_k = 0;
        double best_gap = -1.0;
        {
            Instance inst(cfg, 1000);
            CandidatePoolOptions opt;
            opt.n = cfg.n;
            opt.m = cfg.n;
            opt.sigma_min = 0.25;  // informative hints relative to the prior variance of CBD(2)
            opt.sigma_max = 2.0;
            opt.seed = 77;
            const auto pool = CandidatePools::pool_b(opt);
            const DBDDStateStatistics prior = inst.prior_stats();
            const BudgetedHintSelector sel(prior);
            const auto prob_after = [&](const BudgetedSelectionResult& r) {
                DBDDStateStatistics post = prior;
                for (size_t i : r.selected_indices) {
                    post.integrate_approximate_hint(pool[i].v_bar, 0.0, std::sqrt(pool[i].variance));
                }
                return SecurityEstimator::probabilistic(post).beta_continuous;
            };
            for (size_t k = 5; k <= std::min<size_t>(60, pool.size()); k += 5) {
                const double pg = prob_after(sel.select_greedy(pool, k));
                const double ps = prob_after(sel.select_static_top_k(pool, k));
                if (g_verbose) std::cerr << "  V3 k=" << k << " prob greedy " << pg << " static " << ps << "\n";
                if (ps - pg > best_gap) {
                    best_gap = ps - pg;
                    best_k = k;
                }
            }
            std::cout << "V3 budget k = " << best_k << " (predicted probabilistic gap " << best_gap << ")\n";
        }
        Summary sg, ss;
        for (size_t t = cfg.trial_begin; t < cfg.trial_begin + cfg.trials; ++t) {
            Instance inst(cfg, 1000 + t);
            CandidatePoolOptions opt;
            opt.n = cfg.n;
            opt.m = cfg.n;
            opt.sigma_min = 0.25;
            opt.sigma_max = 2.0;
            opt.seed = 77 + t;
            const auto pool = CandidatePools::pool_b(opt);
            const BudgetedHintSelector sel(inst.prior_stats());
            const auto greedy = sel.select_greedy(pool, best_k);
            const auto fixed = sel.select_static_top_k(pool, best_k);
            const auto [pg, og] = attack_selection(inst, pool, greedy.selected_indices, 7000 + t, cfg);
            const auto [ps, os] = attack_selection(inst, pool, fixed.selected_indices, 7000 + t, cfg);
            row("V3-greedy", best_k, t, 2 * cfg.n + 1, pg, og);
            row("V3-static", best_k, t, 2 * cfg.n + 1, ps, os);
            add(sg, 2 * cfg.n + 1, pg, og);
            add(ss, 2 * cfg.n + 1, ps, os);
        }
        print_summary("V3-greedy", best_k, sg);
        print_summary("V3-static", best_k, ss);
    }

    std::cout << "Wrote " << cfg.out << "\n";
    return 0;
}
