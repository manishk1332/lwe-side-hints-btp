// examples/06_budgeted_selection.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 4.3 (approximate hints) and Section 5.2 (benchmark
//                  instance); thesis extension "Budgeted Hint Selection as D-Optimal Design".
// Mathematical Purpose: Benchmark of budgeted hint selection on the paper instance n = m = 70, q = 3301,
// sigma^2 = 20 (d = 141). An attacker may integrate at most k approximate hints out of a pool of N candidates.
//
//   E1  quality      : f(S) and beta versus the budget k for random, static top-k, greedy and lookahead-2.
//   E2  optimality   : greedy versus brute force on tiny pools (N = 14, k = 4).
//   E3  certificate  : the bound U_best reported with every greedy row of E1.
//   E4  cost         : gain evaluations and time of greedy versus lazy greedy as N grows.
//
// Usage: example_06_budgeted_selection [--quick] [--out <csv path>]

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/hints/budgeted_selection.hpp"
#include "dbdd/hints/candidate_pools.hpp"
#include "dbdd/security/lightweight_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dbdd;

namespace {

struct Config {
    bool quick = false;
    std::string out = "data/budgeted_selection_results.csv";
};

struct Mean {
    double sum = 0.0;
    size_t count = 0;
    void add(double x) { sum += x; ++count; }
    [[nodiscard]] double value() const { return count == 0 ? 0.0 : sum / static_cast<double>(count); }
};

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) {
        const std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        file_.open(path);
        if (!file_) {
            throw std::runtime_error("cannot open output file: " + path);
        }
        file_ << "experiment,pool,N,k,strategy,trial,f,U_best,beta_before,beta_cont,beta_int,"
                 "beta_opt_lower_cont,beta_opt_lower_int,evaluations,ms\n";
        file_ << std::setprecision(12);
    }

    void row(const std::string& experiment, const std::string& pool, size_t pool_size, size_t k,
             const std::string& strategy, size_t trial, const BudgetedSelectionResult& r) {
        file_ << experiment << ',' << pool << ',' << pool_size << ',' << k << ',' << strategy << ',' << trial << ','
              << r.total_gain << ',' << r.upper_bound_gain << ',' << r.beta_before << ',' << r.beta_after << ','
              << r.beta_after_int << ',' << r.beta_opt_lower << ',' << r.beta_opt_lower_int << ','
              << r.gain_evaluations << ',' << r.elapsed_ms << '\n';
        ++rows_;
    }

    [[nodiscard]] size_t rows() const noexcept { return rows_; }

private:
    std::ofstream file_;
    size_t rows_{0};
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quick") {
            cfg.quick = true;
        } else if (arg == "--out" && i + 1 < argc) {
            cfg.out = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " [--quick] [--out <csv path>]\n";
            std::exit(arg == "--help" ? 0 : 2);
        }
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);

    std::cout << "==================================================================================\n";
    std::cout << "      DBDD Framework: 06_budgeted_selection (hint selection as D-optimal design)\n";
    std::cout << "==================================================================================\n";

    // Paper benchmark instance (Section 5.2): CBD(40) has variance 20.
    const size_t n = 70;
    const size_t m = 70;
    const BigInt q(3301);
    CenteredBinomial secret_dist(40, 101);
    CenteredBinomial noise_dist(40, 102);
    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 20200292);
    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    const DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
    const DBDDStateStatistics prior = dbdd.to_state_statistics();
    const BudgetedHintSelector selector(prior);

    const size_t beta0 = prior.estimate_attack_gsa().beta;
    std::cout << "Instance: n=" << n << ", m=" << m << ", q=3301, variance=20, d=" << prior.dimension()
              << ", baseline beta=" << beta0 << "\n";
    std::cout << "Mode: " << (cfg.quick ? "quick" : "full") << ", output: " << cfg.out << "\n\n";

    CsvWriter csv(cfg.out);
    const std::vector<CandidatePoolKind> kinds = {
        CandidatePoolKind::A, CandidatePoolKind::B, CandidatePoolKind::C, CandidatePoolKind::D};

    // ---------------------------------------------------------------------------------------------------------
    // E1 / E3: quality of each strategy and the certificate of the greedy set.
    // ---------------------------------------------------------------------------------------------------------
    const std::vector<size_t> budgets = {5, 10, 20, 30, 40, 60};
    const size_t e1_trials = cfg.quick ? 3 : 30;
    const size_t random_seeds = cfg.quick ? 5 : 20;
    const size_t lookahead_width = 16;
    const std::vector<std::string> strategies = {"random", "static", "greedy", "lookahead2"};

    // (pool, k, strategy) -> mean continuous beta; (pool, k) -> mean certified lower bound on the optimum.
    std::map<std::string, Mean> beta_mean;
    std::map<std::string, Mean> gain_mean;
    std::map<std::string, Mean> bound_mean;
    const auto key = [](const std::string& pool, size_t k, const std::string& strategy) {
        return pool + "/" + std::to_string(k) + "/" + strategy;
    };

    for (CandidatePoolKind kind : kinds) {
        const std::string pool_name = CandidatePools::name(kind);
        for (size_t trial = 0; trial < e1_trials; ++trial) {
            CandidatePoolOptions options;
            options.n = n;
            options.m = m;
            options.seed = 1000 * (static_cast<uint64_t>(kind) + 1) + trial;
            const std::vector<HintCandidate> pool = CandidatePools::make(kind, options);

            for (size_t k : budgets) {
                if (k > pool.size()) {
                    continue;
                }
                const auto record = [&](const std::string& strategy, const BudgetedSelectionResult& r) {
                    csv.row("E1", pool_name, pool.size(), k, strategy, trial, r);
                    beta_mean[key(pool_name, k, strategy)].add(r.beta_after);
                    gain_mean[key(pool_name, k, strategy)].add(r.total_gain);
                };
                for (size_t s = 0; s < random_seeds; ++s) {
                    record("random", selector.select_random(pool, k, 77 + 1000 * trial + s));
                }
                record("static", selector.select_static_top_k(pool, k));
                const BudgetedSelectionResult greedy = selector.select_greedy(pool, k);
                record("greedy", greedy);
                bound_mean[key(pool_name, k, "bound")].add(greedy.beta_opt_lower);
                record("lookahead2", selector.select_lookahead2(pool, k, lookahead_width));
            }
        }
    }

    std::cout << "E1/E3: mean continuous beta after selection (certified lower bound on the optimum in brackets)\n";
    std::cout << std::left << std::setw(6) << "pool" << std::setw(6) << "k" << std::right
              << std::setw(11) << "random" << std::setw(11) << "static" << std::setw(11) << "greedy"
              << std::setw(13) << "lookahead2" << std::setw(14) << "[opt >=]" << "\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    for (CandidatePoolKind kind : kinds) {
        const std::string pool_name = CandidatePools::name(kind);
        for (size_t k : budgets) {
            if (beta_mean.find(key(pool_name, k, "greedy")) == beta_mean.end()) {
                continue;
            }
            std::cout << std::left << std::setw(6) << pool_name << std::setw(6) << k << std::right;
            for (const std::string& strategy : strategies) {
                std::cout << std::setw(strategy == "lookahead2" ? 13 : 11)
                          << beta_mean[key(pool_name, k, strategy)].value();
            }
            std::cout << std::setw(14) << bound_mean[key(pool_name, k, "bound")].value() << "\n";
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // E2: greedy versus brute force on tiny pools.
    // ---------------------------------------------------------------------------------------------------------
    // Fourteen candidates spread over the 140 coordinates of the benchmark instance hardly overlap, which makes
    // the problem almost additive and greedy trivially optimal. To stress the greedy rule the tiny pools live on a
    // compact instance (n = m = 6, same q and variance), where every candidate interacts with several others.
    // Only the gain ratio is reported here; block sizes are not meaningful in dimension 13.
    const size_t e2_trials = cfg.quick ? 20 : 200;
    const size_t tiny_n = 14;
    const size_t tiny_k = 4;
    const size_t compact = 6;
    const size_t compact_amb = 2 * compact + 1;
    MatrixDouble compact_cov(compact_amb, compact_amb, 0.0);
    for (size_t i = 0; i + 1 < compact_amb; ++i) {
        compact_cov(i, i) = 20.0;
    }
    VectorDouble compact_mean(compact_amb, 0.0);
    compact_mean[compact_amb - 1] = 1.0;
    DBDDStateStatistics compact_prior(compact_amb, compact_amb, BigReal(1.0), compact_mean, compact_cov);
    compact_prior.set_log_volume(BigReal(static_cast<double>(compact) * std::log(3301.0)));
    const BudgetedHintSelector compact_selector(compact_prior);

    const std::vector<CandidatePoolKind> tiny_kinds = {
        CandidatePoolKind::B, CandidatePoolKind::C, CandidatePoolKind::D};
    double worst_ratio = 1.0;
    Mean ratio_mean;
    size_t greedy_optimal = 0;
    for (size_t trial = 0; trial < e2_trials; ++trial) {
        const CandidatePoolKind kind = tiny_kinds[trial % tiny_kinds.size()];
        CandidatePoolOptions options;
        options.n = compact;
        options.m = compact;
        options.target_size = tiny_n;
        options.seed = 50000 + trial;
        const std::vector<HintCandidate> pool = CandidatePools::make(kind, options);

        const BudgetedSelectionResult greedy = compact_selector.select_greedy(pool, tiny_k);
        const BudgetedSelectionResult exact = compact_selector.select_exact(pool, tiny_k);
        csv.row("E2", CandidatePools::name(kind), pool.size(), tiny_k, "greedy", trial, greedy);
        csv.row("E2", CandidatePools::name(kind), pool.size(), tiny_k, "exact", trial, exact);

        const double ratio = exact.total_gain > 0.0 ? greedy.total_gain / exact.total_gain : 1.0;
        worst_ratio = std::min(worst_ratio, ratio);
        ratio_mean.add(ratio);
        if (ratio > 1.0 - 1e-9) {
            ++greedy_optimal;
        }
    }
    std::cout << std::setprecision(4);
    std::cout << "\nE2: f(greedy)/f(exact) over " << e2_trials << " pools with N=" << tiny_n << ", k=" << tiny_k
              << ": mean " << ratio_mean.value() << ", worst " << worst_ratio << " (theoretical floor "
              << (1.0 - std::exp(-1.0)) << "), greedy optimal in " << greedy_optimal << " pools\n";

    // ---------------------------------------------------------------------------------------------------------
    // E4: cost of greedy versus lazy greedy as the pool grows.
    // ---------------------------------------------------------------------------------------------------------
    const std::vector<size_t> pool_sizes = cfg.quick ? std::vector<size_t>{100, 250, 500}
                                                     : std::vector<size_t>{100, 250, 500, 1000, 2000};
    const size_t e4_trials = cfg.quick ? 2 : 5;
    const size_t e4_k = 40;
    std::cout << "\nE4: gain evaluations and time, greedy versus lazy greedy (k=" << e4_k << ")\n";
    std::cout << std::left << std::setw(6) << "pool" << std::setw(8) << "N" << std::right
              << std::setw(14) << "greedy evals" << std::setw(12) << "lazy evals"
              << std::setw(13) << "greedy ms" << std::setw(11) << "lazy ms" << std::setw(12) << "same set" << "\n";
    std::cout << std::string(76, '-') << "\n";
    for (CandidatePoolKind kind : {CandidatePoolKind::B, CandidatePoolKind::D}) {
        for (size_t pool_size : pool_sizes) {
            Mean greedy_evals, lazy_evals, greedy_ms, lazy_ms;
            size_t identical = 0;
            for (size_t trial = 0; trial < e4_trials; ++trial) {
                CandidatePoolOptions options;
                options.n = n;
                options.m = m;
                options.target_size = pool_size;
                options.seed = 90000 + 100 * pool_size + trial;
                const std::vector<HintCandidate> pool = CandidatePools::make(kind, options);

                const BudgetedSelectionResult greedy = selector.select_greedy(pool, e4_k);
                const BudgetedSelectionResult lazy = selector.select_lazy_greedy(pool, e4_k);
                csv.row("E4", CandidatePools::name(kind), pool.size(), e4_k, "greedy", trial, greedy);
                csv.row("E4", CandidatePools::name(kind), pool.size(), e4_k, "lazy", trial, lazy);
                greedy_evals.add(static_cast<double>(greedy.gain_evaluations));
                lazy_evals.add(static_cast<double>(lazy.gain_evaluations));
                greedy_ms.add(greedy.elapsed_ms);
                lazy_ms.add(lazy.elapsed_ms);
                if (greedy.selected_indices == lazy.selected_indices) {
                    ++identical;
                }
            }
            std::cout << std::left << std::setw(6) << CandidatePools::name(kind) << std::setw(8) << pool_size
                      << std::right << std::setprecision(0) << std::setw(14) << greedy_evals.value()
                      << std::setw(12) << lazy_evals.value() << std::setprecision(3) << std::setw(13)
                      << greedy_ms.value() << std::setw(11) << lazy_ms.value() << std::setw(9) << identical << "/"
                      << e4_trials << "\n";
        }
    }

    std::cout << "\nWrote " << csv.rows() << " rows to " << cfg.out << "\n";
    return 0;
}
