// src/cli/main.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) / main_plan.txt Section 53
// Mathematical Purpose: Official Command-Line Interface (`dbdd_cli`) providing an end-to-end
// workflow for LWE generation, Kannan embedding inspection, hardness estimation (GSA/Probabilistic),
// BKZ lattice reduction attacks, and benchmark execution.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/secret_recovery.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/worthiness.hpp"
#include "dbdd/hints/greedy_selection.hpp"
#include "dbdd/security/security_estimator.hpp"
#include "dbdd/reduction/bkz.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include "dbdd/reduction/lll_float.hpp"
#include "dbdd/reduction/enumeration.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>

using namespace dbdd;

static void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <subcommand> [options]\n\n"
              << "Subcommands:\n"
              << "  generate-lwe    Generate an LWE instance and output parameters\n"
              << "  inspect-lwe     Inspect an LWE instance and its Kannan embedding\n"
              << "  estimate        Estimate hardness using GSA and Probabilistic estimators\n"
              << "  attack          Execute BKZ lattice reduction attack to recover secret\n"
              << "  benchmark       Run internal performance benchmark suites\n\n"
              << "Common Options:\n"
              << "  --n <int>             Secret dimension n (default: 50)\n"
              << "  --m <int>             Number of samples m (default: 50)\n"
              << "  --q <int>             Modulus q (default: 3301)\n"
              << "  --sigma <float>       Noise standard deviation (default: 3.2)\n"
              << "  --dist <string>       Distribution: gaussian, ternary, binomial (default: gaussian)\n"
              << "  --seed <uint64>       Deterministic RNG seed (default: 1337)\n\n"
              << "Estimation / Attack Options:\n"
              << "  --perfect-hints <int> Number of perfect hints to integrate (default: 0)\n"
              << "  --modular-hints <int> Number of modular hints to integrate (default: 0)\n"
              << "  --approx-hints <int>  Number of approximate hints to integrate (default: 0)\n"
              << "  --method <string>     Estimator method: gsa, prob, both (default: both)\n"
              << "  --blocksize <int>     BKZ block size beta (default: auto from estimator)\n"
              << "  --tours <int>         Maximum BKZ tours (default: 10)\n"
              << "  --suite <string>      Benchmark suite: lll, enumeration, bkz, estimator, all\n\n";
}

struct CLIOptions {
    std::string subcommand;
    size_t n = 50;
    size_t m = 50;
    BigInt q = BigInt(3301);
    double sigma = 3.2;
    std::string dist = "gaussian";
    uint64_t seed = 1337;

    size_t perfect_hints = 0;
    size_t modular_hints = 0;
    size_t approx_hints = 0;
    BigInt mod_k = BigInt(3);
    double approx_sigma = 3.2;

    std::string method = "both";
    size_t blocksize = 0; // 0 = auto
    size_t tours = 10;
    std::string suite = "all";
};

static CLIOptions parse_arguments(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(1);
    }

    CLIOptions opts;
    opts.subcommand = argv[1];

    if (opts.subcommand == "-h" || opts.subcommand == "--help") {
        print_usage(argv[0]);
        std::exit(0);
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            opts.n = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--m" && i + 1 < argc) {
            opts.m = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--q" && i + 1 < argc) {
            opts.q = BigInt(argv[++i]);
        } else if (arg == "--sigma" && i + 1 < argc) {
            opts.sigma = std::stod(argv[++i]);
        } else if (arg == "--dist" && i + 1 < argc) {
            opts.dist = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = std::stoull(argv[++i]);
        } else if (arg == "--perfect-hints" && i + 1 < argc) {
            opts.perfect_hints = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--modular-hints" && i + 1 < argc) {
            opts.modular_hints = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--approx-hints" && i + 1 < argc) {
            opts.approx_hints = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--method" && i + 1 < argc) {
            opts.method = argv[++i];
        } else if (arg == "--blocksize" && i + 1 < argc) {
            opts.blocksize = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--tours" && i + 1 < argc) {
            opts.tours = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--suite" && i + 1 < argc) {
            opts.suite = argv[++i];
        }
    }

    return opts;
}

static std::unique_ptr<Distribution> make_distribution(const std::string& name, double sigma, uint64_t seed) {
    if (name == "ternary") {
        return std::make_unique<UniformTernary>(2.0 / 3.0, seed);
    } else if (name == "binomial") {
        size_t k = static_cast<size_t>(std::max(1.0, std::round(2.0 * sigma * sigma)));
        return std::make_unique<CenteredBinomial>(k, seed);
    } else {
        return std::make_unique<DiscreteGaussian>(sigma, seed);
    }
}

// Subcommand: generate-lwe
static int cmd_generate_lwe(const CLIOptions& opts) {
    auto s_dist = make_distribution(opts.dist, opts.sigma, opts.seed);
    auto e_dist = make_distribution(opts.dist, opts.sigma, opts.seed + 1);

    auto lwe = LWEInstance::generate_random(opts.n, opts.m, opts.q, *s_dist, *e_dist, opts.seed);

    std::cout << "{\n"
              << "  \"n\": " << lwe.n() << ",\n"
              << "  \"m\": " << lwe.m() << ",\n"
              << "  \"q\": \"" << lwe.q().to_string() << "\",\n"
              << "  \"distribution\": \"" << opts.dist << "\",\n"
              << "  \"sigma\": " << opts.sigma << ",\n"
              << "  \"seed\": " << opts.seed << ",\n"
              << "  \"matrix_shape\": [" << lwe.m() << ", " << lwe.n() << "],\n"
              << "  \"vector_b_size\": " << lwe.m() << "\n"
              << "}\n";
    return 0;
}

// Subcommand: inspect-lwe
static int cmd_inspect_lwe(const CLIOptions& opts) {
    auto s_dist = make_distribution(opts.dist, opts.sigma, opts.seed);
    auto e_dist = make_distribution(opts.dist, opts.sigma, opts.seed + 1);

    auto lwe = LWEInstance::generate_random(opts.n, opts.m, opts.q, *s_dist, *e_dist, opts.seed);
    auto kannan = KannanEmbedding::from_lwe(lwe, *e_dist, *s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    std::cout << "LWE Instance Inspection:\n"
              << "  n:                 " << opts.n << "\n"
              << "  m:                 " << opts.m << "\n"
              << "  q:                 " << opts.q.to_string() << "\n"
              << "  ambient_dimension: " << dbdd.ambient_dimension() << "\n"
              << "  lattice_rank:      " << dbdd.dimension() << "\n"
              << "  volume:            " << dbdd.volume().to_double() << "\n"
              << "  log2_volume:       " << (dbdd.log_volume().to_double() / std::log(2.0)) << "\n"
              << "  covariance_rank:   " << dbdd.ellipsoid().rank() << "\n";

    auto gsa = SecurityEstimator::gsa_intersect(dbdd);
    std::cout << "  initial_delta:     " << gsa.delta() << "\n"
              << "  initial_beta_gsa:  " << gsa.beta << "\n";
    return 0;
}

// Subcommand: estimate
static int cmd_estimate(const CLIOptions& opts) {
    auto s_dist = make_distribution(opts.dist, opts.sigma, opts.seed);
    auto e_dist = make_distribution(opts.dist, opts.sigma, opts.seed + 1);

    auto lwe = LWEInstance::generate_random(opts.n, opts.m, opts.q, *s_dist, *e_dist, opts.seed);
    auto kannan = KannanEmbedding::from_lwe(lwe, *e_dist, *s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
    DBDDStateStatistics stats = dbdd.to_state_statistics();

    // Apply perfect hints if requested
    for (size_t i = 0; i < opts.perfect_hints; ++i) {
        VectorDouble v(dbdd.ambient_dimension(), 0.0);
        v[opts.m + (i % opts.n)] = 1.0;
        stats.integrate_perfect_hint(v, lwe.test_only_secret()[i % opts.n].to_double(), 1.0);
    }

    // Apply modular hints if requested
    for (size_t i = 0; i < opts.modular_hints; ++i) {
        VectorDouble v(dbdd.ambient_dimension(), 0.0);
        v[opts.m + (i % opts.n)] = 1.0;
        BigInt rem = lwe.test_only_secret()[i % opts.n] % opts.mod_k;
        if (rem < BigInt(0)) rem += opts.mod_k;
        stats.integrate_modular_hint(v, rem.to_double(), opts.mod_k.to_double());
    }

    // Apply approximate hints if requested
    for (size_t i = 0; i < opts.approx_hints; ++i) {
        VectorDouble v(dbdd.ambient_dimension(), 0.0);
        v[opts.m + (i % opts.n)] = 1.0;
        double s_val = lwe.test_only_secret()[i % opts.n].to_double();
        stats.integrate_approximate_hint(v, s_val, opts.approx_sigma * opts.approx_sigma);
    }

    SecurityEstimate est = (opts.method == "prob")
        ? SecurityEstimator::probabilistic(stats)
        : stats.estimate_attack_gsa();

    // Format matching main_plan.txt Section 53
    std::cout << std::left << std::setw(19) << "dimension:"        << stats.dimension() << "\n"
              << std::left << std::setw(19) << "volume:"           << stats.volume().to_double() << "\n"
              << std::left << std::setw(19) << "covariance-rank:"  << stats.dimension() << "\n"
              << std::left << std::setw(19) << "isotropic-volume:" << est.volume().to_double() << "\n"
              << std::left << std::setw(19) << "delta:"            << std::fixed << std::setprecision(6) << est.delta() << "\n"
              << std::left << std::setw(19) << "predicted-beta:"   << est.beta << " (continuous: " << est.beta_continuous << ")\n"
              << std::left << std::setw(19) << "method:"           << est.method << "\n";

    if (opts.method == "both") {
        auto prob_est = SecurityEstimator::probabilistic(stats);
        std::cout << "\n[Probabilistic Simulation Comparison]\n"
                  << "  predicted-beta:  " << prob_est.beta << "\n"
                  << "  success-prob:    " << prob_est.success_probability << "\n";
    }

    return 0;
}

// Subcommand: attack
static int cmd_attack(const CLIOptions& opts) {
    auto s_dist = make_distribution(opts.dist, opts.sigma, opts.seed);
    auto e_dist = make_distribution(opts.dist, opts.sigma, opts.seed + 1);

    auto lwe = LWEInstance::generate_random(opts.n, opts.m, opts.q, *s_dist, *e_dist, opts.seed);
    SecretReconstructor reconstructor(lwe);
    auto kannan = KannanEmbedding::from_lwe(lwe, *e_dist, *s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    for (size_t i = 0; i < opts.perfect_hints; ++i) {
        VectorZZ v(opts.n, BigInt(0));
        v[i % opts.n] = BigInt(1);
        dbdd.integrate_perfect_hint(v, lwe.test_only_secret()[i % opts.n]);
    }

    size_t target_beta = opts.blocksize;
    if (target_beta == 0) {
        auto est = SecurityEstimator::gsa_intersect(dbdd);
        target_beta = std::max(size_t(2), est.beta);
    }

    std::cout << "Starting BKZ Attack:\n"
              << "  Lattice Dimension: " << dbdd.dimension() << "\n"
              << "  BKZ Blocksize:     " << target_beta << "\n"
              << "  Max Tours:         " << opts.tours << "\n";

    BKZParameters params(target_beta, opts.tours, 0.99);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto bkz_res = bkz_reduce(dbdd.basis(), params);
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    bool recovered = false;
    VectorZZ final_secret(opts.n);

    for (size_t r = 0; r < bkz_res.basis.dimension(); ++r) {
        const VectorZZ& candidate = bkz_res.basis.row(r);
        size_t amb = dbdd.ambient_dimension();
        if (candidate[amb - 1].abs() == BigInt(1)) {
            VectorZZ cand_s = reconstructor.recover_secret(candidate);
            if (reconstructor.verify_recovery(cand_s)) {
                recovered = true;
                final_secret = cand_s;
                break;
            }
        }
    }

    if (!recovered) {
        auto rec_dbdd = BKZReduction::recover_dbdd_secret(bkz_res.basis, dbdd);
        if (rec_dbdd.has_value()) {
            final_secret = reconstructor.recover_secret(*rec_dbdd);
            recovered = reconstructor.verify_recovery(final_secret);
        }
    }

    std::cout << "\nAttack Results:\n"
              << "  Status:      " << (recovered ? "SUCCESS" : "FAILED") << "\n"
              << "  Runtime:     " << duration_ms << " ms\n"
              << "  Tours Run:   " << bkz_res.tours_completed << "\n"
              << "  SVP Calls:   " << bkz_res.total_svp_calls << "\n";

    if (recovered) {
        std::cout << "  Secret: [ ";
        for (size_t j = 0; j < opts.n; ++j) {
            std::cout << final_secret[j].to_string() << " ";
        }
        std::cout << "]\n";
        return 0;
    } else {
        std::cerr << "  Secret was not recovered in allotted tours.\n";
        return 1;
    }
}

static BasisZZ make_scrambled_basis(size_t d, uint64_t seed = 42) {
    MatrixZZ B = MatrixZZ::identity(d);
    for (size_t i = 0; i < d; ++i) {
        B(i, i) = BigInt(static_cast<long long>(10 + (i % 7) * 3));
    }

    std::mt19937_64 rng(seed);
    for (size_t step = 0; step < 4 * d; ++step) {
        size_t r1 = rng() % d;
        size_t r2 = rng() % d;
        if (r1 != r2) {
            long long factor = static_cast<long long>((rng() % 4) + 1);
            if (rng() % 2 == 0) factor = -factor;
            B.row_add_multiple(r1, r2, BigInt(factor));
        }
        if (step % 3 == 0) {
            size_t s1 = rng() % d;
            size_t s2 = rng() % d;
            B.swap_rows(s1, s2);
        }
    }
    return BasisZZ(B);
}

// Subcommand: benchmark
static int cmd_benchmark(const CLIOptions& opts) {
    std::cout << "Executing DBDD Performance Benchmark Suite (" << opts.suite << ")...\n\n";

    // 1. LLL Exact vs Float benchmark
    if (opts.suite == "lll" || opts.suite == "all") {
        std::cout << "--- Benchmark: Exact vs Floating LLL ---\n";
        for (size_t d : {10, 20, 30}) {
            BasisZZ basis = make_scrambled_basis(d, 42);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto res_exact = lll_exact(basis);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms_exact = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto t2 = std::chrono::high_resolution_clock::now();
            auto res_float = lll_float(basis);
            auto t3 = std::chrono::high_resolution_clock::now();
            double ms_float = std::chrono::duration<double, std::milli>(t3 - t2).count();

            std::cout << "  Dim " << std::setw(2) << d 
                      << " | LLLExact: " << std::setw(8) << ms_exact << " ms"
                      << " | LLLFloat: " << std::setw(8) << ms_float << " ms\n";
        }
        std::cout << "\n";
    }

    // 2. Enumeration SVP benchmark
    if (opts.suite == "enumeration" || opts.suite == "all") {
        std::cout << "--- Benchmark: SVP Enumeration Node Counts ---\n";
        EnumerationSVP svp;
        for (size_t d : {10, 15, 20}) {
            BasisZZ basis = make_scrambled_basis(d, 101);
            basis = lll_exact(basis).basis;

            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = svp.find_shortest_vector(basis);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

            std::cout << "  Dim " << std::setw(2) << d
                      << " | Nodes: " << std::setw(6) << res.nodes_visited
                      << " | Time: " << std::setw(8) << us << " us"
                      << " | ||v_min||: " << res.length << "\n";
        }
        std::cout << "\n";
    }

    // 3. Security Estimator Throughput
    if (opts.suite == "estimator" || opts.suite == "all") {
        std::cout << "--- Benchmark: Security Estimator Throughput ---\n";
        DiscreteGaussian s_dist(3.2, 42);
        DiscreteGaussian e_dist(3.2, 43);
        auto lwe = LWEInstance::generate_random(50, 50, BigInt(3301), s_dist, e_dist, 42);
        auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

        const size_t N_EVALS = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N_EVALS; ++i) {
            (void)SecurityEstimator::gsa_intersect(dbdd);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_gsa = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < 5; ++i) {
            (void)SecurityEstimator::probabilistic(dbdd);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_prob = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "  GSA-Intersect: " << (N_EVALS / (ms_gsa / 1000.0)) << " evals/sec\n";
        std::cout << "  Probabilistic: " << (5 / (ms_prob / 1000.0)) << " evals/sec\n\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {
    try {
        CLIOptions opts = parse_arguments(argc, argv);

        if (opts.subcommand == "generate-lwe") {
            return cmd_generate_lwe(opts);
        } else if (opts.subcommand == "inspect-lwe") {
            return cmd_inspect_lwe(opts);
        } else if (opts.subcommand == "estimate") {
            return cmd_estimate(opts);
        } else if (opts.subcommand == "attack") {
            return cmd_attack(opts);
        } else if (opts.subcommand == "benchmark") {
            return cmd_benchmark(opts);
        } else {
            std::cerr << "Error: Unknown subcommand '" << opts.subcommand << "'\n\n";
            print_usage(argv[0]);
            return 1;
        }
    } catch (const DBDDException& ex) {
        std::cerr << "DBDD Error: " << ex.what() << "\n";
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "Standard Error: " << ex.what() << "\n";
        return 3;
    }
}
