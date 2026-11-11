// examples/02_paper_benchmark.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 5.2 / Figure 5 & Table 1 / main_plan.txt Section 62
// Mathematical Purpose: Concrete security estimation degradation under varying counts of
// perfect hints, modular hints, and approximate hints on the paper benchmark configuration:
//   n = m = 70, q = 3301, sigma^2 = 20 (CBD(40)).
// Hint vectors: Random ternary vectors of weight 5.
// Compares GSA-Intersect and Probabilistic Simulation estimates against published trends.
// Additionally benchmarks concrete end-to-end secret recovery under BKZ-4 reduction.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/secret_recovery.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/reduction/bkz.hpp"
#include "dbdd/security/security_estimator.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>

using namespace dbdd;

struct BenchmarkResult {
    std::string hint_type;
    size_t count;
    size_t dimension;
    double log2_vol;
    double delta;
    size_t beta_gsa;
    double beta_gsa_cont;
    size_t beta_prob;
    double success_prob;
};

static VectorDouble sample_weight5_ternary(size_t dim, std::mt19937_64& rng) {
    VectorDouble v(dim, 0.0);
    std::vector<size_t> indices(dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (size_t i = 0; i < 5 && i < dim; ++i) {
        v[indices[i]] = (rng() % 2 == 0) ? 1.0 : -1.0;
    }
    return v;
}

static void print_table_header() {
    std::cout << "\n"
              << std::string(98, '-') << "\n"
              << std::left << std::setw(14) << "Hint Type"
              << std::right << std::setw(8) << "Count"
              << std::setw(8) << "Dim"
              << std::setw(14) << "Log2(Vol)"
              << std::setw(12) << "delta"
              << std::setw(12) << "beta (GSA)"
              << std::setw(14) << "beta (Prob)"
              << std::setw(16) << "P(Success)"
              << "\n"
              << std::string(98, '-') << "\n";
}

static void print_table_row(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(14) << r.hint_type
              << std::right << std::setw(8) << r.count
              << std::setw(8) << r.dimension
              << std::setw(14) << std::fixed << std::setprecision(2) << r.log2_vol
              << std::setw(12) << std::fixed << std::setprecision(5) << r.delta
              << std::setw(12) << r.beta_gsa
              << std::setw(14) << r.beta_prob
              << std::setw(16) << std::fixed << std::setprecision(4) << r.success_prob
              << "\n";
}

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "          DBDD Framework: 02_paper_benchmark (Dachman-Soled et al. Section 5.2)        \n";
    std::cout << "==================================================================================\n";

    const size_t n = 70;
    const size_t m = 70;
    const BigInt q(3301);
    const uint64_t base_seed = 20200292;

    // Centered binomial CBD(40) has variance k/2 = 20 (Dachman-Soled et al. Section 5.2)
    CenteredBinomial secret_dist(40, 101);
    CenteredBinomial noise_dist(40, 102);

    std::cout << "Base Configuration: n=" << n << ", m=" << m 
              << ", q=" << q.to_string() << ", distribution=CBD(40) (sigma^2 = 20)\n";
    std::cout << "Initial Kannan Lattice Dimension: " << (m + n + 1) << "\n";

    std::vector<BenchmarkResult> all_results;
    std::ofstream csv("data/paper_validation_decay.csv");
    csv << "hint_type,count,dimension,log2_vol,delta,beta_gsa,beta_gsa_cont,beta_prob,success_prob\n";

    print_table_header();

    // -------------------------------------------------------------------------
    // 1. Baseline: 0 Hints
    // -------------------------------------------------------------------------
    {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

        auto gsa = SecurityEstimator::gsa_intersect(dbdd);
        auto prob = SecurityEstimator::probabilistic(dbdd);

        BenchmarkResult res{
            "Baseline", 0, dbdd.dimension(),
            dbdd.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        all_results.push_back(res);
        print_table_row(res);

        csv << res.hint_type << "," << res.count << "," << res.dimension << ","
            << res.log2_vol << "," << res.delta << "," << res.beta_gsa << ","
            << res.beta_gsa_cont << "," << res.beta_prob << "," << res.success_prob << "\n";
    }

    // -------------------------------------------------------------------------
    // 2. Perfect Hints Sweep (0 to 35 hints)
    // -------------------------------------------------------------------------
    std::vector<size_t> perfect_counts = {1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35};
    for (size_t num_h : perfect_counts) {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        std::mt19937_64 rng(base_seed + 1000 + num_h);
        for (size_t i = 0; i < num_h; ++i) {
            VectorDouble v = sample_weight5_ternary(stats.ambient_dimension(), rng);
            stats.integrate_perfect_hint(v, 0.0, 1.0);
        }

        auto gsa = stats.estimate_attack_gsa();
        auto prob = SecurityEstimator::probabilistic(stats);

        BenchmarkResult res{
            "Perfect", num_h, stats.dimension(),
            stats.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        all_results.push_back(res);
        print_table_row(res);

        csv << res.hint_type << "," << res.count << "," << res.dimension << ","
            << res.log2_vol << "," << res.delta << "," << res.beta_gsa << ","
            << res.beta_gsa_cont << "," << res.beta_prob << "," << res.success_prob << "\n";
    }

    // -------------------------------------------------------------------------
    // 3. Modular Hints Sweep (mod 11, 0 to 120 hints)
    // -------------------------------------------------------------------------
    const double mod_k = 11.0;
    std::vector<size_t> mod_counts = {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    for (size_t num_h : mod_counts) {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        std::mt19937_64 rng(base_seed + 2000 + num_h);
        for (size_t i = 0; i < num_h; ++i) {
            VectorDouble v = sample_weight5_ternary(stats.ambient_dimension(), rng);
            stats.integrate_modular_hint(v, 0.0, mod_k);
        }

        auto gsa = stats.estimate_attack_gsa();
        auto prob = SecurityEstimator::probabilistic(stats);

        BenchmarkResult res{
            "Modular", num_h, stats.dimension(),
            stats.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        all_results.push_back(res);
        print_table_row(res);

        csv << res.hint_type << "," << res.count << "," << res.dimension << ","
            << res.log2_vol << "," << res.delta << "," << res.beta_gsa << ","
            << res.beta_gsa_cont << "," << res.beta_prob << "," << res.success_prob << "\n";
    }

    // -------------------------------------------------------------------------
    // 4. Approximate Hints Sweep (sigma_epsilon^2 = 3.0, weight-5 ternary, 0 to 300 hints)
    // -------------------------------------------------------------------------
    const double noise_sigma = std::sqrt(3.0);
    std::vector<size_t> approx_counts = {10, 25, 50, 75, 100, 150, 200, 250, 300};
    for (size_t num_h : approx_counts) {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        std::mt19937_64 rng(base_seed + 3000 + num_h);
        for (size_t i = 0; i < num_h; ++i) {
            VectorDouble v = sample_weight5_ternary(stats.ambient_dimension(), rng);
            stats.integrate_approximate_hint(v, 0.0, noise_sigma);
        }

        auto gsa = stats.estimate_attack_gsa();
        auto prob = SecurityEstimator::probabilistic(stats);

        BenchmarkResult res{
            "Approximate", num_h, stats.dimension(),
            stats.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        all_results.push_back(res);
        print_table_row(res);

        csv << res.hint_type << "," << res.count << "," << res.dimension << ","
            << res.log2_vol << "," << res.delta << "," << res.beta_gsa << ","
            << res.beta_gsa_cont << "," << res.beta_prob << "," << res.success_prob << "\n";
    }
    csv.close();
    std::cout << "\n[OK] Paper validation decay data written to data/paper_validation_decay.csv\n";

    // -------------------------------------------------------------------------
    // 5. Concrete End-to-End Secret Recovery under BKZ Reduction
    // -------------------------------------------------------------------------
    std::cout << "\n==================================================================================\n";
    std::cout << "     Tractable End-to-End BKZ Secret Recovery Benchmark (n=m=12, q=257, CBD(3))   \n";
    std::cout << "==================================================================================\n";

    const size_t e2e_n = 12;
    const size_t e2e_m = 12;
    const BigInt e2e_q(257);
    CenteredBinomial e2e_s_dist(3, 101);
    CenteredBinomial e2e_e_dist(3, 102);

    std::ofstream e2e_csv("data/e2e_bkz_recovery.csv");
    e2e_csv << "k,dimension,mean_time_ms,mean_tours,success_rate,predicted_beta\n";

    std::cout << std::left << std::setw(8) << "Hints"
              << std::setw(8) << "Dim"
              << std::setw(16) << "Mean Time (ms)"
              << std::setw(14) << "Mean Tours"
              << std::setw(16) << "Success Rate"
              << std::setw(16) << "Predicted Beta"
              << "\n"
              << std::string(78, '-') << "\n";

    std::vector<size_t> e2e_hints = {0, 2, 4, 6, 8, 10, 12};
    const size_t trials = 15;

    for (size_t num_h : e2e_hints) {
        double total_ms = 0.0;
        double total_tours = 0.0;
        size_t success_count = 0;
        size_t final_dim = 0;
        size_t pred_beta = 0;

        for (size_t t = 0; t < trials; ++t) {
            uint64_t seed = 2026 + t * 47;
            auto lwe = LWEInstance::generate_random(e2e_n, e2e_m, e2e_q, e2e_s_dist, e2e_e_dist, seed);
            SecretReconstructor reconstructor(lwe);

            auto kannan = KannanEmbedding::from_lwe(lwe, e2e_e_dist, e2e_s_dist);
            DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

            for (size_t h = 0; h < num_h && h < e2e_n; ++h) {
                VectorZZ v(e2e_n, BigInt(0));
                v[h] = BigInt(1);
                dbdd.integrate_perfect_hint(v, lwe.test_only_secret()[h]);
            }

            dbdd.homogenize();
            dbdd.isotropize();

            final_dim = dbdd.dimension();
            auto gsa = SecurityEstimator::gsa_intersect(dbdd);
            pred_beta = gsa.beta;

            auto t0 = std::chrono::high_resolution_clock::now();
            BKZParameters params(4, 15, 0.99);
            auto bkz_res = bkz_reduce(dbdd.basis(), params);
            auto t1 = std::chrono::high_resolution_clock::now();

            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_tours += bkz_res.tours_completed;

            bool rec = false;
            for (size_t r = 0; r < bkz_res.basis.dimension(); ++r) {
                const VectorZZ& cand = bkz_res.basis.row(r);
                size_t amb = dbdd.ambient_dimension();
                if (cand[amb - 1].abs() == BigInt(1)) {
                    VectorZZ cand_s = reconstructor.recover_secret(cand);
                    if (reconstructor.verify_recovery(cand_s)) {
                        rec = true;
                        break;
                    }
                }
            }
            if (!rec) {
                auto rec_dbdd = BKZReduction::recover_dbdd_secret(bkz_res.basis, dbdd);
                if (rec_dbdd.has_value()) {
                    VectorZZ cand_s = reconstructor.recover_secret(*rec_dbdd);
                    if (reconstructor.verify_recovery(cand_s)) {
                        rec = true;
                    }
                }
            }
            if (rec) success_count++;
        }

        double mean_ms = total_ms / trials;
        double mean_tours = total_tours / trials;
        double succ_rate = 100.0 * success_count / trials;

        std::cout << std::left << std::setw(8) << num_h
                  << std::setw(8) << final_dim
                  << std::setw(16) << std::fixed << std::setprecision(2) << mean_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << mean_tours
                  << std::setw(16) << (std::to_string(int(succ_rate)) + "%")
                  << std::setw(16) << pred_beta
                  << "\n";

        e2e_csv << num_h << "," << final_dim << "," << mean_ms << ","
                << mean_tours << "," << succ_rate << "," << pred_beta << "\n";
    }
    e2e_csv.close();
    std::cout << "\n[OK] End-to-end BKZ recovery data written to data/e2e_bkz_recovery.csv\n";

    std::cout << std::string(98, '-') << "\n";
    std::cout << "All benchmarks completed successfully.\n\n";
    return 0;
}
