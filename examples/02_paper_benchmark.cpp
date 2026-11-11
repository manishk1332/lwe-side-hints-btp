// examples/02_paper_benchmark.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 5.2 / Figure 5 & Table 1 / main_plan.txt Section 62
// Mathematical Purpose: Concrete security estimation degradation under varying counts of
// perfect hints, modular hints, and approximate hints on the paper benchmark configuration:
//   n = m = 70, q = 3301, sigma = 20.
// Compares GSA-Intersect and Probabilistic Simulation estimates against published trends.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/security/security_estimator.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

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

    // Centered binomial CBD(40) has variance k/2 = 20 (Dachman-Soled et al. Section 5.2 / README.md)
    CenteredBinomial secret_dist(40, 101);
    CenteredBinomial noise_dist(40, 102);

    std::cout << "Base Configuration: n=" << n << ", m=" << m 
              << ", q=" << q.to_string() << ", distribution=CBD(40) (sigma^2 = 20)\n";
    std::cout << "Initial Kannan Lattice Dimension: " << (m + n + 1) << "\n";

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
        print_table_row(res);
    }

    // -------------------------------------------------------------------------
    // 2. Perfect Hints Sweep
    // -------------------------------------------------------------------------
    std::vector<size_t> perfect_counts = {1, 2, 4, 8, 12, 16, 20};
    for (size_t num_h : perfect_counts) {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        for (size_t i = 0; i < num_h; ++i) {
            VectorDouble v(dbdd.ambient_dimension(), 0.0);
            v[m + (i % n)] = 1.0;
            stats.integrate_perfect_hint(v, lwe.test_only_secret()[i % n].to_double(), 1.0);
        }

        auto gsa = stats.estimate_attack_gsa();
        auto prob = SecurityEstimator::probabilistic(stats);

        BenchmarkResult res{
            "Perfect", num_h, stats.dimension(),
            stats.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        print_table_row(res);
    }

    // -------------------------------------------------------------------------
    // 3. Modular Hints Sweep (k = 3)
    // -------------------------------------------------------------------------
    const BigInt mod_k(3);
    std::vector<size_t> mod_counts = {2, 4, 8, 12, 16, 20};
    for (size_t num_h : mod_counts) {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        for (size_t i = 0; i < num_h; ++i) {
            VectorDouble v(dbdd.ambient_dimension(), 0.0);
            v[m + (i % n)] = 1.0;
            BigInt s_val = lwe.test_only_secret()[i % n];
            BigInt rem = s_val % mod_k;
            if (rem < BigInt(0)) rem += mod_k;
            stats.integrate_modular_hint(v, rem.to_double(), 3.0);
        }

        auto gsa = stats.estimate_attack_gsa();
        auto prob = SecurityEstimator::probabilistic(stats);

        BenchmarkResult res{
            "Modular", num_h, stats.dimension(),
            stats.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        print_table_row(res);
    }

    // -------------------------------------------------------------------------
    // 4. Approximate Hints Sweep (variance = 20.0)
    // -------------------------------------------------------------------------
    std::vector<size_t> approx_counts = {2, 4, 8, 12, 16, 20};
    std::mt19937_64 rng(base_seed + 999);
    std::normal_distribution<double> noise_gen(0.0, std::sqrt(20.0));

    for (size_t num_h : approx_counts) {
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, base_seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        for (size_t i = 0; i < num_h; ++i) {
            VectorDouble v(dbdd.ambient_dimension(), 0.0);
            v[m + (i % n)] = 1.0;
            double s_val = lwe.test_only_secret()[i % n].to_double();
            double noisy_obs = s_val + noise_gen(rng);
            stats.integrate_approximate_hint(v, noisy_obs, 1.0); // sigma = 1.0
        }

        auto gsa = stats.estimate_attack_gsa();
        auto prob = SecurityEstimator::probabilistic(stats);

        BenchmarkResult res{
            "Approximate", num_h, stats.dimension(),
            stats.log_volume().to_double() / std::log(2.0),
            gsa.delta(), gsa.beta, gsa.beta_continuous,
            prob.beta, prob.success_probability
        };
        print_table_row(res);
    }

    std::cout << std::string(98, '-') << "\n";
    std::cout << "Benchmark completed successfully.\n\n";
    return 0;
}
