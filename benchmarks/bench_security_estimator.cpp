// benchmarks/bench_security_estimator.cpp
//
// Paper Reference: Albrecht et al. Section 3.4, 4.4 / main_plan.txt Section 58
// Mathematical Purpose: Throughput benchmark for security estimation engines:
//   - GSA-Intersect Security Estimator (SecurityEstimator::gsa_intersect)
//   - Probabilistic Simulation Estimator (SecurityEstimator::probabilistic)
//   - Lightweight Dense Statistics Predictor (LightweightPredictor)
// Measures operations per second across increasing lattice dimensions.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/security/security_estimator.hpp"
#include "dbdd/security/lightweight_predictor.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace dbdd;

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "          DBDD Framework: Security Estimator Throughput Benchmark                       \n";
    std::cout << "========================================================================================\n";

    std::cout << "\n"
              << std::string(84, '-') << "\n"
              << std::left << std::setw(8) << "n"
              << std::right << std::setw(8) << "Dim"
              << std::setw(20) << "GSA (evals/sec)"
              << std::setw(22) << "Prob (evals/sec)"
              << std::setw(24) << "Lightweight (evals/sec)"
              << "\n"
              << std::string(84, '-') << "\n";

    std::vector<size_t> ns = {30, 50, 70, 100};

    for (size_t n : ns) {
        size_t m = n;
        BigInt q(3301);
        double sigma = 3.2;

        DiscreteGaussian s_dist(sigma, 42);
        DiscreteGaussian e_dist(sigma, 43);

        auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, 42);
        auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();

        // 1. GSA-Intersect Throughput
        const size_t gsa_runs = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < gsa_runs; ++i) {
            (void)SecurityEstimator::gsa_intersect(dbdd);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_gsa = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gsa_tps = (ms_gsa > 0.0) ? (gsa_runs / (ms_gsa / 1000.0)) : 0.0;

        // 2. Probabilistic Simulation Throughput
        const size_t prob_runs = 10;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < prob_runs; ++i) {
            (void)SecurityEstimator::probabilistic(dbdd);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_prob = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double prob_tps = (ms_prob > 0.0) ? (prob_runs / (ms_prob / 1000.0)) : 0.0;

        // 3. Lightweight Predictor Throughput
        const size_t light_runs = 100;
        auto t4 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < light_runs; ++i) {
            (void)stats.estimate_attack_gsa();
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double ms_light = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double light_tps = (ms_light > 0.0) ? (light_runs / (ms_light / 1000.0)) : 0.0;

        std::cout << std::left << std::setw(8) << n
                  << std::right << std::setw(8) << dbdd.dimension()
                  << std::setw(20) << std::fixed << std::setprecision(1) << gsa_tps
                  << std::setw(22) << std::fixed << std::setprecision(1) << prob_tps
                  << std::setw(24) << std::fixed << std::setprecision(1) << light_tps
                  << "\n";
    }

    std::cout << std::string(84, '-') << "\n";
    std::cout << "Security Estimator Benchmark completed successfully.\n\n";
    return 0;
}
