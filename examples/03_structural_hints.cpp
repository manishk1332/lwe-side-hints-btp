// examples/03_structural_hints.cpp
//
// Paper Reference: Albrecht et al. (CRYPTO 2020) Section 4.5 & Section 6 / main_plan.txt Section 63
// Mathematical Purpose: Demonstrates structural secret constraints and short-vector worthiness optimization:
//   1. Balanced ternary secret: sum_{i=0}^{n-1} s_i = 0
//   2. Integration of structural perfect hint: dimension drop and volume expansion
//   3. Residual short-vector search via QVectorGenerator
//   4. Analytical worthiness analysis (ShortVectorWorthinessAnalyzer)
//   5. Greedy short-vector selection (GreedyHintSelector) and cumulative bikz degradation.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/worthiness.hpp"
#include "dbdd/hints/greedy_selection.hpp"
#include "dbdd/lattice/q_vectors.hpp"
#include "dbdd/security/security_estimator.hpp"

#include <iostream>
#include <iomanip>
#include <random>

using namespace dbdd;

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "            DBDD Framework: 03_structural_hints Demonstration                     \n";
    std::cout << "==================================================================================\n";

    // -------------------------------------------------------------------------
    // 1. Setup LWE with Balanced Ternary Secret: sum(s_i) = 0
    // -------------------------------------------------------------------------
    const size_t n = 20;
    const size_t m = 20;
    const BigInt q(257);
    const uint64_t seed = 2020;

    std::cout << "LWE Parameters: n=" << n << ", m=" << m << ", q=" << q.to_string() << "\n";
    std::cout << "Secret Type: Balanced Ternary (sum_i s_i = 0)\n";

    // Construct balanced secret: 8 +1's, 8 -1's, 4 0's -> sum = 0
    VectorZZ secret(n, BigInt(0));
    for (size_t i = 0; i < 8; ++i) secret[i] = BigInt(1);
    for (size_t i = 8; i < 16; ++i) secret[i] = BigInt(-1);

    // Verify balance
    BigInt s_sum(0);
    for (size_t i = 0; i < n; ++i) s_sum += secret[i];
    std::cout << "Ground-Truth Secret Sum: " << s_sum.to_string() << " (balanced)\n";

    // Generate random public matrix A and small error e
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long long> unif_q(0, 256);
    MatrixZZ A(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = BigInt(unif_q(rng));
        }
    }

    DiscreteGaussian noise_dist(1.2, 777);
    VectorZZ error(m);
    for (size_t i = 0; i < m; ++i) {
        error[i] = BigInt(noise_dist.sample(rng));
    }

    // Compute b = s * A^T + e (mod q)
    VectorZZ sA = secret * A.transpose();
    VectorZZ b(m);
    for (size_t i = 0; i < m; ++i) {
        BigInt val = (sA[i] + error[i]) % q;
        if (val < BigInt(0)) val += q;
        b[i] = val;
    }

    LWEInstance lwe(n, m, q, A, b, secret, error);
    auto kannan = KannanEmbedding::from_lwe(lwe, 1.2, 0.816);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    std::cout << "\n[Initial Kannan State]\n";
    std::cout << "  Dimension:       " << dbdd.dimension() << "\n";
    std::cout << "  Volume:          " << dbdd.volume().to_double() << "\n";
    std::cout << "  Covariance Rank: " << dbdd.ellipsoid().rank() << "\n";

    auto gsa_init = SecurityEstimator::gsa_intersect(dbdd);
    std::cout << "  Initial Hardness: beta = " << gsa_init.beta << ", delta = " << gsa_init.delta() << "\n";

    // -------------------------------------------------------------------------
    // 2. Integrate Structural Perfect Hint: sum_{i=0}^{n-1} s_i = 0
    // -------------------------------------------------------------------------
    std::cout << "\n[Integrating Structural Perfect Hint: <s, 1> = 0]\n";
    VectorZZ v_all_ones(n, BigInt(1));
    dbdd.integrate_perfect_hint(v_all_ones, BigInt(0));

    std::cout << "  New Dimension:       " << dbdd.dimension() << " (dropped by 1)\n";
    std::cout << "  New Volume:          " << dbdd.volume().to_double() << " (volume increased)\n";
    std::cout << "  New Covariance Rank: " << dbdd.ellipsoid().rank() << "\n";

    auto gsa_struct = SecurityEstimator::gsa_intersect(dbdd);
    std::cout << "  Hardness After Structural Hint: beta = " << gsa_struct.beta 
              << ", delta = " << gsa_struct.delta() << "\n";

    // -------------------------------------------------------------------------
    // 3. Search for Worthy Residual Short Vectors (q-vectors)
    // -------------------------------------------------------------------------
    std::cout << "\n[Searching for Worthy Short Vectors via Greedy Optimization]\n";
    auto greedy_result = dbdd.integrate_q_vectors(q, 5, false, true);

    std::cout << "  Candidates Evaluated: " << greedy_result.candidates_evaluated << "\n";
    std::cout << "  Worthy Vectors Found: " << greedy_result.selected_vectors.size() << "\n";
    std::cout << "  Cumulative Bikz Drop: " << greedy_result.bikz_reduction() << " bikz\n";

    if (!greedy_result.history.empty()) {
        std::cout << "\n  Progression History:\n";
        std::cout << "  " << std::string(72, '-') << "\n";
        std::cout << "  " << std::left << std::setw(6) << "Step"
                  << std::right << std::setw(12) << "Var Score"
                  << std::setw(14) << "Proxy Ratio"
                  << std::setw(14) << "beta After"
                  << std::setw(14) << "delta After"
                  << "\n";
        std::cout << "  " << std::string(72, '-') << "\n";

        for (size_t step = 0; step < greedy_result.history.size(); ++step) {
            const auto& h = greedy_result.history[step];
            std::cout << "  " << std::left << std::setw(6) << (step + 1)
                      << std::right << std::setw(12) << std::fixed << std::setprecision(4) << h.incremental_variance
                      << std::setw(14) << std::fixed << std::setprecision(6) << h.incremental_proxy_ratio
                      << std::setw(14) << h.beta_after
                      << std::setw(14) << std::fixed << std::setprecision(5) << h.delta_after
                      << "\n";
        }
        std::cout << "  " << std::string(72, '-') << "\n";
    }

    std::cout << "\n[Final Instance State]\n";
    std::cout << "  Final Dimension: " << dbdd.dimension() << "\n";
    std::cout << "  Final Volume:    " << dbdd.volume().to_double() << "\n";

    auto gsa_final = SecurityEstimator::gsa_intersect(dbdd);
    auto prob_final = SecurityEstimator::probabilistic(dbdd);

    std::cout << "  Final GSA Beta:  " << gsa_final.beta << " (delta = " << gsa_final.delta() << ")\n";
    std::cout << "  Final Prob Beta: " << prob_final.beta << "\n\n";

    return 0;
}
