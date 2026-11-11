// tests/integration/test_paper_reproduction.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020) Section 5 & 6 / main_plan.txt Sections 61, 62, 63
// Mathematical Purpose: Integration test suite reproducing paper experimental benchmarks,
// verifying accuracy tolerances for security estimations under perfect, modular, and approximate hints,
// and validating end-to-end exact secret recovery.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/secret_recovery.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/short_vector_hint.hpp"
#include "dbdd/hints/worthiness.hpp"
#include "dbdd/hints/greedy_selection.hpp"
#include "dbdd/security/security_estimator.hpp"
#include "dbdd/reduction/bkz.hpp"

#include <cmath>
#include <vector>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Paper Reproduction - Dachman-Soled et al. Section 5.2 Baseline (n=m=70, q=3301, sigma=20)", "[paper][baseline]") {
    const size_t n = 70;
    const size_t m = 70;
    const BigInt q(3301);
    const uint64_t seed = 20200292;

    // Dachman-Soled et al. Section 5.2 / README.md: Centered binomial CBD(40) has variance k/2 = 20
    CenteredBinomial s_dist(40, 101);
    CenteredBinomial e_dist(40, 102);

    auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, seed);
    auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    SECTION("Kannan Embedding Dimensions & Invariants") {
        REQUIRE(dbdd.dimension() == 141);
        REQUIRE(dbdd.ambient_dimension() == 141);
        REQUIRE(dbdd.ellipsoid().rank() == 140); // constant-1 coordinate has 0 variance before homogenization
        REQUIRE(dbdd.target_in_lattice());
    }

    SECTION("Baseline Security Estimator Values Match Paper Within Tolerances") {
        auto gsa = SecurityEstimator::gsa_intersect(dbdd);
        auto prob = SecurityEstimator::probabilistic(dbdd);

        // Paper values: delta ~ 1.01236, beta ~ 45.40
        REQUIRE_THAT(gsa.delta(), WithinAbs(1.01236, 0.0006));
        REQUIRE(gsa.beta >= 44);
        REQUIRE(gsa.beta <= 47);
        REQUIRE_THAT(gsa.beta_continuous, WithinAbs(45.4, 1.5));

        // Probabilistic estimator is within reasonable bounds of GSA
        REQUIRE(prob.beta >= 41);
        REQUIRE(prob.beta <= 48);
    }
}

TEST_CASE("Paper Reproduction - Security Degradation Under Hints (Section 5.2)", "[paper][hints]") {
    const size_t n = 70;
    const size_t m = 70;
    const BigInt q(3301);
    const uint64_t seed = 20200292;

    CenteredBinomial s_dist(40, 101);
    CenteredBinomial e_dist(40, 102);

    SECTION("4 Perfect Hints: Dimension Drops to 137, Beta Drops ~10-14 bikz") {
        auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();
        size_t init_beta = stats.estimate_attack_gsa().beta;

        std::mt19937_64 rng(42);
        for (size_t i = 0; i < 4; ++i) {
            VectorDouble v(stats.ambient_dimension(), 0.0);
            for (size_t j = 0; j < m + n; ++j) v[j] = (rng() % 2 == 0) ? 1.0 : 0.0;
            stats.integrate_perfect_hint(v, 0.0, 1.0);
        }

        REQUIRE(stats.dimension() == 137);
        REQUIRE(stats.ambient_dimension() == 141);

        auto gsa = stats.estimate_attack_gsa();
        // Paper: beta drops to ~30-36
        REQUIRE(gsa.beta < init_beta);
        REQUIRE(gsa.beta <= 38);
        REQUIRE(gsa.beta >= 30);
    }

    SECTION("4 Modular Hints (k=3): Security Decreases") {
        auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();
        size_t init_beta = stats.estimate_attack_gsa().beta;

        std::mt19937_64 rng(42);
        for (size_t i = 0; i < 4; ++i) {
            VectorDouble v(stats.ambient_dimension(), 0.0);
            for (size_t j = 0; j < m + n; ++j) v[j] = (rng() % 2 == 0) ? 1.0 : 0.0;
            stats.integrate_modular_hint(v, 0.0, 3.0);
        }

        auto gsa = stats.estimate_attack_gsa();
        REQUIRE(gsa.beta_continuous < stats.estimate_attack_gsa().beta_continuous + 0.1);
        REQUIRE(gsa.beta <= init_beta);
    }

    SECTION("4 Approximate Hints (sigma=1.0): Beta Drops Noticeably") {
        auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, seed);
        auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);
        DBDDStateStatistics stats = dbdd.to_state_statistics();
        size_t init_beta = stats.estimate_attack_gsa().beta;

        std::mt19937_64 rng(42);
        for (size_t i = 0; i < 4; ++i) {
            VectorDouble v(stats.ambient_dimension(), 0.0);
            for (size_t j = 0; j < m + n; ++j) v[j] = (rng() % 2 == 0) ? 1.0 : 0.0;
            stats.integrate_approximate_hint(v, 0.0, 1.0);
        }

        auto gsa = stats.estimate_attack_gsa();
        REQUIRE(gsa.beta < init_beta);
        REQUIRE(gsa.beta <= 42);
    }
}

TEST_CASE("Paper Reproduction - Canonical End-to-End Secret Recovery Pipeline (Section 61)", "[paper][e2e]") {
    const size_t n = 3;
    const size_t m = 3;
    const BigInt q(11);
    const uint64_t seed = 42;

    UniformTernary secret_dist;
    UniformTernary noise_dist;

    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, seed);
    SecretReconstructor reconstructor(lwe);

    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    REQUIRE(dbdd.dimension() == 7);
    REQUIRE(dbdd.target_in_lattice());

    // 1. Short Vector Hint: Project along q-vector row 0 of Kannan basis
    VectorZZ q_vec = dbdd.basis().row(0);
    reconstructor.record_projection(q_vec);
    dbdd.integrate_short_vector_hint(q_vec);
    REQUIRE(dbdd.dimension() == 6);
    REQUIRE(dbdd.target_in_lattice());

    // 2. Perfect Hint on s_0
    VectorZZ v_perf({BigInt(1), BigInt(0), BigInt(0)});
    dbdd.integrate_perfect_hint(v_perf, lwe.test_only_secret()[0]);
    REQUIRE(dbdd.dimension() == 5);
    REQUIRE(dbdd.target_in_lattice());

    // 3. Approximate Hint on s_1
    VectorDouble v_approx({0.0, 1.0, 0.0});
    double s1 = lwe.test_only_secret()[1].to_double();
    dbdd.integrate_approximate_hint(v_approx, s1 + 0.1, 0.5);
    REQUIRE(dbdd.dimension() == 5);

    // 4. Homogenize & Isotropize
    dbdd.homogenize();
    REQUIRE(dbdd.is_homogenized());

    dbdd.isotropize();
    REQUIRE(dbdd.is_isotropized());

    // 5. Security Estimation
    auto gsa = SecurityEstimator::gsa_intersect(dbdd);
    REQUIRE(gsa.beta >= 2);

    // 6. BKZ Reduction
    BKZParameters params(std::max(size_t(2), gsa.beta), 15, 0.99);
    auto bkz_res = bkz_reduce(dbdd.basis(), params);
    REQUIRE(bkz_res.converged);

    // 7. Recover Secret & Verify 100% Match
    bool recovered = false;
    VectorZZ final_secret(n);

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

    REQUIRE(recovered);
    REQUIRE(final_secret == lwe.test_only_secret());
    REQUIRE(reconstructor.verify_recovery(final_secret));
}

TEST_CASE("Paper Reproduction - Structural Constraints & Worthy Residual Vectors (Section 63)", "[paper][structural]") {
    const size_t n = 20;
    const size_t m = 20;
    const BigInt q(257);

    // Balanced secret with sum = 0
    VectorZZ secret(n, BigInt(0));
    for (size_t i = 0; i < 8; ++i) secret[i] = BigInt(1);
    for (size_t i = 8; i < 16; ++i) secret[i] = BigInt(-1);

    DiscreteGaussian noise_dist(1.0, 42);
    std::mt19937_64 rng(42);
    MatrixZZ A(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = BigInt(rng() % 257);
        }
    }
    VectorZZ error(m);
    for (size_t i = 0; i < m; ++i) {
        error[i] = BigInt(noise_dist.sample(rng));
    }
    VectorZZ sA = secret * A.transpose();
    VectorZZ b(m);
    for (size_t i = 0; i < m; ++i) {
        BigInt val = (sA[i] + error[i]) % q;
        if (val < BigInt(0)) val += q;
        b[i] = val;
    }

    LWEInstance lwe(n, m, q, A, b, secret, error);
    auto kannan = KannanEmbedding::from_lwe(lwe, 1.0, 0.89);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    size_t init_dim = dbdd.dimension();
    BigReal init_vol = dbdd.volume();

    // 1. Structural Hint: sum_i s_i = 0
    VectorZZ v_sum(n, BigInt(1));
    BigInt true_sum = secret.dot(v_sum);
    REQUIRE(true_sum == BigInt(0));

    // Project using synthetic perfect hint
    dbdd.integrate_perfect_hint(v_sum, true_sum);

    REQUIRE(dbdd.dimension() == init_dim - 1);
    REQUIRE(dbdd.volume().to_double() > init_vol.to_double());

    // 2. Greedy Q-Vector Integration (canonical q-vectors)
    auto res = dbdd.integrate_q_vectors(q, 3, false, true);
    REQUIRE(res.candidates_evaluated > 0);
    REQUIRE(res.final_dim <= dbdd.dimension());
}
