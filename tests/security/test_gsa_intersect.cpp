#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/security/root_hermite.hpp"
#include "dbdd/security/gsa.hpp"
#include "dbdd/security/security_estimate.hpp"
#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/dbdd/homogenization.hpp"
#include "dbdd/dbdd/isotropization.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <numeric>

using namespace dbdd;
using Catch::Approx;

TEST_CASE("GSAModel - Profile and Decay Invariants", "[security][gsa]") {
    const size_t d = 30;
    const double ln_vol = 75.0; // Vol = exp(75)
    const double beta = 60.0;

    SECTION("Sum of log GS lengths preserves volume") {
        auto log_prof = GSAModel::log_profile(beta, d, ln_vol);
        REQUIRE(log_prof.size() == d);

        double sum_log = 0.0;
        for (double val : log_prof) {
            sum_log += val;
        }

        // Sum_{i=1}^d ln(||b_i^*||) == ln(Vol)
        REQUIRE(sum_log == Approx(ln_vol).epsilon(1e-12));
    }

    SECTION("Geometric decay ratio equals delta_beta^(-2)") {
        auto prof = GSAModel::profile(beta, d, ln_vol);
        const double delta_b = RootHermite::delta(beta);
        const double expected_ratio = 1.0 / (delta_b * delta_b);

        for (size_t i = 0; i < d - 1; ++i) {
            const double ratio = prof[i + 1] / prof[i];
            REQUIRE(ratio == Approx(expected_ratio).epsilon(1e-9));
        }

        REQUIRE(GSAModel::slope(beta) == Approx(expected_ratio).epsilon(1e-9));
        REQUIRE(GSAModel::log_slope(beta) == Approx(std::log(expected_ratio)).epsilon(1e-9));
    }

    SECTION("Projected block head consistency") {
        // For beta <= d, projected block head is at index i = d - beta + 1
        const size_t beta_int = 20;
        const size_t head_idx = d - beta_int + 1;
        const double log_head = GSAModel::log_projected_block_head(static_cast<double>(beta_int), d, ln_vol);
        const double log_i = GSAModel::log_gs_norm(static_cast<double>(beta_int), d, ln_vol, head_idx);

        REQUIRE(log_head == Approx(log_i).epsilon(1e-12));
    }

    SECTION("Target projection: Isotropic vs General") {
        const double beta_val = 55.0;
        const double iso_proj = GSAModel::log_projected_target_norm(beta_val, d);
        const double gen_proj = GSAModel::log_projected_target_norm(beta_val, d, std::sqrt(static_cast<double>(d)));

        REQUIRE(iso_proj == Approx(0.5 * std::log(beta_val)));
        REQUIRE(gen_proj == Approx(iso_proj).epsilon(1e-12));
    }
}

TEST_CASE("GSAIntersect - Objective Function and Monotonicity", "[security][gsa_intersect]") {
    const size_t d = 141;
    const double ln_vol = 147.736;

    SECTION("Monotonic increase with beta") {
        double prev_F = GSAIntersect::objective(3.0, d, ln_vol);

        for (int b = 4; b <= 100; ++b) {
            const double curr_F = GSAIntersect::objective(static_cast<double>(b), d, ln_vol);
            REQUIRE(curr_F > prev_F);
            prev_F = curr_F;
        }

        for (int b = 150; b <= 300; b += 10) {
            const double curr_F = GSAIntersect::objective(static_cast<double>(b), d, ln_vol);
            REQUIRE(curr_F > prev_F);
            prev_F = curr_F;
        }
    }

    SECTION("Equivalence between objective >= 0 and GSAModel::intersects") {
        for (double b : {20.0, 50.0, 100.0, 197.0, 198.0, 250.0}) {
            const bool cond = GSAIntersect::condition(b, d, ln_vol);
            const bool inter = GSAModel::intersects(b, d, ln_vol);
            REQUIRE(cond == inter);
        }
    }

    SECTION("General objective matches isotropic when ||s|| = sqrt(d)") {
        const double s_norm = std::sqrt(static_cast<double>(d));
        for (double b : {10.0, 50.0, 150.0, 200.0}) {
            const double f_iso = GSAIntersect::objective(b, d, ln_vol);
            const double f_gen = GSAIntersect::objective_general(b, d, ln_vol, s_norm);
            REQUIRE(f_gen == Approx(f_iso).epsilon(1e-12));
        }
    }
}

TEST_CASE("GSAIntersect - Continuous and Integer Root Solving", "[security][gsa_intersect]") {
    const size_t d = 141;
    const double ln_vol = 70.0 * std::log(3301.0) - 140.0 * std::log(20.0); // ~147.736

    const double b_cont = GSAIntersect::estimate_continuous_beta(d, ln_vol);
    const size_t b_int  = GSAIntersect::estimate_integer_beta(d, ln_vol);

    SECTION("Continuous root accurately zeros the objective") {
        const double f_at_root = GSAIntersect::objective(b_cont, d, ln_vol);
        REQUIRE(std::abs(f_at_root) < 1e-8);
    }

    SECTION("Integer root bracket condition") {
        // b_int = ceil(b_cont)
        REQUIRE(b_int == static_cast<size_t>(std::ceil(b_cont)));

        // F(b_int) >= 0 and F(b_int - 1) < 0
        REQUIRE(GSAIntersect::objective(static_cast<double>(b_int), d, ln_vol) >= 0.0);
        REQUIRE(GSAIntersect::objective(static_cast<double>(b_int - 1), d, ln_vol) < 0.0);
    }

    SECTION("Trivial / Low-hardness instance bracket") {
        // Enormous volume: already solved at beta = 2
        const double huge_vol = 5000.0;
        const double b_triv = GSAIntersect::estimate_continuous_beta(20, huge_vol);
        const size_t b_int_triv = GSAIntersect::estimate_integer_beta(20, huge_vol);

        REQUIRE(b_triv == 2.0);
        REQUIRE(b_int_triv == 2);
    }

    SECTION("Infeasible instance bracket") {
        // Extremely small volume in high dimension
        const double tiny_vol = 0.0;
        const double b_infeas = GSAIntersect::estimate_continuous_beta(500, tiny_vol, 2.0, 100.0);
        REQUIRE(b_infeas > 100.0);

        auto est = GSAIntersect::estimate(500, tiny_vol, 2.0, 100.0);
        REQUIRE_FALSE(est.solved);
    }
}

TEST_CASE("GSAIntersect - Dimension Scalability Stress Test", "[security][gsa_intersect]") {
    // Test robustness across dimensions from 10 to 1000
    const size_t dimensions[] = {10, 50, 100, 200, 400, 600, 800, 1000};

    for (size_t d : dimensions) {
        CAPTURE(d);
        // Realistic normalized LWE volume: ln(Vol) ~ d * ln(q / sigma)
        const double ln_vol = static_cast<double>(d) * std::log(10.0);

        const double b_cont = GSAIntersect::estimate_continuous_beta(d, ln_vol);
        REQUIRE(std::isfinite(b_cont));
        REQUIRE(b_cont >= 2.0);

        const size_t b_int = GSAIntersect::estimate_integer_beta(d, ln_vol);
        REQUIRE(b_int >= 2);
        REQUIRE(GSAIntersect::objective(static_cast<double>(b_int), d, ln_vol) >= 0.0);
    }
}

TEST_CASE("GSAIntersect - Monotonic Hardness Reduction Under Hints", "[security][gsa_intersect][hints]") {
    SECTION("Toy LWE with Perfect Hints") {
        // Toy LWE: n=3, m=3, q=11
        const size_t n = 3;
        const size_t m = 3;
        const BigInt q(11);

        DiscreteGaussian noise_dist(1.5, 101);
        DiscreteGaussian secret_dist(2.0, 102);

        auto lwe_inst = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 2002);
        auto kannan = KannanEmbedding::from_lwe(lwe_inst, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        auto est_init = dbdd.estimate_attack_gsa();
        size_t prev_bikz = est_init.bikz();

        const auto& s_bar = dbdd.target();

        // Reveal secret coordinates one by one as perfect hints: <s_bar, v> = l
        // Coordinate s_0 is at index m = 3 in target (e, s, 1)
        VectorZZ v0(dbdd.ambient_dimension(), BigInt(0));
        v0[3] = BigInt(1);
        dbdd.integrate_perfect_hint(v0, s_bar[3]);

        auto est_h1 = dbdd.estimate_attack_gsa();
        REQUIRE(est_h1.bikz() <= prev_bikz);
        prev_bikz = est_h1.bikz();

        // Reveal s_1 at index 4
        VectorZZ v1(dbdd.ambient_dimension(), BigInt(0));
        v1[4] = BigInt(1);
        dbdd.integrate_perfect_hint(v1, s_bar[4]);

        auto est_h2 = dbdd.estimate_attack_gsa();
        REQUIRE(est_h2.bikz() <= prev_bikz);
        prev_bikz = est_h2.bikz();

        // Reveal s_2 at index 5
        VectorZZ v2(dbdd.ambient_dimension(), BigInt(0));
        v2[5] = BigInt(1);
        dbdd.integrate_perfect_hint(v2, s_bar[5]);

        auto est_h3 = dbdd.estimate_attack_gsa();
        REQUIRE(est_h3.bikz() <= prev_bikz);
    }
}

TEST_CASE("GSAIntersect - Direct Instance Methods & Cost Models", "[security][gsa_intersect]") {
    const size_t n = 20;
    const size_t m = 20;
    const BigInt q(127);
    const double sigma = 2.5;

    // Construct diagonal covariance DBDD instance
    const size_t d = m + n + 1;
    MatrixZZ B(d, d, BigInt(0));
    for (size_t i = 0; i < d; ++i) {
        B(i, i) = (i < m) ? q : BigInt(1);
    }

    VectorDouble mu(d, 0.0);
    MatrixDouble cov(d, d, 0.0);
    for (size_t i = 0; i < d - 1; ++i) {
        cov(i, i) = sigma * sigma;
    }
    cov(d - 1, d - 1) = 1.0;

    DBDDInstance dbdd(B, mu, cov);

    SECTION("DBDDInstance and UsvpInstance estimation match") {
        auto est_dbdd = dbdd.estimate_attack_gsa();
        auto usvp = dbdd.to_usvp_instance();
        auto est_usvp = usvp.estimate_attack_gsa();

        REQUIRE(est_dbdd.dimension == est_usvp.dimension);
        REQUIRE(est_dbdd.log_volume == Approx(est_usvp.log_volume).epsilon(1e-9));
        REQUIRE(est_dbdd.beta_continuous == Approx(est_usvp.beta_continuous).epsilon(1e-9));
        REQUIRE(est_dbdd.bikz() == est_usvp.bikz());
        REQUIRE(dbdd.estimate_beta_gsa() == usvp.estimate_beta_gsa());
    }

    SECTION("Cost Model Evaluations") {
        auto est = dbdd.estimate_attack_gsa();
        const double b = static_cast<double>(est.bikz());

        REQUIRE(est.cost_bits(CostModel::CoreSVPClassical) == Approx(0.292 * b));
        REQUIRE(est.cost_bits(CostModel::CoreSVPQuantum) == Approx(0.265 * b));

        if (b > 2.0) {
            const double expected_enum = 0.187 * b * std::log2(b) - 1.019 * b + 16.1;
            REQUIRE(est.cost_bits(CostModel::EnumerationClassical) == Approx(expected_enum));
        }
    }
}
