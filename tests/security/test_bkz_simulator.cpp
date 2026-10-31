#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/security/bkz_simulator.hpp"
#include "dbdd/security/gsa.hpp"
#include "dbdd/reduction/bkz.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include <chrono>
#include <cmath>

using namespace dbdd;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("BKZSimulator - Parameter Validation & Edge Cases", "[security][simulator]") {
    SECTION("Invalid Parameters") {
        BKZSimulatorParameters params;

        params.block_size = 1;
        REQUIRE_THROWS_AS(params.validate(), InvalidParameterException);

        params.block_size = 5;
        params.pruning_proba = 0.0;
        REQUIRE_THROWS_AS(params.validate(), InvalidParameterException);

        params.pruning_proba = 1.5;
        REQUIRE_THROWS_AS(params.validate(), InvalidParameterException);

        params.pruning_proba = 0.5;
        params.early_abort_threshold = -1.0;
        REQUIRE_THROWS_AS(params.validate(), InvalidParameterException);
    }

    SECTION("Dimension 0 and 1 Edge Cases") {
        BKZSimulatorParameters params(5, 10);

        REQUIRE_THROWS_AS(BKZSimulator::simulate(std::vector<double>{}, params), DimensionMismatchException);
        REQUIRE_THROWS_AS(BKZSimulator::simulate(0, 10.0, params), DimensionMismatchException);

        std::vector<double> profile1 = {2.5};
        auto res1 = BKZSimulator::simulate(profile1, params);
        REQUIRE(res1.log_profile.size() == 1);
        REQUIRE_THAT(res1.log_profile[0], WithinAbs(2.5, 1e-12));
        REQUIRE_THAT(res1.profile[0], WithinAbs(std::exp(2.5), 1e-12));
        REQUIRE(res1.converged);
        REQUIRE(res1.tours_completed == 0);
        REQUIRE_THAT(res1.final_slope, WithinAbs(0.0, 1e-12));
    }

    SECTION("Projected Block Head Validation") {
        std::vector<double> prof = {1.0, 2.0, 3.0, 4.0, 5.0};
        BKZSimulatorParameters params(3, 5);
        auto res = BKZSimulator::simulate(prof, params);

        REQUIRE_THROWS_AS(res.projected_block_head(0), IndexOutOfBoundsException);
        REQUIRE_THROWS_AS(res.projected_block_head(6), IndexOutOfBoundsException);
        // k = 2 at tail corresponds to index 5 - 2 = 3
        REQUIRE_THAT(res.projected_block_head(2), WithinAbs(res.log_profile[3], 1e-12));
    }
}

TEST_CASE("BKZSimulator - Chen-Nguyen Constant Invariants", "[security][simulator]") {
    SECTION("Base Cases & Known Values") {
        REQUIRE_THAT(BKZSimulator::chen_nguyen_constant_ln(1), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(BKZSimulator::chen_nguyen_constant_log2(1), WithinAbs(0.0, 1e-12));

        // Dimension 2: c_log2[2] ~= 0.00109348
        REQUIRE_THAT(BKZSimulator::chen_nguyen_constant_log2(2), WithinRel(0.00109348, 1e-3));

        // Dimension 45: c_log2[45] ~= 0.789528
        REQUIRE_THAT(BKZSimulator::chen_nguyen_constant_log2(45), WithinRel(0.789528, 1e-3));
    }

    SECTION("Smooth Continuity Across Boundary d = 45 and d = 46") {
        double c45 = BKZSimulator::chen_nguyen_constant_ln(45);
        double c46 = BKZSimulator::chen_nguyen_constant_ln(46);
        REQUIRE(c46 > c45);
        // Jump must be smooth (< 0.05)
        REQUIRE(std::abs(c46 - c45) < 0.05);
    }

    SECTION("Positivity, General Trend, and Base-2 Identity") {
        const double ln2 = std::log(2.0);
        for (size_t d = 2; d <= 100; ++d) {
            double c_ln = BKZSimulator::chen_nguyen_constant_ln(d);
            double c_log2 = BKZSimulator::chen_nguyen_constant_log2(d);

            REQUIRE(c_ln > 0.0);
            REQUIRE_THAT(c_ln, WithinAbs(c_log2 * ln2, 1e-12));

            // For d >= 46, Gaussian heuristic formula is strictly monotonic
            if (d > 46) {
                double c_prev = BKZSimulator::chen_nguyen_constant_ln(d - 1);
                REQUIRE(c_ln >= c_prev);
            }
        }
        // General trend across empirical table is increasing
        REQUIRE(BKZSimulator::chen_nguyen_constant_ln(45) > BKZSimulator::chen_nguyen_constant_ln(2));
    }
}

TEST_CASE("BKZSimulator - Exact Volume Preservation", "[security][simulator]") {
    const std::vector<size_t> test_dims = {10, 25, 50, 100};
    const std::vector<size_t> test_betas = {3, 10, 20};

    for (size_t d : test_dims) {
        for (size_t beta : test_betas) {
            if (beta > d) continue;

            double ln_vol = 2.5 * static_cast<double>(d);
            BKZSimulatorParameters params(beta, 15);
            auto res = BKZSimulator::simulate(d, ln_vol, params);

            // Verify sum_{i=0}^{d-1} ln(||b_i^*||) matches initial ln_vol to machine precision
            double simulated_ln_vol = 0.0;
            for (double v : res.log_profile) {
                simulated_ln_vol += v;
            }
            REQUIRE_THAT(simulated_ln_vol, WithinAbs(ln_vol, 1e-11));
            REQUIRE_THAT(res.log_volume, WithinAbs(ln_vol, 1e-11));
        }
    }
}

TEST_CASE("BKZSimulator - Slope Flattening & Profile Convergence", "[security][simulator]") {
    const size_t d = 30;
    const double ln_vol = 45.0;

    SECTION("Slope Flattens as Tours Increase") {
        BKZSimulatorParameters params_t1(10, 1, 0.5, 1e-12, SimulationModel::ChenNguyen, true);
        auto res_t1 = BKZSimulator::simulate(d, ln_vol, params_t1);

        BKZSimulatorParameters params_t5(10, 5, 0.5, 1e-12, SimulationModel::ChenNguyen, true);
        auto res_t5 = BKZSimulator::simulate(d, ln_vol, params_t5);

        BKZSimulatorParameters params_t20(10, 20, 0.5, 1e-12, SimulationModel::ChenNguyen, true);
        auto res_t20 = BKZSimulator::simulate(d, ln_vol, params_t20);

        // Slopes are negative: flattening means slope values become less negative (increase)
        REQUIRE(res_t1.final_slope < 0.0);
        REQUIRE(res_t5.final_slope < 0.0);
        REQUIRE(res_t20.final_slope < 0.0);

        REQUIRE(res_t1.final_slope < res_t5.final_slope);
        REQUIRE(res_t5.final_slope <= res_t20.final_slope);

        // First vector length ||b_0^*|| decreases with tours
        REQUIRE(res_t1.head_length() > res_t5.head_length());
        REQUIRE(res_t5.head_length() >= res_t20.head_length());

        // History recorded
        REQUIRE(res_t1.tour_log_profiles.size() == 2); // init + tour 1
        REQUIRE(res_t5.tour_log_profiles.size() == 6);
    }

    SECTION("Larger Block Size Produces Stronger Tail Reduction") {
        double init_slope = BKZSimulator::compute_profile_slope(
            BKZSimulator::initial_lll_log_profile(d, ln_vol));

        BKZSimulatorParameters params_beta5(5, 20);
        auto res_beta5 = BKZSimulator::simulate(d, ln_vol, params_beta5);

        BKZSimulatorParameters params_beta15(15, 20);
        auto res_beta15 = BKZSimulator::simulate(d, ln_vol, params_beta15);

        // Both BKZ block sizes significantly flatten the initial LLL slope
        REQUIRE(res_beta5.final_slope > init_slope);
        REQUIRE(res_beta15.final_slope > init_slope);

        // Larger block size beta=15 achieves deeper tail reduction than beta=5
        REQUIRE(res_beta15.tail_length() < res_beta5.tail_length());
    }

    SECTION("Convergence Detection") {
        // High max_tours should trigger early convergence
        BKZSimulatorParameters params(8, 100, 0.5, 1e-6);
        auto res = BKZSimulator::simulate(d, ln_vol, params);

        REQUIRE(res.converged);
        REQUIRE(res.tours_completed < 100);
    }
}

TEST_CASE("BKZSimulator - Comparison Against Actual BKZ Reduction", "[security][simulator]") {
    // End-to-end lattice instance from Kannan embedding (n=3, m=6, q=23)
    const size_t n = 3;
    const size_t m = 6;
    const BigInt q(23);

    DiscreteGaussian secret_dist(1.2, 101);
    DiscreteGaussian noise_dist(1.0, 102);

    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 4004);
    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    BasisZZ basis = kannan.basis();

    // Actual BKZ reduction
    BKZParameters bkz_params(4, 10, 0.99);
    auto bkz_res = bkz_reduce(basis, bkz_params);

    // BKZ simulation on the exact same basis
    BKZSimulatorParameters sim_params(4, 10);
    auto sim_res = BKZSimulator::simulate(basis, sim_params);

    CAPTURE(bkz_res.final_profile);
    CAPTURE(sim_res.log_profile);
    CAPTURE(bkz_res.final_slope);
    CAPTURE(sim_res.final_slope);

    // Total volume matches
    REQUIRE_THAT(sim_res.volume(), WithinRel(compute_volume_exact(bkz_res.basis).to_double(), 1e-4));

    // Target vector at row 0 is the planted short secret (detected below lattice background)
    REQUIRE(bkz_res.final_profile[0] < sim_res.log_profile[0]);

    // Background lattice vectors (indices 1 to 8) show negative downward slope in actual BKZ
    double bkz_bg_slope = (bkz_res.final_profile[8] - bkz_res.final_profile[1]) / 7.0;
    double sim_bg_slope = (sim_res.log_profile[8] - sim_res.log_profile[1]) / 7.0;
    REQUIRE(bkz_bg_slope < 0.0);
    REQUIRE(sim_bg_slope < 0.0);

    // Background slopes agree within 25%
    REQUIRE_THAT(sim_bg_slope, WithinRel(bkz_bg_slope, 0.25));

    // Background log-lengths agree with simulation within 10%
    for (size_t i = 1; i <= 8; ++i) {
        REQUIRE_THAT(sim_res.log_profile[i], WithinRel(bkz_res.final_profile[i], 0.10));
    }
}

TEST_CASE("BKZSimulator - Cryptographic Scalability Benchmark", "[security][simulator]") {
    // Cryptographic dimension d = 500, beta = 100, 20 tours
    const size_t d = 500;
    const double ln_vol = 1200.0;
    BKZSimulatorParameters params(100, 20);

    auto start = std::chrono::steady_clock::now();
    auto res = BKZSimulator::simulate(d, ln_vol, params);
    auto end = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    REQUIRE(res.log_profile.size() == d);
    REQUIRE_THAT(res.log_volume, WithinAbs(ln_vol, 1e-10));
    REQUIRE(res.tours_completed > 0);

    // Must execute in under 15 ms
    REQUIRE(elapsed_ms < 15.0);
}
