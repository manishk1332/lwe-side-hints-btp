#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/hints/worthiness.hpp"
#include "dbdd/hints/greedy_selection.hpp"
#include "dbdd/lattice/q_vectors.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/hints/perfect_hint.hpp"

#include <cmath>
#include <vector>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Worthiness - Analytical Scores & Gram Determinants", "[hints][worthiness]") {
    SECTION("Single Vector on Identity Covariance") {
        VectorDouble v(3, 0.0);
        v[0] = 3.0;
        v[1] = 4.0;
        v[2] = 0.0; // ||v||^2 = 25

        MatrixDouble I(3, 3, 0.0);
        I(0, 0) = 1.0; I(1, 1) = 1.0; I(2, 2) = 1.0;

        double g = ShortVectorWorthinessAnalyzer::gram_determinant(v);
        REQUIRE_THAT(g, WithinAbs(25.0, 1e-12));

        double qform = ShortVectorWorthinessAnalyzer::covariance_form(v, I);
        REQUIRE_THAT(qform, WithinAbs(25.0, 1e-12));

        double v_score = ShortVectorWorthinessAnalyzer::variance_score(v, I);
        REQUIRE_THAT(v_score, WithinAbs(1.0, 1e-12));

        double proxy_ratio = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(v, I);
        REQUIRE_THAT(proxy_ratio, WithinAbs(25.0, 1e-12));

        double log_delta = ShortVectorWorthinessAnalyzer::log_hardness_delta(v, I);
        REQUIRE_THAT(log_delta, WithinAbs(std::log(25.0), 1e-12));

        // When Sigma = I and ||v|| >= 1, proxy_ratio >= 1.0, so not worthy
        REQUIRE_FALSE(ShortVectorWorthinessAnalyzer::is_worthy_proxy(I, v));
    }

    SECTION("Diagonal Covariance with High Variance (Worthy vs Unworthy)") {
        MatrixDouble Sigma(3, 3, 0.0);
        Sigma(0, 0) = 100.0;
        Sigma(1, 1) = 4.0;
        Sigma(2, 2) = 1.0;

        VectorDouble v0(3, 0.0);
        v0[0] = 2.0; // ||v0||^2 = 4, v0 Sigma v0 = 400

        VectorDouble v2(3, 0.0);
        v2[2] = 2.0; // ||v2||^2 = 4, v2 Sigma v2 = 4

        // v0 has large variance: proxy ratio = 16 / 400 = 0.04 < 1.0 (WORTHY)
        double ratio0 = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(v0, Sigma);
        REQUIRE_THAT(ratio0, WithinAbs(0.04, 1e-12));
        REQUIRE(ShortVectorWorthinessAnalyzer::is_worthy_proxy(Sigma, v0));
        REQUIRE(ShortVectorWorthinessAnalyzer::log_hardness_delta(v0, Sigma) < 0.0);

        // v2 has small variance: proxy ratio = 16 / 4 = 4.0 > 1.0 (UNWORTHY)
        double ratio2 = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(v2, Sigma);
        REQUIRE_THAT(ratio2, WithinAbs(4.0, 1e-12));
        REQUIRE_FALSE(ShortVectorWorthinessAnalyzer::is_worthy_proxy(Sigma, v2));
        REQUIRE(ShortVectorWorthinessAnalyzer::log_hardness_delta(v2, Sigma) > 0.0);
    }

    SECTION("Multi-Vector Matrix Gram and Covariance Determinants") {
        MatrixDouble Sigma(3, 3, 0.0);
        Sigma(0, 0) = 16.0;
        Sigma(1, 1) = 9.0;
        Sigma(2, 2) = 1.0;

        MatrixDouble V(2, 3, 0.0);
        V(0, 0) = 1.0; // row 0 = e0
        V(1, 1) = 1.0; // row 1 = e1

        // Orthogonal rows -> det(V V^T) = 1 * 1 = 1
        double det_g = ShortVectorWorthinessAnalyzer::gram_determinant(V);
        REQUIRE_THAT(det_g, WithinAbs(1.0, 1e-12));

        // det(V Sigma V^T) = 16 * 9 = 144
        double det_q = ShortVectorWorthinessAnalyzer::covariance_determinant(V, Sigma);
        REQUIRE_THAT(det_q, WithinAbs(144.0, 1e-12));

        double v_score = ShortVectorWorthinessAnalyzer::variance_score(V, Sigma);
        REQUIRE_THAT(v_score, WithinAbs(144.0, 1e-12));

        double proxy_ratio = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(V, Sigma);
        REQUIRE_THAT(proxy_ratio, WithinAbs(1.0 / 144.0, 1e-12));
    }
}

TEST_CASE("Worthiness - Q-Vector Generation & Lattice Membership", "[hints][q_vectors]") {
    const size_t m = 4;
    const size_t amb_dim = 9;
    const BigInt q(17);

    SECTION("Canonical and Difference Generation") {
        auto canonical = QVectorGenerator::generate_canonical(q, m, amb_dim);
        REQUIRE(canonical.size() == m);
        for (size_t i = 0; i < m; ++i) {
            REQUIRE(canonical[i].size() == amb_dim);
            REQUIRE(canonical[i][i] == q);
            for (size_t j = 0; j < amb_dim; ++j) {
                if (j != i) {
                    REQUIRE(canonical[i][j] == BigInt(0));
                }
            }
        }

        auto diffs = QVectorGenerator::generate_differences(q, m, amb_dim);
        REQUIRE(diffs.size() == (m * (m - 1)) / 2); // 6
        for (const auto& d : diffs) {
            REQUIRE(d.size() == amb_dim);
            BigInt pos_cnt = 0, neg_cnt = 0;
            for (size_t j = 0; j < amb_dim; ++j) {
                if (d[j] == q) pos_cnt += 1;
                else if (d[j] == -q) neg_cnt += 1;
                else REQUIRE(d[j] == BigInt(0));
            }
            REQUIRE(pos_cnt == BigInt(1));
            REQUIRE(neg_cnt == BigInt(1));
        }
    }

    SECTION("Lattice Membership Before and After Perfect Hint") {
        const size_t n = 3;
        const size_t m_lwe = 4;
        const BigInt q_lwe(23);

        DiscreteGaussian s_dist(1.0, 10);
        DiscreteGaussian e_dist(1.0, 11);
        auto lwe = LWEInstance::generate_random(n, m_lwe, q_lwe, s_dist, e_dist, 12);
        auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
        DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

        // On pristine Kannan lattice, all canonical q-vectors q * e_i for i < m_lwe belong to Lambda
        for (size_t i = 0; i < m_lwe; ++i) {
            VectorZZ qe(dbdd.ambient_dimension(), BigInt(0));
            qe[i] = q_lwe;
            REQUIRE(QVectorGenerator::is_in_lattice(dbdd.lattice(), qe));
        }

        auto valid_initial = QVectorGenerator::generate_valid(dbdd, q_lwe, true);
        REQUIRE(valid_initial.size() >= m_lwe);

        // Integrate a perfect hint: e_0 + e_1 = leak
        VectorZZ v_hint(dbdd.ambient_dimension(), BigInt(0));
        v_hint[0] = BigInt(1);
        v_hint[1] = BigInt(1);
        BigInt leak = dbdd.target()[0] + dbdd.target()[1];
        dbdd.integrate_perfect_hint(v_hint, leak);

        // Now canonical q * e_0 and q * e_1 should NO LONGER be in the lattice,
        // because <q * e_0, (1, 1, 0, ...)> = q != 0 mod leak.
        VectorZZ qe0(dbdd.ambient_dimension(), BigInt(0)); qe0[0] = q_lwe;
        VectorZZ qe1(dbdd.ambient_dimension(), BigInt(0)); qe1[1] = q_lwe;
        REQUIRE_FALSE(QVectorGenerator::is_in_lattice(dbdd.lattice(), qe0));
        REQUIRE_FALSE(QVectorGenerator::is_in_lattice(dbdd.lattice(), qe1));

        // BUT the difference q * (e_0 - e_1) satisfies <q*(e0-e1), (1, 1, ...)> = q - q = 0!
        // Thus it DOES survive and belong to the projected lattice!
        VectorZZ q_diff(dbdd.ambient_dimension(), BigInt(0));
        q_diff[0] = q_lwe;
        q_diff[1] = -q_lwe;
        REQUIRE(QVectorGenerator::is_in_lattice(dbdd.lattice(), q_diff));

        // Filter confirms only valid survivors remain
        auto filtered = QVectorGenerator::filter_in_lattice(dbdd.lattice(), {qe0, qe1, q_diff});
        REQUIRE(filtered.size() == 1);
        REQUIRE(filtered[0] == q_diff);
    }
}

TEST_CASE("Worthiness - Exact vs Greedy Subset Selection", "[hints][greedy_selection]") {
    const size_t n = 3;
    const size_t m = 5;
    const BigInt q(29);

    DiscreteGaussian s_dist(1.0, 501);
    DiscreteGaussian e_dist(1.0, 502);
    auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, 503);
    auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    auto candidates = QVectorGenerator::generate_canonical(q, m, dbdd.ambient_dimension());

    SECTION("Exact Selection on Small Set (k=2)") {
        auto res_exact = GreedyShortVectorSelector::select_exact(dbdd, candidates, 2);
        REQUIRE(res_exact.selected_vectors.size() == 2);
        REQUIRE(res_exact.candidates_evaluated == 10); // 5 choose 2 = 10
        REQUIRE(res_exact.final_dim == dbdd.dimension() - 2);
    }

    SECTION("Greedy Selection Agreement with Exact") {
        DBDDInstance dbdd_greedy = dbdd;
        auto res_greedy = GreedyShortVectorSelector::select_greedy(dbdd_greedy, candidates, 2, false);
        REQUIRE(res_greedy.selected_vectors.size() == 2);

        auto res_exact = GreedyShortVectorSelector::select_exact(dbdd, candidates, 2);

        // On symmetric q-vectors, both greedy and exact achieve equal proxy ratio
        REQUIRE_THAT(res_greedy.total_proxy_ratio, WithinAbs(res_exact.total_proxy_ratio, 1e-9));
    }
}

TEST_CASE("Worthiness - End-to-End Q-Vector Integration", "[hints][integration]") {
    const size_t n = 4;
    const size_t m = 6;
    const BigInt q(31);

    DiscreteGaussian s_dist(1.2, 771);
    DiscreteGaussian e_dist(1.0, 772);
    auto lwe = LWEInstance::generate_random(n, m, q, s_dist, e_dist, 773);
    auto kannan = KannanEmbedding::from_lwe(lwe, e_dist, s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    SECTION("DBDDInstance Integration & Dimension Drop") {
        const size_t init_dim = dbdd.dimension();
        auto result = dbdd.integrate_q_vectors(q, 3, false, true);

        REQUIRE(result.initial_dim == init_dim);
        REQUIRE(result.selected_vectors.size() <= 3);
        REQUIRE(dbdd.dimension() == init_dim - result.selected_vectors.size());
        REQUIRE(dbdd.target_in_lattice());

        // Covariance remains PSD
        const auto& cov = dbdd.covariance();
        for (size_t i = 0; i < cov.rows(); ++i) {
            REQUIRE(cov(i, i) >= 0.0);
        }
    }

    SECTION("Consistency with DBDDStateStatistics") {
        DBDDStateStatistics stats = dbdd.to_state_statistics();
        const size_t init_dim = stats.dimension();

        auto res_stats = GreedyShortVectorSelector::integrate_q_vectors(stats, q, m, 2, true);
        REQUIRE(res_stats.selected_vectors.size() <= 2);
        REQUIRE(stats.dimension() == init_dim - res_stats.selected_vectors.size());
    }
}
