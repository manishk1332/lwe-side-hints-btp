#include "dbdd/hints/greedy_selection.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/lattice/q_vectors.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <limits>
#include <algorithm>

namespace dbdd {

namespace {

VectorDouble to_double_vector(const VectorZZ& v) {
    VectorDouble res(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        res[i] = v[i].to_double();
    }
    return res;
}

// Generates all combinations of k indices from 0..n-1
void generate_combinations_recursive(
    size_t offset,
    size_t k,
    size_t n,
    std::vector<size_t>& current,
    std::vector<std::vector<size_t>>& all_combos) {
    if (k == 0) {
        all_combos.push_back(current);
        return;
    }
    for (size_t i = offset; i <= n - k; ++i) {
        current.push_back(i);
        generate_combinations_recursive(i + 1, k - 1, n, current, all_combos);
        current.pop_back();
    }
}

} // anonymous namespace

SelectionResult GreedyShortVectorSelector::select_exact(
    const DBDDInstance& dbdd,
    const std::vector<VectorZZ>& candidates,
    size_t k) {
    const size_t N = candidates.size();
    if (k == 0) {
        SelectionResult res;
        res.initial_dim = dbdd.dimension();
        res.final_dim = dbdd.dimension();
        auto est = dbdd.estimate_attack_gsa();
        res.initial_beta = est.beta;
        res.final_beta = est.beta;
        return res;
    }
    if (k > N) {
        throw InvalidParameterException("GreedyShortVectorSelector::select_exact: k exceeds candidate count");
    }
    if (N > 15) {
        throw InvalidParameterException("GreedyShortVectorSelector::select_exact: candidate pool exceeds 15 (combinatorial explosion)");
    }

    auto est_init = dbdd.estimate_attack_gsa();

    std::vector<std::vector<size_t>> combos;
    std::vector<size_t> cur;
    generate_combinations_recursive(0, k, N, cur, combos);

    double best_proxy_ratio = std::numeric_limits<double>::infinity();
    std::vector<size_t> best_combo;

    const size_t amb_dim = dbdd.ambient_dimension();
    const auto& cov = dbdd.covariance();

    size_t evals = 0;
    for (const auto& combo : combos) {
        evals++;
        MatrixDouble V_sub(k, amb_dim, 0.0);
        for (size_t r = 0; r < k; ++r) {
            const auto& vec = candidates[combo[r]];
            for (size_t c = 0; c < amb_dim; ++c) {
                V_sub(r, c) = vec[c].to_double();
            }
        }

        double det_g = ShortVectorWorthinessAnalyzer::gram_determinant(V_sub);
        if (det_g <= 1e-12) {
            // Linearly dependent subset
            continue;
        }

        double ratio = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(V_sub, cov);
        if (ratio < best_proxy_ratio) {
            best_proxy_ratio = ratio;
            best_combo = combo;
        }
    }

    SelectionResult result;
    result.candidates_evaluated = evals;
    result.initial_dim = dbdd.dimension();
    result.initial_beta = est_init.beta;
    result.total_proxy_ratio = best_proxy_ratio;

    if (!best_combo.empty()) {
        result.selected_indices = best_combo;
        for (size_t idx : best_combo) {
            result.selected_vectors.push_back(candidates[idx]);
        }
        result.final_dim = dbdd.dimension() >= k ? (dbdd.dimension() - k) : 0;

        // Estimate final beta via predictor
        DBDDStateStatistics stats = dbdd.to_state_statistics();
        for (const auto& v : result.selected_vectors) {
            stats.integrate_short_vector_hint(to_double_vector(v));
        }
        auto est_final = stats.estimate_attack_gsa();
        result.final_beta = est_final.beta;
    } else {
        result.final_dim = dbdd.dimension();
        result.final_beta = est_init.beta;
    }

    return result;
}

SelectionResult GreedyShortVectorSelector::select_greedy(
    DBDDInstance& dbdd,
    const std::vector<VectorZZ>& candidates,
    size_t max_k,
    bool verify_beta) {
    SelectionResult result;
    result.initial_dim = dbdd.dimension();
    auto est_cur = dbdd.estimate_attack_gsa();
    result.initial_beta = est_cur.beta;
    result.final_beta = est_cur.beta;
    result.final_dim = dbdd.dimension();

    const size_t N = candidates.size();
    if (N == 0 || dbdd.dimension() <= 1) {
        return result;
    }

    const size_t limit_k = (max_k == 0 || max_k >= dbdd.dimension()) ? (dbdd.dimension() - 1) : max_k;

    std::vector<bool> used(N, false);
    double cumulative_proxy_ratio = 1.0;

    for (size_t step = 0; step < limit_k; ++step) {
        if (dbdd.dimension() <= 1) {
            break;
        }

        size_t best_idx = N;
        double best_score = -1.0;
        double best_proxy_ratio = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < N; ++i) {
            if (used[i]) {
                continue;
            }
            result.candidates_evaluated++;

            // Must be contained in current transformed lattice
            if (!QVectorGenerator::is_in_lattice(dbdd.lattice(), candidates[i])) {
                continue;
            }

            double v_score = ShortVectorWorthinessAnalyzer::variance_score(candidates[i], dbdd.covariance());
            double p_ratio = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(candidates[i], dbdd.covariance());

            if (v_score > best_score) {
                best_score = v_score;
                best_proxy_ratio = p_ratio;
                best_idx = i;
            }
        }

        if (best_idx == N) {
            // No valid candidates remaining in lattice
            result.terminated_early = true;
            break;
        }

        // Test if candidate is worthy
        if (verify_beta) {
            if (!ShortVectorWorthinessAnalyzer::is_worthy_beta(dbdd, candidates[best_idx])) {
                result.terminated_early = true;
                break;
            }
        }

        // Apply chosen hint
        used[best_idx] = true;
        result.selected_indices.push_back(best_idx);
        result.selected_vectors.push_back(candidates[best_idx]);

        dbdd.integrate_short_vector_hint(candidates[best_idx]);
        auto est_after = dbdd.estimate_attack_gsa();

        cumulative_proxy_ratio *= best_proxy_ratio;

        SelectionStep s_record;
        s_record.vector = candidates[best_idx];
        s_record.candidate_index = best_idx;
        s_record.incremental_variance = best_score;
        s_record.incremental_proxy_ratio = best_proxy_ratio;
        s_record.beta_after = est_after.beta;
        s_record.delta_after = est_after.delta_beta;
        result.history.push_back(s_record);

        result.final_dim = dbdd.dimension();
        result.final_beta = est_after.beta;
    }

    result.total_proxy_ratio = cumulative_proxy_ratio;
    return result;
}

SelectionResult GreedyShortVectorSelector::select_greedy(
    DBDDStateStatistics& stats,
    const std::vector<VectorZZ>& candidates,
    size_t max_k,
    bool verify_beta) {
    SelectionResult result;
    result.initial_dim = stats.dimension();
    auto est_cur = stats.estimate_attack_gsa();
    result.initial_beta = est_cur.beta;
    result.final_beta = est_cur.beta;
    result.final_dim = stats.dimension();

    const size_t N = candidates.size();
    if (N == 0 || stats.dimension() <= 1) {
        return result;
    }

    const size_t limit_k = (max_k == 0 || max_k >= stats.dimension()) ? (stats.dimension() - 1) : max_k;

    std::vector<bool> used(N, false);
    double cumulative_proxy_ratio = 1.0;

    for (size_t step = 0; step < limit_k; ++step) {
        if (stats.dimension() <= 1) {
            break;
        }

        size_t best_idx = N;
        double best_score = -1.0;
        double best_proxy_ratio = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < N; ++i) {
            if (used[i]) {
                continue;
            }
            result.candidates_evaluated++;

            VectorDouble v_d = to_double_vector(candidates[i]);
            double v_score = ShortVectorWorthinessAnalyzer::variance_score(v_d, stats.covariance());
            double p_ratio = ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(v_d, stats.covariance());

            if (v_score > best_score) {
                best_score = v_score;
                best_proxy_ratio = p_ratio;
                best_idx = i;
            }
        }

        if (best_idx == N) {
            result.terminated_early = true;
            break;
        }

        VectorDouble best_vd = to_double_vector(candidates[best_idx]);
        if (verify_beta) {
            if (!ShortVectorWorthinessAnalyzer::is_worthy_beta(stats, best_vd)) {
                result.terminated_early = true;
                break;
            }
        }

        used[best_idx] = true;
        result.selected_indices.push_back(best_idx);
        result.selected_vectors.push_back(candidates[best_idx]);

        stats.integrate_short_vector_hint(best_vd);
        auto est_after = stats.estimate_attack_gsa();

        cumulative_proxy_ratio *= best_proxy_ratio;

        SelectionStep s_record;
        s_record.vector = candidates[best_idx];
        s_record.candidate_index = best_idx;
        s_record.incremental_variance = best_score;
        s_record.incremental_proxy_ratio = best_proxy_ratio;
        s_record.beta_after = est_after.beta;
        s_record.delta_after = est_after.delta_beta;
        result.history.push_back(s_record);

        result.final_dim = stats.dimension();
        result.final_beta = est_after.beta;
    }

    result.total_proxy_ratio = cumulative_proxy_ratio;
    return result;
}

SelectionResult GreedyShortVectorSelector::integrate_q_vectors(
    DBDDInstance& dbdd,
    const BigInt& q,
    size_t max_k,
    bool include_differences,
    bool verify_beta) {
    auto valid_q_vecs = QVectorGenerator::generate_valid(dbdd, q, include_differences);
    return select_greedy(dbdd, valid_q_vecs, max_k, verify_beta);
}

SelectionResult GreedyShortVectorSelector::integrate_q_vectors(
    DBDDStateStatistics& stats,
    const BigInt& q,
    size_t m,
    size_t max_k,
    bool verify_beta) {
    auto q_vecs = QVectorGenerator::generate_canonical(q, m, stats.ambient_dimension());
    return select_greedy(stats, q_vecs, max_k, verify_beta);
}

} // namespace dbdd
