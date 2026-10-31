#include "dbdd/reduction/bkz.hpp"
#include "dbdd/reduction/lll_float.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include "dbdd/core/exceptions.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

namespace dbdd {

namespace {

/**
 * @brief Constructs a unimodular matrix M in GL_k(Z) whose first row is the primitive vector x.
 * Uses extended Euclidean algorithm across coordinates to eliminate off-diagonal terms.
 */
MatrixZZ complete_first_row_unimodular(const VectorZZ& a) {
    size_t k = a.size();
    if (k < 2) {
        throw DimensionMismatchException("complete_first_row_unimodular requires dimension >= 2");
    }

    VectorZZ v = a;
    MatrixZZ M = MatrixZZ::identity(k);

    // If v[0] == 0, find first non-zero coordinate and swap columns
    if (v[0] == BigInt(0)) {
        size_t swap_idx = k;
        for (size_t i = 1; i < k; ++i) {
            if (v[i] != BigInt(0)) {
                swap_idx = i;
                break;
            }
        }
        if (swap_idx == k) {
            throw InvalidParameterException("Cannot complete the all-zero vector to a unimodular matrix");
        }
        std::swap(v[0], v[swap_idx]);
        // Permute columns of M by swapping row 0 and swap_idx in column basis
        // M has row 0 as e_0, row swap_idx as e_{swap_idx}
        M.swap_rows(0, swap_idx);
    }

    for (size_t i = 1; i < k; ++i) {
        if (v[i] == BigInt(0)) {
            continue;
        }

        BigInt x_val = v[0];
        BigInt y_val = v[i];

        BigInt s, t;
        BigInt g = BigInt::xgcd(x_val, y_val, s, t);

        BigInt a00 = x_val / g;
        BigInt a01 = y_val / g;
        BigInt a10 = -t;
        BigInt a11 = s;

        // Multiply coordinates 0 and i of M
        VectorZZ row_0 = M.row(0);
        VectorZZ row_i = M.row(i);

        for (size_t c = 0; c < k; ++c) {
            BigInt new_0 = a00 * row_0[c] + a01 * row_i[c];
            BigInt new_i = a10 * row_0[c] + a11 * row_i[c];
            M(0, c) = new_0;
            M(i, c) = new_i;
        }

        v[0] = g;
        v[i] = BigInt(0);
    }

    if (v[0] != BigInt(1)) {
        if (v[0] == BigInt(-1)) {
            for (size_t c = 0; c < k; ++c) {
                M(0, c) = -M(0, c);
            }
            v[0] = BigInt(1);
        } else {
            throw InvalidParameterException("Input vector is not primitive (gcd > 1)");
        }
    }

    return M;
}

MatrixDouble to_double_matrix(const BasisZZ& basis) {
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();
    MatrixDouble M(d, m);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < m; ++j) {
            M(i, j) = basis.matrix()(i, j).to_double();
        }
    }
    return M;
}

double compute_profile_slope(const std::vector<double>& profile) {
    size_t d = profile.size();
    if (d < 2) {
        return 0.0;
    }
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;

    for (size_t i = 0; i < d; ++i) {
        double x = static_cast<double>(i);
        double y = profile[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double n = static_cast<double>(d);
    double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) {
        return 0.0;
    }
    return (n * sum_xy - sum_x * sum_y) / denom;
}

} // anonymous namespace

BKZReduction::BKZReduction(
    BKZParameters params,
    std::shared_ptr<SVPOracle> oracle)
    : params_(std::move(params)),
      oracle_(std::move(oracle)) {
    params_.validate();
    if (!oracle_) {
        oracle_ = std::make_shared<EnumerationSVP>();
    }
}

size_t BKZReduction::run_tour(
    BasisZZ& basis,
    MatrixZZ& U,
    size_t& total_svp_calls,
    size_t& total_svp_nodes) const {

    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();
    size_t beta = std::min(params_.block_size, d);

    if (d < 2 || beta < 2) {
        return 0;
    }

    size_t insertions = 0;
    const double delta_f = params_.delta.to_double();

    for (size_t j = 0; j + 1 < d; ++j) {
        size_t k = std::min(beta, d - j);
        if (k < 2) {
            continue;
        }

        // Compute Gram-Schmidt of current basis
        GramSchmidtStateDouble gs = compute_gram_schmidt_mgs<double>(to_double_matrix(basis));

        // Solve projected SVP on block [j, j + k - 1]
        SVPResult svp_res = oracle_->solve_projected(gs, j, k, params_.svp_params);
        total_svp_calls++;
        total_svp_nodes += svp_res.nodes_visited;

        if (!svp_res.found) {
            continue;
        }

        double head_norm_sq = gs.bstar_norm_sq[j];
        double new_norm_sq = svp_res.length * svp_res.length;

        // Check Lovász improvement condition: ||v_proj||^2 < delta * ||b_j*||^2
        if (new_norm_sq < delta_f * head_norm_sq - 1e-10) {
            const auto& x = svp_res.coordinates;

            // Check if coordinates are trivial unit vector (+/- 1, 0, ..., 0)
            bool is_unit = (x[0].abs() == BigInt(1));
            if (is_unit) {
                for (size_t l = 1; l < k; ++l) {
                    if (x[l] != BigInt(0)) {
                        is_unit = false;
                        break;
                    }
                }
            }
            if (is_unit) {
                continue;
            }

            // Primitive extraction: ensure gcd = 1
            BigInt g(0);
            for (size_t l = 0; l < k; ++l) {
                g = BigInt::gcd(g, x[l].abs());
            }
            if (g == BigInt(0)) {
                continue;
            }

            VectorZZ x_prim(k);
            for (size_t l = 0; l < k; ++l) {
                x_prim[l] = x[l] / g;
            }

            // Construct unimodular completion matrix M in GL_k(Z)
            MatrixZZ M = complete_first_row_unimodular(x_prim);

            // Update basis rows [j, j + k - 1]: B_block' = M * B_block
            MatrixZZ B_block(k, m);
            for (size_t r = 0; r < k; ++r) {
                B_block.set_row(r, basis.row(j + r));
            }
            MatrixZZ B_block_new = M * B_block;
            for (size_t r = 0; r < k; ++r) {
                basis.set_row(j + r, B_block_new.row(r));
            }

            // Update unimodular transformation U rows [j, j + k - 1]
            size_t u_cols = U.cols();
            MatrixZZ U_block(k, u_cols);
            for (size_t r = 0; r < k; ++r) {
                U_block.set_row(r, U.row(j + r));
            }
            MatrixZZ U_block_new = M * U_block;
            for (size_t r = 0; r < k; ++r) {
                U.set_row(j + r, U_block_new.row(r));
            }

            // Clean up with LLL reduction on the updated basis
            auto lll_res = lll_float(basis);
            basis = std::move(lll_res.basis);
            U = lll_res.transformation * U;

            insertions++;
        }
    }

    return insertions;
}

BKZResult BKZReduction::reduce(const BasisZZ& basis) const {
    params_.validate();

    size_t d = basis.dimension();

    if (d == 0) {
        throw DimensionMismatchException("BKZReduction::reduce: Basis dimension cannot be 0");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Base cases
    if (d < 2) {
        BKZResult res;
        res.basis = basis;
        res.transformation = MatrixZZ::identity(d);
        res.tours_completed = 0;
        res.converged = true;
        if (d == 1) {
            double norm = std::sqrt(basis.row(0).squared_norm().to_double());
            res.final_profile = {std::log(std::max(1e-12, norm))};
        }
        auto end_time = std::chrono::steady_clock::now();
        res.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return res;
    }

    // Initial LLL reduction
    auto lll_init = lll_float(basis);
    BasisZZ current_basis = std::move(lll_init.basis);
    MatrixZZ U = std::move(lll_init.transformation);

    size_t total_insertions = 0;
    size_t total_svp_calls = 0;
    size_t total_svp_nodes = 0;
    size_t tours_completed = 0;
    bool converged = false;

    std::vector<std::vector<double>> tour_profiles;
    std::vector<double> prev_profile;

    for (size_t tour = 1; tour <= params_.max_tours; ++tour) {
        tours_completed = tour;

        size_t insertions = run_tour(current_basis, U, total_svp_calls, total_svp_nodes);
        total_insertions += insertions;

        // Compute current Gram-Schmidt profile
        GramSchmidtStateDouble gs = compute_gram_schmidt_mgs<double>(to_double_matrix(current_basis));
        std::vector<double> cur_profile(d);
        for (size_t i = 0; i < d; ++i) {
            cur_profile[i] = 0.5 * std::log(std::max(1e-20, gs.bstar_norm_sq[i]));
        }

        if (params_.log_tours) {
            tour_profiles.push_back(cur_profile);
        }

        // Convergence check: 0 insertions in a tour
        if (insertions == 0 && tour >= params_.min_tours) {
            converged = true;
            break;
        }

        // Early abort check: negligible change across tours
        if (tour >= 2 && !prev_profile.empty()) {
            double max_diff = 0.0;
            for (size_t i = 0; i < d; ++i) {
                double diff = std::abs(cur_profile[i] - prev_profile[i]) /
                              std::max(1.0, std::abs(prev_profile[i]));
                max_diff = std::max(max_diff, diff);
            }
            if (max_diff < params_.early_abort_threshold && tour >= params_.min_tours) {
                converged = true;
                break;
            }
        }

        prev_profile = std::move(cur_profile);

        // Overall timeout check
        if (params_.timeout_seconds > 0.0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed >= params_.timeout_seconds) {
                converged = false;
                break;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // Final profile & slope
    GramSchmidtStateDouble gs_final = compute_gram_schmidt_mgs<double>(to_double_matrix(current_basis));
    std::vector<double> final_profile(d);
    for (size_t i = 0; i < d; ++i) {
        final_profile[i] = 0.5 * std::log(std::max(1e-20, gs_final.bstar_norm_sq[i]));
    }
    double final_slope = compute_profile_slope(final_profile);

    BKZResult res;
    res.basis = std::move(current_basis);
    res.transformation = std::move(U);
    res.tours_completed = tours_completed;
    res.total_svp_calls = total_svp_calls;
    res.total_svp_nodes = total_svp_nodes;
    res.total_insertions = total_insertions;
    res.converged = converged;
    res.elapsed_time_seconds = total_time;
    res.tour_profiles = std::move(tour_profiles);
    res.final_profile = std::move(final_profile);
    res.final_slope = final_slope;

    return res;
}

LatticeZZ BKZReduction::reduce(const LatticeZZ& lattice) const {
    BKZResult res = reduce(lattice.basis());
    return LatticeZZ(std::move(res.basis), lattice.volume());
}

std::optional<VectorZZ> BKZReduction::recover_lwe_secret(
    const BasisZZ& reduced_basis,
    const LWEInstance& lwe) {

    size_t d = reduced_basis.dimension();
    size_t m = lwe.m();
    size_t n = lwe.n();
    const BigInt& q = lwe.q();

    if (reduced_basis.ambient_dimension() != m + n + 1) {
        return std::nullopt;
    }

    for (size_t row = 0; row < d; ++row) {
        const VectorZZ& v = reduced_basis.row(row);

        BigInt last_coord = v[m + n];
        if (last_coord == BigInt(1) || last_coord == BigInt(-1)) {
            int sign = (last_coord == BigInt(1)) ? 1 : -1;

            VectorZZ s_cand(n);
            for (size_t j = 0; j < n; ++j) {
                s_cand[j] = BigInt(sign) * v[m + j];
            }

            VectorZZ e_cand(m);
            for (size_t j = 0; j < m; ++j) {
                e_cand[j] = BigInt(sign) * v[j];
            }

            // Verify: b = s * A^T + e (mod q)
            VectorZZ sA = s_cand * lwe.A().transpose();
            bool valid = true;
            for (size_t j = 0; j < m; ++j) {
                BigInt computed = (sA[j] + e_cand[j]) % q;
                if (computed < BigInt(0)) computed += q;
                BigInt expected = lwe.b()[j] % q;
                if (expected < BigInt(0)) expected += q;

                if (computed != expected) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                return s_cand;
            }
        }
    }

    return std::nullopt;
}

std::optional<VectorZZ> BKZReduction::recover_dbdd_secret(
    const BasisZZ& reduced_basis,
    const DBDDInstance& dbdd) {

    if (!dbdd.has_target()) {
        return std::nullopt;
    }

    const VectorZZ& target = dbdd.target();
    size_t d = reduced_basis.dimension();

    for (size_t row = 0; row < d; ++row) {
        const VectorZZ& v = reduced_basis.row(row);
        if (v == target) {
            return v;
        }
        VectorZZ neg_v = v;
        for (size_t j = 0; j < neg_v.size(); ++j) {
            neg_v[j] = -neg_v[j];
        }
        if (neg_v == target) {
            return target;
        }
    }

    return std::nullopt;
}

} // namespace dbdd
