#include "dbdd/reduction/enumeration.hpp"
#include "dbdd/reduction/lll_float.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include "dbdd/core/exceptions.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

namespace dbdd {

namespace {

struct LevelState {
    long long x{0};
    double center{0.0};
    long long x_0{0};
    long long step{0};
    int dir{1};
    bool all_upper_zero{true};

    // For Fincke-Pohst linear interval:
    long long low{0};
    long long high{0};
};

/**
 * @brief Advances to the next coordinate candidate at the given level.
 * @return true if a candidate within the current level bound exists, false if exhausted.
 */
bool advance_candidate(
    LevelState& st,
    EnumerationStrategy strategy,
    double partial_norm_upper,
    double B_k,
    double R_sq) {

    double rem = R_sq - partial_norm_upper;
    if (rem < -1e-12) {
        return false;
    }
    double max_dist = (rem > 0.0) ? std::sqrt(rem / B_k) : 0.0;

    if (strategy == EnumerationStrategy::SchnorrEuchner) {
        st.step++;
        if (st.all_upper_zero) {
            // Highest non-zero coordinate must be positive: 0, 1, 2, 3, ...
            st.x = st.step;
        } else {
            // Zigzag sequence around x_0
            long long delta = (st.step % 2 == 1)
                ? ((st.step + 1) / 2) * st.dir
                : -(st.step / 2) * st.dir;
            st.x = st.x_0 + delta;
        }

        double dist = std::abs(static_cast<double>(st.x) - st.center);
        return dist <= max_dist + 1e-12;
    } else {
        // Fincke-Pohst: linear iteration from low to high
        st.x++;
        return st.x <= st.high;
    }
}

/**
 * @brief Initializes the first coordinate candidate when entering a level.
 * @return true if an initial valid candidate exists, false if level bound is empty.
 */
bool init_level_candidate(
    LevelState& st,
    EnumerationStrategy strategy,
    double partial_norm_upper,
    double B_k,
    double R_sq,
    bool is_level_zero,
    bool prune_zero) {

    double rem = R_sq - partial_norm_upper;
    if (rem < -1e-12) {
        return false;
    }
    double max_dist = (rem > 0.0) ? std::sqrt(rem / B_k) : 0.0;

    if (strategy == EnumerationStrategy::SchnorrEuchner) {
        st.step = 0;
        if (st.all_upper_zero) {
            // Upper coordinates are all zero
            if (is_level_zero && prune_zero) {
                // At level 0 with upper coords zero, x=0 is the zero vector; start at x=1
                st.step = 1;
                st.x = 1;
            } else {
                st.x = 0;
            }
        } else {
            st.x_0 = static_cast<long long>(std::round(st.center));
            st.dir = (st.center >= static_cast<double>(st.x_0)) ? 1 : -1;
            st.x = st.x_0;
        }

        double dist = std::abs(static_cast<double>(st.x) - st.center);
        if (dist > max_dist + 1e-12) {
            return false;
        }
        return true;
    } else {
        // Fincke-Pohst
        long long low = static_cast<long long>(std::ceil(st.center - max_dist - 1e-12));
        long long high = static_cast<long long>(std::floor(st.center + max_dist + 1e-12));

        if (st.all_upper_zero) {
            if (is_level_zero && prune_zero) {
                low = std::max(1LL, low);
            } else {
                low = std::max(0LL, low);
            }
        }

        if (low > high) {
            return false;
        }

        st.low = low;
        st.high = high;
        st.x = low;
        return true;
    }
}

} // anonymous namespace

SVPResult EnumerationSVP::solve(
    const BasisZZ& basis,
    const SVPParameters& params) {

    params.validate();

    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();

    if (d == 0) {
        throw DimensionMismatchException("EnumerationSVP::solve: Basis dimension cannot be 0");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Base case: rank 1
    if (d == 1) {
        SVPResult res;
        res.found = true;
        res.coordinates = VectorZZ(1, BigInt(1));
        res.vector = basis.row(0);
        res.squared_norm = res.vector.squared_norm();
        res.length = std::sqrt(res.squared_norm.to_double());
        res.nodes_visited = 1;
        res.complete_search = true;
        auto end_time = std::chrono::steady_clock::now();
        res.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return res;
    }

    // Optional LLL preprocessing
    BasisZZ search_basis = basis;
    MatrixZZ U = MatrixZZ::identity(d);

    if (params.preprocess_lll) {
        auto lll_res = lll_float(basis);
        search_basis = std::move(lll_res.basis);
        U = std::move(lll_res.transformation);
    }

    // Modified Gram-Schmidt on search basis in double precision
    MatrixDouble B_double(d, m);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < m; ++j) {
            B_double(i, j) = search_basis.matrix()(i, j).to_double();
        }
    }

    GramSchmidtStateDouble gs = compute_gram_schmidt_mgs<double>(B_double);

    // Initial radius determination
    double R_sq = 0.0;
    VectorZZ best_coords_search;
    VectorZZ best_vector;
    BigInt best_sq_norm;
    bool found_candidate = false;

    if (params.radius_bound > 0.0) {
        double r = params.radius_bound * params.radius_expansion_factor;
        R_sq = r * r;
    } else {
        // Default: test all basis vectors as initial bounds, take the minimum
        size_t best_idx = 0;
        BigInt min_norm = search_basis.row(0).squared_norm();
        for (size_t i = 1; i < d; ++i) {
            BigInt cur_norm = search_basis.row(i).squared_norm();
            if (cur_norm < min_norm) {
                min_norm = cur_norm;
                best_idx = i;
            }
        }
        R_sq = min_norm.to_double();
        best_coords_search = VectorZZ(d, BigInt(0));
        best_coords_search[best_idx] = BigInt(1);
        best_vector = search_basis.row(best_idx);
        best_sq_norm = min_norm;
        found_candidate = true;
    }

    // Core iterative enumeration
    std::vector<LevelState> levels(d);
    std::vector<double> partial_norm(d + 1, 0.0);
    size_t nodes_visited = 0;
    bool complete = true;

    // Initialize top level d - 1
    levels[d - 1].center = 0.0;
    levels[d - 1].all_upper_zero = true;

    if (!init_level_candidate(
            levels[d - 1],
            params.strategy,
            0.0,
            gs.bstar_norm_sq[d - 1],
            R_sq,
            false,
            params.prune_zero)) {
        // Empty tree
        SVPResult res;
        res.found = found_candidate;
        if (found_candidate) {
            res.coordinates = best_coords_search * U;
            res.vector = std::move(best_vector);
            res.squared_norm = best_sq_norm;
            res.length = std::sqrt(res.squared_norm.to_double());
        }
        res.nodes_visited = 1;
        res.complete_search = true;
        auto end_time = std::chrono::steady_clock::now();
        res.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return res;
    }

    size_t k = d - 1;

    while (true) {
        nodes_visited++;

        if (params.max_nodes > 0 && nodes_visited >= params.max_nodes) {
            complete = false;
            break;
        }

        if (params.timeout_seconds > 0.0 && (nodes_visited & 31) == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed >= params.timeout_seconds) {
                complete = false;
                break;
            }
        }

        // Current contribution at level k
        double c_k = static_cast<double>(levels[k].x) - levels[k].center;
        double current_norm = partial_norm[k + 1] + c_k * c_k * gs.bstar_norm_sq[k];

        if (k == 0) {
            // Reached leaf node: test candidate
            bool is_zero = levels[0].all_upper_zero && (levels[0].x == 0);
            if (!is_zero || !params.prune_zero) {
                if (current_norm <= R_sq + 1e-12) {
                    // Form exact integer vector to verify and update
                    VectorZZ coords_search(d);
                    for (size_t i = 0; i < d; ++i) {
                        coords_search[i] = BigInt(levels[i].x);
                    }
                    VectorZZ v_candidate = coords_search * search_basis.matrix();
                    BigInt exact_norm = v_candidate.squared_norm();

                    if (!found_candidate || exact_norm < best_sq_norm) {
                        best_sq_norm = exact_norm;
                        best_vector = std::move(v_candidate);
                        best_coords_search = std::move(coords_search);
                        found_candidate = true;
                        R_sq = exact_norm.to_double(); // Dynamic radius shrinking
                    }
                }
            }

            // Advance at level 0
            while (!advance_candidate(
                levels[k],
                params.strategy,
                partial_norm[k + 1],
                gs.bstar_norm_sq[k],
                R_sq)) {
                // Ascend
                k++;
                if (k >= d) {
                    break;
                }
            }
            if (k >= d) {
                break;
            }
        } else {
            // Descend to level k - 1
            partial_norm[k] = current_norm;
            size_t next_k = k - 1;

            // Compute center for level next_k: - sum_{j = next_k + 1}^{d - 1} x_j * mu(j, next_k)
            double center_val = 0.0;
            for (size_t j = next_k + 1; j < d; ++j) {
                center_val -= static_cast<double>(levels[j].x) * gs.mu(j, next_k);
            }
            levels[next_k].center = center_val;
            levels[next_k].all_upper_zero = levels[k].all_upper_zero && (levels[k].x == 0);

            if (init_level_candidate(
                    levels[next_k],
                    params.strategy,
                    partial_norm[k],
                    gs.bstar_norm_sq[next_k],
                    R_sq,
                    next_k == 0,
                    params.prune_zero)) {
                k = next_k;
            } else {
                // Cannot descend; advance current level k
                while (!advance_candidate(
                    levels[k],
                    params.strategy,
                    partial_norm[k + 1],
                    gs.bstar_norm_sq[k],
                    R_sq)) {
                    // Ascend
                    k++;
                    if (k >= d) {
                        break;
                    }
                }
                if (k >= d) {
                    break;
                }
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    SVPResult res;
    res.found = found_candidate;
    if (found_candidate) {
        // Transform coordinates back to original basis B: x = x_search * U
        res.coordinates = best_coords_search * U;
        res.vector = std::move(best_vector);
        res.squared_norm = best_sq_norm;
        res.length = std::sqrt(best_sq_norm.to_double());
    }
    res.nodes_visited = nodes_visited;
    res.complete_search = complete;
    res.elapsed_time_seconds = total_time;

    return res;
}

SVPResult EnumerationSVP::solve_projected(
    const GramSchmidtStateDouble& gs,
    size_t start_idx,
    size_t block_size,
    const SVPParameters& params) {

    params.validate();

    size_t total_dim = gs.dimension();
    if (start_idx >= total_dim) {
        throw IndexOutOfBoundsException("EnumerationSVP::solve_projected: start_idx out of bounds");
    }
    if (block_size == 0 || start_idx + block_size > total_dim) {
        throw DimensionMismatchException("EnumerationSVP::solve_projected: Invalid block_size");
    }

    auto start_time = std::chrono::steady_clock::now();
    size_t beta = block_size;

    // Base case: rank 1 block
    if (beta == 1) {
        SVPResult res;
        res.found = true;
        res.coordinates = VectorZZ(1, BigInt(1));
        res.squared_norm = BigInt(static_cast<long long>(std::round(gs.bstar_norm_sq[start_idx])));
        res.length = std::sqrt(gs.bstar_norm_sq[start_idx]);
        res.nodes_visited = 1;
        res.complete_search = true;
        auto end_time = std::chrono::steady_clock::now();
        res.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return res;
    }

    // Initial radius: default to projected head norm ||b_{start_idx}*||^2
    double R_sq = gs.bstar_norm_sq[start_idx];
    if (params.radius_bound > 0.0) {
        double r = params.radius_bound * params.radius_expansion_factor;
        R_sq = r * r;
    }

    VectorZZ best_coords(beta, BigInt(0));
    best_coords[0] = BigInt(1);
    double best_proj_sq_norm = gs.bstar_norm_sq[start_idx];
    bool found_candidate = (params.radius_bound <= 0.0);

    std::vector<LevelState> levels(beta);
    std::vector<double> partial_norm(beta + 1, 0.0);
    size_t nodes_visited = 0;
    bool complete = true;

    // Initialize top level beta - 1
    levels[beta - 1].center = 0.0;
    levels[beta - 1].all_upper_zero = true;

    if (!init_level_candidate(
            levels[beta - 1],
            params.strategy,
            0.0,
            gs.bstar_norm_sq[start_idx + beta - 1],
            R_sq,
            false,
            params.prune_zero)) {
        SVPResult res;
        res.found = found_candidate;
        if (found_candidate) {
            res.coordinates = std::move(best_coords);
            res.squared_norm = BigInt(static_cast<long long>(std::round(best_proj_sq_norm)));
            res.length = std::sqrt(best_proj_sq_norm);
        }
        res.nodes_visited = 1;
        res.complete_search = true;
        auto end_time = std::chrono::steady_clock::now();
        res.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        return res;
    }

    size_t k = beta - 1;

    while (true) {
        nodes_visited++;

        if (params.max_nodes > 0 && nodes_visited >= params.max_nodes) {
            complete = false;
            break;
        }

        if (params.timeout_seconds > 0.0 && (nodes_visited & 31) == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed >= params.timeout_seconds) {
                complete = false;
                break;
            }
        }

        size_t global_k = start_idx + k;
        double c_k = static_cast<double>(levels[k].x) - levels[k].center;
        double current_norm = partial_norm[k + 1] + c_k * c_k * gs.bstar_norm_sq[global_k];

        if (k == 0) {
            bool is_zero = levels[0].all_upper_zero && (levels[0].x == 0);
            if (!is_zero || !params.prune_zero) {
                if (current_norm <= R_sq + 1e-12) {
                    VectorZZ coords(beta);
                    for (size_t i = 0; i < beta; ++i) {
                        coords[i] = BigInt(levels[i].x);
                    }
                    best_coords = std::move(coords);
                    best_proj_sq_norm = current_norm;
                    found_candidate = true;
                    R_sq = current_norm; // Dynamic radius shrinking
                }
            }

            while (!advance_candidate(
                levels[k],
                params.strategy,
                partial_norm[k + 1],
                gs.bstar_norm_sq[global_k],
                R_sq)) {
                k++;
                if (k >= beta) {
                    break;
                }
            }
            if (k >= beta) {
                break;
            }
        } else {
            partial_norm[k] = current_norm;
            size_t next_k = k - 1;
            size_t global_next_k = start_idx + next_k;

            // Center: - sum_{j = next_k + 1}^{beta - 1} x_j * mu(start_idx + j, global_next_k)
            double center_val = 0.0;
            for (size_t j = next_k + 1; j < beta; ++j) {
                center_val -= static_cast<double>(levels[j].x) * gs.mu(start_idx + j, global_next_k);
            }
            levels[next_k].center = center_val;
            levels[next_k].all_upper_zero = levels[k].all_upper_zero && (levels[k].x == 0);

            if (init_level_candidate(
                    levels[next_k],
                    params.strategy,
                    partial_norm[k],
                    gs.bstar_norm_sq[global_next_k],
                    R_sq,
                    next_k == 0,
                    params.prune_zero)) {
                k = next_k;
            } else {
                while (!advance_candidate(
                    levels[k],
                    params.strategy,
                    partial_norm[k + 1],
                    gs.bstar_norm_sq[global_k],
                    R_sq)) {
                    k++;
                    if (k >= beta) {
                        break;
                    }
                }
                if (k >= beta) {
                    break;
                }
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    SVPResult res;
    res.found = found_candidate;
    if (found_candidate) {
        res.coordinates = std::move(best_coords);
        res.squared_norm = BigInt(static_cast<long long>(std::round(best_proj_sq_norm)));
        res.length = std::sqrt(best_proj_sq_norm);
    }
    res.nodes_visited = nodes_visited;
    res.complete_search = complete;
    res.elapsed_time_seconds = total_time;

    return res;
}

SVPResult EnumerationSVP::find_shortest_vector(
    const BasisZZ& basis,
    const SVPParameters& params) {
    EnumerationSVP oracle;
    return oracle.solve(basis, params);
}

SVPResult EnumerationSVP::find_shortest_vector_projected(
    const GramSchmidtStateDouble& gs,
    size_t start_idx,
    size_t block_size,
    const SVPParameters& params) {
    EnumerationSVP oracle;
    return oracle.solve_projected(gs, start_idx, block_size, params);
}

} // namespace dbdd
