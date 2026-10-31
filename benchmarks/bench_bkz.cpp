// benchmarks/bench_bkz.cpp
//
// Paper Reference: Albrecht et al. Section 2, 3 / Schnorr (1987) / main_plan.txt Section 58
// Mathematical Purpose: BKZ lattice reduction tour performance and block size scaling benchmark.
// Evaluates:
//   - Wall-clock time per tour (ms)
//   - Number of tours until termination
//   - Total SVP oracle calls
//   - Total SVP search tree nodes
//   - Shortest basis vector Euclidean norm ||b_1||

#include "dbdd/lattice/basis.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include "dbdd/reduction/enumeration.hpp"
#include "dbdd/reduction/bkz.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using namespace dbdd;

static BasisZZ make_test_basis(size_t d, uint64_t seed = 42) {
    MatrixZZ B = MatrixZZ::identity(d);
    for (size_t i = 0; i < d; ++i) {
        B(i, i) = BigInt(static_cast<long long>(25 + (i % 9) * 4));
    }

    std::mt19937_64 rng(seed);
    for (size_t step = 0; step < 4 * d; ++step) {
        size_t r1 = rng() % d;
        size_t r2 = rng() % d;
        if (r1 != r2) {
            long long factor = static_cast<long long>((rng() % 4) + 1);
            if (rng() % 2 == 0) factor = -factor;
            B.row_add_multiple(r1, r2, BigInt(factor));
        }
        if (step % 2 == 0) {
            size_t s1 = rng() % d;
            size_t s2 = rng() % d;
            B.swap_rows(s1, s2);
        }
    }
    return BasisZZ(B);
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "          DBDD Framework: BKZ Reduction Tour & Block Size Scaling Benchmark             \n";
    std::cout << "========================================================================================\n";

    const size_t d = 20;
    std::cout << "Fixed Lattice Dimension: d = " << d << "\n\n";

    std::cout << std::string(92, '-') << "\n"
              << std::left << std::setw(8) << "beta"
              << std::right << std::setw(12) << "Tours"
              << std::setw(14) << "Total Time(ms)"
              << std::setw(14) << "ms / Tour"
              << std::setw(14) << "SVP Calls"
              << std::setw(16) << "SVP Nodes"
              << std::setw(14) << "||b_1||"
              << "\n"
              << std::string(92, '-') << "\n";

    std::vector<size_t> betas = {2, 4, 6, 8, 10, 12, 14};

    for (size_t beta : betas) {
        BasisZZ basis = make_test_basis(d, 500 + beta);
        basis = lll_exact(basis).basis;

        BKZParameters params(beta, 10, 0.99);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = bkz_reduce(basis, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_per_tour = (res.tours_completed > 0) ? (total_ms / res.tours_completed) : 0.0;
        double b1_norm = std::sqrt(res.basis.row(0).dot(res.basis.row(0)).to_double());

        std::cout << std::left << std::setw(8) << beta
                  << std::right << std::setw(12) << res.tours_completed
                  << std::setw(14) << std::fixed << std::setprecision(2) << total_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << ms_per_tour
                  << std::setw(14) << res.total_svp_calls
                  << std::setw(16) << res.total_svp_nodes
                  << std::setw(14) << std::fixed << std::setprecision(2) << b1_norm
                  << "\n";
    }

    std::cout << std::string(92, '-') << "\n";
    std::cout << "BKZ Benchmark completed successfully.\n\n";
    return 0;
}
