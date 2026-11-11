// benchmarks/bench_enumeration.cpp
//
// Paper Reference: Dachman-Soled et al. Section 2 / Schnorr-Euchner (1994) / main_plan.txt Section 58
// Mathematical Purpose: SVP enumeration search tree complexity benchmark across dimensions d in [10, 30].
// Evaluates:
//   - Search tree nodes visited
//   - Execution time in microseconds
//   - Node throughput (nodes/sec)
//   - Shortest non-zero vector length ||v_min||

#include "dbdd/lattice/basis.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include "dbdd/reduction/enumeration.hpp"

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
        B(i, i) = BigInt(static_cast<long long>(15 + (i % 7) * 4));
    }

    std::mt19937_64 rng(seed);
    for (size_t step = 0; step < 4 * d; ++step) {
        size_t r1 = rng() % d;
        size_t r2 = rng() % d;
        if (r1 != r2) {
            long long factor = static_cast<long long>((rng() % 3) + 1);
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
    std::cout << "          DBDD Framework: SVP Enumeration Search Complexity Benchmark                   \n";
    std::cout << "========================================================================================\n";

    std::cout << "\n"
              << std::string(82, '-') << "\n"
              << std::left << std::setw(6) << "Dim"
              << std::right << std::setw(14) << "Nodes Visited"
              << std::setw(14) << "Time (us)"
              << std::setw(18) << "Throughput(kNode/s)"
              << std::setw(16) << "||v_min||"
              << std::setw(14) << "Success"
              << "\n"
              << std::string(82, '-') << "\n";

    EnumerationSVP svp;
    std::vector<size_t> dims = {10, 12, 14, 16, 18, 20, 22, 24};

    for (size_t d : dims) {
        BasisZZ raw = make_test_basis(d, 100 + d);
        BasisZZ lll_basis = lll_exact(raw).basis;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto res = svp.find_shortest_vector(lll_basis);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        double knodes_per_sec = (us > 0.0) ? (res.nodes_visited / (us / 1000.0)) : 0.0;

        std::cout << std::left << std::setw(6) << d
                  << std::right << std::setw(14) << res.nodes_visited
                  << std::setw(14) << std::fixed << std::setprecision(1) << us
                  << std::setw(18) << std::fixed << std::setprecision(2) << knodes_per_sec
                  << std::setw(16) << std::fixed << std::setprecision(4) << res.length
                  << std::setw(14) << (res.found ? "YES" : "NO")
                  << "\n";
    }

    std::cout << std::string(82, '-') << "\n";
    std::cout << "SVP Enumeration Benchmark completed successfully.\n\n";
    return 0;
}
