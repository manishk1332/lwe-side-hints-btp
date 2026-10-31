// benchmarks/bench_lll.cpp
//
// Paper Reference: Albrecht et al. Section 2 / main_plan.txt Section 58
// Mathematical Purpose: Rigorous performance and quality benchmark comparing
// exact rational arithmetic LLL (LLLExact) with floating-point MPFR LLL (LLLFloat)
// across dimensions d in [10, 50]. Measures:
//   - Execution runtime (ms)
//   - First basis vector Euclidean norm ||b_1||
//   - Hermite factor ||b_1|| / det(Lambda)^(1/d)
//   - Basis orthogonality defect prod_i ||b_i*|| / det(Lambda)

#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include "dbdd/reduction/lll_float.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using namespace dbdd;

static BasisZZ make_scrambled_basis(size_t d, uint64_t seed = 42) {
    MatrixZZ B = MatrixZZ::identity(d);
    for (size_t i = 0; i < d; ++i) {
        B(i, i) = BigInt(static_cast<long long>(20 + (i % 11) * 5));
    }

    std::mt19937_64 rng(seed);
    for (size_t step = 0; step < 5 * d; ++step) {
        size_t r1 = rng() % d;
        size_t r2 = rng() % d;
        if (r1 != r2) {
            long long factor = static_cast<long long>((rng() % 5) + 1);
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

static double compute_first_vector_norm(const BasisZZ& basis) {
    if (basis.dimension() == 0) return 0.0;
    return std::sqrt(basis.row(0).dot(basis.row(0)).to_double());
}

static double compute_orthogonality_defect(const BasisZZ& basis) {
    size_t d = basis.dimension();
    if (d <= 1) return 1.0;

    auto gs = compute_gram_schmidt_cgs(basis.matrix().cast<double>());
    double log_prod_star = 0.0;
    for (size_t i = 0; i < d; ++i) {
        log_prod_star += 0.5 * std::log(gs.bstar_norm_sq[i]);
    }

    BigReal vol = compute_volume(basis);
    double log_vol = std::log(vol.to_double());

    return std::exp(log_prod_star - log_vol);
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "          DBDD Framework: LLL Benchmark (LLLExact vs LLLFloat)                         \n";
    std::cout << "========================================================================================\n";

    std::cout << "\n"
              << std::string(88, '-') << "\n"
              << std::left << std::setw(6) << "Dim"
              << std::right << std::setw(14) << "Exact Time(ms)"
              << std::setw(14) << "Float Time(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Exact ||b1||"
              << std::setw(14) << "Float ||b1||"
              << std::setw(14) << "Orth Defect"
              << "\n"
              << std::string(88, '-') << "\n";

    std::vector<size_t> dims = {10, 15, 20, 25, 30, 40};

    for (size_t d : dims) {
        BasisZZ orig = make_scrambled_basis(d, 42 + d);

        // 1. Exact LLL
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res_exact = lll_exact(orig);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_exact = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. Float LLL
        auto t2 = std::chrono::high_resolution_clock::now();
        auto res_float = lll_float(orig);
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_float = std::chrono::duration<double, std::milli>(t3 - t2).count();

        double speedup = (ms_float > 0.0) ? (ms_exact / ms_float) : 1.0;
        double norm_exact = compute_first_vector_norm(res_exact.basis);
        double norm_float = compute_first_vector_norm(res_float.basis);
        double orth_defect = compute_orthogonality_defect(res_exact.basis);

        std::cout << std::left << std::setw(6) << d
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2) << ms_exact
                  << std::setw(14) << std::fixed << std::setprecision(2) << ms_float
                  << std::setw(12) << std::fixed << std::setprecision(1) << (std::to_string(speedup).substr(0, 4) + "x")
                  << std::setw(14) << std::fixed << std::setprecision(2) << norm_exact
                  << std::setw(14) << std::fixed << std::setprecision(2) << norm_float
                  << std::setw(14) << std::fixed << std::setprecision(3) << orth_defect
                  << "\n";
    }

    std::cout << std::string(88, '-') << "\n";
    std::cout << "LLL Benchmark completed successfully.\n\n";
    return 0;
}
