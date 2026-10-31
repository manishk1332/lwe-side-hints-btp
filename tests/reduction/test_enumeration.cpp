#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dbdd/reduction/svp.hpp"
#include "dbdd/reduction/enumeration.hpp"
#include "dbdd/reduction/lll_float.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/core/exceptions.hpp"

#include <chrono>
#include <cmath>
#include <vector>

using namespace dbdd;
using Catch::Approx;

namespace {

/**
 * @brief Brute-force exhaustive coordinate search oracle for small dimensions (d <= 4).
 */
BigInt brute_force_shortest_norm(const BasisZZ& basis, int search_bound = 4) {
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();
    BigInt min_sq_norm = basis.row(0).squared_norm();

    std::vector<int> x(d, -search_bound);

    while (true) {
        bool all_zero = true;
        for (int val : x) {
            if (val != 0) {
                all_zero = false;
                break;
            }
        }

        if (!all_zero) {
            VectorZZ v(m, BigInt(0));
            for (size_t i = 0; i < d; ++i) {
                if (x[i] != 0) {
                    BigInt c(x[i]);
                    for (size_t j = 0; j < m; ++j) {
                        v[j] += c * basis.matrix()(i, j);
                    }
                }
            }
            BigInt sq_norm = v.squared_norm();
            if (sq_norm < min_sq_norm) {
                min_sq_norm = sq_norm;
            }
        }

        // Increment odometer
        size_t pos = 0;
        while (pos < d) {
            x[pos]++;
            if (x[pos] <= search_bound) {
                break;
            }
            x[pos] = -search_bound;
            pos++;
        }
        if (pos >= d) {
            break;
        }
    }

    return min_sq_norm;
}

} // anonymous namespace

TEST_CASE("EnumerationSVP - Base Cases and Error Handling", "[svp][enumeration]") {
    SECTION("Dimension 1 Base Case") {
        MatrixZZ B(1, 3);
        B(0, 0) = BigInt(3); B(0, 1) = BigInt(-4); B(0, 2) = BigInt(0);
        BasisZZ basis(B);

        auto res = EnumerationSVP::find_shortest_vector(basis);
        REQUIRE(res.found);
        REQUIRE(res.squared_norm == BigInt(25));
        REQUIRE(res.length == Approx(5.0));
        REQUIRE(res.coordinates.size() == 1);
        REQUIRE(res.coordinates[0] == BigInt(1));
        REQUIRE(res.vector[0] == BigInt(3));
        REQUIRE(res.vector[1] == BigInt(-4));
        REQUIRE(res.nodes_visited == 1);
        REQUIRE(res.complete_search);
    }

    SECTION("Invalid Dimensions and Parameters") {
        MatrixZZ B_empty(0, 0);
        BasisZZ empty_basis(B_empty);
        REQUIRE_THROWS_AS(EnumerationSVP::find_shortest_vector(empty_basis), DimensionMismatchException);

        SVPParameters invalid_params;
        invalid_params.radius_bound = -1.0;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        invalid_params.radius_bound = 0.0;
        invalid_params.timeout_seconds = -0.5;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);

        invalid_params.timeout_seconds = 0.0;
        invalid_params.radius_expansion_factor = 0.0;
        REQUIRE_THROWS_AS(invalid_params.validate(), InvalidParameterException);
    }
}

TEST_CASE("EnumerationSVP - Brute-Force Oracle Equivalence", "[svp][enumeration]") {
    SECTION("2D Lattice Equivalence") {
        MatrixZZ B(2, 2);
        B(0, 0) = BigInt(7); B(0, 1) = BigInt(11);
        B(1, 0) = BigInt(5); B(1, 1) = BigInt(8);
        BasisZZ basis(B);

        BigInt expected_min_sq = brute_force_shortest_norm(basis, 12);
        auto res = EnumerationSVP::find_shortest_vector(basis);

        REQUIRE(res.found);
        REQUIRE(res.squared_norm == expected_min_sq);
        REQUIRE(res.vector == res.coordinates * basis.matrix());
        REQUIRE(res.vector.squared_norm() == expected_min_sq);
    }

    SECTION("3D Lattice Equivalence") {
        MatrixZZ B(3, 3);
        B(0, 0) = BigInt(12); B(0, 1) = BigInt(2);  B(0, 2) = BigInt(5);
        B(1, 0) = BigInt(4);  B(1, 1) = BigInt(15); B(1, 2) = BigInt(1);
        B(2, 0) = BigInt(3);  B(2, 1) = BigInt(6);  B(2, 2) = BigInt(14);
        BasisZZ basis(B);

        BigInt expected_min_sq = brute_force_shortest_norm(basis, 5);
        auto res = EnumerationSVP::find_shortest_vector(basis);

        REQUIRE(res.found);
        REQUIRE(res.squared_norm == expected_min_sq);
        REQUIRE(res.vector == res.coordinates * basis.matrix());
        REQUIRE(res.vector.squared_norm() == expected_min_sq);
    }

    SECTION("4D Lattice Equivalence") {
        MatrixZZ B(4, 4);
        B(0, 0) = BigInt(10); B(0, 1) = BigInt(1);  B(0, 2) = BigInt(3);  B(0, 3) = BigInt(5);
        B(1, 0) = BigInt(2);  B(1, 1) = BigInt(11); B(1, 2) = BigInt(4);  B(1, 3) = BigInt(1);
        B(2, 0) = BigInt(1);  B(2, 1) = BigInt(3);  B(2, 2) = BigInt(12); B(2, 3) = BigInt(2);
        B(3, 0) = BigInt(4);  B(3, 1) = BigInt(2);  B(3, 2) = BigInt(1);  B(3, 3) = BigInt(13);
        BasisZZ basis(B);

        BigInt expected_min_sq = brute_force_shortest_norm(basis, 3);
        auto res = EnumerationSVP::find_shortest_vector(basis);

        REQUIRE(res.found);
        REQUIRE(res.squared_norm == expected_min_sq);
        REQUIRE(res.vector == res.coordinates * basis.matrix());
        REQUIRE(res.vector.squared_norm() == expected_min_sq);
    }
}

TEST_CASE("EnumerationSVP - Root Lattices with Known Shortest Norms", "[svp][enumeration]") {
    SECTION("A2 Hexagonal Root Lattice (lambda_1^2 = 2)") {
        // A2 root lattice basis: rows with squared norm 2 and inner product -1
        MatrixZZ B(2, 3);
        B(0, 0) = BigInt(1); B(0, 1) = BigInt(-1); B(0, 2) = BigInt(0);
        B(1, 0) = BigInt(0); B(1, 1) = BigInt(1);  B(1, 2) = BigInt(-1);
        BasisZZ basis(B);

        auto res = EnumerationSVP::find_shortest_vector(basis);
        REQUIRE(res.found);
        REQUIRE(res.squared_norm == BigInt(2));
        REQUIRE(res.length == Approx(std::sqrt(2.0)));
    }

    SECTION("A3 FCC Root Lattice (lambda_1^2 = 2)") {
        MatrixZZ B(3, 4);
        B(0, 0) = BigInt(1); B(0, 1) = BigInt(-1); B(0, 2) = BigInt(0);  B(0, 3) = BigInt(0);
        B(1, 0) = BigInt(0); B(1, 1) = BigInt(1);  B(1, 2) = BigInt(-1); B(1, 3) = BigInt(0);
        B(2, 0) = BigInt(0); B(2, 1) = BigInt(0);  B(2, 2) = BigInt(1);  B(2, 3) = BigInt(-1);
        BasisZZ basis(B);

        auto res = EnumerationSVP::find_shortest_vector(basis);
        REQUIRE(res.found);
        REQUIRE(res.squared_norm == BigInt(2));
        REQUIRE(res.length == Approx(std::sqrt(2.0)));
    }

    SECTION("D4 Checkerboard Root Lattice (lambda_1^2 = 2)") {
        MatrixZZ B(4, 4);
        B(0, 0) = BigInt(1); B(0, 1) = BigInt(1);  B(0, 2) = BigInt(0); B(0, 3) = BigInt(0);
        B(1, 0) = BigInt(1); B(1, 1) = BigInt(-1); B(1, 2) = BigInt(0); B(1, 3) = BigInt(0);
        B(2, 0) = BigInt(0); B(2, 1) = BigInt(1);  B(2, 2) = BigInt(1); B(2, 3) = BigInt(0);
        B(3, 0) = BigInt(0); B(3, 1) = BigInt(0);  B(3, 2) = BigInt(1); B(3, 3) = BigInt(1);
        BasisZZ basis(B);

        auto res = EnumerationSVP::find_shortest_vector(basis);
        REQUIRE(res.found);
        REQUIRE(res.squared_norm == BigInt(2));
        REQUIRE(res.length == Approx(std::sqrt(2.0)));
    }

    SECTION("D8 Root Lattice (lambda_1^2 = 2)") {
        MatrixZZ B(8, 8, BigInt(0));
        B(0, 0) = BigInt(1); B(0, 1) = BigInt(-1);
        B(1, 1) = BigInt(1); B(1, 2) = BigInt(-1);
        B(2, 2) = BigInt(1); B(2, 3) = BigInt(-1);
        B(3, 3) = BigInt(1); B(3, 4) = BigInt(-1);
        B(4, 4) = BigInt(1); B(4, 5) = BigInt(-1);
        B(5, 5) = BigInt(1); B(5, 6) = BigInt(-1);
        B(6, 6) = BigInt(1); B(6, 7) = BigInt(-1);
        B(7, 6) = BigInt(1); B(7, 7) = BigInt(1);

        BasisZZ basis(B);
        auto res = EnumerationSVP::find_shortest_vector(basis);
        REQUIRE(res.found);
        REQUIRE(res.squared_norm == BigInt(2));
        REQUIRE(res.length == Approx(std::sqrt(2.0)));
    }
}

TEST_CASE("EnumerationSVP - Invariants and Minkowski Bound", "[svp][enumeration]") {
    MatrixZZ B(5, 5);
    B(0, 0) = BigInt(15); B(0, 1) = BigInt(2);  B(0, 2) = BigInt(1);  B(0, 3) = BigInt(4);  B(0, 4) = BigInt(7);
    B(1, 0) = BigInt(3);  B(1, 1) = BigInt(19); B(1, 2) = BigInt(5);  B(1, 3) = BigInt(2);  B(1, 4) = BigInt(1);
    B(2, 0) = BigInt(2);  B(2, 1) = BigInt(1);  B(2, 2) = BigInt(17); B(2, 3) = BigInt(3);  B(2, 4) = BigInt(6);
    B(3, 0) = BigInt(6);  B(3, 1) = BigInt(4);  B(3, 2) = BigInt(2);  B(3, 3) = BigInt(22); B(3, 4) = BigInt(3);
    B(4, 0) = BigInt(1);  B(4, 1) = BigInt(5);  B(4, 2) = BigInt(3);  B(4, 3) = BigInt(1);  B(4, 4) = BigInt(18);
    BasisZZ basis(B);

    auto res = EnumerationSVP::find_shortest_vector(basis);

    REQUIRE(res.found);
    REQUIRE(res.vector == res.coordinates * basis.matrix());
    REQUIRE(res.squared_norm == res.vector.squared_norm());

    // Minkowski First Theorem: lambda_1 <= sqrt(d) * Vol^(1/d)
    BigReal vol(compute_volume_exact(basis));
    double minkowski_bound = std::sqrt(5.0) * std::pow(vol.to_double(), 1.0 / 5.0);
    REQUIRE(res.length <= minkowski_bound + 1e-9);

    // Coordinate non-triviality: x != 0
    bool non_zero_coords = false;
    for (size_t i = 0; i < res.coordinates.size(); ++i) {
        if (res.coordinates[i] != BigInt(0)) {
            non_zero_coords = true;
            break;
        }
    }
    REQUIRE(non_zero_coords);
}

TEST_CASE("EnumerationSVP - Projected Block Sublattice (solve_projected)", "[svp][enumeration]") {
    // 6-dimensional lattice
    MatrixZZ B(6, 6);
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            B(i, j) = BigInt(static_cast<long long>((i + 1) * (j + 2) + ((i == j) ? 20 : 0)));
        }
    }
    BasisZZ basis(B);

    // Compute Gram-Schmidt of LLL-reduced basis
    auto lll_res = lll_float(basis);
    MatrixDouble B_double(6, 6);
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            B_double(i, j) = lll_res.basis.matrix()(i, j).to_double();
        }
    }
    auto gs = compute_gram_schmidt_mgs<double>(B_double);

    SECTION("Projected Block [2, 5] (beta = 4)") {
        size_t start_idx = 2;
        size_t beta = 4;

        auto res = EnumerationSVP::find_shortest_vector_projected(gs, start_idx, beta);

        REQUIRE(res.found);
        REQUIRE(res.coordinates.size() == beta);
        REQUIRE(res.complete_search);

        // Projected squared norm must be <= ||b_{start_idx}*||^2
        double head_norm_sq = gs.bstar_norm_sq[start_idx];
        REQUIRE(res.length * res.length <= head_norm_sq + 1e-8);

        // Non-zero coordinates
        bool non_zero = false;
        for (size_t i = 0; i < beta; ++i) {
            if (res.coordinates[i] != BigInt(0)) non_zero = true;
        }
        REQUIRE(non_zero);
    }
}

TEST_CASE("EnumerationSVP - Strategy Comparison & Pruning", "[svp][enumeration]") {
    MatrixZZ B(5, 5);
    B(0, 0) = BigInt(18); B(0, 1) = BigInt(3);  B(0, 2) = BigInt(2);  B(0, 3) = BigInt(5);  B(0, 4) = BigInt(1);
    B(1, 0) = BigInt(2);  B(1, 1) = BigInt(15); B(1, 2) = BigInt(4);  B(1, 3) = BigInt(1);  B(1, 4) = BigInt(6);
    B(2, 0) = BigInt(4);  B(2, 1) = BigInt(1);  B(2, 2) = BigInt(21); B(2, 3) = BigInt(2);  B(2, 4) = BigInt(3);
    B(3, 0) = BigInt(1);  B(3, 1) = BigInt(5);  B(3, 2) = BigInt(3);  B(3, 3) = BigInt(17); B(3, 4) = BigInt(2);
    B(4, 0) = BigInt(3);  B(4, 1) = BigInt(2);  B(4, 2) = BigInt(1);  B(4, 3) = BigInt(4);  B(4, 4) = BigInt(19);
    BasisZZ basis(B);

    SVPParameters params_se;
    params_se.strategy = EnumerationStrategy::SchnorrEuchner;

    SVPParameters params_fp;
    params_fp.strategy = EnumerationStrategy::FinckePohst;

    auto res_se = EnumerationSVP::find_shortest_vector(basis, params_se);
    auto res_fp = EnumerationSVP::find_shortest_vector(basis, params_fp);

    REQUIRE(res_se.found);
    REQUIRE(res_fp.found);
    // Both strategies must find the exact same shortest vector length
    REQUIRE(res_se.squared_norm == res_fp.squared_norm);
    REQUIRE(res_se.length == Approx(res_fp.length).epsilon(1e-9));

    // Schnorr-Euchner dynamic shrinking should visit <= nodes compared to basic Fincke-Pohst
    REQUIRE(res_se.nodes_visited <= res_fp.nodes_visited);
}

TEST_CASE("EnumerationSVP - Node and Timeout Limits", "[svp][enumeration]") {
    // Generate a 15-dimensional lattice
    size_t d = 15;
    MatrixZZ B(d, d, BigInt(0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            B(i, j) = BigInt(static_cast<long long>((i + 2) * (j + 3) + ((i == j) ? 50 : 0)));
        }
    }
    BasisZZ basis(B);

    SECTION("Node Limit") {
        SVPParameters params;
        params.max_nodes = 50;
        params.preprocess_lll = false; // Force search on raw unreduced basis to generate large tree

        auto res = EnumerationSVP::find_shortest_vector(basis, params);
        REQUIRE_FALSE(res.complete_search);
        // Checks are periodic every 2048 nodes or at completion
        REQUIRE(res.nodes_visited >= 50);
    }

    SECTION("Timeout Limit") {
        SVPParameters params;
        params.timeout_seconds = 1e-6; // 1 microsecond
        params.preprocess_lll = false;

        auto res = EnumerationSVP::find_shortest_vector(basis, params);
        REQUIRE_FALSE(res.complete_search);
    }
}

TEST_CASE("EnumerationSVP - Performance Benchmark", "[svp][benchmark]") {
    // 20-dimensional lattice
    size_t d = 20;
    MatrixZZ B(d, d, BigInt(0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            B(i, j) = BigInt(static_cast<long long>(((i + 1) * 31 + (j + 1) * 17) % 50 + ((i == j) ? 60 : 0)));
        }
    }
    BasisZZ basis(B);

    auto start = std::chrono::high_resolution_clock::now();
    auto res = EnumerationSVP::find_shortest_vector(basis);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    REQUIRE(res.found);
    REQUIRE(res.complete_search);
    // Dimension 20 SVP with LLL preprocessing must complete in < 50 ms
    REQUIRE(elapsed_ms < 50);
}
