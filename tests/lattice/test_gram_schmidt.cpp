#include <catch2/catch_test_macros.hpp>
#include "dbdd/linalg/gram_schmidt.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"

using namespace dbdd;

TEST_CASE("Gram-Schmidt orthogonalization satisfies mathematical invariants", "[gram_schmidt]") {
    SECTION("Exact rational Gram-Schmidt on 4x4 matrix") {
        MatrixQQ B(4, 4, {
            BigRational(1), BigRational(2), BigRational(3), BigRational(4),
            BigRational(0), BigRational(1), BigRational(-1), BigRational(2),
            BigRational(2), BigRational(0), BigRational(1), BigRational(1),
            BigRational(-1), BigRational(1), BigRational(0), BigRational(3)
        });

        auto state = compute_gram_schmidt_exact(B);
        size_t d = state.dimension();

        // Invariant 1: Orthogonality: <b_i*, b_j*> == 0 for all i != j
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < i; ++j) {
                BigRational dot_prod = state.bstar[i].dot(state.bstar[j]);
                REQUIRE(dot_prod == BigRational(0));
            }
        }

        // Invariant 2: Triangular reconstruction: b_i == b_i* + sum_{j < i} mu_{i,j} b_j*
        for (size_t i = 0; i < d; ++i) {
            VectorQQ reconstructed = state.bstar[i];
            for (size_t j = 0; j < i; ++j) {
                reconstructed += state.mu(i, j) * state.bstar[j];
            }
            REQUIRE(reconstructed == B.row(i));
        }

        // Invariant 3: Squared norm consistency: ||b_i*||^2 == B_i
        for (size_t i = 0; i < d; ++i) {
            REQUIRE(state.bstar[i].squared_norm() == state.bstar_norm_sq[i]);
        }

        // Invariant 4: MGS vs CGS match exactly on rational matrices
        auto state_cgs = compute_gram_schmidt_cgs(B);
        for (size_t i = 0; i < d; ++i) {
            REQUIRE(state.bstar[i] == state_cgs.bstar[i]);
            REQUIRE(state.bstar_norm_sq[i] == state_cgs.bstar_norm_sq[i]);
            for (size_t j = 0; j < d; ++j) {
                REQUIRE(state.mu(i, j) == state_cgs.mu(i, j));
            }
        }
    }

    SECTION("Floating-point Gram-Schmidt with BigReal") {
        MatrixRR B(3, 3, {
            BigReal(1.0), BigReal(0.5), BigReal(0.2),
            BigReal(0.3), BigReal(2.0), BigReal(0.1),
            BigReal(0.4), BigReal(0.6), BigReal(3.0)
        });

        auto state = compute_gram_schmidt_mgs(B);
        size_t d = state.dimension();

        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < i; ++j) {
                BigReal dot_prod = state.bstar[i].dot(state.bstar[j]);
                REQUIRE(dot_prod.abs() < BigReal(1e-15));
            }
        }
    }

    SECTION("Linear dependence detection throws SingularMatrixException") {
        // Row 2 is 2 * Row 0
        MatrixQQ singular(2, 2, {
            BigRational(1), BigRational(2),
            BigRational(2), BigRational(4)
        });
        REQUIRE_THROWS_AS(compute_gram_schmidt_exact(singular), SingularMatrixException);
    }
}

TEST_CASE("Gram-Schmidt local updates match full recomputation", "[gram_schmidt]") {
    MatrixQQ B(4, 4, {
        BigRational(2), BigRational(1), BigRational(0), BigRational(1),
        BigRational(1), BigRational(3), BigRational(1), BigRational(0),
        BigRational(0), BigRational(1), BigRational(4), BigRational(2),
        BigRational(1), BigRational(0), BigRational(2), BigRational(5)
    });

    SECTION("Local swap update matches recomputation") {
        for (size_t k = 0; k < 3; ++k) {
            auto state = compute_gram_schmidt_exact(B);
            update_gram_schmidt_swap(state, k);

            // Recompute from swapped basis
            MatrixQQ B_swapped = B;
            B_swapped.swap_rows(k, k + 1);
            auto state_recomputed = compute_gram_schmidt_exact(B_swapped);

            REQUIRE(state.basis == state_recomputed.basis);
            for (size_t i = 0; i < 4; ++i) {
                REQUIRE(state.bstar[i] == state_recomputed.bstar[i]);
                REQUIRE(state.bstar_norm_sq[i] == state_recomputed.bstar_norm_sq[i]);
                for (size_t j = 0; j < 4; ++j) {
                    REQUIRE(state.mu(i, j) == state_recomputed.mu(i, j));
                }
            }
        }
    }

    SECTION("Row addition update matches recomputation") {
        auto state = compute_gram_schmidt_exact(B);
        BigRational factor(3, 2);
        update_gram_schmidt_row_add(state, 2, 0, factor);

        auto state_recomputed = compute_gram_schmidt_exact(state.basis);

        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(state.bstar[i] == state_recomputed.bstar[i]);
            REQUIRE(state.bstar_norm_sq[i] == state_recomputed.bstar_norm_sq[i]);
            for (size_t j = 0; j < 4; ++j) {
                REQUIRE(state.mu(i, j) == state_recomputed.mu(i, j));
            }
        }
    }
}
