#include <catch2/catch_test_macros.hpp>
#include "dbdd/linalg/projection.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"

using namespace dbdd;

TEST_CASE("Vector orthogonal projection satisfies invariants", "[projection]") {
    SECTION("Exact rational projection of integer vector") {
        VectorZZ v({BigInt(1), BigInt(2), BigInt(-1), BigInt(3)});
        MatrixQQ P = orthogonal_projection_matrix(v);
        MatrixQQ P_par = parallel_projection_matrix(v);

        size_t m = v.size();
        MatrixQQ I = MatrixQQ::identity(m);

        // Invariant 1: P_par + P == I
        REQUIRE(P_par + P == I);

        // Invariant 2: P^2 == P (Idempotence)
        MatrixQQ P_sq = P * P;
        REQUIRE(P_sq == P);

        // Invariant 3: P^T == P (Symmetry)
        REQUIRE(P.transpose() == P);

        // Invariant 4: v * P == 0 (Annihilation in row-vector convention)
        VectorQQ v_qq = v.cast<BigRational>();
        VectorQQ vP = v_qq * P;
        for (size_t i = 0; i < m; ++i) {
            REQUIRE(vP[i] == BigRational(0));
        }

        // Test project_vector helper
        VectorQQ proj = project_vector(v_qq, P);
        REQUIRE(proj == vP);
    }

    SECTION("High-precision BigReal projection") {
        VectorRR v({BigReal(3.5), BigReal(-1.2), BigReal(2.8)});
        MatrixRR P = orthogonal_projection_matrix(v);
        size_t m = v.size();

        // Idempotence: P * P ≈ P
        MatrixRR P_sq = P * P;
        for (size_t r = 0; r < m; ++r) {
            for (size_t c = 0; c < m; ++c) {
                REQUIRE((P_sq(r, c) - P(r, c)).abs() < BigReal(1e-15));
            }
        }

        // Symmetry: P^T ≈ P
        MatrixRR Pt = P.transpose();
        for (size_t r = 0; r < m; ++r) {
            for (size_t c = 0; c < m; ++c) {
                REQUIRE((Pt(r, c) - P(r, c)).abs() < BigReal(1e-15));
            }
        }

        // Annihilation: v * P ≈ 0
        VectorRR vP = v * P;
        for (size_t i = 0; i < m; ++i) {
            REQUIRE(vP[i].abs() < BigReal(1e-15));
        }
    }

    SECTION("Zero vector throws DivisionByZeroException") {
        VectorQQ zero({BigRational(0), BigRational(0)});
        REQUIRE_THROWS_AS(orthogonal_projection_matrix(zero), DivisionByZeroException);
    }
}

TEST_CASE("Matrix subspace projection satisfies invariants", "[projection]") {
    SECTION("Exact rational subspace projection") {
        // Subspace of dimension 2 in R^4
        MatrixQQ V(2, 4, {
            BigRational(1), BigRational(0), BigRational(2), BigRational(-1),
            BigRational(0), BigRational(1), BigRational(-1), BigRational(1)
        });

        MatrixQQ P_perp = orthogonal_complement_projection_matrix(V);
        MatrixQQ P_par = parallel_projection_matrix(V);
        size_t m = V.cols();
        MatrixQQ I = MatrixQQ::identity(m);

        // Invariant 1: P_par + P_perp == I
        REQUIRE(P_par + P_perp == I);

        // Invariant 2: Idempotence: P^2 == P
        REQUIRE(P_perp * P_perp == P_perp);
        REQUIRE(P_par * P_par == P_par);

        // Invariant 3: Symmetry: P^T == P
        REQUIRE(P_perp.transpose() == P_perp);
        REQUIRE(P_par.transpose() == P_par);

        // Invariant 4: V * P_perp == 0 (rows of V are annihilated)
        MatrixQQ VP = V * P_perp;
        for (size_t r = 0; r < V.rows(); ++r) {
            for (size_t c = 0; c < m; ++c) {
                REQUIRE(VP(r, c) == BigRational(0));
            }
        }

        // Invariant 5: V * P_par == V (rows of V are preserved)
        REQUIRE(V * P_par == V);
    }
}
