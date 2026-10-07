#include <catch2/catch_test_macros.hpp>
#include "dbdd/lattice/basis.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"

using namespace dbdd;

TEST_CASE("Basis representation and invariants", "[basis]") {
    SECTION("Construction and rank validation") {
        MatrixZZ M(3, 3, {
            BigInt(1), BigInt(0), BigInt(0),
            BigInt(2), BigInt(3), BigInt(0),
            BigInt(1), BigInt(1), BigInt(5)
        });

        BasisZZ B(M);
        REQUIRE(B.dimension() == 3);
        REQUIRE(B.ambient_dimension() == 3);
        REQUIRE(B.is_full_rank());
        REQUIRE(B.rank() == 3);

        // Dependent matrix throws SingularMatrixException
        MatrixZZ M_dep(2, 2, {
            BigInt(2), BigInt(4),
            BigInt(1), BigInt(2)
        });
        REQUIRE_THROWS_AS(BasisZZ(M_dep), SingularMatrixException);

        // Rows > cols throws DimensionMismatchException
        MatrixZZ M_over(3, 2, {
            BigInt(1), BigInt(0),
            BigInt(0), BigInt(1),
            BigInt(1), BigInt(1)
        });
        REQUIRE_THROWS_AS(BasisZZ(M_over), DimensionMismatchException);
    }

    SECTION("Linear combinations and coordinate solving") {
        MatrixZZ M(3, 3, {
            BigInt(2), BigInt(1), BigInt(0),
            BigInt(0), BigInt(3), BigInt(1),
            BigInt(1), BigInt(0), BigInt(4)
        });
        BasisZZ B(M);

        VectorZZ x({BigInt(2), BigInt(-1), BigInt(3)});
        VectorZZ v = B.linear_combination(x);

        // Verify v = 2*b0 - b1 + 3*b2
        VectorZZ expected({
            BigInt(2)*BigInt(2) - BigInt(0) + BigInt(3)*BigInt(1),
            BigInt(2)*BigInt(1) - BigInt(3) + BigInt(3)*BigInt(0),
            BigInt(2)*BigInt(0) - BigInt(1) + BigInt(3)*BigInt(4)
        });
        REQUIRE(v == expected);

        // Invert coordinates: coordinates_of(v) == x
        auto coords = B.coordinates_of(v);
        REQUIRE(coords == x.cast<BigRational>());
    }

    SECTION("Subspace lattice coordinates (d < m)") {
        // 2 vectors in R^3
        MatrixZZ M(2, 3, {
            BigInt(1), BigInt(0), BigInt(2),
            BigInt(0), BigInt(1), BigInt(-1)
        });
        BasisZZ B(M);
        REQUIRE(B.dimension() == 2);
        REQUIRE(B.ambient_dimension() == 3);
        REQUIRE_FALSE(B.is_full_rank());

        VectorZZ x({BigInt(4), BigInt(-3)});
        VectorZZ v = B.linear_combination(x);

        auto coords = B.coordinates_of(v);
        REQUIRE(coords[0] == BigRational(4));
        REQUIRE(coords[1] == BigRational(-3));
    }

    SECTION("Elementary unimodular operations") {
        BasisZZ B = BasisZZ::identity(3);
        B.row_add_multiple(0, 1, BigInt(5));
        REQUIRE(B[0] == VectorZZ({BigInt(1), BigInt(5), BigInt(0)}));

        B.swap_rows(0, 2);
        REQUIRE(B[0] == VectorZZ({BigInt(0), BigInt(0), BigInt(1)}));
        REQUIRE(B[2] == VectorZZ({BigInt(1), BigInt(5), BigInt(0)}));
    }
}
