#include <catch2/catch_test_macros.hpp>
#include "dbdd/lattice/primitive.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/dual.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"

using namespace dbdd;

TEST_CASE("Integer vector primitivity", "[primitive]") {
    SECTION("Primitive vs non-primitive integer vectors") {
        VectorZZ v_prim({BigInt(2), BigInt(3), BigInt(5)});
        REQUIRE(is_primitive_vector(v_prim));

        VectorZZ v_nonprim({BigInt(4), BigInt(6), BigInt(10)});
        REQUIRE_FALSE(is_primitive_vector(v_nonprim));
        REQUIRE(vector_gcd(v_nonprim) == BigInt(2));

        VectorZZ made_prim = make_primitive(v_nonprim);
        REQUIRE(made_prim == v_prim);
        REQUIRE(is_primitive_vector(made_prim));
    }

    SECTION("Zero vector is not primitive") {
        VectorZZ v_zero({BigInt(0), BigInt(0), BigInt(0)});
        REQUIRE_FALSE(is_primitive_vector(v_zero));
        REQUIRE_THROWS_AS(make_primitive(v_zero), DBDDException);
    }
}

TEST_CASE("Lattice vector primitivity", "[primitive]") {
    MatrixZZ M(3, 3, {
        BigInt(2), BigInt(1), BigInt(0),
        BigInt(0), BigInt(3), BigInt(1),
        BigInt(1), BigInt(0), BigInt(4)
    });
    BasisZZ B(M);

    SECTION("Primitive vector in lattice") {
        // v = 1*b0 + 2*b1 - 1*b2
        VectorZZ coords({BigInt(1), BigInt(2), BigInt(-1)});
        VectorZZ v = B.linear_combination(coords);

        REQUIRE(is_primitive_in_lattice(v, B));
    }

    SECTION("Non-primitive vector in lattice") {
        // v = 2*b0 + 4*b1 - 2*b2  (gcd = 2)
        VectorZZ coords({BigInt(2), BigInt(4), BigInt(-2)});
        VectorZZ v = B.linear_combination(coords);

        REQUIRE_FALSE(is_primitive_in_lattice(v, B));

        VectorZZ v_prim = make_primitive_in_lattice(v, B);
        REQUIRE(is_primitive_in_lattice(v_prim, B));
        REQUIRE(B.linear_combination(VectorZZ({BigInt(1), BigInt(2), BigInt(-1)})) == v_prim);
    }

    SECTION("Vector outside lattice is not primitive") {
        // Coordinates not integer
        VectorZZ v_outside({BigInt(1), BigInt(1), BigInt(1)});
        auto coords = B.coordinates_of<BigRational>(v_outside);
        bool all_int = true;
        for (size_t i = 0; i < coords.size(); ++i) {
            if (coords[i].denominator() != BigInt(1)) all_int = false;
        }
        if (!all_int) {
            REQUIRE_FALSE(is_primitive_in_lattice(v_outside, B));
        }
    }

    SECTION("Dual vector primitivity") {
        BasisQQ D = dual_basis(B);
        VectorZZ y_prim({BigInt(1), BigInt(0), BigInt(0)});
        VectorQQ w_prim = D.linear_combination(y_prim.cast<BigRational>());
        REQUIRE(is_primitive_dual_vector(w_prim, D));

        VectorZZ y_nonprim({BigInt(3), BigInt(6), BigInt(9)});
        VectorQQ w_nonprim = D.linear_combination(y_nonprim.cast<BigRational>());
        REQUIRE_FALSE(is_primitive_dual_vector(w_nonprim, D));
    }
}

TEST_CASE("Smith Normal Form and sublattice primitivity", "[primitive]") {
    SECTION("Smith Normal Form invariant factors") {
        MatrixZZ X(2, 3, {
            BigInt(2), BigInt(4), BigInt(4),
            BigInt(-6), BigInt(6), BigInt(12)
        });
        auto factors = compute_smith_normal_form_invariant_factors(X);
        REQUIRE(factors.size() == 2);
        REQUIRE(factors[0] == BigInt(2));
        REQUIRE(factors[1] == BigInt(6));
    }

    SECTION("Sublattice primitivity in Z^3") {
        BasisZZ B = BasisZZ::identity(3);

        // Sublattice spanned by (1, 2, 0) and (0, 1, 1) -> invariant factors [1, 1] -> primitive
        MatrixZZ V_prim(2, 3, {
            BigInt(1), BigInt(2), BigInt(0),
            BigInt(0), BigInt(1), BigInt(1)
        });
        REQUIRE(is_primitive_set_in_lattice(V_prim, B));

        // Sublattice spanned by (2, 4, 0) and (0, 2, 2) -> factors [2, 2] -> not primitive
        MatrixZZ V_nonprim(2, 3, {
            BigInt(2), BigInt(4), BigInt(0),
            BigInt(0), BigInt(2), BigInt(2)
        });
        REQUIRE_FALSE(is_primitive_set_in_lattice(V_nonprim, B));
    }
}
