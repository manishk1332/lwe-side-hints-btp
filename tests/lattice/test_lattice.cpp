#include <catch2/catch_test_macros.hpp>
#include "dbdd/lattice/lattice.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"

using namespace dbdd;

TEST_CASE("Lattice class invariants and operations", "[lattice]") {
    MatrixZZ M(3, 3, {
        BigInt(2), BigInt(1), BigInt(0),
        BigInt(0), BigInt(3), BigInt(1),
        BigInt(1), BigInt(0), BigInt(4)
    });
    LatticeZZ L(M);

    SECTION("Dimensions and basic properties") {
        REQUIRE(L.dimension() == 3);
        REQUIRE(L.ambient_dimension() == 3);
        REQUIRE(L.is_full_rank());
        REQUIRE(L.rank() == 3);
        REQUIRE_FALSE(L.empty());
    }

    SECTION("Volume caching and preservation") {
        BigReal vol1 = L.volume();
        REQUIRE(L.exact_volume_int() == BigInt(25)); // det = 2*(12-0) - 1*(-1) = 24 + 1 = 25
        REQUIRE((vol1 - BigReal(25)).abs() < BigReal(1e-15));

        // Volume cache hit
        BigReal vol2 = L.volume();
        REQUIRE(vol1 == vol2);

        // Perform unimodular row operations: volume cache preserved!
        L.row_add_multiple(0, 1, BigInt(10));
        L.swap_rows(1, 2);
        REQUIRE(L.exact_volume_int() == BigInt(25));
        REQUIRE((L.volume() - BigReal(25)).abs() < BigReal(1e-15));
    }

    SECTION("Containment and coordinate solving") {
        VectorZZ in_lat = L.basis().linear_combination(VectorZZ({BigInt(2), BigInt(-3), BigInt(1)}));
        REQUIRE(L.contains(in_lat));

        auto coords = L.coordinates_of(in_lat);
        REQUIRE(coords[0] == BigRational(2));
        REQUIRE(coords[1] == BigRational(-3));
        REQUIRE(coords[2] == BigRational(1));

        // Vector with non-integer coordinates is not in lattice
        VectorZZ not_in_lat({BigInt(1), BigInt(1), BigInt(1)});
        auto coords_not = L.coordinates_of(not_in_lat);
        bool all_integers = true;
        for (size_t i = 0; i < coords_not.size(); ++i) {
            if (coords_not[i].denominator() != BigInt(1)) all_integers = false;
        }
        if (!all_integers) {
            REQUIRE_FALSE(L.contains(not_in_lat));
        }
    }

    SECTION("Dual lattice accessor") {
        LatticeQQ L_dual = L.dual();
        REQUIRE(L_dual.dimension() == 3);
        REQUIRE(L_dual.ambient_dimension() == 3);

        BigReal vol_primal = L.volume();
        BigReal vol_dual = L_dual.volume();
        REQUIRE(((vol_primal * vol_dual) - BigReal(1.0)).abs() < BigReal(1e-15));
    }

    SECTION("Orthogonal projection") {
        VectorZZ v({BigInt(1), BigInt(0), BigInt(0)});
        MatrixQQ P = orthogonal_projection_matrix(v);

        LatticeQQ L_qq = L.cast<BigRational>();
        BasisQQ B_proj = L_qq.project(P);
        REQUIRE(B_proj.dimension() == 3);
        REQUIRE(B_proj.ambient_dimension() == 3);
    }
}
