#include <catch2/catch_test_macros.hpp>
#include "dbdd/lattice/volume.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"

using namespace dbdd;

TEST_CASE("Lattice volume computation and invariance", "[volume]") {
    SECTION("Full-rank integer basis exact volume") {
        MatrixZZ M(3, 3, {
            BigInt(2), BigInt(1), BigInt(0),
            BigInt(1), BigInt(3), BigInt(1),
            BigInt(0), BigInt(1), BigInt(4)
        });
        BasisZZ B(M);

        BigInt exact_vol = compute_volume_exact(B);
        // det = 2*(12 - 1) - 1*(4 - 0) + 0 = 22 - 4 = 18
        REQUIRE(exact_vol == BigInt(18));

        BigReal vol = compute_volume(B);
        REQUIRE((vol - BigReal(18)).abs() < BigReal(1e-15));

        // Invariance under unimodular elementary row operations
        BasisZZ B_trans = B;
        B_trans.row_add_multiple(0, 1, BigInt(7));
        B_trans.swap_rows(1, 2);
        B_trans.row_add_multiple(2, 0, BigInt(-3));

        REQUIRE(compute_volume_exact(B_trans) == BigInt(18));
        REQUIRE((compute_volume(B_trans) - BigReal(18)).abs() < BigReal(1e-15));
    }

    SECTION("Subspace lattice intrinsic volume (d < m)") {
        // 2 vectors in R^3: b0 = (1, 2, 0), b1 = (0, 1, 2)
        // B * B^T = [[5, 2], [2, 5]]
        // det(B B^T) = 25 - 4 = 21
        // Vol = sqrt(21)
        MatrixZZ M(2, 3, {
            BigInt(1), BigInt(2), BigInt(0),
            BigInt(0), BigInt(1), BigInt(2)
        });
        BasisZZ B(M);

        BigReal vol = compute_volume(B);
        BigReal expected_vol = BigReal(21).sqrt();
        REQUIRE((vol - expected_vol).abs() < BigReal(1e-15));

        // Volume via Gram-Schmidt
        auto gs = compute_gram_schmidt_exact(M);
        BigReal gs_vol = compute_volume_from_gs(gs);
        REQUIRE((gs_vol - expected_vol).abs() < BigReal(1e-15));

        // Unimodular row operation on subspace basis
        BasisZZ B_mod = B;
        B_mod.row_add_multiple(0, 1, BigInt(4));
        BigReal mod_vol = compute_volume(B_mod);
        REQUIRE((mod_vol - expected_vol).abs() < BigReal(1e-15));
    }

    SECTION("Log-domain volume consistency") {
        MatrixZZ M(3, 3, {
            BigInt(5), BigInt(0), BigInt(0),
            BigInt(2), BigInt(3), BigInt(0),
            BigInt(1), BigInt(1), BigInt(2)
        });
        BasisZZ B(M);

        BigReal log_vol = compute_log_volume(B);
        BigReal vol = compute_volume(B); // vol = 30
        REQUIRE((vol - BigReal(30)).abs() < BigReal(1e-15));
        REQUIRE((log_vol.exp() - BigReal(30)).abs() < BigReal(1e-15));
    }
}
