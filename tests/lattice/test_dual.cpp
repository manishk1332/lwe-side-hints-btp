#include <catch2/catch_test_macros.hpp>
#include "dbdd/lattice/dual.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"

using namespace dbdd;

TEST_CASE("Dual lattice properties and involution", "[dual]") {
    SECTION("Full-rank basis duality") {
        MatrixZZ M(3, 3, {
            BigInt(2), BigInt(1), BigInt(0),
            BigInt(0), BigInt(3), BigInt(1),
            BigInt(1), BigInt(0), BigInt(4)
        });
        BasisZZ B(M);

        BasisQQ D = dual_basis(B);
        size_t d = B.dimension();

        // Invariant 1: B * D^T == I_d
        MatrixQQ B_qq = M.cast<BigRational>();
        MatrixQQ I = MatrixQQ::identity(d);
        REQUIRE(B_qq * D.matrix().transpose() == I);

        // Invariant 2: Involution: primal_from_dual(D) == B
        BasisQQ B_recovered = primal_from_dual(D);
        REQUIRE(B_recovered.matrix() == B_qq);

        // Invariant 3: Dual volume identity: Vol(Lambda*) == 1 / Vol(Lambda)
        BigReal vol_primal = compute_volume(B);
        BigReal vol_dual = compute_volume(D);
        BigReal product = vol_primal * vol_dual;
        REQUIRE((product - BigReal(1.0)).abs() < BigReal(1e-15));

        // Invariant 4: For any x in Lambda and y in Lambda*, <x, y> is an integer
        VectorZZ a({BigInt(3), BigInt(-2), BigInt(5)});
        VectorZZ c({BigInt(4), BigInt(1), BigInt(-3)});
        VectorQQ x = B.linear_combination(a).cast<BigRational>();
        VectorQQ y = D.linear_combination(c.cast<BigRational>());

        BigRational ip = x.dot(y);
        REQUIRE(ip.denominator() == BigInt(1));
        REQUIRE(ip.numerator() == a.dot(c));
    }

    SECTION("Subspace lattice duality (d < m)") {
        // 2 vectors in R^3
        MatrixZZ M(2, 3, {
            BigInt(1), BigInt(2), BigInt(0),
            BigInt(0), BigInt(1), BigInt(2)
        });
        BasisZZ B(M);

        BasisQQ D = dual_basis(B);
        size_t d = B.dimension();

        // Invariant 1: B * D^T == I_d
        MatrixQQ B_qq = M.cast<BigRational>();
        MatrixQQ I = MatrixQQ::identity(d);
        REQUIRE(B_qq * D.matrix().transpose() == I);

        // Invariant 2: Involution: primal_from_dual(D) == B
        BasisQQ B_recovered = primal_from_dual(D);
        REQUIRE(B_recovered.matrix() == B_qq);

        // Invariant 3: Dual volume: Vol(Lambda*) == 1 / Vol(Lambda)
        BigReal vol_primal = compute_volume(B);
        BigReal vol_dual = compute_volume(D);
        BigReal product = vol_primal * vol_dual;
        REQUIRE((product - BigReal(1.0)).abs() < BigReal(1e-15));
    }
}
