#include <catch2/catch_test_macros.hpp>
#include "dbdd/reduction/lll_exact.hpp"
#include <random>

using namespace dbdd;

namespace {

BasisZZ make_scrambled_basis(size_t d, uint64_t seed = 42) {
    MatrixZZ B = MatrixZZ::identity(d);
    for (size_t i = 0; i < d; ++i) {
        B(i, i) = BigInt(static_cast<long long>(10 + (i % 7) * 3));
    }

    std::mt19937_64 rng(seed);
    for (size_t step = 0; step < 4 * d; ++step) {
        size_t r1 = rng() % d;
        size_t r2 = rng() % d;
        if (r1 != r2) {
            long long factor = static_cast<long long>((rng() % 4) + 1);
            if (rng() % 2 == 0) factor = -factor;
            B.row_add_multiple(r1, r2, BigInt(factor));
        }
        if (step % 3 == 0) {
            size_t s1 = rng() % d;
            size_t s2 = rng() % d;
            B.swap_rows(s1, s2);
        }
    }
    return BasisZZ(B);
}

} // namespace

TEST_CASE("LLLExact edge cases and parameter validation", "[lll_exact]") {
    SECTION("Empty basis (dimension 0)") {
        BasisZZ empty;
        auto res = lll_exact(empty);
        REQUIRE(res.basis.dimension() == 0);
        REQUIRE(res.converged);
        REQUIRE(res.swaps == 0);
    }

    SECTION("1D basis") {
        BasisZZ b1({VectorZZ({BigInt(42), BigInt(-17)})});
        auto res = lll_exact(b1);
        REQUIRE(res.basis.dimension() == 1);
        REQUIRE(res.basis[0] == b1[0]);
        REQUIRE(res.transformation.rows() == 1);
        REQUIRE(res.transformation(0, 0) == BigInt(1));
        REQUIRE(res.converged);
    }

    SECTION("Invalid delta parameter throws InvalidParameterException") {
        REQUIRE_THROWS_AS(LLLParams(0.25), InvalidParameterException);
        REQUIRE_THROWS_AS(LLLParams(0.1), InvalidParameterException);
        REQUIRE_THROWS_AS(LLLParams(1.05), InvalidParameterException);
    }

    SECTION("Invalid eta parameter throws InvalidParameterException") {
        REQUIRE_THROWS_AS(LLLParams(0.75, 0.49), InvalidParameterException);
        REQUIRE_THROWS_AS(LLLParams(0.75, 1.0), InvalidParameterException);
    }

    SECTION("Linearly dependent basis throws SingularMatrixException") {
        MatrixZZ singular(2, 2, {
            BigInt(1), BigInt(2),
            BigInt(2), BigInt(4)
        });
        REQUIRE_THROWS_AS(BasisZZ(singular), SingularMatrixException);
    }
}

TEST_CASE("LLLExact 2D Gauss-Lagrange reduction equivalence", "[lll_exact]") {
    SECTION("Heavily skewed 2D basis") {
        MatrixZZ M(2, 2, {
            BigInt(1), BigInt(100),
            BigInt(0), BigInt(1)
        });
        BasisZZ B(M);

        LLLParams params(0.99);
        auto res = lll_exact(B, params);

        REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
        REQUIRE(verify_volume_invariance(B, res.basis));
        REQUIRE(is_lll_reduced(res.basis, params.delta, params.eta));
        REQUIRE(verify_hermite_bound(res.basis, params.delta));

        // Shortest vector of lattice Z^2 has squared norm 1
        BigInt norm_sq = res.basis[0].squared_norm();
        REQUIRE(norm_sq == BigInt(1));
    }

    SECTION("Arbitrary 2D skewed integer basis") {
        MatrixZZ M(2, 2, {
            BigInt(201), BigInt(560),
            BigInt(561), BigInt(1560)
        });
        BasisZZ B(M);

        LLLParams params(0.75);
        auto res = lll_exact(B, params);

        REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
        REQUIRE(verify_volume_invariance(B, res.basis));
        REQUIRE(is_lll_reduced(res.basis, params.delta, params.eta));
        REQUIRE(verify_hermite_bound(res.basis, params.delta));

        // Reduced basis vectors must be significantly shorter than initial rows
        REQUIRE(res.basis[0].squared_norm() < B[0].squared_norm());
    }
}

TEST_CASE("LLLExact toy 3D and 4D lattices", "[lll_exact]") {
    SECTION("3D Upper Triangular Basis") {
        MatrixZZ M(3, 3, {
            BigInt(1), BigInt(2), BigInt(3),
            BigInt(0), BigInt(4), BigInt(5),
            BigInt(0), BigInt(0), BigInt(6)
        });
        BasisZZ B(M);

        auto res = lll_exact(B);

        REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
        REQUIRE(verify_volume_invariance(B, res.basis));
        REQUIRE(is_lll_reduced(res.basis));
        REQUIRE(verify_hermite_bound(res.basis));
    }

    SECTION("4D Arbitrary Basis") {
        MatrixZZ M(4, 4, {
            BigInt(10), BigInt(12), BigInt(15), BigInt(18),
            BigInt(0),  BigInt(8),  BigInt(14), BigInt(19),
            BigInt(0),  BigInt(0),  BigInt(7),  BigInt(13),
            BigInt(0),  BigInt(0),  BigInt(0),  BigInt(5)
        });
        BasisZZ B(M);

        LLLParams params(0.99);
        auto res = lll_exact(B, params);

        REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
        REQUIRE(verify_volume_invariance(B, res.basis));
        REQUIRE(is_lll_reduced(res.basis, params.delta, params.eta));
        REQUIRE(verify_hermite_bound(res.basis, params.delta));
    }
}

TEST_CASE("LLLExact scrambled bases across dimensions d = 5, 10, 20, 50", "[lll_exact]") {
    for (size_t d : {5UL, 10UL, 20UL, 50UL}) {
        DYNAMIC_SECTION("Dimension d = " << d) {
            BasisZZ B = make_scrambled_basis(d, 100 + d);
            BigInt initial_norm_sq = B[0].squared_norm();

            LLLParams params(0.75);
            auto res = lll_exact(B, params);

            REQUIRE(res.converged);
            REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
            REQUIRE(verify_volume_invariance(B, res.basis));
            REQUIRE(is_lll_reduced(res.basis, params.delta, params.eta));
            REQUIRE(verify_hermite_bound(res.basis, params.delta));

            BigInt reduced_norm_sq = res.basis[0].squared_norm();
            REQUIRE(reduced_norm_sq <= initial_norm_sq);
        }
    }
}

TEST_CASE("LLLExact over rational lattice BasisQQ", "[lll_exact]") {
    MatrixQQ M(3, 3, {
        BigRational(1, 2), BigRational(3, 4), BigRational(5, 6),
        BigRational(0),    BigRational(2, 3), BigRational(7, 5),
        BigRational(0),    BigRational(0),    BigRational(3, 2)
    });
    BasisQQ B(M);

    LLLParams params(0.75);
    auto res = lll_exact(B, params);

    REQUIRE(res.converged);
    REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
    REQUIRE(verify_volume_invariance(B, res.basis));
    REQUIRE(is_lll_reduced(res.basis, params.delta, params.eta));
}

TEST_CASE("LLLExact LatticeZZ volume preservation and cache integrity", "[lll_exact]") {
    BasisZZ B = make_scrambled_basis(6, 777);
    LatticeZZ lat(B);
    BigReal orig_vol = lat.volume();

    LatticeZZ reduced_lat = lll_exact(lat);

    REQUIRE(reduced_lat.volume() == orig_vol);
    REQUIRE(is_lll_reduced(reduced_lat.basis()));
}
