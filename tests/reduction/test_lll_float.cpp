#include <catch2/catch_test_macros.hpp>
#include "dbdd/reduction/lll_float.hpp"
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

TEST_CASE("LLLFloat multi-precision MPFR validation", "[lll_float]") {
    BasisZZ B = make_scrambled_basis(8, 2024);

    for (prec_t prec : {128U, 256U, 512U}) {
        DYNAMIC_SECTION("MPFR Precision = " << prec << " bits") {
            LLLFloat lll_engine;
            auto res = lll_engine.reduce_mpfr(B, prec);

            REQUIRE(res.converged);
            REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
            REQUIRE(verify_volume_invariance(B, res.basis));
            REQUIRE(is_lll_reduced(res.basis));
            REQUIRE(verify_hermite_bound(res.basis));
        }
    }
}

TEST_CASE("LLLFloat double precision reduction", "[lll_float]") {
    for (size_t d : {2UL, 5UL, 10UL, 20UL}) {
        DYNAMIC_SECTION("Double precision on dimension d = " << d) {
            BasisZZ B = make_scrambled_basis(d, 500 + d);
            LLLFloat lll_engine;
            auto res = lll_engine.reduce_double(B);

            REQUIRE(res.converged);
            REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
            REQUIRE(verify_volume_invariance(B, res.basis));
            REQUIRE(is_lll_reduced(res.basis));
            REQUIRE(verify_hermite_bound(res.basis));
        }
    }
}

TEST_CASE("LLLFloat cross-validation against LLLExact", "[lll_float]") {
    for (size_t d : {5UL, 10UL, 15UL}) {
        DYNAMIC_SECTION("Dimension d = " << d) {
            BasisZZ B = make_scrambled_basis(d, 999 + d);

            auto res_exact = lll_exact(B);
            auto res_float = lll_float(B);

            // Both reductions must preserve the exact same lattice volume
            REQUIRE(compute_volume_exact(res_exact.basis) == compute_volume_exact(res_float.basis));

            // Both transformations must be unimodular
            REQUIRE(verify_unimodular_transformation(B, res_exact.basis, res_exact.transformation));
            REQUIRE(verify_unimodular_transformation(B, res_float.basis, res_float.transformation));

            // Both must be strictly LLL-reduced
            REQUIRE(is_lll_reduced(res_exact.basis));
            REQUIRE(is_lll_reduced(res_float.basis));

            // Both must satisfy the Hermite bound
            REQUIRE(verify_hermite_bound(res_exact.basis));
            REQUIRE(verify_hermite_bound(res_float.basis));
        }
    }
}

TEST_CASE("LLLFloat scaling on dimension d = 50", "[lll_float]") {
    BasisZZ B = make_scrambled_basis(50, 4242);
    BigInt initial_norm_sq = B[0].squared_norm();

    LLLFloat lll_engine;
    auto res = lll_engine.reduce(B);

    REQUIRE(res.converged);
    REQUIRE(verify_unimodular_transformation(B, res.basis, res.transformation));
    REQUIRE(verify_volume_invariance(B, res.basis));
    REQUIRE(is_lll_reduced(res.basis));
    REQUIRE(verify_hermite_bound(res.basis));

    BigInt reduced_norm_sq = res.basis[0].squared_norm();
    REQUIRE(reduced_norm_sq < initial_norm_sq);
}
