#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include <random>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Unimodular Primitive Vector Completion", "[perfect_hint][unimodular]") {
    SECTION("2D Primitive vectors") {
        VectorZZ a({BigInt(3), BigInt(5)});
        MatrixZZ U = PerfectHint::complete_primitive_vector_unimodular(a);

        REQUIRE(U.rows() == 2);
        REQUIRE(U.cols() == 2);
        REQUIRE(U.row(1) == a);
        BigInt det = U.determinant();
        REQUIRE((det == BigInt(1) || det == BigInt(-1)));
    }

    SECTION("3D Primitive vectors with zeros and negatives") {
        VectorZZ a({BigInt(0), BigInt(-7), BigInt(11)});
        MatrixZZ U = PerfectHint::complete_primitive_vector_unimodular(a);

        REQUIRE(U.rows() == 3);
        REQUIRE(U.cols() == 3);
        REQUIRE(U.row(2) == a);
        BigInt det = U.determinant();
        REQUIRE((det == BigInt(1) || det == BigInt(-1)));
    }

    SECTION("5D and 8D Primitive vectors") {
        std::vector<VectorZZ> test_vectors = {
            VectorZZ({BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)}),
            VectorZZ({BigInt(2), BigInt(0), BigInt(0), BigInt(0), BigInt(3)}),
            VectorZZ({BigInt(1), BigInt(-1), BigInt(1), BigInt(-1), BigInt(1), BigInt(-1), BigInt(1), BigInt(2)})
        };

        for (const auto& a : test_vectors) {
            size_t d = a.size();
            MatrixZZ U = PerfectHint::complete_primitive_vector_unimodular(a);
            REQUIRE(U.rows() == d);
            REQUIRE(U.cols() == d);
            REQUIRE(U.row(d - 1) == a);
            BigInt det = U.determinant();
            REQUIRE((det == BigInt(1) || det == BigInt(-1)));
        }
    }

    SECTION("Non-primitive vector throws InvalidParameterException") {
        VectorZZ non_prim({BigInt(2), BigInt(4), BigInt(6)});
        REQUIRE_THROWS_AS(PerfectHint::complete_primitive_vector_unimodular(non_prim), InvalidParameterException);
    }
}

TEST_CASE("Conditional Gaussian Distribution Update Invariants", "[perfect_hint][distribution]") {
    size_t d = 5;
    VectorDouble mean({0.5, -1.0, 2.0, 0.0, 1.0});

    // Create a symmetric positive definite covariance matrix
    MatrixDouble A(d, d);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            A(i, j) = dist(rng);
        }
    }
    MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d) * 0.5;

    VectorDouble v_bar({1.0, 2.0, -1.0, 0.5, -2.5});

    auto [mu_prime, cov_prime] = PerfectHint::update_distribution(mean, cov, v_bar);

    SECTION("Exact Mean Orthogonality: <mu', v_bar> == 0") {
        double dot_prod = 0.0;
        for (size_t i = 0; i < d; ++i) {
            dot_prod += mu_prime[i] * v_bar[i];
        }
        REQUIRE_THAT(dot_prod, WithinAbs(0.0, 1e-12));
    }

    SECTION("Covariance Null-Space: v_bar * Sigma' == 0") {
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < d; ++i) {
                sum += v_bar[i] * cov_prime(i, j);
            }
            REQUIRE_THAT(sum, WithinAbs(0.0, 1e-12));
        }
    }

    SECTION("Covariance Symmetry & PSD Preservation") {
        // Symmetry
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                REQUIRE_THAT(cov_prime(i, j), WithinAbs(cov_prime(j, i), 1e-14));
            }
        }

        // Positive semidefiniteness via Ellipsoid abstraction
        Ellipsoid ell(mu_prime, cov_prime);
        for (size_t i = 0; i < ell.eigenvalues().size(); ++i) {
            REQUIRE(ell.eigenvalues()[i] >= -1e-10);
        }
        // Rank should drop by exactly 1
        REQUIRE(ell.rank() == d - 1);
    }

    SECTION("Deterministic Prior Edge Case") {
        // v_bar is in null-space of cov (variance = 0)
        MatrixDouble cov_deg = cov_prime; // already has v_bar in null space
        // Consistent hint (<mu', v_bar> == 0)
        auto [mu_same, cov_same] = PerfectHint::update_distribution(mu_prime, cov_deg, v_bar);
        for (size_t i = 0; i < d; ++i) {
            REQUIRE_THAT(mu_same[i], WithinAbs(mu_prime[i], 1e-12));
        }

        // Contradictory hint (<mu_bad, v_bar> != 0 with variance 0)
        VectorDouble mu_bad = mu_prime;
        mu_bad[0] += 10.0;
        REQUIRE_THROWS_AS(PerfectHint::update_distribution(mu_bad, cov_deg, v_bar), InvalidParameterException);
    }
}

TEST_CASE("Synthetic Lattice Hyperplane Intersection (d = 3, 4, 5)", "[perfect_hint][lattice]") {
    SECTION("3D Integer Lattice: Z^3 with target (1, 2, 3)") {
        size_t d = 3;
        BasisZZ basis = BasisZZ::identity(d);
        LatticeZZ lattice(basis);

        VectorZZ target({BigInt(1), BigInt(2), BigInt(3)});
        VectorZZ v_bar({BigInt(1), BigInt(1), BigInt(-1)}); // 1*1 + 2*1 + 3*(-1) = 0
        REQUIRE(target.dot(v_bar) == BigInt(0));

        LatticeZZ intersected = PerfectHint::update_lattice(lattice, v_bar);

        REQUIRE(intersected.dimension() == 2);
        REQUIRE(intersected.ambient_dimension() == 3);

        // Every basis vector of intersected lattice must be orthogonal to v_bar
        const auto& B_prime = intersected.basis();
        for (size_t i = 0; i < B_prime.dimension(); ++i) {
            REQUIRE(B_prime[i].dot(v_bar) == BigInt(0));
            // Basis vectors must be in original lattice
            REQUIRE(lattice.contains(B_prime[i]));
        }

        // Target must be in intersected lattice
        REQUIRE(intersected.contains(target));

        // Volume formula: Vol(Lambda') = Vol(Lambda) * ||v_bar*_{Lambda^*}|| = 1 * sqrt(3)
        double expected_vol = std::sqrt(3.0);
        REQUIRE_THAT(intersected.volume().to_double(), WithinAbs(expected_vol, 1e-10));
    }

    SECTION("4D Scrambled Integer Lattice with known target") {
        size_t d = 4;
        MatrixZZ B(d, d, {
            BigInt(2), BigInt(1), BigInt(0), BigInt(0),
            BigInt(1), BigInt(3), BigInt(1), BigInt(0),
            BigInt(0), BigInt(1), BigInt(2), BigInt(1),
            BigInt(0), BigInt(0), BigInt(1), BigInt(2)
        });
        BasisZZ basis(B);
        LatticeZZ lattice(basis);
        BigReal initial_vol = lattice.volume();

        // Target x * B with x = (1, -1, 2, 1)
        VectorZZ x({BigInt(1), BigInt(-1), BigInt(2), BigInt(1)});
        VectorZZ target = x * B;

        // Choose v_bar orthogonal to target: <target, v_bar> = 0
        // target = 1*(2,1,0,0) - 1*(1,3,1,0) + 2*(0,1,2,1) + 1*(0,0,1,2)
        //        = (2-1+0+0, 1-3+2+0, 0-1+4+1, 0-0+2+2) = (1, 0, 4, 4)
        // v_bar = (0, 1, 1, -1) => 1*0 + 0*1 + 4*1 + 4*(-1) = 0
        VectorZZ v_bar({BigInt(0), BigInt(1), BigInt(1), BigInt(-1)});
        REQUIRE(target.dot(v_bar) == BigInt(0));

        LatticeZZ intersected = PerfectHint::update_lattice(lattice, v_bar);

        REQUIRE(intersected.dimension() == 3);
        REQUIRE(intersected.ambient_dimension() == 4);
        REQUIRE(intersected.contains(target));

        for (size_t i = 0; i < intersected.basis().dimension(); ++i) {
            REQUIRE(intersected.basis()[i].dot(v_bar) == BigInt(0));
            REQUIRE(lattice.contains(intersected.basis()[i]));
        }

        BigReal w_norm = PerfectHint::primitive_dual_norm(basis, v_bar);
        BigReal expected_vol = initial_vol * w_norm;
        REQUIRE_THAT(intersected.volume().to_double(), WithinAbs(expected_vol.to_double(), 1e-8));
    }
}

TEST_CASE("Brute-Force Oracle for Tiny Lattices (d <= 4)", "[perfect_hint][brute_force]") {
    size_t d = 3;
    BasisZZ basis = BasisZZ::identity(d);
    VectorZZ v_bar({BigInt(1), BigInt(2), BigInt(-1)}); // x + 2y - z = 0

    BasisZZ intersected_basis = PerfectHint::intersect_hyperplane(basis, v_bar);
    LatticeZZ intersected_lattice(intersected_basis);

    REQUIRE(intersected_basis.dimension() == 2);

    // Enumerate all lattice vectors in [-3, 3]^3
    for (long long c0 = -3; c0 <= 3; ++c0) {
        for (long long c1 = -3; c1 <= 3; ++c1) {
            for (long long c2 = -3; c2 <= 3; ++c2) {
                VectorZZ v({BigInt(c0), BigInt(c1), BigInt(c2)});
                if (v.dot(v_bar) == BigInt(0)) {
                    // Every vector satisfying the hyperplane constraint MUST be in the intersected lattice
                    REQUIRE(intersected_lattice.contains(v));
                }
            }
        }
    }
}

TEST_CASE("DBDDInstance Integration on Toy LWE (n=3, m=3, q=11, d=7)", "[perfect_hint][toy_lwe]") {
    size_t n = 3;
    size_t m = 3;
    BigInt q(11);
    size_t d = m + n + 1; // 7

    double sigma_e = 1.5;
    double sigma_s = 2.0;

    DiscreteGaussian noise_dist(sigma_e, 301);
    DiscreteGaussian secret_dist(sigma_s, 302);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 4004);
    auto kannan = KannanEmbedding::from_lwe(instance, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    const auto& s = instance.test_only_secret();

    SECTION("Single Coordinate Secret Hint: s_0 = l_0") {
        VectorZZ v({BigInt(1), BigInt(0), BigInt(0)});
        BigInt l = s[0];

        dbdd.integrate_perfect_hint(v, l);

        REQUIRE(dbdd.dimension() == d - 1); // 6
        REQUIRE(dbdd.ambient_dimension() == d); // 7
        REQUIRE(dbdd.ellipsoid().rank() == d - 2); // 5

        REQUIRE(dbdd.has_target());
        REQUIRE(dbdd.has_target_coordinates());
        REQUIRE(dbdd.target_in_lattice());
        REQUIRE(dbdd.verify_target());

        // Target coordinates in new basis
        VectorZZ reconstructed = dbdd.target_coordinates() * dbdd.basis().matrix();
        REQUIRE(reconstructed == dbdd.target());
    }

    SECTION("Two Sequential Perfect Hints: s_0 and s_1") {
        VectorZZ v0({BigInt(1), BigInt(0), BigInt(0)});
        dbdd.integrate_perfect_hint(v0, s[0]);

        REQUIRE(dbdd.dimension() == 6);
        REQUIRE(dbdd.ellipsoid().rank() == 5);
        REQUIRE(dbdd.verify_target());

        VectorZZ v1({BigInt(0), BigInt(1), BigInt(0)});
        dbdd.integrate_perfect_hint(v1, s[1]);

        REQUIRE(dbdd.dimension() == 5);
        REQUIRE(dbdd.ellipsoid().rank() == 4);
        REQUIRE(dbdd.target_in_lattice());
        REQUIRE(dbdd.verify_target());

        VectorZZ reconstructed = dbdd.target_coordinates() * dbdd.basis().matrix();
        REQUIRE(reconstructed == dbdd.target());
    }

    SECTION("Floating-point hint overload") {
        VectorDouble v_d({0.0, 0.0, 1.0});
        double l_d = s[2].to_double();

        dbdd.integrate_perfect_hint(v_d, l_d);

        REQUIRE(dbdd.dimension() == 6);
        REQUIRE(dbdd.ellipsoid().rank() == 5);
        REQUIRE(dbdd.verify_target());
    }
}

TEST_CASE("DBDDInstance Integration on Medium LWE Instance (n=8, m=8, q=97, sigma=2.0, d=17)", "[perfect_hint][medium_lwe]") {
    size_t n = 8;
    size_t m = 8;
    BigInt q(97);
    double sigma = 2.0;
    size_t d = m + n + 1; // 17

    DiscreteGaussian noise_dist(sigma, 601);
    DiscreteGaussian secret_dist(sigma, 602);

    auto instance = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 8888);
    auto kannan = KannanEmbedding::from_lwe(instance, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    const auto& s = instance.test_only_secret();

    // Create a linear hint on secret: <s, v> = l
    VectorZZ v(n);
    v[0] = BigInt(1);
    v[1] = BigInt(-1);
    v[2] = BigInt(2);
    BigInt l = s[0] - s[1] + BigInt(2) * s[2];

    dbdd.integrate_perfect_hint(v, l);

    SECTION("Dimensions and Rank Drops") {
        REQUIRE(dbdd.dimension() == d - 1); // 16
        REQUIRE(dbdd.ambient_dimension() == d); // 17
        REQUIRE(dbdd.ellipsoid().rank() == d - 2); // 15
    }

    SECTION("Target Verification and Ground Truth Preservation") {
        REQUIRE(dbdd.has_target());
        REQUIRE(dbdd.has_target_coordinates());
        REQUIRE(dbdd.target_in_lattice());
        REQUIRE(dbdd.verify_target(0.999));

        VectorZZ reconstructed = dbdd.target_coordinates() * dbdd.basis().matrix();
        REQUIRE(reconstructed == dbdd.target());
    }
}
