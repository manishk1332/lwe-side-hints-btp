#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include "dbdd/lattice/dual.hpp"
#include <random>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Modular Hint Construction and Homogenization", "[modular_hint][homogenize]") {
    SECTION("Constructor parameter validation") {
        VectorZZ v({BigInt(1), BigInt(2)});
        REQUIRE_THROWS_AS(ModularHint(v, BigInt(1), BigInt(1)), InvalidParameterException); // k < 2
        REQUIRE_THROWS_AS(ModularHint(v, BigInt(1), BigInt(-3)), InvalidParameterException); // k < 0
        REQUIRE_THROWS_AS(ModularHint(VectorZZ{}, BigInt(1), BigInt(3)), InvalidParameterException); // empty v

        VectorDouble vd({1.0, 2.0});
        REQUIRE_THROWS_AS(ModularHint(vd, 1.0, 1.5), InvalidParameterException); // k < 2
        REQUIRE_THROWS_AS(ModularHint(VectorDouble{}, 1.0, 3.0), InvalidParameterException); // empty
    }

    SECTION("Homogenization from secret vector") {
        size_t n = 3;
        size_t m = 4;
        size_t ambient_dim = m + n + 1; // 8

        VectorZZ v({BigInt(2), BigInt(-1), BigInt(5)});
        BigInt l(3);
        BigInt k(7);

        ModularHint hint(v, l, k);
        REQUIRE(hint.type() == HintType::Modular);
        REQUIRE(!hint.is_already_homogenized());
        REQUIRE(hint.modulus_int() == BigInt(7));

        VectorZZ v_bar = hint.homogenize_int(ambient_dim);
        REQUIRE(v_bar.size() == ambient_dim);

        // Layout: (0_m, v, -l)
        for (size_t i = 0; i < m; ++i) {
            REQUIRE(v_bar[i] == BigInt(0));
        }
        for (size_t j = 0; j < n; ++j) {
            REQUIRE(v_bar[m + j] == v[j]);
        }
        REQUIRE(v_bar[ambient_dim - 1] == -l);

        // Target congruence check: s_bar = (e, s, 1)
        VectorZZ e({BigInt(1), BigInt(0), BigInt(-1), BigInt(2)});
        VectorZZ s({BigInt(4), BigInt(2), BigInt(1)});
        VectorZZ s_bar(ambient_dim);
        for (size_t i = 0; i < m; ++i) s_bar[i] = e[i];
        for (size_t j = 0; j < n; ++j) s_bar[m + j] = s[j];
        s_bar[ambient_dim - 1] = BigInt(1);

        // <s, v> = 4*2 + 2*(-1) + 1*5 = 8 - 2 + 5 = 11 = 4 (mod 7).
        // Since l = 3 != 4 (mod 7), inner product <s_bar, v_bar> = 11 - 3 = 8 != 0 (mod 7).
        BigInt inner = s_bar.dot(v_bar);
        REQUIRE((inner % k) != BigInt(0));

        // Now with correct residue l = 11 mod 7 = 4:
        ModularHint correct_hint(v, BigInt(11), k);
        VectorZZ correct_v_bar = correct_hint.homogenize_int(ambient_dim);
        BigInt correct_inner = s_bar.dot(correct_v_bar);
        REQUIRE((correct_inner % k) == BigInt(0));
    }
}

TEST_CASE("1D Discrete Gaussian Coset Invariants", "[modular_hint][discrete_gaussian]") {
    SECTION("Symmetric prior over coset k*Z has zero mean") {
        double mu = 0.0;
        double sigma_sq = 4.0;
        double k = 2.0;

        auto [mu_scalar, var_scalar] = ModularHint::discrete_gaussian_1d_coset(mu, sigma_sq, k, 0.0);
        REQUIRE_THAT(mu_scalar, WithinAbs(0.0, 1e-12));
        REQUIRE(var_scalar <= sigma_sq);
        REQUIRE(var_scalar > 0.0);
    }

    SECTION("Variance reduction: var_scalar <= prior variance") {
        std::vector<double> sigmas = {0.5, 1.0, 2.0, 3.5, 5.0};
        std::vector<double> moduli = {2.0, 3.0, 5.0, 11.0};

        for (double s : sigmas) {
            double s2 = s * s;
            for (double k : moduli) {
                auto [mu_scalar, var_scalar] = ModularHint::discrete_gaussian_1d_coset(1.23, s2, k, 0.0);
                REQUIRE(var_scalar <= s2 + 1e-14);
                REQUIRE(var_scalar >= 0.0);
            }
        }
    }

    SECTION("Smooth limit: sigma >> k leaves distribution unchanged") {
        double mu = 3.7;
        double sigma = 50.0;
        double sigma_sq = sigma * sigma;
        double k = 2.0;

        auto [mu_scalar, var_scalar] = ModularHint::discrete_gaussian_1d_coset(mu, sigma_sq, k, 0.0);
        REQUIRE_THAT(mu_scalar, WithinAbs(mu, 1e-8));
        REQUIRE_THAT(var_scalar, WithinAbs(sigma_sq, 1e-8));
    }

    SECTION("Single coset point limit: sigma << k collapses to closest coset point") {
        double mu = 4.1;
        double sigma_sq = 0.0001; // tiny variance
        double k = 2.0; // cosets: ..., 2, 4, 6, ... closest is 4

        auto [mu_scalar, var_scalar] = ModularHint::discrete_gaussian_1d_coset(mu, sigma_sq, k, 0.0);
        REQUIRE_THAT(mu_scalar, WithinAbs(4.0, 1e-3));
        REQUIRE_THAT(var_scalar, WithinAbs(0.0, 1e-4));
    }

    SECTION("Shift by integer multiple of k shifts mean by that multiple") {
        double mu = 1.4;
        double sigma_sq = 1.5;
        double k = 3.0;

        auto [m1, v1] = ModularHint::discrete_gaussian_1d_coset(mu, sigma_sq, k, 0.0);
        auto [m2, v2] = ModularHint::discrete_gaussian_1d_coset(mu + 2.0 * k, sigma_sq, k, 0.0);

        REQUIRE_THAT(m2, WithinAbs(m1 + 2.0 * k, 1e-10));
        REQUIRE_THAT(v2, WithinAbs(v1, 1e-10));
    }
}

TEST_CASE("Distribution Update Invariants (Non-Smooth Case)", "[modular_hint][distribution]") {
    size_t d = 5;
    VectorDouble mean({0.5, -1.0, 2.0, 0.0, 1.0});

    // Create a symmetric positive definite covariance matrix
    MatrixDouble A(d, d);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            A(i, j) = dist(rng);
        }
    }
    MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d) * 0.5;

    VectorDouble v_bar({1.0, -1.0, 0.5, 2.0, -0.5});
    double k = 5.0; // Large modulus to ensure non-smooth regime

    // Compute expected scalar statistics along direction
    double mu_Y = 0.0;
    for (size_t i = 0; i < d; ++i) mu_Y += mean[i] * v_bar[i];

    VectorDouble w(d);
    for (size_t j = 0; j < d; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < d; ++i) sum += v_bar[i] * cov(i, j);
        w[j] = sum;
    }
    double sigma_Y_sq = 0.0;
    for (size_t j = 0; j < d; ++j) sigma_Y_sq += w[j] * v_bar[j];

    auto [expected_mu_scalar, expected_var_scalar] =
        ModularHint::discrete_gaussian_1d_coset(mu_Y, sigma_Y_sq, k, 0.0);

    auto [mu_prime, cov_prime] = ModularHint::update_distribution(mean, cov, v_bar, k);

    SECTION("Directional mean matches scalar discrete Gaussian mean") {
        double actual_mu_Y = 0.0;
        for (size_t i = 0; i < d; ++i) actual_mu_Y += mu_prime[i] * v_bar[i];
        REQUIRE_THAT(actual_mu_Y, WithinAbs(expected_mu_scalar, 1e-11));
    }

    SECTION("Directional variance matches scalar discrete Gaussian variance") {
        VectorDouble w_new(d);
        for (size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < d; ++i) sum += v_bar[i] * cov_prime(i, j);
            w_new[j] = sum;
        }
        double actual_var_Y = 0.0;
        for (size_t j = 0; j < d; ++j) actual_var_Y += w_new[j] * v_bar[j];
        REQUIRE_THAT(actual_var_Y, WithinAbs(expected_var_scalar, 1e-10));
    }

    SECTION("Covariance symmetry is strictly preserved") {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                REQUIRE_THAT(cov_prime(i, j), WithinAbs(cov_prime(j, i), 1e-14));
            }
        }
    }

    SECTION("Monotonic uncertainty reduction: Sigma - Sigma' is symmetric PSD") {
        MatrixDouble diff(d, d);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                diff(i, j) = cov(i, j) - cov_prime(i, j);
            }
        }

        // Test with random projection vectors x
        for (int trial = 0; trial < 20; ++trial) {
            VectorDouble x(d);
            for (size_t i = 0; i < d; ++i) x[i] = dist(rng);

            double quad = 0.0;
            for (size_t i = 0; i < d; ++i) {
                double row_sum = 0.0;
                for (size_t j = 0; j < d; ++j) row_sum += x[j] * diff(j, i);
                quad += row_sum * x[i];
            }
            REQUIRE(quad >= -1e-12); // Non-negative quadratic form
        }
    }
}

TEST_CASE("Dual Lattice Densification and Primal Sparsification (Smooth Case)", "[modular_hint][sparsification]") {
    SECTION("3D Integer Lattice with primitive modular hint") {
        MatrixZZ B(3, 3);
        B(0, 0) = BigInt(3); B(0, 1) = BigInt(1); B(0, 2) = BigInt(0);
        B(1, 0) = BigInt(0); B(1, 1) = BigInt(2); B(1, 2) = BigInt(1);
        B(2, 0) = BigInt(1); B(2, 1) = BigInt(0); B(2, 2) = BigInt(4);

        BasisZZ basis(B);
        LatticeZZ lattice(basis);
        BigReal vol_orig = lattice.volume();

        VectorZZ v_bar({BigInt(1), BigInt(2), BigInt(-1)});
        BigInt k(3);

        BasisZZ sparsified_basis = ModularHint::sparsify_lattice(basis, v_bar, k);
        LatticeZZ sparsified_lattice = ModularHint::update_lattice(lattice, v_bar, k);

        // 1. Dimension preserved
        REQUIRE(sparsified_basis.dimension() == 3);
        REQUIRE(sparsified_basis.ambient_dimension() == 3);
        REQUIRE(sparsified_lattice.dimension() == 3);

        // 2. Volume scaled by k = 3
        BigReal expected_vol = vol_orig * BigReal(3);
        REQUIRE_THAT(sparsified_lattice.volume().to_double(),
                     WithinAbs(expected_vol.to_double(), 1e-8));

        // 3. Every basis vector of sparsified lattice satisfies congruence mod k
        for (size_t i = 0; i < 3; ++i) {
            VectorZZ row_i = sparsified_basis.row(i);
            BigInt dot_mod = row_i.dot(v_bar) % k;
            if (dot_mod < BigInt(0)) dot_mod += k;
            REQUIRE(dot_mod == BigInt(0));

            // Also verify row_i belongs to original lattice
            REQUIRE(lattice.contains(row_i));
        }
    }

    SECTION("4D Subspace Lattice with ground truth target vector") {
        // d = 3 in ambient m = 4
        MatrixZZ B(3, 4);
        B(0, 0) = BigInt(2); B(0, 1) = BigInt(1); B(0, 2) = BigInt(0); B(0, 3) = BigInt(1);
        B(1, 0) = BigInt(0); B(1, 1) = BigInt(3); B(1, 2) = BigInt(2); B(1, 3) = BigInt(0);
        B(2, 0) = BigInt(1); B(2, 1) = BigInt(0); B(2, 2) = BigInt(1); B(2, 3) = BigInt(2);

        BasisZZ basis(B);
        LatticeZZ lattice(basis);
        BigReal vol_orig = lattice.volume();

        // Target vector in lattice: s_bar = 2 * b_0 - 1 * b_1 + 3 * b_2
        VectorZZ coords({BigInt(2), BigInt(-1), BigInt(3)});
        VectorZZ target = coords * B;
        REQUIRE(lattice.contains(target));

        // Choose v_bar such that <target, v_bar> = 0 (mod 5)
        BigInt k(5);
        VectorZZ v_bar({BigInt(1), BigInt(1), BigInt(2), BigInt(-1)});
        BigInt target_inner = target.dot(v_bar);
        BigInt rem = target_inner % k;
        if (rem < BigInt(0)) rem += k;

        // Adjust last coordinate of v_bar so <target, v_bar> == 0 (mod 5)
        // target[3] = 2*1 - 1*0 + 3*2 = 8
        // If we add offset to v_bar:
        while ((target.dot(v_bar) % k) != BigInt(0)) {
            v_bar[0] += BigInt(1);
        }
        REQUIRE((target.dot(v_bar) % k) == BigInt(0));

        BasisZZ sparsified_basis = ModularHint::sparsify_lattice(basis, v_bar, k);
        LatticeZZ sparsified_lattice = ModularHint::update_lattice(lattice, v_bar, k);

        REQUIRE(sparsified_basis.dimension() == 3);
        REQUIRE(sparsified_basis.ambient_dimension() == 4);

        // Target MUST belong to the sparsified lattice!
        REQUIRE(sparsified_lattice.contains(target));

        // Coordinates of target in sparsified basis must be exact integers
        MatrixQQ D_prime = compute_dual_basis(sparsified_basis.matrix());
        VectorQQ target_q = target.cast<BigRational>();
        VectorQQ x_new_q = target_q * D_prime.transpose();
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(x_new_q[i].denominator() == BigInt(1));
        }

        VectorZZ x_new({x_new_q[0].numerator(), x_new_q[1].numerator(), x_new_q[2].numerator()});
        REQUIRE(x_new * sparsified_basis.matrix() == target);
    }
}

TEST_CASE("Small Lattice Brute-Force Congruence Verification", "[modular_hint][brute_force]") {
    // 3D lattice
    MatrixZZ B(3, 3);
    B(0, 0) = BigInt(4); B(0, 1) = BigInt(1); B(0, 2) = BigInt(1);
    B(1, 0) = BigInt(1); B(1, 1) = BigInt(3); B(1, 2) = BigInt(0);
    B(2, 0) = BigInt(0); B(2, 1) = BigInt(2); B(2, 2) = BigInt(5);

    BasisZZ basis(B);
    VectorZZ v_bar({BigInt(1), BigInt(2), BigInt(-1)});
    BigInt k(3);

    BasisZZ B_prime = ModularHint::sparsify_lattice(basis, v_bar, k);
    MatrixQQ D_prime = compute_dual_basis(B_prime.matrix());

    // Exhaustive enumeration of all integer combinations with coefficients in [-3, 3]
    for (long long c0 = -3; c0 <= 3; ++c0) {
        for (long long c1 = -3; c1 <= 3; ++c1) {
            for (long long c2 = -3; c2 <= 3; ++c2) {
                VectorZZ c({BigInt(c0), BigInt(c1), BigInt(c2)});
                VectorZZ x = c * B;

                BigInt dot = x.dot(v_bar);
                BigInt rem = dot % k;
                if (rem < BigInt(0)) rem += k;

                // Check membership in B_prime via dual basis coordinates:
                VectorQQ x_q = x.cast<BigRational>();
                VectorQQ coords_q = x_q * D_prime.transpose();
                bool in_B_prime = (coords_q[0].denominator() == BigInt(1) &&
                                   coords_q[1].denominator() == BigInt(1) &&
                                   coords_q[2].denominator() == BigInt(1));

                if (rem == BigInt(0)) {
                    // Vectors satisfying the congruence MUST belong to B'
                    REQUIRE(in_B_prime);
                } else {
                    // Vectors violating the congruence MUST NOT belong to B'
                    REQUIRE(!in_B_prime);
                }
            }
        }
    }
}

TEST_CASE("LWE Kannan Embedding Modular Hint Integration", "[modular_hint][lwe]") {
    SECTION("Toy LWE: n=3, m=3, q=11") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(11);
        double sigma_e = 1.0;
        double sigma_s = 2.0;

        DiscreteGaussian noise_dist(sigma_e, 12345);
        DiscreteGaussian secret_dist(sigma_s, 12346);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 7777);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto instance = DBDDInstance::from_kannan(kannan);

        size_t orig_dim = instance.dimension();
        BigReal orig_vol = instance.volume();
        REQUIRE(orig_dim == m + n + 1); // 7
        REQUIRE(instance.has_target());
        REQUIRE(instance.target_in_lattice());

        // Extract secret s
        const auto& s = lwe.test_only_secret();
        VectorZZ v({BigInt(1), BigInt(0), BigInt(1)});
        BigInt inner = s.dot(v);
        BigInt k1(2);
        BigInt l1 = inner % k1;
        if (l1 < BigInt(0)) l1 += k1;

        // Integrate first modular hint: <s, v> = l1 (mod 2)
        instance.integrate_modular_hint(v, l1, k1);

        // Verify invariants:
        REQUIRE(instance.dimension() == orig_dim);
        REQUIRE_THAT(instance.volume().to_double(), WithinAbs((orig_vol * BigReal(2)).to_double(), 1e-6));
        REQUIRE(instance.target_in_lattice());
        REQUIRE(instance.verify_target(0.999));

        // Integrate second modular hint with coprime modulus k2 = 3
        VectorZZ v2({BigInt(0), BigInt(1), BigInt(1)});
        BigInt inner2 = s.dot(v2);
        BigInt k2(3);
        BigInt l2 = inner2 % k2;
        if (l2 < BigInt(0)) l2 += k2;

        instance.integrate_modular_hint(v2, l2, k2);

        REQUIRE(instance.dimension() == orig_dim);
        REQUIRE_THAT(instance.volume().to_double(), WithinAbs((orig_vol * BigReal(6)).to_double(), 1e-5));
        REQUIRE(instance.target_in_lattice());
        REQUIRE(instance.verify_target(0.999));
    }

    SECTION("Medium LWE: n=8, m=8, q=97") {
        size_t n = 8;
        size_t m = 8;
        BigInt q(97);
        double sigma_e = 1.5;
        double sigma_s = 2.0;

        DiscreteGaussian noise_dist(sigma_e, 54321);
        DiscreteGaussian secret_dist(sigma_s, 54322);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 8888);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto instance = DBDDInstance::from_kannan(kannan);

        size_t orig_dim = instance.dimension(); // 8 + 8 + 1 = 17
        BigReal orig_vol = instance.volume();

        const auto& s = lwe.test_only_secret();
        VectorZZ v({BigInt(1), BigInt(-1), BigInt(2), BigInt(0), BigInt(1), BigInt(0), BigInt(-1), BigInt(1)});
        BigInt k(5);
        BigInt l = s.dot(v) % k;
        if (l < BigInt(0)) l += k;

        instance.integrate_modular_hint(v, l, k);

        REQUIRE(instance.dimension() == orig_dim);
        REQUIRE_THAT(instance.volume().to_double(), WithinAbs((orig_vol * BigReal(5)).to_double(), 1e-4));
        REQUIRE(instance.target_in_lattice());
        REQUIRE(instance.verify_target(0.999));
    }
}

TEST_CASE("Modular Hint Error Handling and Edge Cases", "[modular_hint][errors]") {
    SECTION("Contradictory hint throws InvalidParameterException") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(11);
        DiscreteGaussian noise_dist(1.0, 999);
        DiscreteGaussian secret_dist(2.0, 998);
        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 1111);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto instance = DBDDInstance::from_kannan(kannan);

        const auto& s = lwe.test_only_secret();
        VectorZZ v({BigInt(1), BigInt(0), BigInt(0)});
        BigInt true_inner = s.dot(v);
        BigInt k(3);
        BigInt wrong_l = (true_inner + BigInt(1)) % k;

        REQUIRE_THROWS_AS(instance.integrate_modular_hint(v, wrong_l, k), InvalidParameterException);
    }
}
