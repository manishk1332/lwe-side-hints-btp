#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "dbdd/hints/short_vector_hint.hpp"
#include "dbdd/lwe/secret_recovery.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include <random>

using namespace dbdd;
using Catch::Matchers::WithinAbs;

TEST_CASE("Short-Vector Projector Invariants", "[short_vector_hint][projector]") {
    SECTION("Single Vector Projector: P^2 = P, P^T = P, v * P = 0") {
        VectorZZ v({BigInt(2), BigInt(-1), BigInt(3), BigInt(5)});
        MatrixQQ P = ShortVectorHint::orthogonal_projector(v);
        size_t m = v.size();

        // 1. Symmetry: P^T == P
        REQUIRE(P.transpose() == P);

        // 2. Idempotence: P^2 == P
        MatrixQQ P2 = P * P;
        REQUIRE(P2 == P);

        // 3. Annihilation: v * P == 0
        VectorQQ v_q = v.cast<BigRational>();
        VectorQQ vP = v_q * P;
        for (size_t j = 0; j < m; ++j) {
            REQUIRE(vP[j] == BigRational(0));
        }
    }

    SECTION("Multi-Vector Projector: P_V^2 = P_V, P_V^T = P_V, V * P_V = 0") {
        MatrixZZ V(2, 4);
        V.set_row(0, VectorZZ({BigInt(1), BigInt(0), BigInt(2), BigInt(1)}));
        V.set_row(1, VectorZZ({BigInt(0), BigInt(3), BigInt(-1), BigInt(2)}));

        MatrixQQ P = ShortVectorHint::orthogonal_projector(V);
        size_t m = 4;

        // Symmetry & Idempotence
        REQUIRE(P.transpose() == P);
        REQUIRE(P * P == P);

        // Annihilation: each row of V projected by P is zero
        for (size_t r = 0; r < V.rows(); ++r) {
            VectorQQ row_q = V.row(r).cast<BigRational>();
            VectorQQ row_proj = row_q * P;
            for (size_t j = 0; j < m; ++j) {
                REQUIRE(row_proj[j] == BigRational(0));
            }
        }
    }
}

TEST_CASE("Single Vector Projection and Exact Volume Drop", "[short_vector_hint][volume]") {
    // 3D Integer lattice with known volume
    MatrixZZ mat(3, 3);
    mat(0, 0) = 3; mat(0, 1) = 0; mat(0, 2) = 0;
    mat(1, 0) = 1; mat(1, 1) = 2; mat(1, 2) = 0;
    mat(2, 0) = 0; mat(2, 1) = 1; mat(2, 2) = 4;
    BasisZZ basis(mat);
    LatticeZZ lattice(basis);

    BigReal orig_vol = lattice.volume(); // det = 3 * 2 * 4 = 24
    REQUIRE_THAT(orig_vol.to_double(), WithinAbs(24.0, 1e-10));

    // Choose primitive lattice vector v = (3, 0, 0) (first basis vector)
    VectorZZ v = basis.row(0);
    REQUIRE(lattice.contains(v));

    BasisZZ B_prime = ShortVectorHint::project_basis(basis, v);
    REQUIRE(B_prime.dimension() == 2);
    REQUIRE(B_prime.ambient_dimension() == 3);

    // Projected basis vectors must be orthogonal to v
    for (size_t r = 0; r < B_prime.dimension(); ++r) {
        REQUIRE(B_prime.row(r).dot(v) == BigInt(0));
    }

    // Volume formula: Vol(Lambda') = Vol(Lambda) / ||v|| = 24 / 3 = 8
    BigReal predicted_vol = ShortVectorHint::projected_volume(orig_vol, v);
    REQUIRE_THAT(predicted_vol.to_double(), WithinAbs(8.0, 1e-10));

    BigReal actual_vol = compute_volume(B_prime);
    REQUIRE_THAT(actual_vol.to_double(), WithinAbs(predicted_vol.to_double(), 1e-10));
}

TEST_CASE("Multi-Vector Projection and Volume Formula", "[short_vector_hint][multivolume]") {
    // 4D diagonal lattice with basis vectors 2*e_0, 3*e_1, 5*e_2, 7*e_3
    MatrixZZ mat = MatrixZZ::identity(4);
    mat(0, 0) = 2; mat(1, 1) = 3; mat(2, 2) = 5; mat(3, 3) = 7;
    BasisZZ basis(mat);
    BigReal orig_vol = compute_volume(basis); // 210
    REQUIRE_THAT(orig_vol.to_double(), WithinAbs(210.0, 1e-10));

    // Choose 2 orthogonal lattice vectors V = [2*e_0; 3*e_1]
    MatrixZZ V(2, 4);
    V.set_row(0, basis.row(0)); // (2, 0, 0, 0)
    V.set_row(1, basis.row(1)); // (0, 3, 0, 0)

    BasisZZ B_prime = ShortVectorHint::project_basis(basis, V);
    REQUIRE(B_prime.dimension() == 2);
    REQUIRE(B_prime.ambient_dimension() == 4);

    // Sublattice volume = sqrt(det(V V^T)) = sqrt(4 * 9) = 6
    // Predicted volume = 210 / 6 = 35 (which is 5 * 7)
    BigReal pred_vol = ShortVectorHint::projected_volume(orig_vol, V);
    REQUIRE_THAT(pred_vol.to_double(), WithinAbs(35.0, 1e-10));

    BigReal actual_vol = compute_volume(B_prime);
    REQUIRE_THAT(actual_vol.to_double(), WithinAbs(35.0, 1e-10));
}

TEST_CASE("Distribution Projection: Symmetry and PSD Preservation", "[short_vector_hint][distribution]") {
    size_t d = 5;
    VectorDouble mean({1.0, -2.0, 0.5, 3.0, -1.0});

    std::mt19937_64 rng(789);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    MatrixDouble A(d, d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            A(i, j) = dist(rng);
        }
    }
    MatrixDouble cov = A * A.transpose() + MatrixDouble::identity(d) * 0.5;

    VectorZZ v({BigInt(1), BigInt(-1), BigInt(2), BigInt(0), BigInt(1)});
    MatrixQQ P_q = ShortVectorHint::orthogonal_projector(v);
    MatrixDouble P = P_q.cast<double>();

    auto [new_mean, new_cov] = ShortVectorHint::project_distribution(mean, cov, P);

    SECTION("Symmetry preserved") {
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                REQUIRE_THAT(new_cov(i, j), WithinAbs(new_cov(j, i), 1e-14));
            }
        }
    }

    SECTION("Positive semidefiniteness preserved") {
        for (int trial = 0; trial < 25; ++trial) {
            VectorDouble x(d);
            for (size_t i = 0; i < d; ++i) x[i] = dist(rng);

            double quad = 0.0;
            for (size_t i = 0; i < d; ++i) {
                double row_sum = 0.0;
                for (size_t j = 0; j < d; ++j) row_sum += x[j] * new_cov(j, i);
                quad += row_sum * x[i];
            }
            REQUIRE(quad >= -1e-12);
        }
    }

    SECTION("Annihilation along v") {
        VectorDouble v_d = v.cast<double>();
        double var_along_v = 0.0;
        for (size_t i = 0; i < d; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < d; ++j) row_sum += v_d[j] * new_cov(j, i);
            var_along_v += v_d[i] * row_sum;
        }
        REQUIRE_THAT(var_along_v, WithinAbs(0.0, 1e-12));

        double mean_along_v = 0.0;
        for (size_t i = 0; i < d; ++i) mean_along_v += new_mean[i] * v_d[i];
        REQUIRE_THAT(mean_along_v, WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("DBDDInstance Integration with q-Vectors", "[short_vector_hint][dbdd]") {
    size_t n = 3;
    size_t m = 3;
    BigInt q(11);

    DiscreteGaussian noise_dist(1.0, 111);
    DiscreteGaussian secret_dist(2.0, 222);

    auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 333);
    auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
    auto dbdd = DBDDInstance::from_kannan(kannan);

    size_t init_dim = dbdd.dimension(); // 3 + 3 + 1 = 7
    size_t init_amb = dbdd.ambient_dimension(); // 7
    BigReal init_vol = dbdd.volume(); // 11^3 = 1331

    // First basis vector of Kannan embedding is the q-vector (q, 0, 0, 0, 0, 0, 0)
    VectorZZ q_vec = dbdd.basis().row(0);
    REQUIRE(q_vec[0] == q);
    REQUIRE(dbdd.lattice().contains(q_vec));

    dbdd.integrate_short_vector_hint(q_vec);

    // Dimension drops by 1
    REQUIRE(dbdd.dimension() == init_dim - 1);
    REQUIRE(dbdd.ambient_dimension() == init_amb);

    // Volume drops by ||q_vec|| = q = 11: 1331 / 11 = 121
    REQUIRE_THAT(dbdd.volume().to_double(), WithinAbs((init_vol / BigReal(11.0)).to_double(), 1e-8));

    // Target remains in the projected lattice
    REQUIRE(dbdd.has_target());
    REQUIRE(dbdd.target_in_lattice());
    REQUIRE(dbdd.verify_target(0.999));
}

TEST_CASE("SecretReconstructor End-to-End on Toy and Medium LWE Instances", "[short_vector_hint][reconstruct]") {
    SECTION("Toy LWE: n=3, m=3, q=97 with 1 q-vector projection") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(97);

        DiscreteGaussian noise_dist(0.5, 444);
        DiscreteGaussian secret_dist(0.5, 555);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 666);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        const VectorZZ& true_secret = lwe.test_only_secret();
        SecretReconstructor reconstructor(lwe);

        // Project out first q-vector
        VectorZZ q_vec = dbdd.basis().row(0);
        reconstructor.record_projection(q_vec);
        dbdd.integrate_short_vector_hint(q_vec);

        REQUIRE(dbdd.dimension() == 6);

        // 1. Direct validation: SecretReconstructor on projected ground truth target
        REQUIRE(dbdd.has_target());
        VectorZZ direct_secret = reconstructor.recover_secret(dbdd.target());
        REQUIRE(direct_secret == true_secret);
        REQUIRE(reconstructor.verify_recovery(direct_secret));

        VectorZZ direct_target = reconstructor.recover_target(dbdd.target());
        REQUIRE(direct_target == kannan.target());

        // 2. End-to-end lattice reduction solving: LLL finds embedded target
        auto lll_res = lll_exact(dbdd.basis());
        const auto& red_basis = lll_res.basis;

        bool found = false;
        for (size_t r = 0; r < red_basis.dimension(); ++r) {
            VectorZZ candidate = red_basis.row(r);
            size_t amb = dbdd.ambient_dimension();
            if (candidate[amb - 1].abs() == BigInt(1)) {
                VectorZZ recovered_secret = reconstructor.recover_secret(candidate);
                if (recovered_secret == true_secret) {
                    found = true;
                    REQUIRE(reconstructor.verify_recovery(recovered_secret));
                    break;
                }
            }
        }
        REQUIRE(found);
    }

    SECTION("Toy LWE: n=3, m=3, q=97 with 2 sequential q-vector projections") {
        size_t n = 3;
        size_t m = 3;
        BigInt q(97);

        DiscreteGaussian noise_dist(0.5, 777);
        DiscreteGaussian secret_dist(0.5, 888);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 999);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        const VectorZZ& true_secret = lwe.test_only_secret();
        SecretReconstructor reconstructor(lwe);

        // Record the two orthogonal q-vectors before projection: row 0 and row 1
        VectorZZ q1 = dbdd.basis().row(0);
        VectorZZ q2 = dbdd.basis().row(1);

        reconstructor.record_projection(q1);
        dbdd.integrate_short_vector_hint(q1);

        reconstructor.record_projection(q2);
        dbdd.integrate_short_vector_hint(q2);

        // Dimension dropped by 2: 7 -> 5
        REQUIRE(dbdd.dimension() == 5);
        REQUIRE_THAT(dbdd.volume().to_double(), WithinAbs(97.0, 1e-8));

        // Direct reconstruction from target
        REQUIRE(reconstructor.recover_secret(dbdd.target()) == true_secret);

        // End-to-end LLL recovery
        auto lll_res = lll_exact(dbdd.basis());
        const auto& red_basis = lll_res.basis;

        bool found = false;
        for (size_t r = 0; r < red_basis.dimension(); ++r) {
            VectorZZ candidate = red_basis.row(r);
            size_t amb = dbdd.ambient_dimension();
            if (candidate[amb - 1].abs() == BigInt(1)) {
                VectorZZ recovered_secret = reconstructor.recover_secret(candidate);
                if (recovered_secret == true_secret) {
                    found = true;
                    REQUIRE(reconstructor.verify_recovery(recovered_secret));
                    break;
                }
            }
        }
        REQUIRE(found);
    }

    SECTION("Medium LWE: n=4, m=4, q=97 with multi-vector projection") {
        size_t n = 4;
        size_t m = 4;
        BigInt q(97);

        DiscreteGaussian noise_dist(0.5, 123);
        DiscreteGaussian secret_dist(0.5, 456);

        auto lwe = LWEInstance::generate_random(n, m, q, secret_dist, noise_dist, 789);
        auto kannan = KannanEmbedding::from_lwe(lwe, noise_dist, secret_dist);
        auto dbdd = DBDDInstance::from_kannan(kannan);

        const VectorZZ& true_secret = lwe.test_only_secret();
        SecretReconstructor reconstructor(lwe);

        // Collect 2 orthogonal q-vectors in matrix V
        MatrixZZ V(2, dbdd.ambient_dimension());
        V.set_row(0, dbdd.basis().row(0));
        V.set_row(1, dbdd.basis().row(1));

        dbdd.integrate_short_vector_hint(V);

        // Dimension dropped by 2: 9 -> 7
        REQUIRE(dbdd.dimension() == 7);
        REQUIRE(dbdd.target_in_lattice());
        REQUIRE(dbdd.verify_target(0.999));

        // Direct reconstruction from target
        REQUIRE(reconstructor.recover_secret(dbdd.target()) == true_secret);
        REQUIRE(reconstructor.verify_recovery(true_secret));

        // End-to-end LLL recovery
        auto lll_res = lll_exact(dbdd.basis());
        const auto& red_basis = lll_res.basis;

        bool found = false;
        for (size_t r = 0; r < red_basis.dimension(); ++r) {
            VectorZZ candidate = red_basis.row(r);
            size_t amb = dbdd.ambient_dimension();
            if (candidate[amb - 1].abs() == BigInt(1)) {
                VectorZZ recovered_secret = reconstructor.recover_secret(candidate);
                if (recovered_secret == true_secret) {
                    found = true;
                    REQUIRE(reconstructor.verify_recovery(recovered_secret));
                    break;
                }
            }
        }
        REQUIRE(found);
    }
}
