#include "dbdd/hints/short_vector_hint.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/linalg/projection.hpp"
#include <cmath>

namespace dbdd {

// Paper Reference: Albrecht et al. Section 3.5 / Equations (27)-(30)
// Mathematical Purpose: Implementation of ShortVectorHint for DBDD.
// Projects lattice Lambda' = Lambda * P_v^perp and distribution (mu', Sigma') = (mu P, P^T Sigma P),
// dropping dimension by 1 (or k) and volume by ||v|| (or sqrt(det(V V^T))).

ShortVectorHint::ShortVectorHint(VectorZZ v)
    : v_(std::move(v)), is_multi_(false) {
    if (v_.empty()) {
        throw InvalidParameterException("ShortVectorHint: hint vector cannot be empty");
    }
    if (v_.squared_norm() == BigInt(0)) {
        throw InvalidParameterException("ShortVectorHint: hint vector cannot be the zero vector");
    }
}

ShortVectorHint::ShortVectorHint(MatrixZZ V)
    : V_(std::move(V)), is_multi_(true) {
    if (V_.empty()) {
        throw InvalidParameterException("ShortVectorHint: hint matrix cannot be empty");
    }
    if (V_.rank() != V_.rows()) {
        throw InvalidParameterException("ShortVectorHint: rows of hint matrix must be linearly independent");
    }
    if (V_.rows() == 1) {
        v_ = V_.row(0);
        is_multi_ = false;
    }
}

ShortVectorHint ShortVectorHint::from_vector(VectorZZ v) {
    return ShortVectorHint(std::move(v));
}

ShortVectorHint ShortVectorHint::from_matrix(MatrixZZ V) {
    return ShortVectorHint(std::move(V));
}

const VectorZZ& ShortVectorHint::hint_vector() const {
    if (is_multi_) {
        throw InvalidParameterException("ShortVectorHint is multi-vector; call hint_matrix() instead");
    }
    return v_;
}

const MatrixZZ& ShortVectorHint::hint_matrix() const {
    return V_;
}

size_t ShortVectorHint::num_vectors() const noexcept {
    return is_multi_ ? V_.rows() : 1;
}

std::string ShortVectorHint::description() const {
    if (is_multi_) {
        return "ShortVectorHint(multi-vector: " + std::to_string(V_.rows()) + " vectors)";
    }
    std::string desc = "(";
    for (size_t i = 0; i < v_.size(); ++i) {
        if (i > 0) desc += ", ";
        desc += v_[i].to_string();
    }
    desc += ")";
    return "ShortVectorHint(v = " + desc + ")";
}

MatrixQQ ShortVectorHint::orthogonal_projector(const VectorZZ& v) {
    return dbdd::orthogonal_projection_matrix(v);
}

MatrixQQ ShortVectorHint::orthogonal_projector(const MatrixZZ& V) {
    return dbdd::orthogonal_complement_projection_matrix(V);
}

std::pair<VectorDouble, MatrixDouble> ShortVectorHint::project_distribution(
    const VectorDouble& mean,
    const MatrixDouble& cov,
    const MatrixDouble& P) {

    size_t d = mean.size();
    if (cov.rows() != d || cov.cols() != d) {
        throw DimensionMismatchException("ShortVectorHint::project_distribution: covariance dimension mismatch");
    }
    if (P.rows() != d || P.cols() != d) {
        throw DimensionMismatchException("ShortVectorHint::project_distribution: projector dimension mismatch");
    }

    // mu' = mean * P (1 x d)
    VectorDouble new_mean = mean * P;

    // Sigma' = P^T * cov * P
    MatrixDouble PT_cov = P.transpose() * cov;
    MatrixDouble new_cov = PT_cov * P;

    // Enforce strict numerical symmetry
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (new_cov(i, j) + new_cov(j, i));
            new_cov(i, j) = avg;
            new_cov(j, i) = avg;
        }
    }

    return {new_mean, new_cov};
}

BasisQQ ShortVectorHint::project_basis_rational(const BasisZZ& basis, const VectorZZ& v) {
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();

    if (v.size() != m) {
        throw DimensionMismatchException(
            "ShortVectorHint::project_basis_rational: vector dimension (" + std::to_string(v.size()) +
            ") does not match basis ambient dimension (" + std::to_string(m) + ")");
    }
    if (d < 2) {
        throw InvalidParameterException("Cannot project basis with dimension < 2");
    }

    // 1. Compute coordinates c of v in basis B: v = c * B
    VectorQQ c_q = basis.coordinates_of<BigRational>(v);
    VectorZZ c(d);
    for (size_t i = 0; i < d; ++i) {
        if (c_q[i].denominator() != BigInt(1)) {
            throw InvalidParameterException("ShortVectorHint: vector v does not belong to the lattice");
        }
        c[i] = c_q[i].numerator();
    }

    if (c.squared_norm() == BigInt(0)) {
        throw InvalidParameterException("ShortVectorHint: vector v is zero in lattice coordinates");
    }

    // 2. Primitivity check & reduction: gcd(c_1, ..., c_d)
    BigInt g(0);
    for (size_t i = 0; i < d; ++i) {
        g = BigInt::gcd(g, c[i].abs());
    }
    VectorZZ c_prim(d);
    for (size_t i = 0; i < d; ++i) {
        c_prim[i] = c[i] / g;
    }

    // 3. Complete c_prim to a unimodular matrix U in GL_d(Z) such that row d - 1 is c_prim
    MatrixZZ U = PerfectHint::complete_primitive_vector_unimodular(c_prim);

    // Alternative unimodular basis: B_tilde = U * B (d x m)
    MatrixZZ B_tilde = U * basis.matrix();

    // 4. Compute exact rational projector P_v^perp
    MatrixQQ P = orthogonal_projector(v);

    // 5. Project first d - 1 rows of B_tilde: row i becomes r_i * P
    MatrixQQ B_prime(d - 1, m);
    for (size_t i = 0; i < d - 1; ++i) {
        VectorQQ r_q = B_tilde.row(i).cast<BigRational>();
        VectorQQ proj_r = r_q * P;
        B_prime.set_row(i, proj_r);
    }

    return BasisQQ(std::move(B_prime), /*validate_rank=*/false);
}

BasisZZ ShortVectorHint::project_basis(const BasisZZ& basis, const VectorZZ& v) {
    BasisQQ B_q = project_basis_rational(basis, v);
    size_t d_prime = B_q.dimension();
    size_t m = B_q.ambient_dimension();

    MatrixZZ B_z(d_prime, m);
    for (size_t i = 0; i < d_prime; ++i) {
        for (size_t j = 0; j < m; ++j) {
            const BigRational& val = B_q.matrix()(i, j);
            if (val.denominator() != BigInt(1)) {
                throw NumericalPrecisionException(
                    "ShortVectorHint::project_basis: projected basis entry (" + std::to_string(i) +
                    ", " + std::to_string(j) + ") has denominator " + val.denominator().to_string() +
                    "; use project_basis_rational instead.");
            }
            B_z(i, j) = val.numerator();
        }
    }

    return BasisZZ(std::move(B_z), /*validate_rank=*/false);
}

BasisQQ ShortVectorHint::project_basis_rational(const BasisZZ& basis, const MatrixZZ& V) {
    BasisQQ cur = basis.cast<BigRational>();
    for (size_t r = 0; r < V.rows(); ++r) {
        VectorZZ v_row = V.row(r);
        // Project current rational basis along v_row
        size_t d = cur.dimension();
        size_t m = cur.ambient_dimension();

        VectorQQ c_q = cur.coordinates_of<BigRational>(v_row.cast<BigRational>());
        VectorZZ c(d);
        for (size_t i = 0; i < d; ++i) {
            if (c_q[i].denominator() != BigInt(1)) {
                throw InvalidParameterException("ShortVectorHint: vector V[r] does not belong to the lattice");
            }
            c[i] = c_q[i].numerator();
        }

        BigInt g(0);
        for (size_t i = 0; i < d; ++i) {
            g = BigInt::gcd(g, c[i].abs());
        }
        VectorZZ c_prim(d);
        for (size_t i = 0; i < d; ++i) {
            c_prim[i] = c[i] / g;
        }

        MatrixZZ U = PerfectHint::complete_primitive_vector_unimodular(c_prim);
        MatrixQQ B_tilde = U.cast<BigRational>() * cur.matrix();
        MatrixQQ P = orthogonal_projector(v_row);

        MatrixQQ B_prime(d - 1, m);
        for (size_t i = 0; i < d - 1; ++i) {
            VectorQQ r_q = B_tilde.row(i);
            B_prime.set_row(i, r_q * P);
        }
        cur = BasisQQ(std::move(B_prime), false);
    }
    return cur;
}

BasisZZ ShortVectorHint::project_basis(const BasisZZ& basis, const MatrixZZ& V) {
    BasisQQ B_q = project_basis_rational(basis, V);
    size_t d_prime = B_q.dimension();
    size_t m = B_q.ambient_dimension();

    MatrixZZ B_z(d_prime, m);
    for (size_t i = 0; i < d_prime; ++i) {
        for (size_t j = 0; j < m; ++j) {
            const BigRational& val = B_q.matrix()(i, j);
            if (val.denominator() != BigInt(1)) {
                throw NumericalPrecisionException(
                    "ShortVectorHint::project_basis: projected basis has non-integer entries; use project_basis_rational.");
            }
            B_z(i, j) = val.numerator();
        }
    }
    return BasisZZ(std::move(B_z), /*validate_rank=*/false);
}

BigReal ShortVectorHint::projected_volume(const BigReal& vol, const VectorZZ& v) {
    BigInt sq_norm = v.squared_norm();
    if (sq_norm == BigInt(0)) {
        throw DivisionByZeroException("ShortVectorHint::projected_volume: zero vector norm");
    }
    BigReal v_norm = BigReal(sq_norm).sqrt();
    return vol / v_norm;
}

BigReal ShortVectorHint::projected_volume(const BigReal& vol, const MatrixZZ& V) {
    // Gram matrix G = V * V^T (k x k)
    MatrixZZ G = V * V.transpose();
    BigInt det_G = G.determinant();
    if (det_G <= BigInt(0)) {
        throw SingularMatrixException("ShortVectorHint::projected_volume: vectors in V are not linearly independent");
    }
    BigReal sublat_vol = BigReal(det_G).sqrt();
    return vol / sublat_vol;
}

void ShortVectorHint::apply(DBDDInstance& instance) const {
    if (!is_multi_) {
        // 1. Project basis & compute exact projected volume
        BasisZZ new_basis = project_basis(instance.basis(), v_);
        BigReal new_vol = projected_volume(instance.volume(), v_);
        LatticeZZ new_lat(std::move(new_basis), new_vol);

        // 2. Project distribution
        MatrixQQ P_q = orthogonal_projector(v_);
        MatrixDouble P_d = P_q.cast<double>();
        auto [new_mean, new_cov] = project_distribution(instance.mean(), instance.covariance(), P_d);

        // 3. Project target vector and coordinates if present
        std::optional<VectorZZ> new_target = std::nullopt;
        std::optional<VectorZZ> new_coords = std::nullopt;
        if (instance.has_target()) {
            VectorQQ target_q = instance.target().cast<BigRational>() * P_q;
            VectorZZ target_z(target_q.size());
            for (size_t i = 0; i < target_q.size(); ++i) {
                target_z[i] = target_q[i].numerator();
            }

            VectorQQ coords_q = new_lat.coordinates_of<BigRational>(target_z);
            VectorZZ coords_z(coords_q.size());
            for (size_t i = 0; i < coords_q.size(); ++i) {
                coords_z[i] = coords_q[i].numerator();
            }

            new_target = std::move(target_z);
            new_coords = std::move(coords_z);
        }

        // 4. Update instance state
        instance.set_lattice(std::move(new_lat));
        instance.set_distribution(std::move(new_mean), std::move(new_cov));
        if (new_target.has_value()) {
            instance.set_target(std::move(*new_target), std::move(new_coords));
        }
    } else {
        // Multi-vector projection applied sequentially
        for (size_t r = 0; r < V_.rows(); ++r) {
            ShortVectorHint single_hint(V_.row(r));
            single_hint.apply(instance);
        }
    }
}

} // namespace dbdd
