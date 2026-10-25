#include "dbdd/linalg/eigendecomposition.hpp"
#include "dbdd/core/exceptions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace dbdd {

SymmetricEigendecomposition SymmetricEigendecomposition::decompose(
    const MatrixDouble& A,
    double rank_tol) {
    if (A.rows() != A.cols()) {
        throw DimensionMismatchException(
            "SymmetricEigendecomposition: Matrix must be square (got " +
            std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + ")");
    }

    SymmetricEigendecomposition res;
    res.dim_ = A.rows();
    res.rank_tol_ = rank_tol;

    if (res.dim_ == 0) {
        return res;
    }

    size_t d = res.dim_;

    // 1. Enforce strict mathematical symmetry
    MatrixDouble A_sym = A;
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double avg = 0.5 * (A_sym(i, j) + A_sym(j, i));
            A_sym(i, j) = avg;
            A_sym(j, i) = avg;
        }
    }

    // 2. Compute spectral decomposition via Eigen
    Eigen::MatrixXd em = A_sym.to_eigen();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(em);
    if (solver.info() != Eigen::Success) {
        throw NumericalPrecisionException(
            "SymmetricEigendecomposition: SelfAdjointEigenSolver failed to converge");
    }

    const auto& raw_eigs = solver.eigenvalues();
    const auto& raw_vecs = solver.eigenvectors();

    // 3. Sort eigenvalues and eigenvectors in descending order
    std::vector<size_t> indices(d);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return raw_eigs(i) > raw_eigs(j);
    });

    res.eigenvalues_ = VectorDouble(d);
    res.eigenvectors_ = MatrixDouble(d, d);
    res.rank_ = 0;

    for (size_t k = 0; k < d; ++k) {
        size_t idx = indices[k];
        double ev = raw_eigs(idx);
        // Clamp small negative numerical noise in [-rank_tol, 0) to 0.0
        if (ev < 0.0 && ev >= -rank_tol) {
            ev = 0.0;
        }
        res.eigenvalues_[k] = ev;
        if (ev > rank_tol) {
            res.rank_++;
        }

        // Copy eigenvector column
        for (size_t r = 0; r < d; ++r) {
            res.eigenvectors_(r, k) = raw_vecs(r, idx);
        }
    }

    return res;
}

MatrixDouble SymmetricEigendecomposition::matrix_sqrt() const {
    if (dim_ == 0) return MatrixDouble(0, 0);

    MatrixDouble res(dim_, dim_, 0.0);
    for (size_t k = 0; k < rank_; ++k) {
        double sqrt_lambda = std::sqrt(eigenvalues_[k]);
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                res(i, j) += sqrt_lambda * eigenvectors_(i, k) * eigenvectors_(j, k);
            }
        }
    }

    // Enforce symmetry
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = i + 1; j < dim_; ++j) {
            double avg = 0.5 * (res(i, j) + res(j, i));
            res(i, j) = avg;
            res(j, i) = avg;
        }
    }
    return res;
}

MatrixDouble SymmetricEigendecomposition::matrix_inv_sqrt() const {
    if (dim_ == 0) return MatrixDouble(0, 0);

    MatrixDouble res(dim_, dim_, 0.0);
    for (size_t k = 0; k < rank_; ++k) {
        double inv_sqrt_lambda = 1.0 / std::sqrt(eigenvalues_[k]);
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                res(i, j) += inv_sqrt_lambda * eigenvectors_(i, k) * eigenvectors_(j, k);
            }
        }
    }

    // Enforce symmetry
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = i + 1; j < dim_; ++j) {
            double avg = 0.5 * (res(i, j) + res(j, i));
            res(i, j) = avg;
            res(j, i) = avg;
        }
    }
    return res;
}

MatrixDouble SymmetricEigendecomposition::projector() const {
    if (dim_ == 0) return MatrixDouble(0, 0);

    MatrixDouble res(dim_, dim_, 0.0);
    for (size_t k = 0; k < rank_; ++k) {
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                res(i, j) += eigenvectors_(i, k) * eigenvectors_(j, k);
            }
        }
    }

    // Enforce symmetry
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = i + 1; j < dim_; ++j) {
            double avg = 0.5 * (res(i, j) + res(j, i));
            res(i, j) = avg;
            res(j, i) = avg;
        }
    }
    return res;
}

BigReal SymmetricEigendecomposition::restricted_determinant() const {
    if (dim_ == 0 || rank_ == 0) {
        return BigReal(1.0);
    }
    BigReal log_rdet = log_restricted_determinant();
    return log_rdet.exp();
}

BigReal SymmetricEigendecomposition::log_restricted_determinant() const {
    if (dim_ == 0 || rank_ == 0) {
        return BigReal(0.0);
    }
    BigReal sum_log(0.0);
    for (size_t k = 0; k < rank_; ++k) {
        sum_log += BigReal(std::log(eigenvalues_[k]));
    }
    return sum_log;
}

// =============================================================================
// MPFR Cyclic Jacobi Eigendecomposition Implementation
// =============================================================================

SymmetricEigendecompositionRR SymmetricEigendecompositionRR::decompose(
    const MatrixRR& A,
    double rank_tol,
    size_t max_sweeps) {
    if (A.rows() != A.cols()) {
        throw DimensionMismatchException(
            "SymmetricEigendecompositionRR: Matrix must be square (got " +
            std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + ")");
    }

    SymmetricEigendecompositionRR res;
    res.dim_ = A.rows();
    res.rank_tol_ = rank_tol;

    if (res.dim_ == 0) {
        return res;
    }

    size_t d = res.dim_;

    // Matrix copy D and eigenvector matrix V initialized to identity
    MatrixRR D(d, d);
    MatrixRR V = MatrixRR::identity(d);

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            D(i, j) = BigReal(0.5) * (A(i, j) + A(j, i));
        }
    }

    BigReal conv_tol("1e-30");

    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        BigReal off_diag_sum(0.0);
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = i + 1; j < d; ++j) {
                off_diag_sum += D(i, j).abs();
            }
        }

        if (off_diag_sum <= conv_tol) {
            break;
        }

        for (size_t p = 0; p < d - 1; ++p) {
            for (size_t q = p + 1; q < d; ++q) {
                if (D(p, q).abs() <= conv_tol) {
                    continue;
                }

                BigReal diff = D(q, q) - D(p, p);
                BigReal tau = diff / (BigReal(2.0) * D(p, q));
                BigReal t;
                if (tau >= BigReal(0.0)) {
                    t = BigReal(1.0) / (tau + (BigReal(1.0) + tau * tau).sqrt());
                } else {
                    t = BigReal(-1.0) / ((-tau) + (BigReal(1.0) + tau * tau).sqrt());
                }

                BigReal c = BigReal(1.0) / (BigReal(1.0) + t * t).sqrt();
                BigReal s = t * c;
                BigReal h = t * D(p, q);

                D(p, p) -= h;
                D(q, q) += h;
                D(p, q) = BigReal(0.0);
                D(q, p) = BigReal(0.0);

                for (size_t j = 0; j < d; ++j) {
                    if (j != p && j != q) {
                        BigReal Dpj = D(p, j);
                        BigReal Dqj = D(q, j);
                        D(p, j) = c * Dpj - s * Dqj;
                        D(j, p) = D(p, j);
                        D(q, j) = s * Dpj + c * Dqj;
                        D(j, q) = D(q, j);
                    }
                }

                for (size_t j = 0; j < d; ++j) {
                    BigReal Vjp = V(j, p);
                    BigReal Vjq = V(j, q);
                    V(j, p) = c * Vjp - s * Vjq;
                    V(j, q) = s * Vjp + c * Vjq;
                }
            }
        }
    }

    // Sort descending
    std::vector<size_t> indices(d);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return D(i, i) > D(j, j);
    });

    res.eigenvalues_ = VectorRR(d);
    res.eigenvectors_ = MatrixRR(d, d);
    res.rank_ = 0;
    BigReal rtol(rank_tol);

    for (size_t k = 0; k < d; ++k) {
        size_t idx = indices[k];
        BigReal ev = D(idx, idx);
        if (ev < BigReal(0.0) && ev >= -rtol) {
            ev = BigReal(0.0);
        }
        res.eigenvalues_[k] = ev;
        if (ev > rtol) {
            res.rank_++;
        }
        for (size_t r = 0; r < d; ++r) {
            res.eigenvectors_(r, k) = V(r, idx);
        }
    }

    return res;
}

MatrixRR SymmetricEigendecompositionRR::matrix_sqrt() const {
    if (dim_ == 0) return MatrixRR(0, 0);

    MatrixRR res(dim_, dim_, BigReal(0.0));
    for (size_t k = 0; k < rank_; ++k) {
        BigReal sqrt_lambda = eigenvalues_[k].sqrt();
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                res(i, j) += sqrt_lambda * eigenvectors_(i, k) * eigenvectors_(j, k);
            }
        }
    }
    return res;
}

MatrixRR SymmetricEigendecompositionRR::matrix_inv_sqrt() const {
    if (dim_ == 0) return MatrixRR(0, 0);

    MatrixRR res(dim_, dim_, BigReal(0.0));
    for (size_t k = 0; k < rank_; ++k) {
        BigReal inv_sqrt_lambda = BigReal(1.0) / eigenvalues_[k].sqrt();
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                res(i, j) += inv_sqrt_lambda * eigenvectors_(i, k) * eigenvectors_(j, k);
            }
        }
    }
    return res;
}

BigReal SymmetricEigendecompositionRR::restricted_determinant() const {
    if (dim_ == 0 || rank_ == 0) return BigReal(1.0);
    BigReal log_rdet = log_restricted_determinant();
    return log_rdet.exp();
}

BigReal SymmetricEigendecompositionRR::log_restricted_determinant() const {
    if (dim_ == 0 || rank_ == 0) return BigReal(0.0);
    BigReal sum_log(0.0);
    for (size_t k = 0; k < rank_; ++k) {
        sum_log += eigenvalues_[k].log();
    }
    return sum_log;
}

} // namespace dbdd
