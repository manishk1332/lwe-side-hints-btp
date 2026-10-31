#include "dbdd/hints/worthiness.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/security/lightweight_predictor.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <limits>

namespace dbdd {

namespace {

VectorDouble to_double_vector(const VectorZZ& v) {
    VectorDouble res(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        res[i] = v[i].to_double();
    }
    return res;
}

} // anonymous namespace

double ShortVectorWorthinessAnalyzer::gram_determinant(const MatrixDouble& V) {
    const size_t k = V.rows();
    if (k == 0) {
        return 1.0;
    }
    if (k == 1) {
        double norm_sq = 0.0;
        for (size_t j = 0; j < V.cols(); ++j) {
            norm_sq += V(0, j) * V(0, j);
        }
        return norm_sq;
    }

    MatrixDouble G(k, k, 0.0);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = i; j < k; ++j) {
            double dot = 0.0;
            for (size_t c = 0; c < V.cols(); ++c) {
                dot += V(i, c) * V(j, c);
            }
            G(i, j) = dot;
            G(j, i) = dot;
        }
    }
    return G.determinant();
}

double ShortVectorWorthinessAnalyzer::covariance_determinant(
    const MatrixDouble& V,
    const MatrixDouble& Sigma) {
    const size_t k = V.rows();
    const size_t m = V.cols();
    if (Sigma.rows() != m || Sigma.cols() != m) {
        throw DimensionMismatchException("ShortVectorWorthinessAnalyzer: V cols and Sigma dimension mismatch");
    }
    if (k == 0) {
        return 1.0;
    }
    if (k == 1) {
        return covariance_form(V.row(0), Sigma);
    }

    // Compute Q = V * Sigma * V^T in R^{k x k}
    // First V_Sigma = V * Sigma (k x m)
    MatrixDouble V_Sigma(k, m, 0.0);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            for (size_t c = 0; c < m; ++c) {
                sum += V(i, c) * Sigma(c, j);
            }
            V_Sigma(i, j) = sum;
        }
    }

    // Then Q = V_Sigma * V^T (k x k)
    MatrixDouble Q(k, k, 0.0);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = i; j < k; ++j) {
            double sum = 0.0;
            for (size_t c = 0; c < m; ++c) {
                sum += V_Sigma(i, c) * V(j, c);
            }
            Q(i, j) = sum;
            Q(j, i) = sum;
        }
    }
    return Q.determinant();
}

double ShortVectorWorthinessAnalyzer::gram_determinant(const VectorDouble& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }
    return sum;
}

double ShortVectorWorthinessAnalyzer::gram_determinant(const VectorZZ& v) {
    return gram_determinant(to_double_vector(v));
}

double ShortVectorWorthinessAnalyzer::covariance_form(
    const VectorDouble& v,
    const MatrixDouble& Sigma) {
    const size_t m = v.size();
    if (Sigma.rows() != m || Sigma.cols() != m) {
        throw DimensionMismatchException("ShortVectorWorthinessAnalyzer: v size and Sigma dimension mismatch");
    }

    double qform = 0.0;
    for (size_t i = 0; i < m; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < m; ++j) {
            row_sum += v[j] * Sigma(i, j);
        }
        qform += v[i] * row_sum;
    }
    return qform;
}

double ShortVectorWorthinessAnalyzer::covariance_form(
    const VectorZZ& v,
    const MatrixDouble& Sigma) {
    return covariance_form(to_double_vector(v), Sigma);
}

double ShortVectorWorthinessAnalyzer::variance_score(
    const MatrixDouble& V,
    const MatrixDouble& Sigma) {
    const double det_g = gram_determinant(V);
    if (det_g <= 1e-15) {
        return 0.0;
    }
    const double det_q = covariance_determinant(V, Sigma);
    return det_q / det_g;
}

double ShortVectorWorthinessAnalyzer::variance_score(
    const VectorDouble& v,
    const MatrixDouble& Sigma) {
    const double g = gram_determinant(v);
    if (g <= 1e-15) {
        return 0.0;
    }
    const double q = covariance_form(v, Sigma);
    return q / g;
}

double ShortVectorWorthinessAnalyzer::variance_score(
    const VectorZZ& v,
    const MatrixDouble& Sigma) {
    return variance_score(to_double_vector(v), Sigma);
}

double ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(
    const MatrixDouble& V,
    const MatrixDouble& Sigma) {
    const double det_q = covariance_determinant(V, Sigma);
    if (det_q <= 1e-15) {
        return std::numeric_limits<double>::infinity();
    }
    const double det_g = gram_determinant(V);
    return (det_g * det_g) / det_q;
}

double ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(
    const VectorDouble& v,
    const MatrixDouble& Sigma) {
    const double q = covariance_form(v, Sigma);
    if (q <= 1e-15) {
        return std::numeric_limits<double>::infinity();
    }
    const double g = gram_determinant(v);
    return (g * g) / q;
}

double ShortVectorWorthinessAnalyzer::hardness_proxy_ratio(
    const VectorZZ& v,
    const MatrixDouble& Sigma) {
    return hardness_proxy_ratio(to_double_vector(v), Sigma);
}

double ShortVectorWorthinessAnalyzer::log_hardness_delta(
    const MatrixDouble& V,
    const MatrixDouble& Sigma) {
    const double det_g = gram_determinant(V);
    const double det_q = covariance_determinant(V, Sigma);
    if (det_g <= 0.0 || det_q <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 2.0 * std::log(det_g) - std::log(det_q);
}

double ShortVectorWorthinessAnalyzer::log_hardness_delta(
    const VectorDouble& v,
    const MatrixDouble& Sigma) {
    const double g = gram_determinant(v);
    const double q = covariance_form(v, Sigma);
    if (g <= 0.0 || q <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 2.0 * std::log(g) - std::log(q);
}

double ShortVectorWorthinessAnalyzer::log_hardness_delta(
    const VectorZZ& v,
    const MatrixDouble& Sigma) {
    return log_hardness_delta(to_double_vector(v), Sigma);
}

bool ShortVectorWorthinessAnalyzer::is_worthy_proxy(
    const MatrixDouble& Sigma,
    const VectorDouble& v) {
    const double q = covariance_form(v, Sigma);
    const double g = gram_determinant(v);
    if (q <= 1e-15 || g <= 1e-15) {
        return false;
    }
    // Hardness proxy ratio < 1.0 <=> (||v||^2)^2 < v Sigma v^T
    return (g * g) < q;
}

bool ShortVectorWorthinessAnalyzer::is_worthy_proxy(
    const MatrixDouble& Sigma,
    const VectorZZ& v) {
    return is_worthy_proxy(Sigma, to_double_vector(v));
}

bool ShortVectorWorthinessAnalyzer::is_worthy_beta(
    const DBDDInstance& dbdd,
    const VectorZZ& v) {
    DBDDStateStatistics stats = dbdd.to_state_statistics();
    return is_worthy_beta(stats, to_double_vector(v));
}

bool ShortVectorWorthinessAnalyzer::is_worthy_beta(
    const DBDDStateStatistics& stats,
    const VectorDouble& v) {
    DBDDStateStatistics copy = stats;
    auto est_before = copy.estimate_attack_gsa();

    copy.integrate_short_vector_hint(v);
    auto est_after = copy.estimate_attack_gsa();

    // Worthy if beta decreases, or beta is equal and delta increases
    return (est_after.beta < est_before.beta) ||
           (est_after.beta == est_before.beta && est_after.delta_beta > est_before.delta_beta);
}

} // namespace dbdd

