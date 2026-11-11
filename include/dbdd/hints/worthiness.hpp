#pragma once

// Paper Reference: Dachman-Soled et al. Section 4.5 / Equations (30)-(31)
// Mathematical Purpose: Analyzes the worthiness of candidate short vectors V in Lambda,
// evaluating the variance ratio det(V Sigma V^T) / det(V V^T) and the hardness proxy
// change rdet(Sigma') / Vol(Lambda')^2 = (rdet(Sigma) / Vol(Lambda)^2) * (det(V V^T)^2 / det(V Sigma V^T)).

#include "dbdd/core/types.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/security/security_estimate.hpp"

#include <vector>
#include <cstddef>

namespace dbdd {

class DBDDInstance;
class DBDDStateStatistics;

/**
 * @brief Evaluates algebraic worthiness scores and hardness proxy changes for short-vector hints.
 */
class ShortVectorWorthinessAnalyzer {
public:
    // =========================================================================
    // Gram Determinants & Covariance Quadratic Forms
    // =========================================================================

    /**
     * @brief Computes Gram determinant det(V * V^T) for a matrix V in R^{k x m}.
     */
    [[nodiscard]] static double gram_determinant(const MatrixDouble& V);

    /**
     * @brief Computes covariance determinant det(V * Sigma * V^T) for V in R^{k x m} and PSD Sigma in R^{m x m}.
     */
    [[nodiscard]] static double covariance_determinant(
        const MatrixDouble& V,
        const MatrixDouble& Sigma);

    /**
     * @brief Computes squared norm ||v||^2 for a single vector.
     */
    [[nodiscard]] static double gram_determinant(const VectorDouble& v);
    [[nodiscard]] static double gram_determinant(const VectorZZ& v);

    /**
     * @brief Computes quadratic form v * Sigma * v^T for a single vector.
     */
    [[nodiscard]] static double covariance_form(
        const VectorDouble& v,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double covariance_form(
        const VectorZZ& v,
        const MatrixDouble& Sigma);

    // =========================================================================
    // Analytical Scores (Equations 30 & 31)
    // =========================================================================

    /**
     * @brief Computes the variance score: det(V * Sigma * V^T) / det(V * V^T) (Eq. 30).
     * Represents the factor by which the restricted determinant ratio rdet(Sigma) / rdet(Sigma') decreases.
     */
    [[nodiscard]] static double variance_score(
        const MatrixDouble& V,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double variance_score(
        const VectorDouble& v,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double variance_score(
        const VectorZZ& v,
        const MatrixDouble& Sigma);

    /**
     * @brief Computes the hardness proxy ratio: (det(V * V^T))^2 / det(V * Sigma * V^T) (Eq. 31).
     * Represents the multiplicative change in (rdet(Sigma') / Vol(Lambda')^2) / (rdet(Sigma) / Vol(Lambda)^2).
     * Values < 1.0 indicate a reduction in attack hardness proxy (i.e. the hint is worthy).
     */
    [[nodiscard]] static double hardness_proxy_ratio(
        const MatrixDouble& V,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double hardness_proxy_ratio(
        const VectorDouble& v,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double hardness_proxy_ratio(
        const VectorZZ& v,
        const MatrixDouble& Sigma);

    /**
     * @brief Computes the log-domain hardness proxy change:
     * Delta ln(proxy) = 2 * ln(det(V V^T)) - ln(det(V Sigma V^T)).
     * Negative values indicate worthiness.
     */
    [[nodiscard]] static double log_hardness_delta(
        const MatrixDouble& V,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double log_hardness_delta(
        const VectorDouble& v,
        const MatrixDouble& Sigma);
    [[nodiscard]] static double log_hardness_delta(
        const VectorZZ& v,
        const MatrixDouble& Sigma);

    // =========================================================================
    // Worthiness Predicates
    // =========================================================================

    /**
     * @brief Evaluates whether candidate vector v is worthy according to the analytical proxy (Eq. 31).
     * Returns true if hardness_proxy_ratio(v, Sigma) < 1.0 (equivalently, v Sigma v^T > ||v||^4).
     */
    [[nodiscard]] static bool is_worthy_proxy(
        const MatrixDouble& Sigma,
        const VectorDouble& v);
    [[nodiscard]] static bool is_worthy_proxy(
        const MatrixDouble& Sigma,
        const VectorZZ& v);

    /**
     * @brief Evaluates whether candidate vector v is worthy according to concrete BKZ security reduction.
     * Compares estimated beta before and after integrating v: worthy if beta decreases,
     * or if beta is unchanged and delta increases (matching Sage undo_if_unworthy).
     */
    [[nodiscard]] static bool is_worthy_beta(
        const DBDDInstance& dbdd,
        const VectorZZ& v);

    [[nodiscard]] static bool is_worthy_beta(
        const DBDDStateStatistics& stats,
        const VectorDouble& v);
};

} // namespace dbdd

