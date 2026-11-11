#pragma once

// Paper Reference: Dachman-Soled et al. Section 3.2, Eq. (10) / Chen-Nguyen 2011 Eq. (1)
// Mathematical Purpose: Computes the root-Hermite factor delta_beta as a function of the BKZ block size beta.
// For beta >= 50, evaluates the Gaussian heuristic formula:
//     delta_beta = ( (pi * beta)^(1/beta) * (beta / (2 * pi * e)) )^(1 / (2*beta - 2))
// For beta in [2, 50), evaluates standard literature interpolation (Gama-Nguyen / Chen-Nguyen / Albrecht)
// ensuring delta_beta > 1 and strict monotonic decrease for all beta >= 2.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/real.hpp"

namespace dbdd {

class RootHermite {
public:
    /**
     * @brief Computes root-Hermite factor delta_beta in double precision.
     * @param beta BKZ block size (must be >= 2.0).
     * @return delta_beta > 1.0.
     */
    [[nodiscard]] static double delta(double beta);

    /**
     * @brief Computes natural logarithm ln(delta_beta) in double precision.
     * Evaluated directly in the log domain for numerical stability.
     * @param beta BKZ block size (must be >= 2.0).
     * @return ln(delta_beta) > 0.0.
     */
    [[nodiscard]] static double log_delta(double beta);

    /**
     * @brief Computes root-Hermite factor delta_beta in arbitrary-precision MPFR.
     * @param beta BKZ block size (must be >= 2.0).
     * @param prec Working precision in bits (0 defaults to current global precision).
     * @return BigReal delta_beta > 1.0.
     */
    [[nodiscard]] static BigReal delta_rr(const BigReal& beta, prec_t prec = 0);

    /**
     * @brief Computes ln(delta_beta) in arbitrary-precision MPFR.
     * @param beta BKZ block size (must be >= 2.0).
     * @param prec Working precision in bits (0 defaults to current global precision).
     * @return BigReal ln(delta_beta) > 0.0.
     */
    [[nodiscard]] static BigReal log_delta_rr(const BigReal& beta, prec_t prec = 0);

    /**
     * @brief Pure Gaussian heuristic formula evaluated without low-beta interpolation.
     * Valid for beta >= 50 (or beta > 1.0 mathematically).
     */
    [[nodiscard]] static double delta_gh(double beta);
    [[nodiscard]] static double log_delta_gh(double beta);
    [[nodiscard]] static BigReal delta_gh_rr(const BigReal& beta, prec_t prec = 0);
    [[nodiscard]] static BigReal log_delta_gh_rr(const BigReal& beta, prec_t prec = 0);

private:
    // Small beta interpolation table from literature: (beta, delta)
    struct TableEntry {
        double beta;
        double delta;
    };

    static constexpr TableEntry kSmallTable[] = {
        {2.0, 1.02190},
        {5.0, 1.01862},
        {10.0, 1.01616},
        {15.0, 1.01485},
        {20.0, 1.01420},
        {25.0, 1.01342},
        {28.0, 1.01331},
        {40.0, 1.01295},
        {50.0, 1.012064863554853}
    };
    static constexpr size_t kSmallTableSize = sizeof(kSmallTable) / sizeof(kSmallTable[0]);
};

} // namespace dbdd

