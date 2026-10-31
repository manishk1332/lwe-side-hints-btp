#include "dbdd/security/root_hermite.hpp"
#include "dbdd/arithmetic/precision.hpp"
#include "dbdd/core/exceptions.hpp"
#include <cmath>
#include <numbers>

namespace dbdd {

double RootHermite::log_delta_gh(double beta) {
    if (beta <= 1.0) {
        throw InvalidParameterException("RootHermite::log_delta_gh: beta must be > 1.0");
    }
    constexpr double pi = std::numbers::pi;
    constexpr double e = std::numbers::e;
    const double ln_two_pi_e = std::log(2.0 * pi * e);

    const double ln_pi_beta = std::log(pi * beta);
    const double ln_beta = std::log(beta);
    const double term = (ln_pi_beta / beta) + ln_beta - ln_two_pi_e;
    return term / (2.0 * beta - 2.0);
}

double RootHermite::delta_gh(double beta) {
    return std::exp(log_delta_gh(beta));
}

double RootHermite::delta(double beta) {
    if (beta < 2.0) {
        throw InvalidParameterException("RootHermite::delta: BKZ block size beta must be >= 2.0");
    }
    if (beta >= 50.0) {
        return delta_gh(beta);
    }

    // Piecewise linear interpolation on empirical small-beta table
    for (size_t i = 0; i < kSmallTableSize - 1; ++i) {
        const auto& p0 = kSmallTable[i];
        const auto& p1 = kSmallTable[i + 1];
        if (beta >= p0.beta && beta <= p1.beta) {
            const double t = (beta - p0.beta) / (p1.beta - p0.beta);
            return p0.delta + t * (p1.delta - p0.delta);
        }
    }
    return delta_gh(50.0);
}

double RootHermite::log_delta(double beta) {
    if (beta < 2.0) {
        throw InvalidParameterException("RootHermite::log_delta: BKZ block size beta must be >= 2.0");
    }
    if (beta >= 50.0) {
        return log_delta_gh(beta);
    }
    return std::log(delta(beta));
}

BigReal RootHermite::log_delta_gh_rr(const BigReal& beta, prec_t prec) {
    if (prec == 0) {
        prec = PrecisionManager::get_default_precision();
    }
    BigReal one(1.0, prec);
    if (beta <= one) {
        throw InvalidParameterException("RootHermite::log_delta_gh_rr: beta must be > 1.0");
    }

    BigReal pi = BigReal::pi(prec);
    BigReal e = BigReal::e(prec);
    BigReal two(2.0, prec);

    BigReal pi_beta = pi * beta;
    BigReal ln_pi_beta = pi_beta.log();
    BigReal ln_beta = beta.log();
    BigReal two_pi_e = two * pi * e;
    BigReal ln_two_pi_e = two_pi_e.log();

    BigReal term = (ln_pi_beta / beta) + ln_beta - ln_two_pi_e;
    BigReal denom = two * beta - two;
    return term / denom;
}

BigReal RootHermite::delta_gh_rr(const BigReal& beta, prec_t prec) {
    return log_delta_gh_rr(beta, prec).exp();
}

BigReal RootHermite::delta_rr(const BigReal& beta, prec_t prec) {
    if (prec == 0) {
        prec = PrecisionManager::get_default_precision();
    }
    BigReal two(2.0, prec);
    if (beta < two) {
        throw InvalidParameterException("RootHermite::delta_rr: BKZ block size beta must be >= 2.0");
    }

    BigReal fifty(50.0, prec);
    if (beta >= fifty) {
        return delta_gh_rr(beta, prec);
    }

    // Convert to double for table interpolation, then construct BigReal with requested precision
    double beta_d = beta.to_double();
    double d_val = delta(beta_d);
    return BigReal(d_val, prec);
}

BigReal RootHermite::log_delta_rr(const BigReal& beta, prec_t prec) {
    if (prec == 0) {
        prec = PrecisionManager::get_default_precision();
    }
    BigReal two(2.0, prec);
    if (beta < two) {
        throw InvalidParameterException("RootHermite::log_delta_rr: BKZ block size beta must be >= 2.0");
    }

    BigReal fifty(50.0, prec);
    if (beta >= fifty) {
        return log_delta_gh_rr(beta, prec);
    }

    return delta_rr(beta, prec).log();
}

} // namespace dbdd
