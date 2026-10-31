#include "dbdd/security/gsa.hpp"
#include "dbdd/security/root_hermite.hpp"
#include "dbdd/core/exceptions.hpp"
#include <cmath>

namespace dbdd {

double GSAModel::log_gs_norm(double beta, size_t d, double ln_vol, size_t i) {
    if (d == 0) {
        throw DimensionMismatchException("GSAModel::log_gs_norm: dimension d must be >= 1");
    }
    if (i == 0 || i > d) {
        throw IndexOutOfBoundsException("GSAModel::log_gs_norm: index i must be between 1 and d");
    }
    const double ln_delta = RootHermite::log_delta(beta);
    const double exponent = static_cast<double>(d) - 2.0 * static_cast<double>(i) + 1.0;
    return exponent * ln_delta + (ln_vol / static_cast<double>(d));
}

double GSAModel::gs_norm(double beta, size_t d, double ln_vol, size_t i) {
    return std::exp(log_gs_norm(beta, d, ln_vol, i));
}

std::vector<double> GSAModel::log_profile(double beta, size_t d, double ln_vol) {
    if (d == 0) {
        throw DimensionMismatchException("GSAModel::log_profile: dimension d must be >= 1");
    }
    std::vector<double> res;
    res.reserve(d);
    for (size_t i = 1; i <= d; ++i) {
        res.push_back(log_gs_norm(beta, d, ln_vol, i));
    }
    return res;
}

std::vector<double> GSAModel::profile(double beta, size_t d, double ln_vol) {
    if (d == 0) {
        throw DimensionMismatchException("GSAModel::profile: dimension d must be >= 1");
    }
    std::vector<double> res;
    res.reserve(d);
    for (size_t i = 1; i <= d; ++i) {
        res.push_back(gs_norm(beta, d, ln_vol, i));
    }
    return res;
}

double GSAModel::slope(double beta) {
    return std::exp(log_slope(beta));
}

double GSAModel::log_slope(double beta) {
    return -2.0 * RootHermite::log_delta(beta);
}

double GSAModel::log_projected_block_head(double beta, size_t d, double ln_vol) {
    if (d == 0) {
        throw DimensionMismatchException("GSAModel::log_projected_block_head: dimension d must be >= 1");
    }
    const double ln_delta = RootHermite::log_delta(beta);
    const double exponent = 2.0 * beta - static_cast<double>(d) - 1.0;
    return exponent * ln_delta + (ln_vol / static_cast<double>(d));
}

double GSAModel::log_projected_target_norm(double beta, size_t d, double s_norm) {
    if (beta <= 0.0) {
        throw InvalidParameterException("GSAModel::log_projected_target_norm: beta must be > 0");
    }
    if (d == 0) {
        throw DimensionMismatchException("GSAModel::log_projected_target_norm: dimension d must be >= 1");
    }

    if (s_norm <= 0.0) {
        // Isotropic assumption: ||s|| = sqrt(d), so sqrt(beta / d) * ||s|| = sqrt(beta)
        return 0.5 * std::log(beta);
    }

    // General case: sqrt(beta / d) * ||s||
    return 0.5 * std::log(beta / static_cast<double>(d)) + std::log(s_norm);
}

bool GSAModel::intersects(double beta, size_t d, double ln_vol, double s_norm) {
    return log_projected_block_head(beta, d, ln_vol) >= log_projected_target_norm(beta, d, s_norm);
}

} // namespace dbdd
