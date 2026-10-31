#include "dbdd/security/security_estimator.hpp"
#include <sstream>
#include <iomanip>

namespace dbdd {

std::string SecurityComparison::to_string() const {
    std::ostringstream oss;
    oss << "=== Security Estimation Comparison ===\n";
    oss << std::left << std::setw(22) << "Metric"
        << std::setw(24) << "GSA-Intersect"
        << std::setw(26) << "Probabilistic-Simulation"
        << "Difference (Prob - GSA)\n";
    oss << std::string(85, '-') << "\n";

    oss << std::left << std::setw(22) << "Continuous Beta"
        << std::setw(24) << std::fixed << std::setprecision(4) << gsa.beta_continuous
        << std::setw(26) << probabilistic.beta_continuous
        << (delta_beta >= 0 ? "+" : "") << delta_beta << "\n";

    oss << std::left << std::setw(22) << "Integer bikz"
        << std::setw(24) << gsa.beta
        << std::setw(26) << probabilistic.beta
        << (delta_bikz >= 0 ? "+" : "") << delta_bikz << "\n";

    oss << std::left << std::setw(22) << "Root-Hermite delta"
        << std::setw(24) << std::setprecision(8) << gsa.delta_beta
        << std::setw(26) << probabilistic.delta_beta
        << (probabilistic.delta_beta - gsa.delta_beta) << "\n";

    oss << std::left << std::setw(22) << "Success Probability"
        << std::setw(24) << std::setprecision(4) << gsa.success_probability
        << std::setw(26) << probabilistic.success_probability
        << "\n";

    oss << std::left << std::setw(22) << "CoreSVP Classical (bits)"
        << std::setw(24) << std::setprecision(2) << gsa.cost_bits(CostModel::CoreSVPClassical)
        << std::setw(26) << probabilistic.cost_bits(CostModel::CoreSVPClassical)
        << "\n";

    oss << std::left << std::setw(22) << "CoreSVP Quantum (bits)"
        << std::setw(24) << std::setprecision(2) << gsa.cost_bits(CostModel::CoreSVPQuantum)
        << std::setw(26) << probabilistic.cost_bits(CostModel::CoreSVPQuantum)
        << "\n";

    if (!probabilistic.warning_flags.empty()) {
        oss << "\nWarnings:\n";
        for (const auto& w : probabilistic.warning_flags) {
            oss << "  - [Probabilistic] " << w << "\n";
        }
    }

    return oss.str();
}

// =============================================================================
// GSA-Intersect Facade
// =============================================================================

SecurityEstimate SecurityEstimator::gsa_intersect(
    size_t dimension,
    double ln_volume,
    double beta_min,
    double beta_max) {
    return GSAIntersect::estimate(dimension, ln_volume, beta_min, beta_max);
}

SecurityEstimate SecurityEstimator::gsa_intersect(
    const UsvpInstance& usvp,
    double beta_min,
    double beta_max) {
    return GSAIntersect::estimate(usvp, beta_min, beta_max);
}

SecurityEstimate SecurityEstimator::gsa_intersect(
    const DBDDInstance& dbdd,
    double beta_min,
    double beta_max,
    double rank_tol) {
    return GSAIntersect::estimate(dbdd, beta_min, beta_max, rank_tol);
}

SecurityEstimate SecurityEstimator::gsa_intersect(
    const DBDDStateStatistics& stats,
    double beta_min,
    double beta_max,
    double rank_tol) {
    const double ln_vol_iso = stats.isotropized_log_volume(rank_tol);
    return GSAIntersect::estimate(stats.dimension(), ln_vol_iso, beta_min, beta_max);
}

// =============================================================================
// Probabilistic Simulation Facade
// =============================================================================

SecurityEstimate SecurityEstimator::probabilistic(
    size_t dimension,
    double ln_volume,
    const ProbabilisticOptions& options) {
    return ProbabilisticEstimator::estimate(dimension, ln_volume, options);
}

SecurityEstimate SecurityEstimator::probabilistic(
    const UsvpInstance& usvp,
    const ProbabilisticOptions& options) {
    return ProbabilisticEstimator::estimate(usvp, options);
}

SecurityEstimate SecurityEstimator::probabilistic(
    const DBDDInstance& dbdd,
    const ProbabilisticOptions& options) {
    return ProbabilisticEstimator::estimate(dbdd, options);
}

SecurityEstimate SecurityEstimator::probabilistic(
    const DBDDStateStatistics& stats,
    const ProbabilisticOptions& options) {
    return ProbabilisticEstimator::estimate(stats, options);
}

// =============================================================================
// Comparative Security Analysis
// =============================================================================

SecurityComparison SecurityEstimator::compare(
    size_t dimension,
    double ln_volume,
    const ProbabilisticOptions& options,
    double beta_min,
    double beta_max) {
    SecurityComparison comp;
    comp.gsa = gsa_intersect(dimension, ln_volume, beta_min, beta_max);
    comp.probabilistic = probabilistic(dimension, ln_volume, options);
    comp.delta_beta = comp.probabilistic.beta_continuous - comp.gsa.beta_continuous;
    comp.delta_bikz = static_cast<int>(comp.probabilistic.beta) - static_cast<int>(comp.gsa.beta);
    return comp;
}

SecurityComparison SecurityEstimator::compare(
    const UsvpInstance& usvp,
    const ProbabilisticOptions& options,
    double beta_min,
    double beta_max) {
    return compare(usvp.dimension(), usvp.log_volume().to_double(), options, beta_min, beta_max);
}

SecurityComparison SecurityEstimator::compare(
    const DBDDInstance& dbdd,
    const ProbabilisticOptions& options,
    double beta_min,
    double beta_max,
    double rank_tol) {
    SecurityComparison comp;
    comp.gsa = gsa_intersect(dbdd, beta_min, beta_max, rank_tol);
    ProbabilisticOptions opts = options;
    opts.rank_tol = rank_tol;
    comp.probabilistic = probabilistic(dbdd, opts);
    comp.delta_beta = comp.probabilistic.beta_continuous - comp.gsa.beta_continuous;
    comp.delta_bikz = static_cast<int>(comp.probabilistic.beta) - static_cast<int>(comp.gsa.beta);
    return comp;
}

SecurityComparison SecurityEstimator::compare(
    const DBDDStateStatistics& stats,
    const ProbabilisticOptions& options,
    double beta_min,
    double beta_max,
    double rank_tol) {
    SecurityComparison comp;
    comp.gsa = gsa_intersect(stats, beta_min, beta_max, rank_tol);
    ProbabilisticOptions opts = options;
    opts.rank_tol = rank_tol;
    comp.probabilistic = probabilistic(stats, opts);
    comp.delta_beta = comp.probabilistic.beta_continuous - comp.gsa.beta_continuous;
    comp.delta_bikz = static_cast<int>(comp.probabilistic.beta) - static_cast<int>(comp.gsa.beta);
    return comp;
}

} // namespace dbdd

