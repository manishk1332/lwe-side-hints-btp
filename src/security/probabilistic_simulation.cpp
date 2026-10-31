#include "dbdd/security/probabilistic_simulation.hpp"
#include "dbdd/security/root_hermite.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace dbdd {

namespace {

// Evaluates regularized lower incomplete gamma P(a, z) = gamma(a, z) / Gamma(a)
// using series expansion for z < a + 1 and continued fraction for z >= a + 1.
double regularized_gamma_lower(double a, double z) {
    if (z <= 0.0) {
        return 0.0;
    }

    if (z < a + 1.0) {
        // Series expansion: P(a, z) = (z^a * e^-z / Gamma(a)) * sum_{n=0}^infty (z^n / (a*(a+1)*...*(a+n)))
        double sum = 1.0 / a;
        double term = sum;
        for (int n = 1; n < 400; ++n) {
            term *= z / (a + static_cast<double>(n));
            sum += term;
            if (std::abs(term) < std::abs(sum) * 1e-16) {
                break;
            }
        }
        double ln_prefix = a * std::log(z) - z - std::lgamma(a);
        return std::clamp(std::exp(ln_prefix) * sum, 0.0, 1.0);
    } else {
        // Legendre continued fraction for Q(a, z) = 1 - P(a, z) via modified Lentz's method
        double b = z + 1.0 - a;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i < 400; ++i) {
            double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
            b += 2.0;
            d = an * d + b;
            if (std::abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (std::abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (std::abs(del - 1.0) < 1e-16) {
                break;
            }
        }
        double ln_prefix = a * std::log(z) - z - std::lgamma(a);
        double q = std::exp(ln_prefix) * h;
        return std::clamp(1.0 - q, 0.0, 1.0);
    }
}

} // anonymous namespace

SecurityEstimate ProbabilisticResult::to_security_estimate(size_t dimension, double ln_volume) const {
    SecurityEstimate est;
    est.dimension = dimension;
    est.log_volume = ln_volume;
    est.beta_continuous = average_beta;
    est.beta = integer_beta;
    est.delta_beta = delta;
    est.solved = solvable;
    est.method = "Probabilistic-Simulation";
    est.success_probability = cumulative_probability;
    est.warning_flags = warnings;

    std::ostringstream oss;
    oss << "tours_steps=" << steps.size()
        << ", rem_prob=" << std::scientific << std::setprecision(3) << remaining_probability;
    est.confidence_metadata = oss.str();

    return est;
}

double ProbabilisticEstimator::chi_squared_cdf(size_t df, double x) {
    if (df == 0) {
        throw InvalidParameterException("ProbabilisticEstimator::chi_squared_cdf: df must be >= 1");
    }
    if (x <= 0.0) {
        return 0.0;
    }

    if (df == 1) {
        return std::erf(std::sqrt(0.5 * x));
    }
    if (df == 2) {
        return 1.0 - std::exp(-0.5 * x);
    }

    const double a = 0.5 * static_cast<double>(df);
    const double z = 0.5 * x;
    return regularized_gamma_lower(a, z);
}

double ProbabilisticEstimator::chi_squared_pdf(size_t df, double x) {
    if (df == 0) {
        throw InvalidParameterException("ProbabilisticEstimator::chi_squared_pdf: df must be >= 1");
    }
    if (x <= 0.0) {
        return 0.0;
    }

    const double k = static_cast<double>(df);
    const double ln_pdf = -0.5 * k * std::log(2.0) - std::lgamma(0.5 * k) + (0.5 * k - 1.0) * std::log(x) - 0.5 * x;
    if (ln_pdf < -700.0) {
        return 0.0;
    }
    return std::exp(ln_pdf);
}

double ProbabilisticEstimator::conditional_chi_squared(
    size_t d1, size_t d2, double lt, double l2, size_t steps) {
    if (d1 == 0 && d2 == 0) {
        throw InvalidParameterException("ProbabilisticEstimator::conditional_chi_squared: dimensions cannot both be 0");
    }
    if (lt <= 0.0 || l2 <= 0.0) {
        return 0.0;
    }

    if (d1 == 0) {
        if (lt >= l2) {
            return 1.0;
        }
        double pe2 = chi_squared_cdf(d2, l2);
        return (pe2 <= 1e-15) ? 1.0 : std::clamp(chi_squared_cdf(d2, lt) / pe2, 0.0, 1.0);
    }
    if (d2 == 0) {
        return chi_squared_cdf(d1, lt);
    }

    const double pe2 = chi_squared_cdf(d2, l2);
    if (pe2 <= 1e-15) {
        // High dimension or small l2 leads to numerical underflow in CDF(l2).
        // Following Albrecht et al. Section 3.4 and utils.sage, we assume lifting is successful.
        return 1.0;
    }

    if (steps == 0) {
        steps = 5 * (d1 + d2);
    }
    if (steps < 10) {
        steps = 10;
    }

    const double max_u = std::min(l2, lt);
    const double h = max_u / static_cast<double>(steps);

    double proba = 0.0;
    for (size_t i = 0; i < steps; ++i) {
        const double u_min = static_cast<double>(i) * h;
        const double u_mid = (static_cast<double>(i) + 0.5) * h;
        const double u_max = static_cast<double>(i + 1) * h;

        if (u_mid >= lt) {
            continue;
        }

        const double pc2 = (chi_squared_cdf(d2, u_max) - chi_squared_cdf(d2, u_min)) / pe2;
        const double pe1 = chi_squared_cdf(d1, lt - u_mid);
        proba += pc2 * pe1;
    }

    return std::clamp(proba, 0.0, 1.0);
}

double ProbabilisticEstimator::single_tour_detection_probability(
    const std::vector<double>& log_profile,
    size_t beta,
    const ProbabilisticOptions& options) {
    const size_t d = log_profile.size();
    if (d < 2) {
        throw DimensionMismatchException("ProbabilisticEstimator: profile dimension must be >= 2");
    }
    if (beta < 2 || beta > d) {
        throw InvalidParameterException("ProbabilisticEstimator: beta must be in [2, dimension]");
    }

    // Tail block position: i = d - beta
    const size_t i_tail = d - beta;
    const double x_tail = std::exp(2.0 * log_profile[i_tail]);

    double proba = chi_squared_cdf(beta, x_tail);

    if (!options.ignore_lift_proba) {
        // Consecutive pull-back lifting events across blocks:
        // i_j = d - j * (beta - 1) - 1 for j = 2, 3, ...
        for (size_t j = 2; j < (d / (beta - 1) + 2); ++j) {
            const size_t span = j * (beta - 1) + 1;
            if (span > d) {
                break;
            }
            const size_t i = d - span;
            const size_t i_prev = i + (beta - 1);

            const double xt = std::exp(2.0 * log_profile[i]);

            if (options.lift_union_bound) {
                proba = std::min(proba, chi_squared_cdf(d - i, xt));
            } else {
                const double x2 = std::exp(2.0 * log_profile[i_prev]);
                const size_t d1 = beta - 1;
                const size_t d2 = options.legacy_sage_d2_precedence
                                      ? (d - i + (beta - 1))
                                      : (d - i_prev);

                proba *= conditional_chi_squared(d1, d2, xt, x2, options.integration_steps);
            }
        }
    }

    return std::clamp(proba, 0.0, 1.0);
}

double ProbabilisticEstimator::estimate_probability(
    size_t beta,
    size_t tours,
    size_t dimension,
    double ln_volume,
    const ProbabilisticOptions& options) {
    if (dimension < 2) {
        throw DimensionMismatchException("ProbabilisticEstimator: dimension must be >= 2");
    }
    if (beta < 2 || beta > dimension) {
        throw InvalidParameterException("ProbabilisticEstimator: beta must be in [2, dimension]");
    }
    if (tours == 0) {
        return 0.0;
    }

    BKZSimulatorParameters params;
    params.block_size = beta;
    params.max_tours = tours;
    params.pruning_proba = options.pruning_proba;
    params.model = options.simulation_model;
    params.record_tours = true;

    auto sim_result = BKZSimulator::simulate(dimension, ln_volume, params);

    double fail_prob = 1.0;
    for (size_t t = 0; t < sim_result.tour_log_profiles.size(); ++t) {
        double p_tour = single_tour_detection_probability(sim_result.tour_log_profiles[t], beta, options);
        fail_prob *= (1.0 - p_tour);
    }

    // If simulation converged before max_tours, the remaining tours face identical profile
    if (sim_result.tours_completed < tours) {
        double p_final = single_tour_detection_probability(sim_result.log_profile, beta, options);
        size_t rem = tours - sim_result.tours_completed;
        fail_prob *= std::pow(1.0 - p_final, static_cast<double>(rem));
    }

    return std::clamp(1.0 - fail_prob, 0.0, 1.0);
}

size_t ProbabilisticEstimator::required_beta_for_probability(
    double target_p,
    size_t tours,
    size_t dimension,
    double ln_volume,
    const ProbabilisticOptions& options) {
    if (dimension < 2) {
        throw DimensionMismatchException("ProbabilisticEstimator: dimension must be >= 2");
    }
    if (target_p <= 0.0) {
        return 2;
    }
    if (target_p > 1.0) {
        throw InvalidParameterException("ProbabilisticEstimator: target_p must be <= 1.0");
    }
    if (tours == 0) {
        tours = 1;
    }

    for (size_t beta = 2; beta <= dimension; ++beta) {
        double p = estimate_probability(beta, tours, dimension, ln_volume, options);
        if (p >= target_p) {
            return beta;
        }
    }
    return dimension + 1;
}

ProbabilisticResult ProbabilisticEstimator::estimate_progressive(
    size_t dimension,
    double ln_volume,
    const ProbabilisticOptions& options) {
    if (dimension < 2) {
        throw DimensionMismatchException("ProbabilisticEstimator: dimension must be >= 2");
    }

    const size_t max_b = (options.max_beta == 0 || options.max_beta >= dimension)
                             ? (dimension - 1)
                             : options.max_beta;

    std::vector<double> log_profile = BKZSimulator::initial_lll_log_profile(dimension, ln_volume);

    BKZSimulatorParameters tour_params;
    tour_params.pruning_proba = options.pruning_proba;
    tour_params.model = options.simulation_model;

    ProbabilisticResult result;
    double remaining_proba = 1.0;
    double cumulated_proba = 0.0;
    double average_beta = 0.0;

    for (size_t beta = 2; beta <= max_b; ++beta) {
        tour_params.block_size = beta;
        for (size_t t = 0; t < options.tours_per_beta; ++t) {
            BKZSimulator::simulate_tour(log_profile, beta, tour_params);
        }

        const double p_tour = single_tour_detection_probability(log_profile, beta, options);

        double step_det = 0.0;
        for (size_t tg = 0; tg < options.number_targets; ++tg) {
            const double proba_step = remaining_proba * p_tour;
            average_beta += static_cast<double>(beta) * proba_step;
            cumulated_proba += proba_step;
            remaining_proba *= (1.0 - p_tour);
            step_det += proba_step;
        }

        ProbabilisticStep step;
        step.beta = static_cast<double>(beta);
        step.tour_probability = p_tour;
        step.step_probability = step_det;
        step.cumulative_probability = cumulated_proba;
        step.remaining_probability = remaining_proba;
        result.steps.push_back(step);

        if (remaining_proba < options.stop_remaining_proba) {
            average_beta += static_cast<double>(beta) * remaining_proba;
            cumulated_proba += remaining_proba;
            remaining_proba = 0.0;
            result.solvable = true;
            break;
        }
    }

    if (!result.solvable) {
        if (remaining_proba <= 0.01) {
            result.solvable = true;
        } else {
            result.solvable = false;
            result.warnings.push_back("Instance may be unsolvable within block size bounds (remaining probability > 0.01).");
        }
    }

    result.average_beta = average_beta;
    result.integer_beta = std::max(size_t{2}, static_cast<size_t>(std::ceil(average_beta)));
    result.delta = RootHermite::delta(static_cast<double>(result.integer_beta));
    result.cumulative_probability = cumulated_proba;
    result.remaining_probability = remaining_proba;

    return result;
}

ProbabilisticResult ProbabilisticEstimator::estimate_progressive(
    const UsvpInstance& usvp,
    const ProbabilisticOptions& options) {
    return estimate_progressive(usvp.dimension(), usvp.log_volume().to_double(), options);
}

ProbabilisticResult ProbabilisticEstimator::estimate_progressive(
    const DBDDInstance& dbdd,
    const ProbabilisticOptions& options) {
    double ln_vol_iso = 0.0;
    if (dbdd.is_homogenized()) {
        ln_vol_iso = compute_isotropized_log_volume(dbdd.log_volume(), dbdd.covariance(), options.rank_tol).to_double();
    } else {
        DBDDInstance homog = dbdd;
        homog.homogenize();
        ln_vol_iso = compute_isotropized_log_volume(homog.log_volume(), homog.covariance(), options.rank_tol).to_double();
    }
    return estimate_progressive(dbdd.dimension(), ln_vol_iso, options);
}

ProbabilisticResult ProbabilisticEstimator::estimate_progressive(
    const DBDDStateStatistics& stats,
    const ProbabilisticOptions& options) {
    const double ln_vol_iso = stats.isotropized_log_volume(options.rank_tol);
    return estimate_progressive(stats.dimension(), ln_vol_iso, options);
}

SecurityEstimate ProbabilisticEstimator::estimate(
    size_t dimension,
    double ln_volume,
    const ProbabilisticOptions& options) {
    return estimate_progressive(dimension, ln_volume, options).to_security_estimate(dimension, ln_volume);
}

SecurityEstimate ProbabilisticEstimator::estimate(
    const UsvpInstance& usvp,
    const ProbabilisticOptions& options) {
    return estimate(usvp.dimension(), usvp.log_volume().to_double(), options);
}

SecurityEstimate ProbabilisticEstimator::estimate(
    const DBDDInstance& dbdd,
    const ProbabilisticOptions& options) {
    double ln_vol_iso = 0.0;
    if (dbdd.is_homogenized()) {
        ln_vol_iso = compute_isotropized_log_volume(dbdd.log_volume(), dbdd.covariance(), options.rank_tol).to_double();
    } else {
        DBDDInstance homog = dbdd;
        homog.homogenize();
        ln_vol_iso = compute_isotropized_log_volume(homog.log_volume(), homog.covariance(), options.rank_tol).to_double();
    }
    return estimate(dbdd.dimension(), ln_vol_iso, options);
}

SecurityEstimate ProbabilisticEstimator::estimate(
    const DBDDStateStatistics& stats,
    const ProbabilisticOptions& options) {
    const double ln_vol_iso = stats.isotropized_log_volume(options.rank_tol);
    return estimate(stats.dimension(), ln_vol_iso, options);
}

} // namespace dbdd

