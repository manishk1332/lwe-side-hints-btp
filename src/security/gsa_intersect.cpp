#include "dbdd/security/gsa_intersect.hpp"
#include "dbdd/security/root_hermite.hpp"
#include "dbdd/security/gsa.hpp"
#include "dbdd/dbdd/usvp_instance.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/core/exceptions.hpp"

#include <cmath>
#include <algorithm>
#include <functional>

namespace dbdd {

namespace {

// Bracketed Brent's root-finder for continuous monotonic functions
double brent_root_find(
    const std::function<double(double)>& f,
    double a,
    double b,
    double tol,
    size_t max_iter = 100) {
    double fa = f(a);
    double fb = f(b);

    if (fa >= 0.0) {
        return a;
    }
    if (fb < 0.0) {
        return b + 1.0; // Infeasible / root exceeds b
    }

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    double d = 0.0;
    bool mflag = true;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fb) <= 1e-15 || std::abs(b - a) < tol) {
            return b;
        }

        double s = 0.0;
        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                (c * fa * fb) / ((fc - fa) * (fc - fb));
        } else {
            // Secant step
            s = b - fb * (b - a) / (fb - fa);
        }

        double min_bound = (3.0 * a + b) / 4.0;
        double max_bound = b;
        if (min_bound > max_bound) {
            std::swap(min_bound, max_bound);
        }

        bool cond1 = (s < min_bound || s > max_bound);
        bool cond2 = mflag && (std::abs(s - b) >= std::abs(b - c) / 2.0);
        bool cond3 = (!mflag) && (std::abs(s - b) >= std::abs(c - d) / 2.0);
        bool cond4 = mflag && (std::abs(b - c) < tol);
        bool cond5 = (!mflag) && (std::abs(c - d) < tol);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            // Bisection step
            s = 0.5 * (a + b);
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    return b;
}

} // namespace

double GSAIntersect::objective(double beta, size_t d, double ln_vol) {
    if (d == 0) {
        throw DimensionMismatchException("GSAIntersect::objective: dimension d must be >= 1");
    }
    if (beta < 2.0) {
        throw InvalidParameterException("GSAIntersect::objective: beta must be >= 2.0");
    }
    const double ln_delta = RootHermite::log_delta(beta);
    const double d_d = static_cast<double>(d);
    return (2.0 * beta - d_d - 1.0) * ln_delta + (ln_vol / d_d) - 0.5 * std::log(beta);
}

double GSAIntersect::objective_general(double beta, size_t d, double ln_vol, double s_norm) {
    if (d == 0) {
        throw DimensionMismatchException("GSAIntersect::objective_general: dimension d must be >= 1");
    }
    if (beta < 2.0) {
        throw InvalidParameterException("GSAIntersect::objective_general: beta must be >= 2.0");
    }
    if (s_norm <= 0.0) {
        return objective(beta, d, ln_vol);
    }
    const double ln_delta = RootHermite::log_delta(beta);
    const double d_d = static_cast<double>(d);
    return (2.0 * beta - d_d - 1.0) * ln_delta + (ln_vol / d_d) - 0.5 * std::log(beta / d_d) - std::log(s_norm);
}

bool GSAIntersect::condition(double beta, size_t d, double ln_vol) {
    return objective(beta, d, ln_vol) >= 0.0;
}

bool GSAIntersect::condition_general(double beta, size_t d, double ln_vol, double s_norm) {
    return objective_general(beta, d, ln_vol, s_norm) >= 0.0;
}

double GSAIntersect::estimate_continuous_beta(
    size_t d,
    double ln_vol,
    double beta_min,
    double beta_max,
    double tol) {
    if (d == 0) {
        throw DimensionMismatchException("GSAIntersect::estimate_continuous_beta: dimension d must be >= 1");
    }
    if (beta_min < 2.0) {
        beta_min = 2.0;
    }
    if (beta_max <= beta_min) {
        throw InvalidParameterException("GSAIntersect::estimate_continuous_beta: beta_max must be > beta_min");
    }

    auto f = [d, ln_vol](double b) {
        return objective(b, d, ln_vol);
    };

    return brent_root_find(f, beta_min, beta_max, tol);
}

double GSAIntersect::estimate_continuous_beta_general(
    size_t d,
    double ln_vol,
    double s_norm,
    double beta_min,
    double beta_max,
    double tol) {
    if (d == 0) {
        throw DimensionMismatchException("GSAIntersect::estimate_continuous_beta_general: dimension d must be >= 1");
    }
    if (beta_min < 2.0) {
        beta_min = 2.0;
    }
    if (beta_max <= beta_min) {
        throw InvalidParameterException("GSAIntersect::estimate_continuous_beta_general: beta_max must be > beta_min");
    }

    auto f = [d, ln_vol, s_norm](double b) {
        return objective_general(b, d, ln_vol, s_norm);
    };

    return brent_root_find(f, beta_min, beta_max, tol);
}

size_t GSAIntersect::estimate_integer_beta(
    size_t d,
    double ln_vol,
    size_t beta_min,
    size_t beta_max) {
    if (beta_min < 2) {
        beta_min = 2;
    }
    double b_cont = estimate_continuous_beta(d, ln_vol, static_cast<double>(beta_min), static_cast<double>(beta_max));
    if (b_cont > static_cast<double>(beta_max)) {
        return beta_max + 1; // Infeasible within beta_max
    }
    size_t b_int = static_cast<size_t>(std::ceil(b_cont));
    return std::max(b_int, beta_min);
}

size_t GSAIntersect::estimate_integer_beta_general(
    size_t d,
    double ln_vol,
    double s_norm,
    size_t beta_min,
    size_t beta_max) {
    if (beta_min < 2) {
        beta_min = 2;
    }
    double b_cont = estimate_continuous_beta_general(d, ln_vol, s_norm, static_cast<double>(beta_min), static_cast<double>(beta_max));
    if (b_cont > static_cast<double>(beta_max)) {
        return beta_max + 1;
    }
    size_t b_int = static_cast<size_t>(std::ceil(b_cont));
    return std::max(b_int, beta_min);
}

SecurityEstimate GSAIntersect::estimate(
    size_t d,
    double ln_vol,
    double beta_min,
    double beta_max) {
    double b_cont = estimate_continuous_beta(d, ln_vol, beta_min, beta_max);
    bool solved = (b_cont <= beta_max);
    size_t b_int = solved ? std::max(static_cast<size_t>(std::ceil(b_cont)), static_cast<size_t>(beta_min)) : static_cast<size_t>(beta_max + 1);
    double delta_val = RootHermite::delta(static_cast<double>(b_int));

    SecurityEstimate est;
    est.dimension = d;
    est.log_volume = ln_vol;
    est.beta_continuous = b_cont;
    est.beta = b_int;
    est.delta_beta = delta_val;
    est.solved = solved;
    return est;
}

SecurityEstimate GSAIntersect::estimate(
    const UsvpInstance& usvp,
    double beta_min,
    double beta_max) {
    return estimate(usvp.dimension(), usvp.log_volume().to_double(), beta_min, beta_max);
}

SecurityEstimate GSAIntersect::estimate(
    const DBDDInstance& dbdd,
    double beta_min,
    double beta_max,
    double rank_tol) {
    double ln_vol_iso = 0.0;
    if (dbdd.is_homogenized()) {
        ln_vol_iso = compute_isotropized_log_volume(dbdd.log_volume(), dbdd.covariance(), rank_tol).to_double();
    } else {
        DBDDInstance homog = dbdd;
        homog.homogenize();
        ln_vol_iso = compute_isotropized_log_volume(homog.log_volume(), homog.covariance(), rank_tol).to_double();
    }
    return estimate(dbdd.dimension(), ln_vol_iso, beta_min, beta_max);
}

} // namespace dbdd
