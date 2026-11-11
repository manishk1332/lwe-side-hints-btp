// Paper Reference: Dachman-Soled et al. Section 3.4 & Figure 4 (CRYPTO 2020 / IACR ePrint 2020/292)
// Chen & Nguyen, "BKZ 2.0: Better Lattice Security Estimates" (ASIACRYPT 2011, Section 3)
// Mathematical Purpose: Implementation of the BKZ basis profile simulator.

#include "dbdd/security/bkz_simulator.hpp"
#include "dbdd/security/gsa.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include <algorithm>
#include <numeric>

namespace dbdd {

namespace {

// Empirical HKZ reduction profile table r_k for k = 1..45 from Chen-Nguyen 2011 Table 1.
// Values represent log2-normalized Gram-Schmidt lengths in empirical HKZ-reduced bases.
constexpr double CN_RK[45] = {
    0.789527997160000,   0.780003183804613,   0.750872218594458,
    0.706520454592593,   0.696345241018901,   0.660533841808400,
    0.626274718790505,   0.581480717333169,   0.553171463433503,
    0.520811087419712,   0.487994338534253,   0.459541470573431,
    0.414638319529319,   0.392811729940846,   0.339090376264829,
    0.306561491936042,   0.276041187709516,   0.236698863270441,
    0.196186341673080,   0.161214212092249,   0.110895134828114,
    0.0678261623920553,  0.0272807162335610, -0.0234609979600137,
   -0.0320527224746912, -0.0940331032784437, -0.129109087817554,
   -0.176965384290173,  -0.209405754915959,  -0.265867993276493,
   -0.299031324494802,  -0.349338597048432,  -0.380428160303508,
   -0.427399405474537,  -0.474944677694975,  -0.530140672818150,
   -0.561625221138784,  -0.612008793872032,  -0.669011014635905,
   -0.713766731570930,  -0.754041787011810,  -0.808609696192079,
   -0.859933249032210,  -0.884479963601658,  -0.886666930030433
};

constexpr double PI_VAL = 3.14159265358979323846;
constexpr double E_VAL  = 2.71828182845904523536;

} // anonymous namespace

double BKZSimulator::gh_factor_squared(size_t d) {
    if (d == 0) return 0.0;
    double d_dbl = static_cast<double>(d);
    double pi_d = PI_VAL * d_dbl;
    double pow_term = std::pow(pi_d, 1.0 / d_dbl);
    return (d_dbl / (2.0 * PI_VAL * E_VAL)) * pow_term;
}

double BKZSimulator::gh_factor(size_t d) {
    return std::sqrt(gh_factor_squared(d));
}

double BKZSimulator::chen_nguyen_constant_log2(size_t d, double pruning_proba) {
    if (d <= 1) return 0.0;

    if (d <= 45) {
        size_t start_idx = 45 - d;
        double head = CN_RK[start_idx];
        double sum = 0.0;
        for (size_t j = 0; j < d; ++j) {
            sum += CN_RK[start_idx + j];
        }
        double mean = sum / static_cast<double>(d);
        return head - mean;
    }

    // d >= 46: Gaussian heuristic formula with pruning probability penalty
    double gamma = gh_factor_squared(d);
    double d_dbl = static_cast<double>(d);
    double val_ln = 0.5 * std::log(gamma) - std::log(pruning_proba) / d_dbl;
    return val_ln / std::log(2.0);
}

double BKZSimulator::chen_nguyen_constant_ln(size_t d, double pruning_proba) {
    if (d <= 1) return 0.0;

    if (d <= 45) {
        return chen_nguyen_constant_log2(d, pruning_proba) * std::log(2.0);
    }

    double gamma = gh_factor_squared(d);
    double d_dbl = static_cast<double>(d);
    return 0.5 * std::log(gamma) - std::log(pruning_proba) / d_dbl;
}

std::vector<double> BKZSimulator::initial_lll_log_profile(size_t dimension, double ln_volume) {
    if (dimension == 0) {
        throw DimensionMismatchException("BKZSimulator::initial_lll_log_profile: dimension cannot be 0");
    }
    return GSAModel::log_profile(2.0, dimension, ln_volume);
}

double BKZSimulator::compute_profile_slope(const std::vector<double>& log_profile) {
    size_t n = log_profile.size();
    if (n < 2) return 0.0;

    double n_dbl = static_cast<double>(n);
    double mean_x = (n_dbl - 1.0) / 2.0;

    double num = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += (static_cast<double>(i) - mean_x) * log_profile[i];
    }

    double denom = n_dbl * (n_dbl * n_dbl - 1.0) / 12.0;
    return num / denom;
}

size_t BKZSimulator::simulate_tour(
    std::vector<double>& log_profile,
    size_t beta,
    const BKZSimulatorParameters& params) {

    size_t n = log_profile.size();
    if (n < 2) return 0;

    size_t updates = 0;
    for (size_t k = 0; k + 1 < n; ++k) {
        size_t d = std::min(beta, n - k);
        if (d < 2) continue;
        size_t f = k + d;

        double logV = 0.0;
        for (size_t i = k; i < f; ++i) {
            logV += log_profile[i];
        }

        double c_offset = 0.0;
        if (params.model == SimulationModel::PureGH) {
            double gamma = gh_factor_squared(d);
            c_offset = 0.5 * std::log(gamma) - std::log(params.pruning_proba) / static_cast<double>(d);
        } else {
            c_offset = chen_nguyen_constant_ln(d, params.pruning_proba);
        }

        double lma = logV / static_cast<double>(d) + c_offset;

        if (lma < log_profile[k]) {
            double diff = log_profile[k] - lma;
            log_profile[k] = lma;
            double boost = diff / static_cast<double>(f - k - 1);
            for (size_t a = k + 1; a < f; ++a) {
                log_profile[a] += boost;
            }
            updates++;
        }
    }
    return updates;
}

BKZSimulationResult BKZSimulator::simulate(
    const std::vector<double>& initial_log_profile,
    const BKZSimulatorParameters& params) {

    params.validate();
    size_t d = initial_log_profile.size();
    if (d == 0) {
        throw DimensionMismatchException("BKZSimulator::simulate: dimension cannot be 0");
    }

    BKZSimulationResult res;
    res.log_profile = initial_log_profile;

    double total_log_vol = 0.0;
    for (double v : initial_log_profile) {
        total_log_vol += v;
    }
    res.log_volume = total_log_vol;

    if (d == 1) {
        res.profile = {std::exp(res.log_profile[0])};
        res.tours_completed = 0;
        res.converged = true;
        res.final_slope = 0.0;
        if (params.record_tours) {
            res.tour_log_profiles.push_back(res.log_profile);
        }
        return res;
    }

    if (params.model == SimulationModel::GSA) {
        res.log_profile = GSAModel::log_profile(static_cast<double>(params.block_size), d, total_log_vol);
        res.tours_completed = 1;
        res.converged = true;
    } else {
        if (params.record_tours) {
            res.tour_log_profiles.push_back(res.log_profile);
        }

        size_t tours = 0;
        bool converged = false;

        for (size_t tour = 0; tour < params.max_tours; ++tour) {
            std::vector<double> prev = res.log_profile;
            size_t updates = simulate_tour(res.log_profile, params.block_size, params);
            tours++;

            if (params.record_tours) {
                res.tour_log_profiles.push_back(res.log_profile);
            }

            if (updates == 0) {
                converged = true;
                break;
            }

            double max_diff = 0.0;
            for (size_t i = 0; i < d; ++i) {
                max_diff = std::max(max_diff, std::abs(res.log_profile[i] - prev[i]));
            }
            if (max_diff < params.early_abort_threshold) {
                converged = true;
                break;
            }
        }
        res.tours_completed = tours;
        res.converged = converged;
    }

    res.profile.resize(d);
    for (size_t i = 0; i < d; ++i) {
        res.profile[i] = std::exp(res.log_profile[i]);
    }
    res.final_slope = compute_profile_slope(res.log_profile);

    return res;
}

BKZSimulationResult BKZSimulator::simulate(
    size_t dimension,
    double ln_volume,
    const BKZSimulatorParameters& params) {
    std::vector<double> init_profile = initial_lll_log_profile(dimension, ln_volume);
    return simulate(init_profile, params);
}

BKZSimulationResult BKZSimulator::simulate(
    size_t dimension,
    const BigReal& volume,
    const BKZSimulatorParameters& params) {
    double ln_vol = std::log(std::max(1e-300, volume.to_double()));
    return simulate(dimension, ln_vol, params);
}

BKZSimulationResult BKZSimulator::simulate(
    const BasisZZ& basis,
    const BKZSimulatorParameters& params) {
    size_t d = basis.dimension();
    size_t m = basis.ambient_dimension();
    MatrixDouble M(d, m);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < m; ++j) {
            M(i, j) = basis.matrix()(i, j).to_double();
        }
    }
    GramSchmidtStateDouble gs = compute_gram_schmidt_mgs<double>(M);
    std::vector<double> init_log_profile(d);
    for (size_t i = 0; i < d; ++i) {
        init_log_profile[i] = 0.5 * std::log(std::max(1e-20, gs.bstar_norm_sq[i]));
    }
    return simulate(init_log_profile, params);
}

BKZSimulationResult BKZSimulator::simulate(
    const LatticeZZ& lattice,
    const BKZSimulatorParameters& params) {
    return simulate(lattice.basis(), params);
}

BKZSimulationResult BKZSimulator::simulate(
    const DBDDInstance& dbdd,
    const BKZSimulatorParameters& params) {
    return simulate(dbdd.lattice().basis(), params);
}

} // namespace dbdd
