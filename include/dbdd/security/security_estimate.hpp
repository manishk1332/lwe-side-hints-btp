#pragma once

// Paper Reference: Dachman-Soled et al. Section 3 & 3.4 / main_plan.txt Section 30 & 43
// Mathematical Purpose: Encapsulates the results of a security estimation run.
// Exposes the required BKZ block size beta as the primary metric ("bikz").

#include "dbdd/core/types.hpp"
#include "dbdd/arithmetic/real.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace dbdd {

enum class CostModel {
    CoreSVPClassical,    // 0.292 * beta (BDGL16 classical sieving)
    CoreSVPQuantum,      // 0.265 * beta (LaaPic15 quantum sieving)
    EnumerationClassical // 0.187 * beta * log2(beta) - 1.019 * beta + 16.1 (Chen-Nguyen 2011)
};

struct SecurityEstimate {
    size_t dimension{0};
    double log_volume{0.0};
    double beta_continuous{0.0};
    size_t beta{0};                     // Primary metric: required integer BKZ block size ("bikz")
    double delta_beta{0.0};              // Root-Hermite factor delta_beta
    bool solved{false};                  // Whether an intersection was found within beta_max
    std::string method{"GSA-Intersect"}; // Estimation method name
    double success_probability{1.0};     // Probability of successful recovery
    std::string confidence_metadata{};   // Simulation parameters, tour history, or solver notes
    std::vector<std::string> warning_flags{}; // Numerical or solvability warnings

    [[nodiscard]] size_t bikz() const noexcept { return beta; }
    [[nodiscard]] size_t beta_integer() const noexcept { return beta; }
    [[nodiscard]] double delta() const noexcept { return delta_beta; }
    [[nodiscard]] BigReal volume() const { return BigReal(std::exp(log_volume)); }

    [[nodiscard]] double cost_bits(CostModel model = CostModel::CoreSVPClassical) const {
        const double b = static_cast<double>(beta);
        switch (model) {
            case CostModel::CoreSVPClassical:
                return 0.292 * b;
            case CostModel::CoreSVPQuantum:
                return 0.265 * b;
            case CostModel::EnumerationClassical:
                if (b <= 2.0) return 0.0;
                return 0.187 * b * std::log2(b) - 1.019 * b + 16.1;
        }
        return 0.292 * b;
    }

    friend std::ostream& operator<<(std::ostream& os, const SecurityEstimate& est) {
        os << "SecurityEstimate { "
           << "method=" << est.method
           << ", d=" << est.dimension
           << ", ln(Vol)=" << std::fixed << std::setprecision(4) << est.log_volume
           << ", beta_cont=" << est.beta_continuous
           << ", bikz=" << est.beta
           << ", delta=" << std::setprecision(8) << est.delta_beta
           << ", solved=" << (est.solved ? "true" : "false")
           << ", p_success=" << std::setprecision(4) << est.success_probability
           << ", cost_bits=" << std::setprecision(2) << est.cost_bits()
           << " }";
        return os;
    }
};

} // namespace dbdd
