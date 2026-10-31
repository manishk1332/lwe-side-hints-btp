#pragma once

// Paper Reference: Albrecht et al. Section 3 / docs/paper_mapping.md / main_plan.txt Section 16
// Mathematical Purpose: Abstract base class establishing the polymorphic interface for
// side-information hint integration into the DBDD instance (Phases 06-10).

#include <string>

namespace dbdd {

class DBDDInstance;

/**
 * @brief Categorization of side-information hint types according to Albrecht et al.
 */
enum class HintType {
    Perfect,      // Section 3.1: Hyperplane intersection <s, v> = l
    Modular,      // Section 3.2: Dual sparsification modulo k: <s, v> = l (mod k)
    Approximate,  // Section 3.3: Gaussian noise conditioning: <s, v> = l + e
    Aposteriori,  // Section 3.3: Bayesian posterior conditioning
    ShortVector   // Section 3.5: Dual lattice vector projection
};

/**
 * @brief Abstract base class for all side information hints.
 * Each concrete hint class implements apply(DBDDInstance&) to update the lattice basis
 * and/or the (mean, covariance) distribution of the target vector.
 */
class Hint {
public:
    virtual ~Hint() = default;

    /**
     * @brief Applies the hint to update the DBDD instance.
     * @param instance DBDD instance to be transformed.
     */
    virtual void apply(DBDDInstance& instance) const = 0;

    /**
     * @brief Returns the classification of this hint.
     */
    [[nodiscard]] virtual HintType type() const = 0;

    /**
     * @brief Returns a human-readable description of the hint.
     */
    [[nodiscard]] virtual std::string description() const = 0;
};

} // namespace dbdd
