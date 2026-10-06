#pragma once

#include "dbdd/core/types.hpp"
#include "dbdd/core/constants.hpp"

namespace dbdd {

/**
 * @brief Thread-safe manager for MPFR default floating-point precision context.
 */
class PrecisionManager {
public:
    static prec_t get_default_precision() noexcept;
    static void set_default_precision(prec_t prec);

private:
    static thread_local prec_t current_default_precision_;
};

/**
 * @brief RAII guard to temporarily override default MPFR precision within a scope.
 */
class ScopedPrecision {
public:
    explicit ScopedPrecision(prec_t new_prec);
    ~ScopedPrecision();

    ScopedPrecision(const ScopedPrecision&) = delete;
    ScopedPrecision& operator=(const ScopedPrecision&) = delete;

    ScopedPrecision(ScopedPrecision&& other) noexcept;
    ScopedPrecision& operator=(ScopedPrecision&& other) noexcept;

    [[nodiscard]] prec_t previous_precision() const noexcept { return prev_prec_; }
    [[nodiscard]] prec_t current_precision() const noexcept { return active_prec_; }

private:
    prec_t prev_prec_;
    prec_t active_prec_;
    bool active_;
};

} // namespace dbdd
