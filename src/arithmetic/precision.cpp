#include "dbdd/arithmetic/precision.hpp"
#include "dbdd/core/exceptions.hpp"
#include <utility>

namespace dbdd {

thread_local prec_t PrecisionManager::current_default_precision_ = DEFAULT_PRECISION_BITS;

prec_t PrecisionManager::get_default_precision() noexcept {
    return current_default_precision_;
}

void PrecisionManager::set_default_precision(prec_t prec) {
    if (prec < MINIMUM_PRECISION_BITS) {
        throw NumericalPrecisionException(
            "Precision must be at least " + std::to_string(MINIMUM_PRECISION_BITS) +
            " bits, requested: " + std::to_string(prec)
        );
    }
    current_default_precision_ = prec;
    mpfr_set_default_prec(prec);
}

ScopedPrecision::ScopedPrecision(prec_t new_prec)
    : prev_prec_(PrecisionManager::get_default_precision()),
      active_prec_(new_prec),
      active_(true) {
    PrecisionManager::set_default_precision(new_prec);
}

ScopedPrecision::~ScopedPrecision() {
    if (active_) {
        PrecisionManager::set_default_precision(prev_prec_);
    }
}

ScopedPrecision::ScopedPrecision(ScopedPrecision&& other) noexcept
    : prev_prec_(other.prev_prec_),
      active_prec_(other.active_prec_),
      active_(other.active_) {
    other.active_ = false;
}

ScopedPrecision& ScopedPrecision::operator=(ScopedPrecision&& other) noexcept {
    if (this != &other) {
        if (active_) {
            PrecisionManager::set_default_precision(prev_prec_);
        }
        prev_prec_ = other.prev_prec_;
        active_prec_ = other.active_prec_;
        active_ = other.active_;
        other.active_ = false;
    }
    return *this;
}

} // namespace dbdd
