#pragma once

// Paper Reference: Dachman-Soled et al. Section 2, Section 3 / docs/mathematical_conventions.md Section 1
// Mathematical Purpose: Encapsulates a lattice Lambda defined by a basis matrix B in K^{d x m}.
// Caches volume and log-volume to avoid redundant determinant computations.
// Supports dual lattice extraction, vector containment checks, and orthogonal projections.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/arithmetic/rational.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/volume.hpp"
#include "dbdd/lattice/dual.hpp"
#include "dbdd/linalg/projection.hpp"
#include <optional>

namespace dbdd {

template <typename T>
class Lattice {
    template <typename U>
    friend class Lattice;

public:
    // Constructors
    Lattice() : basis_() {}

    explicit Lattice(Basis<T> basis)
        : basis_(std::move(basis)) {}

    explicit Lattice(Matrix<T> basis_matrix)
        : basis_(std::move(basis_matrix)) {}

    Lattice(Basis<T> basis, BigReal known_volume)
        : basis_(std::move(basis)), cached_volume_(known_volume) {
        if (known_volume > BigReal(0.0)) {
            cached_log_volume_ = known_volume.log();
        }
    }

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return basis_.dimension(); }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return basis_.ambient_dimension(); }
    [[nodiscard]] size_t rank() const { return basis_.rank(); }
    [[nodiscard]] bool empty() const noexcept { return basis_.empty(); }
    [[nodiscard]] bool is_full_rank() const noexcept { return basis_.is_full_rank(); }

    // Basis access
    [[nodiscard]] const Basis<T>& basis() const noexcept { return basis_; }

    void set_basis(Basis<T> new_basis) {
        basis_ = std::move(new_basis);
        invalidate_cache();
    }

    // Volume with caching (Architectural Invariant: Do not permanently recompute volume)
    [[nodiscard]] BigReal volume() const {
        if (cached_volume_.has_value()) {
            return *cached_volume_;
        }
        BigReal vol = compute_volume(basis_);
        cached_volume_ = vol;
        if (vol > BigReal(0.0)) {
            cached_log_volume_ = vol.log();
        }
        return vol;
    }

    [[nodiscard]] BigReal log_volume() const {
        if (cached_log_volume_.has_value()) {
            return *cached_log_volume_;
        }
        BigReal log_vol = compute_log_volume(basis_);
        cached_log_volume_ = log_vol;
        return log_vol;
    }

    [[nodiscard]] BigInt exact_volume_int() const {
        if constexpr (std::is_same_v<T, BigInt>) {
            return compute_volume_exact(basis_);
        } else {
            throw DBDDException("exact_volume_int is only available for integer lattices LatticeZZ");
        }
    }

    [[nodiscard]] BigRational exact_volume_rat() const {
        if constexpr (std::is_same_v<T, BigRational>) {
            return compute_volume_exact(basis_);
        } else if constexpr (std::is_same_v<T, BigInt>) {
            return BigRational(compute_volume_exact(basis_));
        } else {
            throw DBDDException("exact_volume_rat is only available for rational or integer lattices");
        }
    }

    // Dual lattice accessor: Lambda* with dual basis D in Q^{d x m}
    [[nodiscard]] Lattice<BigRational> dual() const {
        BasisQQ D = dual_basis(basis_);
        Lattice<BigRational> lat_dual(std::move(D));
        if (cached_volume_.has_value() && *cached_volume_ > BigReal(0.0)) {
            lat_dual.cached_volume_ = BigReal(1.0) / (*cached_volume_);
        }
        if (cached_log_volume_.has_value()) {
            lat_dual.cached_log_volume_ = -(*cached_log_volume_);
        }
        return lat_dual;
    }

    // Containment and coordinate evaluation
    [[nodiscard]] bool contains(const Vector<T>& v) const {
        if (v.size() != ambient_dimension()) return false;
        Vector<BigRational> coords = basis_.template coordinates_of<BigRational>(v);
        for (size_t i = 0; i < coords.size(); ++i) {
            if (coords[i].denominator() != BigInt(1)) {
                return false;
            }
        }
        return true;
    }

    template <typename OutT = BigRational>
    [[nodiscard]] Vector<OutT> coordinates_of(const Vector<T>& v) const {
        return basis_.template coordinates_of<OutT>(v);
    }

    // Unimodular elementary operations (preserve volume cache!)
    void swap_rows(size_t i, size_t j) {
        basis_.swap_rows(i, j);
        // Note: Volume is invariant under row swap; cache remains valid!
    }

    void row_add_multiple(size_t target, size_t source, const T& factor) {
        basis_.row_add_multiple(target, source, factor);
        // Note: Volume is invariant under elementary row addition; cache remains valid!
    }

    // Orthogonal projection of the lattice: B' = B * P
    [[nodiscard]] Basis<T> project(const Matrix<T>& P) const {
        Matrix<T> proj_matrix = project_matrix(basis_.matrix(), P);
        return Basis<T>(proj_matrix, false);
    }

    // Type casting
    template <typename U>
    [[nodiscard]] Lattice<U> cast() const {
        Lattice<U> res(basis_.template cast<U>());
        res.cached_volume_ = cached_volume_;
        res.cached_log_volume_ = cached_log_volume_;
        return res;
    }

    bool operator==(const Lattice& rhs) const noexcept {
        return basis_ == rhs.basis_;
    }

    bool operator!=(const Lattice& rhs) const noexcept {
        return basis_ != rhs.basis_;
    }

private:
    void invalidate_cache() noexcept {
        cached_volume_.reset();
        cached_log_volume_.reset();
    }

    Basis<T> basis_;
    mutable std::optional<BigReal> cached_volume_;
    mutable std::optional<BigReal> cached_log_volume_;
};

using LatticeZZ = Lattice<BigInt>;
using LatticeQQ = Lattice<BigRational>;
using LatticeRR = Lattice<BigReal>;
using LatticeDouble = Lattice<double>;

} // namespace dbdd
