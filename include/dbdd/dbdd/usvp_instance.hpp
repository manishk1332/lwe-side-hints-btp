#pragma once

// Paper Reference: Albrecht et al. Section 2, Section 3.3 / docs/mathematical_conventions.md Section 6
// Mathematical Purpose: Encapsulates an isotropized lattice instance Lambda_iso = Lambda * M,
// where M = (Sigma_h^{1/2})^dagger. Represents the normalized uSVP instance for security estimation
// and lattice reduction, with normalized target s_bar * M satisfying E[||s_bar * M||^2] = d.

#include "dbdd/core/types.hpp"
#include "dbdd/arithmetic/real.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/lattice/basis.hpp"
#include "dbdd/lattice/lattice.hpp"
#include <optional>

namespace dbdd {

class UsvpInstance {
public:
    UsvpInstance(
        MatrixDouble basis_matrix,
        BigReal volume,
        BigReal log_volume,
        MatrixDouble transformation_matrix,
        std::optional<VectorDouble> target = std::nullopt);

    // Dimensions
    [[nodiscard]] size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] size_t ambient_dimension() const noexcept { return amb_dim_; }
    [[nodiscard]] bool empty() const noexcept { return dim_ == 0; }

    // Volumes
    [[nodiscard]] const BigReal& volume() const noexcept { return volume_; }
    [[nodiscard]] const BigReal& log_volume() const noexcept { return log_volume_; }

    // Lattice Representations
    [[nodiscard]] const MatrixDouble& basis_matrix() const noexcept { return basis_matrix_; }
    [[nodiscard]] BasisDouble basis() const;
    [[nodiscard]] BasisRR basis_rr() const;
    [[nodiscard]] LatticeDouble lattice() const;

    // Transformation Matrix M = (Sigma_h^{1/2})^dagger
    [[nodiscard]] const MatrixDouble& transformation_matrix() const noexcept { return M_; }

    // Target Vector
    [[nodiscard]] bool has_target() const noexcept { return target_.has_value(); }
    [[nodiscard]] const VectorDouble& target() const;
    [[nodiscard]] double target_norm() const;
    [[nodiscard]] double target_squared_norm() const;

    // Discretization to Integer Basis for discrete solvers
    [[nodiscard]] BasisZZ to_integer_basis(double scale_factor = 1.0) const;

    // Mapping back reduced candidate vector w_iso -> w = w_iso * Sigma_h^{1/2}
    [[nodiscard]] VectorDouble reconstruct_vector(const VectorDouble& w_iso, const MatrixDouble& sqrt_cov) const;

private:
    size_t dim_{0};
    size_t amb_dim_{0};
    MatrixDouble basis_matrix_;
    BigReal volume_;
    BigReal log_volume_;
    MatrixDouble M_;
    std::optional<VectorDouble> target_;
};

} // namespace dbdd
