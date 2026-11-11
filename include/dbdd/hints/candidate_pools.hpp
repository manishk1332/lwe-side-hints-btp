#pragma once

// Paper Reference: thesis extension "Budgeted Hint Selection as D-Optimal Design"
//                  (innovation/proposal_3_budgeted_hint_selection.md, Section 6).
// Mathematical Purpose: Synthetic pools of candidate approximate hints on an LWE instance with n secret and
// m error coordinates. Hint vectors are expressed in the ambient coordinates (e, s, 1) of the Kannan embedding;
// the homogeneous coordinate is 0, because the posterior covariance (and therefore the selection) does not
// depend on the observed values. Each candidate's noise standard deviation is log-uniform in
// [sigma_min, sigma_max].
//
//     Pool A: one hint per coordinate. The gains do not interact (control case).
//     Pool B: Pool A plus repeated measurements of a random half of its coordinates (redundancy).
//     Pool C: Pool A plus pair sums and window sums over the secret block (overlapping directions).
//     Pool D: random +-1 combinations of 8 to 32 coordinates (strong interaction).

#include "dbdd/hints/budgeted_selection.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dbdd {

struct CandidatePoolOptions {
    size_t n = 70;            ///< Number of secret coordinates.
    size_t m = 70;            ///< Number of error coordinates (LWE samples).
    size_t target_size = 0;   ///< Desired pool size N for pools B, C, D; 0 selects the default size.
    double sigma_min = 0.5;   ///< Smallest noise standard deviation.
    double sigma_max = 16.0;  ///< Largest noise standard deviation.
    uint64_t seed = 1;        ///< Seed of the pool's pseudo-random generator.
};

enum class CandidatePoolKind { A, B, C, D };

class CandidatePools {
public:
    // Inputs:
    //     options: instance shape, pool size, noise range and seed.
    // Outputs:
    //     Candidates of length m + n + 1 with strictly positive noise variance.
    // Preconditions:
    //     n >= 1, m >= 1, 0 < sigma_min <= sigma_max.
    // Postconditions:
    //     Deterministic for a fixed seed. Pool A has exactly n + m candidates; pools B, C, D have
    //     target_size candidates when target_size > 0.
    // Precision & Complexity:
    //     double precision; O(N (n + m)).
    [[nodiscard]] static std::vector<HintCandidate> pool_a(const CandidatePoolOptions& options);
    [[nodiscard]] static std::vector<HintCandidate> pool_b(const CandidatePoolOptions& options);
    [[nodiscard]] static std::vector<HintCandidate> pool_c(const CandidatePoolOptions& options);
    [[nodiscard]] static std::vector<HintCandidate> pool_d(const CandidatePoolOptions& options);

    [[nodiscard]] static std::vector<HintCandidate> make(CandidatePoolKind kind, const CandidatePoolOptions& options);
    [[nodiscard]] static const char* name(CandidatePoolKind kind) noexcept;
};

} // namespace dbdd
