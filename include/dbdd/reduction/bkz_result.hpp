#pragma once

// Paper Reference: Schnorr (1987) / Schnorr-Euchner (1994) / Chen-Nguyen (2011) / Dachman-Soled et al. Section 2
// Mathematical Purpose: Result container for BKZ reduction profiling and validation.

#include "dbdd/lattice/basis.hpp"
#include "dbdd/linalg/matrix.hpp"
#include <vector>

namespace dbdd {

/**
 * @brief Result of a BKZ reduction.
 */
struct BKZResult {
    BasisZZ basis;                                // BKZ-reduced basis B'
    MatrixZZ transformation;                       // Unimodular matrix U in GL_d(Z) such that B' = U * B
    size_t tours_completed{0};                     // Number of tours executed
    size_t total_svp_calls{0};                     // Number of calls to SVP oracle
    size_t total_svp_nodes{0};                     // Total search nodes across all SVP calls
    size_t total_insertions{0};                    // Number of successful vector insertions
    bool converged{true};                          // Whether reduction terminated cleanly
    double elapsed_time_seconds{0.0};              // Total runtime in seconds
    std::vector<std::vector<double>> tour_profiles;// History of log Gram-Schmidt profiles after each tour
    std::vector<double> final_profile;             // Final log Gram-Schmidt profile ln ||b_i*||
    double final_slope{0.0};                       // Final GSA slope
};

} // namespace dbdd

