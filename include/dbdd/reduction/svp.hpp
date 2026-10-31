#pragma once

// Paper Reference: Albrecht et al. Section 3 / Fincke-Pohst (1985) / Schnorr-Euchner (1994)
// Mathematical Purpose: Abstract SVPOracle interface, parameter container,
// and result structure for shortest vector algorithms in lattices.

#include "dbdd/core/types.hpp"
#include "dbdd/core/exceptions.hpp"
#include "dbdd/arithmetic/integer.hpp"
#include "dbdd/linalg/vector.hpp"
#include "dbdd/linalg/matrix.hpp"
#include "dbdd/linalg/gram_schmidt.hpp"
#include "dbdd/lattice/basis.hpp"
#include <chrono>
#include <optional>

namespace dbdd {

/**
 * @brief Strategy for exploring coordinate tree during SVP enumeration.
 */
enum class EnumerationStrategy {
    SchnorrEuchner,  // Zigzag order around center with dynamic radius shrinking (recommended)
    FinckePohst      // Linear interval scanning
};

/**
 * @brief Configuration parameters for SVP solvers and enumeration oracles.
 */
struct SVPParameters {
    double radius_bound{0.0};                     // Upper bound on ||v|| (0.0 = default to ||b_0||)
    size_t max_nodes{0};                          // Maximum search tree nodes to visit (0 = unlimited)
    double timeout_seconds{0.0};                  // Timeout in seconds (0.0 = unlimited)
    EnumerationStrategy strategy{EnumerationStrategy::SchnorrEuchner};
    bool prune_zero{true};                        // Enforce x != 0
    double radius_expansion_factor{1.0};          // Expansion factor for initial radius bound
    bool preprocess_lll{true};                    // Preprocess with LLL for fast enumeration

    SVPParameters() = default;

    void validate() const {
        if (radius_bound < 0.0) {
            throw InvalidParameterException("SVPParameters: radius_bound cannot be negative");
        }
        if (timeout_seconds < 0.0) {
            throw InvalidParameterException("SVPParameters: timeout_seconds cannot be negative");
        }
        if (radius_expansion_factor <= 0.0) {
            throw InvalidParameterException("SVPParameters: radius_expansion_factor must be positive");
        }
    }
};

/**
 * @brief Result of an SVP search.
 */
struct SVPResult {
    bool found{false};                            // Whether a short non-zero vector was found
    VectorZZ coordinates;                         // Linear combination x in Z^d such that v = x * B
    VectorZZ vector;                              // Shortest non-zero lattice vector v in Z^m
    BigInt squared_norm{0};                       // Exact integer squared norm ||v||^2
    double length{0.0};                           // Euclidean norm ||v||
    size_t nodes_visited{0};                      // Number of search nodes visited
    bool complete_search{true};                   // False if aborted due to max_nodes or timeout
    double elapsed_time_seconds{0.0};             // Wall-clock runtime in seconds
};

/**
 * @brief Abstract base class for Shortest Vector Problem (SVP) oracles.
 * 
 * Decouples lattice reduction algorithms (like BKZ in Phase 16) from the
 * specific underlying solver (e.g., EnumerationSVP, future SieveSVP).
 */
class SVPOracle {
public:
    virtual ~SVPOracle() = default;

    /**
     * @brief Solves exact SVP on the full lattice basis.
     * @param basis Lattice basis B.
     * @param params Configuration parameters.
     * @return SVPResult containing the shortest non-zero vector.
     */
    virtual SVPResult solve(
        const BasisZZ& basis,
        const SVPParameters& params = {}) = 0;

    /**
     * @brief Solves exact SVP on a projected sub-block [start_idx, start_idx + block_size - 1].
     * Uses existing Gram-Schmidt state without basis re-orthogonalization.
     * @param gs Gram-Schmidt state of the encompassing lattice.
     * @param start_idx Index of the first vector in the block.
     * @param block_size Number of vectors in the block.
     * @param params Configuration parameters.
     * @return SVPResult containing the shortest non-zero projected vector.
     */
    virtual SVPResult solve_projected(
        const GramSchmidtStateDouble& gs,
        size_t start_idx,
        size_t block_size,
        const SVPParameters& params = {}) = 0;
};

} // namespace dbdd
