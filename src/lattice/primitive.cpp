#include "dbdd/lattice/primitive.hpp"
#include <algorithm>

namespace dbdd {

std::vector<BigInt> compute_smith_normal_form_invariant_factors(MatrixZZ X) {
    size_t k = X.rows();
    size_t d = X.cols();
    size_t min_dim = std::min(k, d);

    for (size_t p = 0; p < min_dim; ++p) {
        while (true) {
            // Step 1: Find minimal non-zero entry in the submatrix
            size_t min_r = p;
            size_t min_c = p;
            BigInt min_val(0);

            for (size_t r = p; r < k; ++r) {
                for (size_t c = p; c < d; ++c) {
                    BigInt val = X(r, c).abs();
                    if (val > BigInt(0)) {
                        if (min_val == BigInt(0) || val < min_val) {
                            min_val = val;
                            min_r = r;
                            min_c = c;
                        }
                    }
                }
            }

            // If submatrix is entirely zero, we are done
            if (min_val == BigInt(0)) {
                break;
            }

            // Move minimal entry to (p, p)
            if (min_r != p) {
                X.swap_rows(p, min_r);
            }
            if (min_c != p) {
                // Swap columns p and min_c
                for (size_t r = 0; r < k; ++r) {
                    std::swap(X(r, p), X(r, min_c));
                }
            }
            if (X(p, p) < BigInt(0)) {
                for (size_t c = 0; c < d; ++c) {
                    X(p, c) = -X(p, c);
                }
            }

            // Step 2: Clear row p and column p
            bool changed = false;

            // Clear column p
            for (size_t r = p + 1; r < k; ++r) {
                if (X(r, p) != BigInt(0)) {
                    BigInt q = X(r, p) / X(p, p);
                    if (q != BigInt(0)) {
                        X.row_add_multiple(r, p, -q);
                    }
                    if (X(r, p) != BigInt(0)) {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed) continue;

            // Clear row p
            for (size_t c = p + 1; c < d; ++c) {
                if (X(p, c) != BigInt(0)) {
                    BigInt q = X(p, c) / X(p, p);
                    if (q != BigInt(0)) {
                        for (size_t r = 0; r < k; ++r) {
                            X(r, c) -= q * X(r, p);
                        }
                    }
                    if (X(p, c) != BigInt(0)) {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed) continue;

            // Step 3: Check if X(p, p) divides all submatrix entries
            bool divisibility_violated = false;
            for (size_t r = p + 1; r < k; ++r) {
                for (size_t c = p + 1; c < d; ++c) {
                    if (X(r, c) % X(p, p) != BigInt(0)) {
                        // Add row r to row p to introduce non-divisible element into row p
                        X.row_add_multiple(p, r, BigInt(1));
                        divisibility_violated = true;
                        break;
                    }
                }
                if (divisibility_violated) break;
            }

            if (!divisibility_violated) {
                // Pivot p is completely reduced and divides all remaining elements
                break;
            }
        }
    }

    std::vector<BigInt> factors;
    for (size_t p = 0; p < min_dim; ++p) {
        if (X(p, p) != BigInt(0)) {
            factors.push_back(X(p, p).abs());
        }
    }
    return factors;
}

bool is_primitive_set_in_lattice(const MatrixZZ& V, const BasisZZ& basis) {
    if (V.empty() || basis.empty()) return false;
    size_t k = V.rows();
    size_t d = basis.dimension();
    if (k > d) return false;

    // Express V in basis coordinates: V = X * B
    MatrixZZ X(k, d);
    for (size_t r = 0; r < k; ++r) {
        VectorZZ row_v = V.row(r);
        Vector<BigRational> coords = basis.coordinates_of<BigRational>(row_v);
        for (size_t c = 0; c < d; ++c) {
            if (coords[c].denominator() != BigInt(1)) {
                return false; // Vector row is not in the integer lattice
            }
            X(r, c) = coords[c].numerator();
        }
    }

    auto factors = compute_smith_normal_form_invariant_factors(X);
    if (factors.size() != k) {
        return false; // Rows of V are linearly dependent
    }

    for (const auto& f : factors) {
        if (f != BigInt(1)) {
            return false;
        }
    }
    return true;
}

} // namespace dbdd
