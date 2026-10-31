#include "dbdd/lattice/q_vectors.hpp"
#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/core/exceptions.hpp"

namespace dbdd {

std::vector<VectorZZ> QVectorGenerator::generate_canonical(
    const BigInt& q,
    size_t m,
    size_t ambient_dim) {
    if (m > ambient_dim) {
        throw InvalidParameterException("QVectorGenerator::generate_canonical: m exceeds ambient dimension");
    }
    if (q <= BigInt(0)) {
        throw InvalidParameterException("QVectorGenerator::generate_canonical: q must be positive");
    }

    std::vector<VectorZZ> result;
    result.reserve(m);

    for (size_t i = 0; i < m; ++i) {
        VectorZZ v(ambient_dim, BigInt(0));
        v[i] = q;
        result.push_back(std::move(v));
    }
    return result;
}

std::vector<VectorZZ> QVectorGenerator::generate_differences(
    const BigInt& q,
    size_t m,
    size_t ambient_dim) {
    if (m > ambient_dim) {
        throw InvalidParameterException("QVectorGenerator::generate_differences: m exceeds ambient dimension");
    }
    if (q <= BigInt(0)) {
        throw InvalidParameterException("QVectorGenerator::generate_differences: q must be positive");
    }

    std::vector<VectorZZ> result;
    if (m >= 2) {
        result.reserve((m * (m - 1)) / 2);
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = i + 1; j < m; ++j) {
            VectorZZ v(ambient_dim, BigInt(0));
            v[i] = q;
            v[j] = -q;
            result.push_back(std::move(v));
        }
    }
    return result;
}

bool QVectorGenerator::is_in_lattice(
    const LatticeZZ& lattice,
    const VectorZZ& v) {
    if (v.size() != lattice.ambient_dimension()) {
        return false;
    }
    if (lattice.dimension() == 0) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] != BigInt(0)) {
                return false;
            }
        }
        return true;
    }

    // Evaluate coordinates x such that x * B = v
    Vector<BigRational> coords = lattice.basis().template coordinates_of<BigRational>(v);

    // 1. All coordinates must be integers
    VectorZZ x_int(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        if (coords[i].denominator() != BigInt(1)) {
            return false;
        }
        x_int[i] = coords[i].numerator();
    }

    // 2. Exact vector reconstruction check: x_int * B == v
    // Necessary for subspace lattices (d < ambient_dim) where coordinates_of
    // projects onto span(B).
    VectorZZ reconstructed = x_int * lattice.basis().matrix();
    return reconstructed == v;
}

std::vector<VectorZZ> QVectorGenerator::filter_in_lattice(
    const LatticeZZ& lattice,
    const std::vector<VectorZZ>& candidates) {
    std::vector<VectorZZ> result;
    result.reserve(candidates.size());

    for (const auto& v : candidates) {
        if (is_in_lattice(lattice, v)) {
            result.push_back(v);
        }
    }
    return result;
}

std::vector<VectorZZ> QVectorGenerator::generate_valid(
    const DBDDInstance& dbdd,
    const BigInt& q,
    bool include_differences) {
    const size_t amb_dim = dbdd.ambient_dimension();
    const auto& lat = dbdd.lattice();

    std::vector<VectorZZ> candidates;

    // Scan all canonical coordinate vectors q * e_i
    for (size_t i = 0; i < amb_dim; ++i) {
        VectorZZ v(amb_dim, BigInt(0));
        v[i] = q;
        candidates.push_back(std::move(v));
    }

    if (include_differences) {
        for (size_t i = 0; i < amb_dim; ++i) {
            for (size_t j = i + 1; j < amb_dim; ++j) {
                VectorZZ v(amb_dim, BigInt(0));
                v[i] = q;
                v[j] = -q;
                candidates.push_back(std::move(v));
            }
        }
    }

    return filter_in_lattice(lat, candidates);
}

} // namespace dbdd
