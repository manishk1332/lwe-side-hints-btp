#include "dbdd/lattice/lattice.hpp"

namespace dbdd {

template class Lattice<BigInt>;
template class Lattice<BigRational>;
template class Lattice<BigReal>;
template class Lattice<double>;

} // namespace dbdd
