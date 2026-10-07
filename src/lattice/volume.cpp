#include "dbdd/lattice/volume.hpp"

namespace dbdd {

template BigReal compute_log_volume_from_gs(const GramSchmidtState<BigRational>&);
template BigReal compute_log_volume_from_gs(const GramSchmidtState<BigReal>&);
template BigReal compute_log_volume_from_gs(const GramSchmidtState<double>&);

template BigReal compute_volume_from_gs(const GramSchmidtState<BigRational>&);
template BigReal compute_volume_from_gs(const GramSchmidtState<BigReal>&);
template BigReal compute_volume_from_gs(const GramSchmidtState<double>&);

template BigReal compute_log_volume(const Basis<BigInt>&);
template BigReal compute_log_volume(const Basis<BigRational>&);
template BigReal compute_log_volume(const Basis<BigReal>&);
template BigReal compute_log_volume(const Basis<double>&);

template BigReal compute_volume(const Basis<BigInt>&);
template BigReal compute_volume(const Basis<BigRational>&);
template BigReal compute_volume(const Basis<BigReal>&);
template BigReal compute_volume(const Basis<double>&);

} // namespace dbdd
