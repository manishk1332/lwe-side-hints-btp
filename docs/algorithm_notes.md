# Algorithm Notes

This document contains algorithmic implementation details, complexity bounds, and numerical considerations.

## 1. Gram-Schmidt Orthogonalization
- **Modified Gram-Schmidt (MGS)**: Preferred over Classical Gram-Schmidt (CGS) for floating-point calculations due to numerical stability.
- **Exact / Rational GS**: Used in exact LLL to avoid floating-point drift and loss of Lovász condition accuracy.
- **Incremental updates**: When vectors are swapped in LLL, update local $\mu_{i,j}$ and $\|b_i^*\|^2$ without full $O(d^3)$ recomputation.

## 2. LLL Reduction
- **Exact LLL**:
  - Size reduction using exact integer/rational quotients: $q = \lfloor \mu_{k,j} \rceil$.
  - Lovász condition: $\|b_k^*\|^2 \ge (\delta - \mu_{k,k-1}^2) \|b_{k-1}^*\|^2$, typically $\delta = 3/4$ or $0.99$.
  - Invariant: $\det(B)$ preserved up to sign; lattice span preserved.
- **Floating LLL**:
  - MPFR-backed or double-precision with periodic precision checks and exact verification.

## 3. SVP Enumeration
- **Fincke-Pohst / Kannan coordinate enumeration**:
  - Enumerates over Gram-Schmidt coordinates: $\bar{v} = \sum_{i=1}^k x_i b_i^*$.
  - Pruning-free baseline solver for small dimensions ($k \le 40$).
  - Tracks node count and supports timeout / node limit aborts.

## 4. BKZ Reduction
- Tour-based structure with block size $\beta$.
- For each local projected block of dimension $\le \beta$:
  - Extract projected local basis.
  - Call local `SVPOracle`.
  - If a strictly shorter vector is found, insert it and size-reduce / LLL-reduce the block.
- Tour termination: no basis vector changes during a full tour, or maximum tours reached.

## 5. Security Estimator & GSA
- Condition for attack success:
  $$\sqrt{\beta} \le \delta_\beta^{2\beta - d - 1} \mathrm{Vol}(\Lambda)^{1/d}$$
- Log-scale objective:
  $$F(\beta, d, V) = (2\beta - d - 1) \ln(\delta_\beta) + \frac{1}{d} \ln(V) - \frac{1}{2}\ln(\beta)$$
- Attack predicted when $F(\beta, d, V) \ge 0$.
- Bisection / root-finding used for continuous $\beta$, followed by $\lceil \beta \rceil$ for integer BKZ block size.
