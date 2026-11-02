# Mathematical Conventions

This document specifies the mandatory mathematical conventions for the C++20 implementation of:
*"LWE with Side Information: Attacks and Concrete Security Estimation"* (Albrecht et al., 2020).

These conventions prevent transpose, sign, and coordinate orientation bugs across the codebase.

---

## 1. Basis & Lattice Convention
- **Rows are basis vectors**:
  If $B \in \mathbb{R}^{d \times m}$ is a basis matrix for lattice $\Lambda$, the basis vectors are rows:
  $$B = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_d \end{pmatrix}$$
- **Lattice representation**:
  Vectors in the lattice are formed by row-vector integer linear combinations:
  $$v = x B, \quad x \in \mathbb{Z}^d$$
- **Dimension vs Ambient Dimension**:
  - `dimension`: number of basis vectors $d$ (rank of basis).
  - `ambient_dimension`: number of coordinates per vector $m$.
  - In full-rank lattices, $d = m$.

---

## 2. Vectors & Inner Products
- **Vector orientation**:
  Public mathematical APIs expose row-vector semantics matching paper notation.
- **Inner product**:
  For vectors $u, v \in \mathbb{R}^m$:
  $$\langle u, v \rangle = u v^T = \sum_{i=1}^m u_i v_i$$
- **Vector norm**:
  $$\|v\|^2 = \langle v, v \rangle = \sum_{i=1}^m v_i^2$$

---

## 3. Matrix Multiplications & Covariance
- **Matrix-Vector multiplication**:
  Row vector $v$ on the left: $v M$.
- **Covariance matrix $\Sigma$**:
  - $\Sigma$ represents $\mathbb{E}[(x - \mu)^T (x - \mu)]$.
  - $\Sigma$ is symmetric positive semidefinite (PSD).
  - Symmetry must always be explicitly enforced: $\Sigma \leftarrow \frac{1}{2}(\Sigma + \Sigma^T)$.
- **Ellipsoid Representation**:
  $$\mathcal{E}(\mu, \Sigma) = \{ x : (x - \mu) \Sigma^\dagger (x - \mu)^T \le 1 \}$$
  where $\Sigma^\dagger$ is the Moore-Penrose pseudoinverse.

---

## 4. Kannan Embedding & Sign Conventions
- **LWE Instance**:
  Given $A \in \mathbb{Z}_q^{m \times n}$, secret $s \in \mathbb{Z}^n$, error $e \in \mathbb{Z}^m$, and $b = s A^T + e \pmod q$ (with $s, e$ as row vectors),
  embedding basis $B \in \mathbb{Z}^{(m+n+1) \times (m+n+1)}$:
  $$B = \begin{pmatrix} q I_m & 0 & 0 \\ A^T & -I_n & 0 \\ b & 0 & 1 \end{pmatrix}$$
- **Embedded target vector**:
  $$\bar{s} = (e, s, 1)$$
- **Target invariant equation**:
  $$(k, -s, 1) B = (e, s, 1) = \bar{s}$$
  where $e = b - s A^T + k q$ for some $k \in \mathbb{Z}^m$.
- **Volume**:
  $$\mathrm{Vol}(\Lambda) = q^m$$

---

## 5. Homogenization
- Center becomes zero: $\mu_h = 0$.
- Covariance envelope: $\Sigma_h = \Sigma + \mu^T \mu$.
- Invariant inclusion: $\mathcal{E}(\mu, \Sigma) \subseteq \mathcal{E}(0, \Sigma + \mu^T \mu)$.

---

## 6. Isotropization
- Eigendecomposition of symmetric PSD $\Sigma_h$:
  $$\Sigma_h = Q \Lambda Q^T$$
- Square root and pseudoinverse:
  $$M = (\Sigma_h^{1/2})^\dagger = Q \Lambda^{-1/2} Q^T$$
  with eigenvalue thresholding for numerical rank.
- Transformed lattice: $\Lambda_{\text{iso}} = \Lambda M$.
- Isotropized volume shortcut:
  $$\mathrm{Vol}(\Lambda_{\text{iso}}) = \frac{\mathrm{Vol}(\Lambda)}{\sqrt{\mathrm{rdet}(\Sigma_h)}}$$
  where $\mathrm{rdet}$ is the restricted determinant on the lattice span.

---

## 7. Orthogonal Projections
- For row vector $v \neq 0$:
  $$P_v^\perp = I - \frac{v^T v}{\|v\|^2}$$
- Invariant properties:
  $$P^2 = P, \quad P^T = P, \quad v P = 0$$

---

## 8. Dual Lattice
- For full-rank basis $B$:
  $$D = (B^{-1})^T, \quad \text{so } B D^T = I$$
- Dual of dual satisfies: $(D^*)^* = B$ up to unimodular transformation.

---

## 9. Numerical Precision & Arithmetic Types
- **BigInt**: exact integer arithmetic via GMP `mpz_class`.
- **BigRational**: exact rational arithmetic via GMP `mpq_class`.
- **BigReal**: high-precision floating-point via MPFR with configurable precision (default 256 bits, tested at 128, 256, 512 bits).
- **No silent downcasting**: never cast BigInt/BigRational to double without explicit `to_double()`.
- **Log-domain evaluation**: all large powers $\delta_\beta^{2\beta - d - 1}$ and determinants must be evaluated in log space.
