# DBDD: LWE with Side Information in C++20

[![Standard: C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![Build System: CMake 3.20+](https://img.shields.io/badge/CMake-3.20%2B-informational.svg)](https://cmake.org)

A self-contained C++20 implementation of the cryptanalytic framework from:
> **"LWE with Side Information: Attacks and Concrete Security Estimation"**  
> *Martin R. Albrecht, Rachel Player, Samed Düzlü, Michael Walter*  
> **CRYPTO 2020** / [IACR ePrint 2020/292](https://eprint.iacr.org/2020/292).

---

## Table of Contents

- [1. Overview & Key Capabilities](#1-overview--key-capabilities)
- [2. System Architecture](#2-system-architecture)
- [3. Repository Structure](#3-repository-structure)
- [4. Prerequisites & Dependencies](#4-prerequisites--dependencies)
- [5. Compilation & Build Instructions](#5-compilation--build-instructions)
- [6. Running the Demonstrations](#6-running-the-demonstrations)
  - [Toy Pipeline (example_01_toy_pipeline)](#toy-pipeline-example_01_toy_pipeline)
  - [Paper Benchmark Reproduction (example_02_paper_benchmark)](#paper-benchmark-reproduction-example_02_paper_benchmark)
  - [Structural Hints Demonstration (example_03_structural_hints)](#structural-hints-demonstration-example_03_structural_hints)
- [7. Running the Command-Line Interface (dbdd_cli)](#7-running-the-command-line-interface-dbdd_cli)
- [8. Running the Micro-Benchmarks](#8-running-the-micro-benchmarks)
- [9. Running the Test Suite](#9-running-the-test-suite)
- [10. Documentation & Citation](#10-documentation--citation)

---

## 1. Overview & Key Capabilities

This library provides an end-to-end framework to analyze and solve Learning With Errors (LWE) instances with side information:
- **LWE to DBDD / uSVP**: Translates LWE instances into Distorted Bounded Distance Decoding (DBDD) instances via Kannan's embedding with exact ellipsoidal error distributions.
- **Four Hint Processors**:
  - **Perfect Hints**: Exact hyperplane projection ($\langle s, v \rangle = c$), dimension drop, and volume expansion.
  - **Modular Hints**: Sublattice restriction ($\langle s, v \rangle \equiv c \pmod k$) with smooth-case volume scaling.
  - **Approximate Hints**: Kalman-filter covariance update ($\langle s, v \rangle + e_h = c, e_h \sim \mathcal{N}(0, \sigma_h^2)$) guaranteeing monotonically decreasing uncertainty.
  - **A-Posteriori Hints**: Coordinate distribution prior integration.
- **Lattice Transformations**:
  - **Homogenization**: Embeds inhomogeneous targets into homogeneous lattices $\bar{\Lambda} = \Lambda \times \mathbb{Z}$.
  - **Isotropization**: Spectral/Cholesky whitening transformation ($M^T \Sigma_h M \approx I$) for Euclidean reduction.
- **Dual Security Estimation**:
  - **GSA-Intersect**: Closed-form Root-Hermite $\delta_\beta$ intersection model.
  - **Probabilistic Simulator**: Chen-Nguyen (CN11) basis profile simulation with non-central $\chi^2$ detection probabilities.
- **Fully Self-Contained Lattice Algorithms**:
  - Native exact-rational and MPFR floating-point LLL.
  - Native depth-first Schnorr-Euchner SVP enumeration with pruning.
  - Native Block Korkine-Zolotarev (BKZ) reduction algorithm with tour tracking.
  - Zero external dependencies on `fplll` or other lattice libraries.

---

## 2. System Architecture

The codebase is organized into five strictly downward-dependent layers:

```
Layer 5: Security Estimation   (include/dbdd/security/)  -> GSA, BKZ Simulator, Probabilistic Estimator
Layer 4: Lattice Algorithms    (include/dbdd/reduction/) -> LLL (exact & float), SVP Enumeration, BKZ
Layer 3: DBDD & LWE            (include/dbdd/dbdd/, hints/, lwe/) -> Kannan, Hints, Homogenization
Layer 2: Lattice Mathematics   (include/dbdd/lattice/)   -> Basis, Volume, Dual, Projections, Gram-Schmidt
Layer 1: Arithmetic & LinAlg   (include/dbdd/arithmetic/, linalg/) -> BigInt, BigRational, BigReal, Matrix
```

The framework maintains three architectural tiers:
1. **Full DBDD (`DBDDInstance`)**: Concrete lattice basis $B \in \mathbb{Z}^{d \times m}$ and covariance $\Sigma$ for real lattice reduction attacks.
2. **Lightweight Predictor (`DBDDStateStatistics`)**: Closed-form analytical updates of $(\operatorname{Vol}, d, \mu, \Sigma)$ in microseconds without basis manipulation.
3. **Diagonal Covariance (`DiagonalCovariance`)**: Fast 1D coordinate variance tracker.

---

## 3. Repository Structure

The repository contains the following source layout:

```
lwe-side-hints-btp/
├── CMakeLists.txt          # Root CMake build configuration
├── CMakePresets.json       # Standard CMake configure & build presets
├── README.md               # Project documentation manual
├── cmake/                  # CMake discovery modules (FindGMP.cmake, FindMPFR.cmake)
├── docs/                   # In-depth architectural & mathematical documentation
│   ├── algorithm_notes.md
│   ├── architecture.md
│   ├── mathematical_conventions.md
│   └── paper_mapping.md
├── include/dbdd/           # Public C++20 header library
│   ├── arithmetic/         # BigInt, BigRational, BigReal MPFR wrappers
│   ├── core/               # Types, constants, structured exceptions
│   ├── dbdd/               # DBDDInstance, Ellipsoid, Homogenizer, Isotropizer
│   ├── hints/              # Perfect, Modular, Approximate, A-Posteriori, Short-Vector
│   ├── lattice/            # Basis, Volume, Dual, Primitive, QVectors
│   ├── linalg/             # Vector, Matrix, GramSchmidt, Projection
│   ├── lwe/                # LWEInstance, Distribution, KannanEmbedding, SecretRecovery
│   ├── reduction/          # LLLExact, LLLFloat, Enumeration, BKZ
│   └── security/           # GSA, RootHermite, BKZSimulator, Probabilistic, Predictor
├── src/                    # Source implementations and CLI
│   ├── cli/main.cpp        # dbdd_cli executable source
│   └── ...                 # Layer implementations
├── examples/               # End-to-end demonstrations
│   ├── 01_toy_pipeline.cpp
│   ├── 02_paper_benchmark.cpp
│   └── 03_structural_hints.cpp
├── benchmarks/             # Micro-benchmark suites (LLL, SVP, BKZ, Estimator)
└── tests/                  # 35 Catch2 test suites covering all layers
```

---

## 4. Prerequisites & Dependencies

### Required Toolchain
- **C++ Compiler**: A C++20 compliant compiler:
  - Clang 18+ (recommended, supports `-fuse-ld=lld` and `-glldb`)
  - GCC 11+
- **Build System**: CMake $\ge$ 3.20 and [Ninja](https://ninja-build.org/)

### Required Libraries
- **GMP** (`libgmp-dev`): Arbitrary-precision integer and rational arithmetic
- **MPFR** (`libmpfr-dev`): Arbitrary-precision floating-point arithmetic
- **Eigen 3** (`libeigen3-dev` $\ge$ 3.4): Dense matrix and vector operations
- **Catch2 v3**: Unit testing framework (automatically fetched via CMake `FetchContent` if not installed)

#### Installing Dependencies (Ubuntu / Debian)
```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build clang lld \
                                       libgmp-dev libmpfr-dev libeigen3-dev
```

---

## 5. Compilation & Build Instructions

### 5.1 Standard Build
Clone, configure, and build all targets using CMake and Ninja:

```bash
# 1. Clone the repository
git clone https://github.com/manishk1332/lwe-side-hints-btp.git
cd lwe-side-hints-btp

# 2. Configure the build directory in Release mode
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 3. Compile all targets (core library, CLI, examples, benchmarks, tests)
cmake --build build -j$(nproc)
```

### 5.2 Build Using CMake Presets
Alternatively, use the presets defined in `CMakePresets.json`:

```bash
# Configure using release preset
cmake --preset release

# Build using release preset
cmake --build --preset release -j$(nproc)
```

Available presets include: `release`, `debug`, `sanitizers` (ASan/UBSan), `clang-release`, `clang-debug`, `gcc-release`, and `gcc-debug`.

### 5.3 Generated Build Artifacts
Upon compilation, the following binaries are produced inside `build/`:
- **CLI Executable**: `build/dbdd_cli`
- **Demonstrations**: `build/example_01_toy_pipeline`, `build/example_02_paper_benchmark`, `build/example_03_structural_hints`
- **Micro-Benchmarks**: `build/bench_lll`, `build/bench_enumeration`, `build/bench_bkz`, `build/bench_security_estimator`
- **Unit Test Binaries**: Located in `build/tests/test_*`

---

## 6. Running the Demonstrations

After compiling, the built demonstration binaries can be executed directly from the project root:

### Toy Pipeline (`example_01_toy_pipeline`)
Executes an end-to-end secret recovery attack on an LWE instance ($n=3, m=3, q=11$) with uniform ternary secret:
1. LWE instance generation & Kannan embedding ($d=7, \operatorname{Vol}=1331$).
2. Short-vector projection & perfect hint integration (drops dimension to 5).
3. Approximate hint integration (Kalman covariance shrinkage).
4. Homogenization & Isotropization (whitening $\Sigma \to I$).
5. GSA & Probabilistic hardness estimation.
6. BKZ lattice reduction & secret extraction.
7. Verification against public instance (guarantees 100% exact recovery).

```bash
./build/example_01_toy_pipeline
```

### Paper Benchmark Reproduction (`example_02_paper_benchmark`)
Reproduces the benchmark sweeps from Albrecht et al. Section 5.2 (Table 1 & Figure 5) on $n=m=70, q=3301, \sigma=\sqrt{20}$ across baseline, perfect hints, modular hints (mod 3), and approximate hints:

```bash
./build/example_02_paper_benchmark
```

### Structural Hints Demonstration (`example_03_structural_hints`)
Demonstrates cryptanalysis of balanced ternary secrets ($\sum s_i = 0$), $q$-vector generation, analytical worthiness evaluation, and greedy short-vector hint selection:

```bash
./build/example_03_structural_hints
```

---

## 7. Running the Command-Line Interface (`dbdd_cli`)

The `dbdd_cli` binary provides an all-in-one interface for experiment automation:

```bash
# Generate synthetic LWE parameters
./build/dbdd_cli generate-lwe --n 50 --m 50 --q 3301 --sigma 3.2 --dist gaussian

# Inspect Kannan embedding and covariance ellipsoid
./build/dbdd_cli inspect-lwe --n 10 --m 10 --q 257 --sigma 2.0 --dist ternary

# Estimate concrete hardness under side information
./build/dbdd_cli estimate --n 70 --m 70 --q 3301 --sigma 4.472 --perfect-hints 10 --modular-hints 5

# Execute BKZ attack to recover secret
./build/dbdd_cli attack --n 5 --m 5 --q 31 --sigma 1.0 --perfect-hints 1 --blocksize 10

# Run internal benchmark suites
./build/dbdd_cli benchmark --suite all
```

---

## 8. Running the Micro-Benchmarks

Run the standalone performance and scaling benchmarks:

- **LLL Performance** (`./build/bench_lll`): Compares exact rational LLL vs float MPFR LLL ($d \in [10, 50]$).
- **SVP Enumeration** (`./build/bench_enumeration`): Evaluates Schnorr-Euchner tree search throughput ($d \in [10, 30]$).
- **BKZ Reduction** (`./build/bench_bkz`): Measures wall-clock time per tour and SVP calls across block sizes $\beta \in [5, 25]$.
- **Security Estimator** (`./build/bench_security_estimator`): Measures operations/sec for GSA-Intersect, Probabilistic, and Lightweight Predictors.

---

## 9. Running the Test Suite

Execute all 35 Catch2 test suites covering all 5 architectural layers and mathematical invariants:

```bash
ctest --test-dir build --output-on-failure
```

To run a specific test suite directly:
```bash
./build/tests/test_paper_reproduction
./build/tests/test_dbdd_instance
```

---

## 10. Documentation & Citation

### Detailed Documentation
For deep dives into design and math, see the tracked documentation in `docs/`:
- [`docs/architecture.md`](docs/architecture.md): Architectural layers and component boundaries.
- [`docs/mathematical_conventions.md`](docs/mathematical_conventions.md): Row-basis convention, PSD constraints, and numerical precision rules.
- [`docs/algorithm_notes.md`](docs/algorithm_notes.md): Algorithmic implementation notes for LLL, BKZ, and hint processors.
- [`docs/paper_mapping.md`](docs/paper_mapping.md): Mapping between the codebase and equations in Albrecht et al. (CRYPTO 2020).

### Citation
If you use this library in your research, please cite the foundational paper:

```bibtex
@inproceedings{DBLP:conf/crypto/AlbrechtPDW20,
  author    = {Martin R. Albrecht and
               Rachel Player and
               Samed D{\"{u}}zl{\"{u}} and
               Michael Walter},
  title     = {LWE with Side Information: Attacks and Concrete Security Estimation},
  booktitle = {Advances in Cryptology - {CRYPTO} 2020},
  series    = {Lecture Notes in Computer Science},
  volume    = {12171},
  pages     = {329--358},
  publisher = {Springer},
  year      = {2020},
  doi       = {10.1007/978-3-030-56784-2\_12}
}
```
