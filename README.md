# DBDD: LWE with Side Information in C++20

[![Standard: C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![Build System: CMake 3.20+](https://img.shields.io/badge/CMake-3.20%2B-informational.svg)](https://cmake.org)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org)

A high-performance C++20 implementation and interactive visualizer for the cryptanalytic framework from:
> **"LWE with Side Information: Attacks and Concrete Security Estimation"**  
> *Dana Dachman-Soled, Léo Ducas, Huijing Gong, Mélissa Rossi*  
> **CRYPTO 2020** / [IACR ePrint 2020/292](https://eprint.iacr.org/2020/292).

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Prerequisites & Dependencies](#3-prerequisites--dependencies)
  - [C++ Toolchain & Libraries](#c-toolchain--libraries)
  - [Python Virtual Environment](#python-virtual-environment)
- [4. Compilation & Build Instructions](#4-compilation--build-instructions)
  - [Standard Build](#standard-build)
  - [Build Using CMake Presets](#build-using-cmake-presets)
  - [Generated Binaries](#generated-binaries)
- [5. Running the Web Demo Frontend](#5-running-the-web-demo-frontend)
  - [Starting the Demo Server](#starting-the-demo-server)
  - [Running Frontend Integration Tests](#running-frontend-integration-tests)
- [6. Running the Command-Line Interface (`dbdd_cli`)](#6-running-the-command-line-interface-dbdd_cli)
- [7. Running the Demonstrations](#7-running-the-demonstrations)
- [8. Running the Micro-Benchmarks](#8-running-the-micro-benchmarks)
- [9. Running the Test Suite](#9-running-the-test-suite)
- [10. Citation](#10-citation)

---

## 1. Overview

This library provides an end-to-end framework to analyze and solve Learning With Errors (LWE) instances with side information:
- **Kannan Embedding**: Embeds inhomogeneous LWE relations into Distorted Bounded Distance Decoding (DBDD) primal lattices with ellipsoidal error distributions.
- **Side Hint Processors**: Integrates Perfect, Modular, Approximate (Kalman Bayesian update), and Short-Vector hints.
- **Lattice Transformations**: Homogenization (mean-centering) and Isotropization (spectral covariance whitening $\Sigma \to I$).
- **Hardness Estimation**: GSA-Intersect root-Hermite security predictor and progressive BKZ hardness estimation.
- **Self-Contained Lattice Reduction**: Native exact-rational and floating-point LLL, Schnorr-Euchner SVP enumeration, and BKZ reduction.
- **Interactive Web Visualizer**: Browser-based slide-ready visualizer to inspect lattice metrics, step-by-step vector slices, and KaTeX mathematical formulations.

---

## 2. Repository Structure

```
lwe-side-hints-btp/
├── CMakeLists.txt          # Root CMake build configuration
├── CMakePresets.json       # Build presets (release, debug, sanitizers)
├── README.md               # Project setup and usage manual
├── cmake/                  # CMake discovery modules (FindGMP.cmake, FindMPFR.cmake)
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
├── src/                    # Source implementations
│   ├── cli/                # Command-line executables
│   │   ├── main.cpp        # dbdd_cli executable source
│   │   └── demo_runner.cpp # dbdd_demo_runner JSON pipeline exporter
│   └── ...                 # Core algorithmic implementations
├── examples/               # End-to-end C++ demonstration examples
│   ├── 01_toy_pipeline.cpp
│   ├── 02_paper_benchmark.cpp
│   └── 03_structural_hints.cpp
├── benchmarks/             # Standalone performance micro-benchmark suites
├── tests/                  # Catch2 unit and integration test suites
├── scripts/                # Python scripts and demo server
│   ├── demo_server.py      # Zero-dependency HTTP server for the web visualizer
│   └── test_demo_server.py # Automated integration tests for the demo server
└── web/                    # Slide-ready interactive web frontend
    ├── index.html          # Main HTML entry point
    ├── styles.css          # UI styles, theme modes, and responsive layout
    └── app.js              # Pipeline runner, KaTeX formulas, and step visualizer
```

---

## 3. Prerequisites & Dependencies

### C++ Toolchain & Libraries
- **C++ Compiler**: A C++20 compliant compiler:
  - Clang 18+ (recommended)
  - GCC 11+
- **Build System**: CMake ≥ 3.20 and [Ninja](https://ninja-build.org/)
- **Required Libraries**:
  - **GMP** (`libgmp-dev`): Arbitrary-precision integer arithmetic
  - **MPFR** (`libmpfr-dev`): Arbitrary-precision floating-point arithmetic
  - **Eigen 3** (`libeigen3-dev` ≥ 3.4): Dense linear algebra
  - **Catch2 v3**: Unit testing (automatically fetched via CMake `FetchContent` if not pre-installed)

#### Installing Dependencies (Ubuntu / Debian)
```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build clang lld \
                                       libgmp-dev libmpfr-dev libeigen3-dev python3 python3-venv
```

### Python Virtual Environment
Python 3.8+ is used to run the web demo server and its integration test harness.

```bash
# 1. Create a virtual environment in the project root
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate
```

> **Note**: `scripts/demo_server.py` relies exclusively on Python standard library modules (`http.server`, `subprocess`, `json`, `urllib`). No external pip dependencies are required.

---

## 4. Compilation & Build Instructions

### Standard Build
Configure and build all targets using CMake and Ninja:

```bash
# 1. Configure in Release mode
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. Compile all targets
cmake --build build -j$(nproc)
```

### Build Using CMake Presets
Alternatively, compile using presets defined in `CMakePresets.json`:

```bash
# Configure using release preset
cmake --preset release

# Build using release preset
cmake --build --preset release -j$(nproc)
```

Available presets include: `release`, `debug`, `sanitizers` (ASan/UBSan), `clang-release`, `clang-debug`, `gcc-release`, and `gcc-debug`.

### Generated Binaries
Upon compilation, the following binaries are generated inside `build/`:
- **CLI Executable**: `build/dbdd_cli`
- **Web Demo Backend**: `build/dbdd_demo_runner`
- **Demonstrations**: `build/example_01_toy_pipeline`, `build/example_02_paper_benchmark`, `build/example_03_structural_hints`
- **Micro-Benchmarks**: `build/bench_lll`, `build/bench_enumeration`, `build/bench_bkz`, `build/bench_security_estimator`
- **Unit Test Binaries**: Located in `build/tests/test_*`

---

## 5. Running the Web Demo Frontend

The web visualizer allows interactive configuration of LWE dimensions, error distribution, and side hints, rendering step-by-step lattice metrics, basis vectors, and KaTeX mathematical equations.

### Starting the Demo Server
Ensure the C++ demo binary `build/dbdd_demo_runner` is compiled, activate the Python environment, and start the local server:

```bash
# 1. Activate python environment
source .venv/bin/activate

# 2. Start the demo server (default port 8080)
python3 scripts/demo_server.py --port 8080
```

Open your browser and navigate to:
```
http://localhost:8080
```

#### Key Frontend Capabilities
- **Step 0 (LWE Setup)**: Inspects the initial public matrix $A$, public vector $b$, ground-truth secret $s$, and noise $e$.
- **Step 1 (Kannan Embedding)**: Validates target vector containment in the embedded primal lattice.
- **Hint Stepper & Timeline**: Navigate step-by-step or view the full timeline of hint integrations.
- **Formula Inspector**: Click `📐 View Formulas & Step Calculations` to view equations rendered via KaTeX.
- **Secret Reconstruction**: Inspects coordinate-by-coordinate modular verification ($\mathbf{b}_{\text{calc}} \equiv \mathbf{b} \pmod q$).
- **Slide Mode**: Toggle presentation mode with `S` or the top-right button for presentation screenshots.

### Running Frontend Integration Tests
Verify the demo server, API endpoints, and C++ pipeline integration:

```bash
python3 scripts/test_demo_server.py
```

---

## 6. Running the Command-Line Interface (`dbdd_cli`)

The `dbdd_cli` tool provides command-line control for generating instances, estimating hardness, and executing attacks:

```bash
# Generate synthetic LWE parameters
./build/dbdd_cli generate-lwe --n 50 --m 50 --q 3301 --sigma 3.2 --dist gaussian

# Inspect Kannan embedding and covariance ellipsoid
./build/dbdd_cli inspect-lwe --n 10 --m 10 --q 257 --sigma 2.0 --dist ternary

# Estimate concrete hardness under side information
./build/dbdd_cli estimate --n 70 --m 70 --q 3301 --sigma 4.472 --perfect-hints 10 --modular-hints 5

# Execute BKZ attack to recover secret
./build/dbdd_cli attack --n 5 --m 5 --q 31 --sigma 1.0 --perfect-hints 1 --blocksize 10

# Run all internal benchmark suites
./build/dbdd_cli benchmark --suite all
```

---

## 7. Running the Demonstrations

Run the built demonstration binaries directly from the project root:

### Toy Pipeline (`example_01_toy_pipeline`)
Executes an end-to-end secret recovery attack on an LWE instance ($n=3, m=3, q=11$) with uniform ternary secret across Kannan embedding, short-vector hints, perfect hints, approximate hints, homogenization, isotropization, and BKZ reduction:
```bash
./build/example_01_toy_pipeline
```

### Paper Benchmark Reproduction (`example_02_paper_benchmark`)
Reproduces the benchmark sweeps from Dachman-Soled et al. Section 5.2 (Table 1 & Figure 5) on $n=m=70, q=3301, \sigma=\sqrt{20}$ across baseline, perfect hints, modular hints (mod 3), and approximate hints:
```bash
./build/example_02_paper_benchmark
```

### Structural Hints Demonstration (`example_03_structural_hints`)
Demonstrates cryptanalysis of balanced ternary secrets ($\sum s_i = 0$), $q$-vector generation, analytical worthiness evaluation, and greedy short-vector hint selection:
```bash
./build/example_03_structural_hints
```

---

## 8. Running the Micro-Benchmarks

Run the standalone performance benchmark binaries inside `build/`:

```bash
# LLL reduction performance (exact rational vs float MPFR)
./build/bench_lll

# Schnorr-Euchner SVP enumeration throughput across dimensions
./build/bench_enumeration

# BKZ reduction timing and tour metrics across block sizes
./build/bench_bkz

# Security estimator operations per second
./build/bench_security_estimator
```

You can also run all benchmarks through the CLI:
```bash
./build/dbdd_cli benchmark --suite all
```

---

## 9. Running the Test Suite

Execute all Catch2 test suites covering mathematical invariants, hint processors, and integration pipelines:

```bash
ctest --test-dir build --output-on-failure
```

To run a specific test suite directly:
```bash
./build/tests/test_paper_reproduction
./build/tests/test_dbdd_instance
./build/tests/test_bkz
```

---

## 10. Citation

If you use this library in your research, please cite the foundational paper:

```bibtex
@inproceedings{DBLP:conf/crypto/Dachman-SoledDG20,
  author    = {Dana Dachman{-}Soled and
               L{\'{e}}o Ducas and
               Huijing Gong and
               M{\'{e}}lissa Rossi},
  title     = {LWE with Side Information: Attacks and Concrete Security Estimation},
  booktitle = {Advances in Cryptology - {CRYPTO} 2020},
  series    = {Lecture Notes in Computer Science},
  volume    = {12171},
  pages     = {329--358},
  publisher = {Springer},
  year      = {2020},
  doi       = {10.1007/978-3-030-56880-1\_12}
}
```
