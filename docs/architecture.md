# System Architecture

## Architectural Principles

The project is structured into 5 strictly downward-dependent layers:

```
+-------------------------------------------------------------+
| Layer 5: Security Estimation                                |
| (GSA-Intersect, BKZ Simulator, Probabilistic Predictor)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Layer 4: Lattice Algorithms                                 |
| (Exact LLL, Float LLL, Exact SVP Enumeration, BKZ)          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Layer 3: DBDD Mathematics & LWE Embedding                   |
| (LWE Instance, Kannan Embedding, DBDDInstance, Hints)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Layer 2: Lattice Mathematics                                |
| (Basis, Volume, Dual Lattice, Projections, Gram-Schmidt)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Layer 1: Low-Level Arithmetic & Linear Algebra Wrappers     |
| (BigInt, BigRational, BigReal, Vector<T>, Matrix<T>)        |
+-------------------------------------------------------------+
```

### Dependency Rules:
1. Dependencies point strictly downward.
2. BKZ does NOT depend on DBDD.
3. DBDD does NOT depend on BKZ.
4. LWE does NOT depend on Security Estimation.
5. External lattice library `fplll` is an optional verification oracle only in tests/benchmarks, never in library code.

### Three-Tier Representation:
1. **Full DBDD (`DBDDInstance`)**:
   Maintains actual lattice basis $(\Lambda)$, full mean vector $(\mu)$, and full dense covariance matrix $(\Sigma)$. Supports all hint integrations and actual reduction/attack execution.
2. **Lightweight Predictor (`DBDDStateStatistics`)**:
   Maintains only dimension $d$, volume $\operatorname{Vol}(\Lambda)$, mean $\mu$, and covariance $\Sigma$ (no lattice basis). Used for fast security sweeps and hint optimization.
3. **Super-Lightweight Diagonal Predictor (`DiagonalCovariance`)**:
   Maintains diagonal entries of $\Sigma$, dimension, and volume for hint types that preserve diagonality.
