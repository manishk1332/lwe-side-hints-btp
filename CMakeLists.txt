cmake_minimum_required(VERSION 3.20)
project(dbdd VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wunused
        -Woverloaded-virtual
    )
endif()

# Compiler-specific enhancements
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fcolor-diagnostics)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-glldb)
    endif()

    option(USE_LLD "Use LLVM lld linker with Clang" ON)
    if(USE_LLD)
        add_link_options(-fuse-ld=lld)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fdiagnostics-color=always)
endif()

# Optional Sanitizers
option(ENABLE_SANITIZERS "Enable Address and Undefined Behavior Sanitizers" OFF)
if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Dependencies
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

# Core Library
add_library(dbdd STATIC
    src/core/exceptions.cpp
    src/arithmetic/precision.cpp
    src/arithmetic/integer.cpp
    src/arithmetic/rational.cpp
    src/arithmetic/real.cpp
    src/linalg/projection.cpp
    src/linalg/gram_schmidt.cpp
    src/lattice/basis.cpp
    src/lattice/volume.cpp
    src/lattice/dual.cpp
    src/lattice/primitive.cpp
    src/lattice/lattice.cpp
    src/lattice/q_vectors.cpp
    src/reduction/lll_exact.cpp
    src/reduction/lll_float.cpp
    src/reduction/svp.cpp
    src/reduction/enumeration.cpp
    src/reduction/bkz.cpp
    src/lwe/distribution.cpp
    src/lwe/lwe_instance.cpp
    src/lwe/embedding.cpp
    src/dbdd/ellipsoid.cpp
    src/dbdd/dbdd_instance.cpp
    src/hints/perfect_hint.cpp
    src/hints/modular_hint.cpp
    src/hints/approximate_hint.cpp
    src/hints/aposteriori_hint.cpp
    src/hints/short_vector_hint.cpp
    src/hints/worthiness.cpp
    src/hints/greedy_selection.cpp
    src/hints/budgeted_selection.cpp
    src/hints/candidate_pools.cpp
    src/lwe/secret_recovery.cpp
    src/linalg/eigendecomposition.cpp
    src/dbdd/homogenization.cpp
    src/dbdd/usvp_instance.cpp
    src/dbdd/isotropization.cpp
    src/security/root_hermite.cpp
    src/security/gsa.cpp
    src/security/gsa_intersect.cpp
    src/security/lightweight_predictor.cpp
    src/security/bkz_simulator.cpp
    src/security/probabilistic_simulation.cpp
    src/security/security_estimator.cpp
)

add_library(dbdd::dbdd ALIAS dbdd)

target_include_directories(dbdd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(dbdd PUBLIC
    GMP::GMPXX
    GMP::GMP
    MPFR::MPFR
    Eigen3::Eigen
)

# Command-Line Interface (Phase 20)
option(BUILD_CLI "Build DBDD command-line interface" ON)
if(BUILD_CLI)
    add_executable(dbdd_cli src/cli/main.cpp)
    target_link_libraries(dbdd_cli PRIVATE dbdd)

    add_executable(dbdd_demo_runner src/cli/demo_runner.cpp)
    target_link_libraries(dbdd_demo_runner PRIVATE dbdd)
endif()

# Examples (Phase 20)
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES)
    add_executable(example_01_toy_pipeline examples/01_toy_pipeline.cpp)
    target_link_libraries(example_01_toy_pipeline PRIVATE dbdd)

    add_executable(example_02_paper_benchmark examples/02_paper_benchmark.cpp)
    target_link_libraries(example_02_paper_benchmark PRIVATE dbdd)

    add_executable(example_03_structural_hints examples/03_structural_hints.cpp)
    target_link_libraries(example_03_structural_hints PRIVATE dbdd)

    add_executable(example_06_budgeted_selection examples/06_budgeted_selection.cpp)
    target_link_libraries(example_06_budgeted_selection PRIVATE dbdd)

    add_executable(example_07_attack_validation examples/07_attack_validation.cpp)
    target_link_libraries(example_07_attack_validation PRIVATE dbdd)
endif()

# Benchmarks (Phase 20)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
if(BUILD_BENCHMARKS)
    add_executable(bench_lll benchmarks/bench_lll.cpp)
    target_link_libraries(bench_lll PRIVATE dbdd)

    add_executable(bench_enumeration benchmarks/bench_enumeration.cpp)
    target_link_libraries(bench_enumeration PRIVATE dbdd)

    add_executable(bench_bkz benchmarks/bench_bkz.cpp)
    target_link_libraries(bench_bkz PRIVATE dbdd)

    add_executable(bench_security_estimator benchmarks/bench_security_estimator.cpp)
    target_link_libraries(bench_security_estimator PRIVATE dbdd)
endif()

# Testing
option(BUILD_TESTING "Build unit tests" ON)
if(BUILD_TESTING)
    include(CTest)
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        message(STATUS "Catch2 v3 not found on system. Fetching via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.5.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    add_subdirectory(tests)
endif()
