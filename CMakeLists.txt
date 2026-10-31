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
    src/lwe/secret_recovery.cpp
    src/linalg/eigendecomposition.cpp
    src/dbdd/homogenization.cpp
    src/dbdd/usvp_instance.cpp
    src/dbdd/isotropization.cpp
    src/security/root_hermite.cpp
    src/security/gsa.cpp
    src/security/gsa_intersect.cpp
    src/security/lightweight_predictor.cpp
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
