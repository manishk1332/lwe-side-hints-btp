# tests/CMakeLists.txt

macro(add_dbdd_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE dbdd Catch2::Catch2WithMain)
    add_test(NAME ${target_name} COMMAND ${target_name})
endmacro()

# Phase 01: Arithmetic Tests
add_dbdd_test(test_integer arithmetic/test_integer.cpp)
add_dbdd_test(test_rational arithmetic/test_rational.cpp)
add_dbdd_test(test_real arithmetic/test_real.cpp)

# Phase 01: LinAlg Tests
add_dbdd_test(test_vector linalg/test_vector.cpp)
add_dbdd_test(test_matrix linalg/test_matrix.cpp)

# Phase 02: Lattice Mathematics Tests
add_dbdd_test(test_projection lattice/test_projection.cpp)
add_dbdd_test(test_gram_schmidt lattice/test_gram_schmidt.cpp)
add_dbdd_test(test_basis lattice/test_basis.cpp)
add_dbdd_test(test_volume lattice/test_volume.cpp)
add_dbdd_test(test_dual lattice/test_dual.cpp)
add_dbdd_test(test_primitive lattice/test_primitive.cpp)
add_dbdd_test(test_lattice lattice/test_lattice.cpp)

# Phase 03: Lattice Reduction Tests
add_dbdd_test(test_lll_exact reduction/test_lll_exact.cpp)
add_dbdd_test(test_lll_float reduction/test_lll_float.cpp)

# Phase 04: LWE Distributions & Kannan Embedding Tests
add_dbdd_test(test_distribution lwe/test_distribution.cpp)
add_dbdd_test(test_instance lwe/test_instance.cpp)
add_dbdd_test(test_embedding lwe/test_embedding.cpp)

# Phase 05: DBDD Core Tests
add_dbdd_test(test_ellipsoid dbdd/test_ellipsoid.cpp)
add_dbdd_test(test_dbdd_instance dbdd/test_dbdd_instance.cpp)

# Phase 06: Perfect Hints Tests
add_dbdd_test(test_perfect_hint hints/test_perfect_hint.cpp)

