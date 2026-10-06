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
