# cmake/FindGMP.cmake
# Find GNU Multiple Precision Arithmetic Library (GMP) and its C++ wrapper (GMPXX)
#
# Output Targets:
#   GMP::GMP   - C interface
#   GMP::GMPXX - C++ interface

find_path(GMP_INCLUDE_DIR NAMES gmp.h PATHS /usr/include /usr/local/include)
find_library(GMP_LIBRARIES NAMES gmp PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

find_path(GMPXX_INCLUDE_DIR NAMES gmpxx.h PATHS /usr/include /usr/local/include)
find_library(GMPXX_LIBRARIES NAMES gmpxx PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP
    REQUIRED_VARS GMP_LIBRARIES GMP_INCLUDE_DIR GMPXX_LIBRARIES GMPXX_INCLUDE_DIR
)

if(GMP_FOUND)
    if(NOT TARGET GMP::GMP)
        add_library(GMP::GMP UNKNOWN IMPORTED)
        set_target_properties(GMP::GMP PROPERTIES
            IMPORTED_LOCATION "${GMP_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
        )
    endif()

    if(NOT TARGET GMP::GMPXX)
        add_library(GMP::GMPXX UNKNOWN IMPORTED)
        set_target_properties(GMP::GMPXX PROPERTIES
            IMPORTED_LOCATION "${GMPXX_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMPXX_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES GMP::GMP
        )
    endif()
endif()
