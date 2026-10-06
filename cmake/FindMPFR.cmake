# cmake/FindMPFR.cmake
# Find Multiple Precision Floating-Point Reliable Library (MPFR)
#
# Output Targets:
#   MPFR::MPFR - C MPFR interface

find_path(MPFR_INCLUDE_DIR NAMES mpfr.h PATHS /usr/include /usr/local/include)
find_library(MPFR_LIBRARIES NAMES mpfr PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR
    REQUIRED_VARS MPFR_LIBRARIES MPFR_INCLUDE_DIR
)

if(MPFR_FOUND)
    if(NOT TARGET MPFR::MPFR)
        add_library(MPFR::MPFR UNKNOWN IMPORTED)
        set_target_properties(MPFR::MPFR PROPERTIES
            IMPORTED_LOCATION "${MPFR_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES GMP::GMP
        )
    endif()
endif()
