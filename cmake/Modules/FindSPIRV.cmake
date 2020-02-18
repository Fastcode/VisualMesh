#[=======================================================================[.rst:
FindSPIRV
---------

Find SPIRV, which is a simple binary intermediate language for graphical
shaders and compute kernels.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``SPIRV::SPIRV``, if
SPIRV has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables::

  SPIRV_FOUND          - "True" if SPIRV was found
  SPIRV_INCLUDE_DIRS   - include directories for SPIRV
  SPIRV_LIBRARIES      - link against this library to use SPIRV

The module will also define three cache variables::

  SPIRV_INCLUDE_DIR     - the SPIRV include directory
  SPIRV_LIBRARY_DEBUG   - the path to the SPIRV debug library
  SPIRV_LIBRARY_RELEASE - the path to the SPIRV optimised library

#]=======================================================================]

find_path(SPIRV_INCLUDE_DIR NAMES "spirv/unified1/spirv.h")
find_library(SPIRV_LIBRARY_DEBUG NAMES SPIRVd SPIRV PATH_SUFFIXES lib)
find_library(SPIRV_LIBRARY_RELEASE NAMES SPIRV PATH_SUFFIXES lib)
set(SPIRV_LIBRARY
    debug
    ${SPIRV_LIBRARY_DEBUG}
    optimized
    ${SPIRV_LIBRARY_RELEASE})

set(SPIRV_LIBRARIES ${SPIRV_LIBRARY})
set(SPIRV_INCLUDE_DIRS ${SPIRV_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SPIRV
                                  REQUIRED_VARS
                                  SPIRV_LIBRARY
                                  SPIRV_INCLUDE_DIR)

mark_as_advanced(SPIRV_INCLUDE_DIR SPIRV_LIBRARY_DEBUG SPIRV_LIBRARY_RELEASE)

if(SPIRV_FOUND AND NOT TARGET SPIRV::SPIRV)
  add_library(SPIRV::SPIRV UNKNOWN IMPORTED)
  set_target_properties(SPIRV::SPIRV
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                   "${SPIRV_INCLUDE_DIRS}")

  if(SPIRV_LIBRARY_DEBUG AND SPIRV_LIBRARY_RELEASE)
    set_target_properties(SPIRV::SPIRV
                          PROPERTIES IMPORTED_LOCATION_DEBUG
                                     "${SPIRV_LIBRARY_DEBUG}"
                                     IMPORTED_LOCATION_RELEASE
                                     "${SPIRV_LIBRARY_RELEASE}")
  else()
    set_target_properties(SPIRV::SPIRV
                          PROPERTIES IMPORTED_LOCATION
                                     "${SPIRV_LIBRARY_RELEASE}")
  endif()
endif()
