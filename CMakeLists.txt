# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.7.0)
project(VisualMesh VERSION 2.0.0)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# If this option is set we are building using continous integration
option(CI_BUILD "Enable build options for building in the CI server" OFF)

# Default not to run the clang-tidy checks, default to whatever our CI_BUILD is
option(ENABLE_CLANG_TIDY "Enable building with clang-tidy checks." OFF)
if(ENABLE_CLANG_TIDY OR CI_BUILD)
    find_package(PythonInterp 3 REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${PYTHON_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/Scripts/clang-tidy.py"
                             "${PROJECT_BINARY_DIR}/clang-tidy-fixes" clang-tidy)
    set(CMAKE_C_CLANG_TIDY ${CMAKE_CXX_CLANG_TIDY})

    # Create a target that will apply clang-tidy fixes to the codebase
    add_custom_target(
        apply-clang-tidy
        COMMAND clang-apply-replacements --format --style=file --style-config="${PROJECT_SOURCE_DIR}"
                --remove-change-desc-files "${PROJECT_BINARY_DIR}/clang-tidy-fixes"
        COMMENT "Applying fixes from clang-tidy to the codebase.")
endif(ENABLE_CLANG_TIDY OR CI_BUILD)

# Add the cmake module path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Configure the c++ header only library and build generated files
add_subdirectory("cpp")

# Build the tensorflow op for training
option(BUILD_TENSORFLOW_OP "Bu ild the tensorflow op used for training" ON)
if(BUILD_TENSORFLOW_OP)
    add_subdirectory("tensorflow")
endif(BUILD_TENSORFLOW_OP)

# Build the c++ examples
option(BUILD_EXAMPLES "Build the c++ examples" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory("example")
endif(BUILD_EXAMPLES)
