# Copyright (C) 2017 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.10)
PROJECT(VisualMesh)

SET(VERSION_MAJOR 1)
SET(VERSION_MINOR 0)
SET(VERSION_PATCH 0)

# We use position independent code
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# We use additional modules that cmake needs to know about
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

FIND_PACKAGE(OpenCL REQUIRED)

# Find our files
FILE(GLOB hdr
    "src/**.hpp")

FILE(GLOB cls
     "src/engine/opencl/kernels/**.cl")

# Convert our opencl files into header files with their contents
FOREACH(cl IN LISTS cls)

    FILE(RELATIVE_PATH cl_path ${CMAKE_CURRENT_SOURCE_DIR} ${cl})

    ADD_CUSTOM_COMMAND(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp"
        COMMAND ${PYTHON_EXECUTABLE}
        ARGS "${PROJECT_SOURCE_DIR}/cmake/Scripts/wrap_opencl.py"
             "${cl}"
             "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp"
        DEPENDS "${cl}"
                "${PROJECT_SOURCE_DIR}/cmake/Scripts/wrap_opencl.py"
        COMMENT "Wrapping OpenCL file ${cl_path} in a header"
    )
    LIST(APPEND hdr "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp")
ENDFOREACH(cl ${cls})

# We must now look in here for headers
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/src")

INCLUDE_DIRECTORIES(SYSTEM ${OpenCL_INCLUDE_DIRS})

OPTION(BUILD_TENSORFLOW_OP "Build the tensorflow op used for training" ON)

IF(BUILD_TENSORFLOW_OP)

    INCLUDE_DIRECTORIES(SYSTEM ${TENSORFLOW_INCLUDE_DIRS})
    FIND_PACKAGE(TensorFlow REQUIRED)

    ADD_LIBRARY(tf_op SHARED "src/tf_op.cpp" ${hdr})
    SET_PROPERTY(TARGET tf_op PROPERTY CXX_STANDARD 14)
    TARGET_COMPILE_OPTIONS(tf_op PRIVATE -fcolor-diagnostics)
    TARGET_LINK_LIBRARIES(tf_op ${TENSORFLOW_LIBRARIES})

ENDIF(BUILD_TENSORFLOW_OP)

# Temp while testing the mesh
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(YAML-CPP REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(SYSTEM ${YAML_CPP_INCLUDE_DIR})
ADD_EXECUTABLE(main "src/main.cpp" ${hdr})
SET_PROPERTY(TARGET main PROPERTY CXX_STANDARD 14)
TARGET_COMPILE_OPTIONS(main PRIVATE -fcolor-diagnostics -march=native -mtune=native)
TARGET_LINK_LIBRARIES(main ${OpenCL_LIBRARIES} ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML_CPP_LIBRARIES})
