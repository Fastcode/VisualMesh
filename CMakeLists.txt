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

ADD_COMPILE_OPTIONS(
                    -std=c++14
                    -march=native
                    -mtune=native
)

# We use additional modules that cmake needs to know about
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Setup our compiler settings
INCLUDE(CompilerSetup)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(YAML-CPP REQUIRED)
FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(fmt REQUIRED)

# Find our files
FILE(GLOB hdr
    "src/**.hpp")

FILE(GLOB cls
     "src/cl/**.cl")

# Convert our opencl files into header files with their contents
FOREACH(cl ${cls})

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

INCLUDE_DIRECTORIES(SYSTEM ${OpenCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(SYSTEM ${fmt_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(SYSTEM ${YAML-CPP_INCLUDE_DIRS})

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/src")

ADD_EXECUTABLE(view_mesh "src/view_mesh.cpp" ${hdr})
TARGET_LINK_LIBRARIES(view_mesh ${OpenCL_LIBRARIES} ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML-CPP_LIBRARIES})

ADD_EXECUTABLE(make_mesh "src/make_mesh.cpp" ${hdr})
TARGET_LINK_LIBRARIES(make_mesh ${OpenCL_LIBRARIES} ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML-CPP_LIBRARIES})
