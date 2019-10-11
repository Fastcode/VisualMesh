# Copyright (C) 2017 Trent Houliston <trent@houliston.me>
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

cmake_minimum_required(VERSION 3.1.0)
project(VisualMesh VERSION 1.0.0)

# Check if we are built as a subproject
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(TOP_LEVEL_PROJECT ON)

  # Force enable diagnostic colours for clang and gcc
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
else()
  set(TOP_LEVEL_PROJECT OFF)
endif()

# Turn on warnings
if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Werror)
endif()

# All the header files for this library
file(GLOB hdr "src/**.hpp")

# Convert our opencl files into header files with their contents and put them in the build folder
file(GLOB cls "src/engine/opencl/kernels/**.cl")
foreach(cl IN LISTS cls)

  file(RELATIVE_PATH cl_path ${CMAKE_CURRENT_SOURCE_DIR} ${cl})

  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp"
    COMMAND
      ${PYTHON_EXECUTABLE} ARGS "${PROJECT_SOURCE_DIR}/cmake/Scripts/wrap_opencl.py" "${cl}"
      "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp"
    DEPENDS "${cl}" "${PROJECT_SOURCE_DIR}/cmake/Scripts/wrap_opencl.py"
    COMMENT "Wrapping OpenCL file ${cl_path} in a header"
  )
  list(APPEND hdr "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp")
endforeach(cl ${cls})

add_custom_target(visualmesh_sources DEPENDS ${hdr})

# Define the VisualMesh library as header only but setup it's links
add_library(libvisualmesh INTERFACE)
target_include_directories(libvisualmesh INTERFACE src)
target_include_directories(libvisualmesh INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/src)
add_dependencies(libvisualmesh visualmesh_sources)

# Grab OpenCL if it's available
find_package(OpenCL)
if(OpenCL_FOUND)
  target_include_directories(libvisualmesh INTERFACE ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(libvisualmesh INTERFACE ${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)

if(TOP_LEVEL_PROJECT)
  # Build the tensorflow op for training
  option(BUILD_TENSORFLOW_OP "Build the tensorflow op used for training" ON)
  if(BUILD_TENSORFLOW_OP)
    # Add this directory so we can find the FindTensorflow.cmake
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")
    find_package(TensorFlow REQUIRED)

    add_library(tf_op SHARED "src/tf_op.cpp" ${hdr})
    target_compile_options(tf_op PRIVATE -march=native -mtune=native)
    set_target_properties(
      tf_op
      PROPERTIES
        CXX_STANDARD
        14
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/training
        PREFIX
        ""
        OUTPUT_NAME
        visualmesh_op
        SUFFIX
        ".so"
    )
    target_include_directories(tf_op SYSTEM PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
    target_include_directories(tf_op PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(tf_op ${TENSORFLOW_LIBRARIES})

  endif(BUILD_TENSORFLOW_OP)

  # Build an example program that classifies using the visual mesh
  option(BUILD_EXAMPLE "Build the example program that executes the mesh" OFF)
  if(BUILD_EXAMPLE)

    find_package(OpenCV REQUIRED)
    find_package(YAML-CPP REQUIRED)
    find_package(OpenCL REQUIRED)
    find_package(Threads REQUIRED)

    add_executable(main "example/main.cpp" ${hdr})
    target_compile_options(main PRIVATE -march=native -mtune=native)
    set_target_properties(main PROPERTIES CXX_STANDARD 14)
    target_include_directories(main SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_include_directories(main SYSTEM PRIVATE ${YAML_CPP_INCLUDE_DIR})
    target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(main PRIVATE ${PROJECT_BINARY_DIR}/src)
    target_link_libraries(
      main
      ${OpenCL_LIBRARIES}
      ${OpenCV_LIBS}
      ${fmt_LIBRARIES}
      ${YAML_CPP_LIBRARIES}
      Threads::Threads
    )

    add_executable(mesh_quality "example/mesh_quality.cpp" ${hdr})
    target_compile_options(mesh_quality PRIVATE -march=native -mtune=native)
    set_target_properties(mesh_quality PROPERTIES CXX_STANDARD 14)
    target_include_directories(mesh_quality PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(mesh_quality PRIVATE ${PROJECT_BINARY_DIR}/src)
  endif(BUILD_EXAMPLE)
endif(TOP_LEVEL_PROJECT)
