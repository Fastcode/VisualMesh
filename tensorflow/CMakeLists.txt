# Add this directory so we can find the FindTensorflow.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")
find_package(TensorFlow REQUIRED)

add_library(tf_op SHARED "map.cpp" "unmap.cpp" "lookup.cpp" "difference.cpp" ${hdr})
target_compile_options(tf_op PRIVATE -march=native -mtune=native)
set_target_properties(
    tf_op
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/training/op"
               PREFIX ""
               OUTPUT_NAME visualmesh_op
               SUFFIX ".so")
target_include_directories(tf_op SYSTEM PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
target_link_libraries(tf_op visualmesh ${TENSORFLOW_LIBRARIES})
