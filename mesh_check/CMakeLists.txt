option(BUILD_MESH_CHECK "Build the mesh check tool" OFF)
if(BUILD_MESH_CHECK)

    # Find our dependencies
    find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)
    find_package(YAML-CPP REQUIRED)

    # Compile the executable
    add_executable(mesh_check "mesh_check.cpp")
    target_compile_options(mesh_check PRIVATE ${compile_options})
    target_link_libraries(mesh_check visualmesh ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES})

endif(BUILD_MESH_CHECK)
