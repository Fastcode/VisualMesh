option(BUILD_MESH_CHECK "Build the mesh check tool" OFF)
if(BUILD_MESH_CHECK)

    # Find our dependencies
    find_package(OpenCV COMPONENTS core highgui imgproc)
    find_package(YAML-CPP)

    # Compile the executable
    add_executable(mesh_types "mesh_types.cpp")
    target_compile_options(mesh_types PRIVATE ${compile_options})
    target_link_libraries(mesh_types visualmesh ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES})

endif(BUILD_MESH_CHECK)
