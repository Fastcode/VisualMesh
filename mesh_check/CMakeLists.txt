option(BUILD_MESH_CHECK "Build the mesh check tool" OFF)
if(BUILD_MESH_CHECK)

    # Find our dependencies
    find_package(Threads REQUIRED)
    find_package(OpenCV COMPONENTS core highgui imgproc)
    find_package(YAML-CPP)
    find_package(fmt)

    # Compile the executable
    add_executable(mesh_types "mesh_types.cpp")
    target_compile_options(mesh_types PRIVATE ${compile_options})
    target_link_libraries(mesh_types visualmesh ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML_CPP_LIBRARIES}
                          Threads::Threads)

endif(BUILD_MESH_CHECK)
