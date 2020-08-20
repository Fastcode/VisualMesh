# Build an program that classifies using the visual mesh
option(BUILD_EXAMPLES "Build the program that executes the mesh" OFF)
if(BUILD_EXAMPLES)

    # Global compiler flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        list(APPEND compile_options /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(
            APPEND
            compile_options
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -march=native
            -mtune=native)

        # Always have the colours otherwise using ninja makes them go away
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND compile_options -fcolor-diagnostics)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            list(APPEND compile_options -fdiagnostics-color=always)
        endif()
    endif()

    # Find packages
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    find_package(OpenCV COMPONENTS core highgui imgproc)
    find_package(YAML-CPP)
    find_package(fmt)

    # We need OpenCV to build all the GUI examples
    if(OpenCV_FOUND)
        add_executable(classified "classified.cpp")
        target_compile_options(classified PRIVATE ${compile_options})
        set_target_properties(classified PROPERTIES CXX_STANDARD 14)
        target_link_libraries(classified visualmesh ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML_CPP_LIBRARIES}
                              Threads::Threads)

        add_executable(mesh_types "mesh_types.cpp")
        target_compile_options(mesh_types PRIVATE ${compile_options})
        set_target_properties(mesh_types PROPERTIES CXX_STANDARD 14)
        target_link_libraries(mesh_types visualmesh ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML_CPP_LIBRARIES}
                              Threads::Threads)

        add_executable(benchmark "benchmark.cpp")
        target_compile_options(benchmark PRIVATE ${compile_options})
        set_target_properties(benchmark PROPERTIES CXX_STANDARD 14)
        target_include_directories(benchmark SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})
        target_link_libraries(benchmark visualmesh ${OpenCV_LIBS} ${fmt_LIBRARIES} ${YAML_CPP_LIBRARIES}
                              Threads::Threads)
    endif(OpenCV_FOUND)

    add_executable(mesh_quality "mesh_quality.cpp")
    target_compile_options(mesh_quality PRIVATE ${compile_options})
    set_target_properties(mesh_quality PROPERTIES CXX_STANDARD 14)
    target_link_libraries(mesh_quality visualmesh)

endif(BUILD_EXAMPLES)
