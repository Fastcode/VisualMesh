# All the header files for this library
file(GLOB hdr "**.hpp")

# Convert our opencl files into header files with their contents and put them in the build folder
file(GLOB cls "visualmesh/engine/opencl/kernels/**.cl")
foreach(cl IN LISTS cls)
    file(RELATIVE_PATH cl_path ${CMAKE_CURRENT_SOURCE_DIR} ${cl})
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp"
        COMMAND ${PYTHON_EXECUTABLE} ARGS "${PROJECT_SOURCE_DIR}/cmake/Scripts/wrap_opencl.py" "${cl}"
                "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp"
        DEPENDS "${cl}" "${PROJECT_SOURCE_DIR}/cmake/Scripts/wrap_opencl.py"
        COMMENT "Wrapping OpenCL file ${cl_path} in a header")
    list(APPEND hdr "${CMAKE_CURRENT_BINARY_DIR}/${cl_path}.hpp")
endforeach(cl ${cls})

add_custom_target(visualmesh_sources DEPENDS ${hdr})

# Define the VisualMesh library as header only but setup its links
add_library(visualmesh INTERFACE)
target_include_directories(visualmesh INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                $<INSTALL_INTERFACE:include>)
target_include_directories(visualmesh INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                                                $<INSTALL_INTERFACE:include>)
add_dependencies(visualmesh visualmesh_sources)

# Find engine libraries so we can link to them
option(BUILD_OPENCL_ENGINE "Should we build the OpenCL engine" ON)
if(BUILD_OPENCL_ENGINE)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        target_include_directories(visualmesh INTERFACE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(visualmesh INTERFACE ${OpenCL_LIBRARIES})
    else()
        target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_OPENCL)
    endif(OpenCL_FOUND)
endif(BUILD_OPENCL_ENGINE)

option(BUILD_VULKAN_ENGINE "Should we build the Vulkan engine" OFF)
if(BUILD_VULKAN_ENGINE)
    find_package(SPIRV)
    find_package(Vulkan)
    if(SPIRV_FOUND AND Vulkan_FOUND)
        target_link_libraries(visualmesh INTERFACE Vulkan::Vulkan SPIRV::SPIRV)
    else()
        target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_VULKAN)
    endif(SPIRV_FOUND AND Vulkan_FOUND)
else()
    target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_VULKAN)
endif(BUILD_VULKAN_ENGINE)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/VisualMeshConfigVersion.cmake"
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/VisualMeshConfig.cmake.in"
                              "${PROJECT_BINARY_DIR}/VisualMeshConfig.cmake" INSTALL_DESTINATION lib/cmake/VisualMesh)

install(
    TARGETS visualmesh
    EXPORT VisualMeshTargets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime)

install(EXPORT VisualMeshTargets DESTINATION lib/cmake/VisualMesh)
install(FILES "${PROJECT_BINARY_DIR}/VisualMeshConfigVersion.cmake" "${PROJECT_BINARY_DIR}/VisualMeshConfig.cmake"
        DESTINATION lib/cmake/VisualMesh)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/visualmesh"
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.hpp")
install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/visualmesh"
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.hpp")
