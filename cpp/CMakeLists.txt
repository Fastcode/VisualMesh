# All the header files for this library
file(GLOB hdr "**/*.hpp")

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

# Add to ALL target so that the wrapped header files are always generated, even if nothing else is being built
add_custom_target(visualmesh_sources ALL DEPENDS ${hdr})

# Define the VisualMesh library as header only but setup its links
add_library(visualmesh INTERFACE)
target_include_directories(visualmesh INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                $<INSTALL_INTERFACE:include>)
target_include_directories(visualmesh INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                                                $<INSTALL_INTERFACE:include>)
add_dependencies(visualmesh visualmesh_sources)
target_compile_features(visualmesh INTERFACE cxx_std_14)

# Find engine libraries so we can link to them
option(BUILD_OPENCL_ENGINE "Should we build the OpenCL engine" ON)
if(BUILD_OPENCL_ENGINE)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        target_link_libraries(visualmesh INTERFACE OpenCL::OpenCL)
    else()
        target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_OPENCL)
    endif(OpenCL_FOUND)
else()
    target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_OPENCL)
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

# Create the VisualMeshConfig files
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_FULL_INCLUDEDIR})
write_basic_package_version_file(${PROJECT_BINARY_DIR}/VisualMeshConfigVersion.cmake COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/VisualMeshConfig.cmake.in" "${PROJECT_BINARY_DIR}/VisualMeshConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VisualMesh"
    PATH_VARS INSTALL_INCLUDE_DIR)

# Install version, config and target files.
install(FILES "${PROJECT_BINARY_DIR}/VisualMeshConfigVersion.cmake" "${PROJECT_BINARY_DIR}/VisualMeshConfig.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VisualMesh")
install(
    EXPORT VisualMeshTargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VisualMesh"
    NAMESPACE visualmesh::)

# Install headers and targets
install(
    TARGETS visualmesh
    EXPORT VisualMeshTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
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
