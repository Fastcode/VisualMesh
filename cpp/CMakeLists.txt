# All the header files for this library
file(GLOB hdr "**.hpp")

# Convert our opencl files into header files with their contents and put them in the build folder
file(GLOB cls "engine/opencl/kernels/**.cl")
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
target_include_directories(visualmesh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(visualmesh INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(visualmesh visualmesh_sources)

# Find engine libraries so we can link to them
find_package(OpenCL)
if(OpenCL_FOUND)
    target_include_directories(visualmesh INTERFACE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(visualmesh INTERFACE ${OpenCL_LIBRARIES})
else()
    target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_OPENCL)
endif(OpenCL_FOUND)

find_package(SPIRV)
find_package(Vulkan)
if(SPIRV_FOUND AND Vulkan_FOUND)
    target_link_libraries(visualmesh INTERFACE Vulkan::Vulkan SPIRV::SPIRV)
else()
    target_compile_definitions(visualmesh INTERFACE VISUALMESH_DISABLE_VULKAN)
endif(SPIRV_FOUND AND Vulkan_FOUND)
