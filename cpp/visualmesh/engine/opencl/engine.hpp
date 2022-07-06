/*
 * Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef VISUALMESH_ENGINE_OPENCL_ENGINE_HPP
#define VISUALMESH_ENGINE_OPENCL_ENGINE_HPP

// If OpenCL is disabled then don't provide this file
#if !defined(VISUALMESH_DISABLE_OPENCL)

#include <stdio.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <tuple>

#include "visualmesh/engine/opencl/kernels/load_image.cl.hpp"
#include "visualmesh/engine/opencl/kernels/project_equidistant.cl.hpp"
#include "visualmesh/engine/opencl/kernels/project_equisolid.cl.hpp"
#include "visualmesh/engine/opencl/kernels/project_rectilinear.cl.hpp"
#include "visualmesh/engine/opencl/operation/make_context.hpp"
#include "visualmesh/engine/opencl/operation/make_network.hpp"
#include "visualmesh/engine/opencl/operation/make_queue.hpp"
#include "visualmesh/engine/opencl/operation/opencl_error_category.hpp"
#include "visualmesh/engine/opencl/operation/scalar_defines.hpp"
#include "visualmesh/engine/opencl/operation/wrapper.hpp"
#include "visualmesh/mesh.hpp"
#include "visualmesh/network_structure.hpp"
#include "visualmesh/projected_mesh.hpp"
#include "visualmesh/utility/math.hpp"
#include "visualmesh/utility/projection.hpp"
#include "visualmesh/visualmesh.hpp"

namespace visualmesh {
namespace engine {
    namespace opencl {

        /**
         * @brief An OpenCL implementation of the visual mesh inference engine
         *
         * @details
         *  The OpenCL implementation is designed to be used for high performance inference. It is able to take
         *  advantage of either GPUs from Intel, AMD, ARM, NVIDIA etc as well as multithreaded CPU implementations.
         *  This allows it to be very flexible with its deployment on devices.
         *
         * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
         */
        template <typename Scalar>
        class Engine {
        private:
            // OpenCL ::clSetKernelArg functions take the sizeof a pointer as their argument, this is correct
            static constexpr size_t MEM_SIZE = sizeof(cl_mem);

        public:
            /**
             * @brief Construct a new OpenCL Engine object
             *
             * @param structure the network structure to use classification
             */

            Engine(const NetworkStructure<Scalar>& structure = {}, std::string cache_directory = "") {
                // Create the OpenCL context and command queue
                cl_int error              = CL_SUCCESS;
                cl_device_id device       = nullptr;
                std::tie(context, device) = operation::make_context();
                queue                     = operation::make_queue(context, device);

                // Get program sources (this does concatenated strings)
                std::stringstream sources;
                sources << operation::get_scalar_defines(Scalar(0.0));
                sources << PROJECT_EQUIDISTANT_CL;
                sources << PROJECT_EQUISOLID_CL;
                sources << PROJECT_RECTILINEAR_CL;
                sources << LOAD_IMAGE_CL;
                sources << operation::make_network(structure);

                std::string source = sources.str();
                const char* cstr   = source.c_str();
                size_t csize       = source.size();
                // The hash of the sources represents the name of the OpenCL compiled program binary file, so that a new
                // binary will be created for different sources
                std::size_t source_hash = std::hash<std::string>{}(source);

                // Variables for reading the binary
                size_t binary_size;

                // If the compiled binary exists, read it
                const std::string binary_path = cache_directory + "/" + std::to_string(source_hash) + ".bin";
                std::ifstream read_binary(binary_path, std::ios::in);
                bool read_failed = false;
                if (read_binary) {
                    std::cout << "reading binary" << std::endl;
                    // Get the length
                    read_binary.seekg(0, read_binary.end);
                    binary_size = read_binary.tellg();
                    read_binary.seekg(0, read_binary.beg);
                    std::vector<char> binary_prog{};
                    binary_prog.reserve(binary_size);
                    // Read the file
                    read_binary.read(binary_prog.data(), binary_size);
                    if (!read_binary) { read_failed = true; }
                    read_binary.close();

                    std::cout << "Compiling with binary" << std::endl;
                    // Load the binary and build
                    cl_int binary_status = CL_SUCCESS;
                    // const unsigned char* thing = reinterpret_cast<unsigned char*>(binary_prog.data());
                    std::vector<const unsigned char*> binary_progs = {
                      reinterpret_cast<unsigned char*>(binary_prog.data())};
                    program =
                      cl::program(::clCreateProgramWithBinary(
                                    context, 1, &device, &binary_size, binary_progs.data(), &binary_status, &error),
                                  ::clReleaseProgram);
                    if ((error != CL_SUCCESS) || (binary_status != CL_SUCCESS)) { read_failed = true; }

                    error = ::clBuildProgram(program,
                                             1,
                                             &device,
                                             "-cl-single-precision-constant -cl-fast-relaxed-math -cl-mad-enable",
                                             nullptr,
                                             nullptr);

                    // If it didn't work, try rebuilding
                    if (error != CL_SUCCESS) { read_failed = true; }
                }

                // If the read failed, remove the file
                if (read_failed) {
                    std::cout << "read failed, remove the file " << std::endl;
                    std::remove(binary_path.c_str());
                }

                // The compiled binary doesn't exist, create it
                if (!read_binary || read_failed) {
                    std::cout << "building program" << std::endl;
                    // Create the program and build
                    program =
                      cl::program(::clCreateProgramWithSource(context, 1, &cstr, &csize, &error), ::clReleaseProgram);
                    throw_cl_error(error, "Error adding sources to OpenCL program");

                    error = ::clBuildProgram(program,
                                             0,
                                             nullptr,
                                             "-cl-single-precision-constant -cl-fast-relaxed-math -cl-mad-enable",
                                             nullptr,
                                             nullptr);

                    // If it didn't work, log and throw an error
                    if (error != CL_SUCCESS) {
                        // Get program build log
                        size_t used = 0;
                        ::clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &used);
                        std::vector<char> log(used);
                        ::clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &used);
                        // Throw an error with the build log
                        throw_cl_error(
                          error, "Error building OpenCL program\n" + std::string(log.begin(), log.begin() + used));
                    }
                    std::cout << "built" << std::endl;
                    // Save the the built program to a file
                    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
                    std::cout << "reserving" << std::endl;
                    std::vector<char> binary_prog{};
                    binary_prog.reserve(binary_size);
                    std::cout << "prog info" << std::endl;
                    clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, binary_prog.data(), nullptr);
                    std::cout << "make var" << std::endl;
                    std::ofstream write_binary(binary_path, std::ofstream::binary);
                    std::cout << "write" << std::endl;
                    write_binary.write(binary_prog.data(), binary_size);
                    std::cout << "close" << std::endl;
                    write_binary.close();
                    std::cout << "done" << std::endl;
                }
                std::cout << "done" << std::endl;
                // Get the kernels
                project_rectilinear =
                  cl::kernel(::clCreateKernel(program, "project_rectilinear", &error), ::clReleaseKernel);
                throw_cl_error(error, "Error getting project_rectilinear kernel");
                project_equidistant =
                  cl::kernel(::clCreateKernel(program, "project_equidistant", &error), ::clReleaseKernel);
                throw_cl_error(error, "Error getting project_equidistant kernel");
                project_equisolid =
                  cl::kernel(::clCreateKernel(program, "project_equisolid", &error), ::clReleaseKernel);
                throw_cl_error(error, "Error getting project_equisolid kernel");
                load_image = cl::kernel(::clCreateKernel(program, "load_image", &error), ::clReleaseKernel);
                throw_cl_error(error, "Failed to create kernel load_image");

                // Grab all the kernels that were generated
                for (unsigned int i = 0; i < structure.size(); ++i) {
                    std::string kernel       = "conv" + std::to_string(i);
                    unsigned int output_size = structure[i].back().biases.size();

                    cl::kernel k(::clCreateKernel(program, kernel.c_str(), &error), ::clReleaseKernel);
                    throw_cl_error(error, "Failed to create kernel " + kernel);
                    conv_layers.emplace_back(k, output_size);
                }

                // Work out what the widest network layer is
                max_width = 4;
                for (const auto& k : conv_layers) {
                    max_width = std::max(max_width, k.second);
                }

                // Function to get the preferred workgroup size for a kernel
                auto workgroup_size_for_kernel = [&device](auto k) {
                    size_t t = 0;
                    ::clGetKernelWorkGroupInfo(
                      k, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(t), &t, nullptr);
                    return t;
                };

                // Go through each of our kernels and see which is the largest preferred size
                workgroup_size = 1;
                workgroup_size = std::max(workgroup_size, workgroup_size_for_kernel(project_rectilinear));
                workgroup_size = std::max(workgroup_size, workgroup_size_for_kernel(project_equisolid));
                workgroup_size = std::max(workgroup_size, workgroup_size_for_kernel(project_equidistant));
                workgroup_size = std::max(workgroup_size, workgroup_size_for_kernel(load_image));
                for (const auto& k : conv_layers) {
                    workgroup_size = std::max(workgroup_size, workgroup_size_for_kernel(k.first));
                }
            }

            /**
             * @brief Projects a provided mesh to pixel coordinates
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh the mesh table that we are projecting to pixel coordinates
             * @param Hoc  the homogenous transformation matrix from the camera to the observation plane
             * @param lens the lens parameters that describe the optics of the camera
             *
             * @return a projected mesh for the provided arguments
             */
            template <template <typename> class Model>
            inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Model>& mesh,
                                                                                 const mat4<Scalar>& Hoc,
                                                                                 const Lens<Scalar>& lens) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                // Perform the projection
                std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
                std::vector<int> indices;
                cl::mem cl_pixels;
                cl::event projected;
                std::tie(neighbourhood, indices, cl_pixels, projected) = do_project(mesh, Hoc, lens);

                // If we didn't get anything, nothing to return
                if (indices.empty()) { return ProjectedMesh<Scalar, N_NEIGHBOURS>(); }

                // Read the pixels off the buffer
                std::vector<std::array<Scalar, 2>> pixels(indices.size());
                std::array<cl_event, 1> events{{projected}};
                cl_int error = ::clEnqueueReadBuffer(queue,
                                                     cl_pixels,
                                                     true,
                                                     0,
                                                     indices.size() * sizeof(std::array<Scalar, 2>),
                                                     pixels.data(),
                                                     events.size(),
                                                     events.data(),
                                                     nullptr);
                throw_cl_error(error, "Failed reading projected pixels from the device");

                return ProjectedMesh<Scalar, N_NEIGHBOURS>{
                  std::move(pixels), std::move(neighbourhood), std::move(indices)};
            }

            /**
             * @brief Projects a provided mesh to pixel coordinates from an aggregate VisualMesh object
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh the mesh table that we are projecting to pixel coordinates
             * @param Hoc  the homogenous transformation matrix from the camera to the observation plane
             * @param lens the lens parameters that describe the optics of the camera
             *
             * @return a projected mesh for the provided arguments
             */
            template <template <typename> class Model>
            inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const VisualMesh<Scalar, Model>& mesh,
                                                                                 const mat4<Scalar>& Hoc,
                                                                                 const Lens<Scalar>& lens) const {
                return operator()(mesh.height(Hoc[2][3]), Hoc, lens);
            }

            /**
             * @brief Project and classify a mesh using the neural network that is loaded into this engine
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh    the mesh table that we are projecting to pixel coordinates
             * @param Hoc     the homogenous transformation matrix from the camera to the observation plane
             * @param lens    the lens parameters that describe the optics of the camera
             * @param image   the data that represents the image the network will run from
             * @param format  the pixel format of this image as a fourcc code
             *
             * @return a classified mesh for the provided arguments
             */
            template <template <typename> class Model>
            ClassifiedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Model>& mesh,
                                                                           const mat4<Scalar>& Hoc,
                                                                           const Lens<Scalar>& lens,
                                                                           const void* image,
                                                                           const uint32_t& format) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;
                cl_int error                      = CL_SUCCESS;

                // Grab the image memory from the cache
                cl::mem cl_image = get_image_memory(lens.dimensions, format);

                // Map our image into device memory
                std::array<size_t, 3> origin = {{0, 0, 0}};
                std::array<size_t, 3> region = {{size_t(lens.dimensions[0]), size_t(lens.dimensions[1]), 1}};

                cl::event cl_image_loaded;
                cl_event ev = nullptr;
                error       = clEnqueueWriteImage(
                  queue, cl_image, false, origin.data(), region.data(), 0, 0, image, 0, nullptr, &ev);
                if (ev) { cl_image_loaded = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error mapping image onto device");

                // Project our visual mesh
                std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
                std::vector<int> indices;
                cl::mem cl_pixels;
                cl::event cl_pixels_loaded;
                std::tie(neighbourhood, indices, cl_pixels, cl_pixels_loaded) = do_project(mesh, Hoc, lens);

                // If there were no points, nothing to project
                if (indices.empty()) { return ClassifiedMesh<Scalar, N_NEIGHBOURS>(); }

                // This includes the offscreen point at the end
                int n_points = neighbourhood.size();

                // Get the neighbourhood memory from cache
                cl::mem cl_neighbourhood = get_neighbourhood_memory(n_points, N_NEIGHBOURS);

                // Upload the neighbourhood buffer
                cl::event cl_neighbourhood_loaded;
                ev    = nullptr;
                error = ::clEnqueueWriteBuffer(queue,
                                               cl_neighbourhood,
                                               false,
                                               0,
                                               n_points * sizeof(std::array<int, N_NEIGHBOURS>),
                                               neighbourhood.data(),
                                               0,
                                               nullptr,
                                               &ev);
                if (ev) { cl_neighbourhood_loaded = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error writing neighbourhood points to the device");

                // Grab our ping pong buffers from the cache
                auto cl_conv_buffers   = get_network_memory(max_width * n_points);
                cl::mem cl_conv_input  = cl_conv_buffers[0];
                cl::mem cl_conv_output = cl_conv_buffers[1];

                // Read the pixels into the buffer
                cl::event img_load_event;
                cl::event network_complete;

                cl_mem arg = nullptr;
                arg        = cl_image;
                throw_cl_error(::clSetKernelArg(load_image, 0, MEM_SIZE, &arg),
                               "Error setting kernel argument 0 for image load kernel");
                throw_cl_error(::clSetKernelArg(load_image, 1, sizeof(format), &format),
                               "Error setting kernel argument 1 for image load kernel");
                arg = cl_pixels;
                throw_cl_error(::clSetKernelArg(load_image, 2, MEM_SIZE, &arg),
                               "Error setting kernel argument 2 for image load kernel");
                arg = cl_conv_input;
                throw_cl_error(::clSetKernelArg(load_image, 3, MEM_SIZE, &arg),
                               "Error setting kernel argument 3 for image load kernel");

                // When calculating global_size we round to the nearest workgroup size
                size_t offset                      = 0;
                size_t global_size                 = (((n_points - 1) / workgroup_size) + 1) * workgroup_size;
                std::array<cl_event, 2> event_list = {cl_pixels_loaded, cl_image_loaded};
                ev                                 = nullptr;
                error                              = ::clEnqueueNDRangeKernel(
                  queue, load_image, 1, &offset, &global_size, &workgroup_size, 2, event_list.data(), &ev);
                if (ev) { img_load_event = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error queueing the image load kernel");

                // The offscreen point gets a value of -1.0 to make it easy to distinguish
                std::array<cl_event, 1> img_loaded_events = {img_load_event};
                cl::event offscreen_fill_event;
                Scalar minus_one(-1.0);
                ev    = nullptr;
                error = ::clEnqueueFillBuffer(queue,
                                              cl_conv_input,
                                              &minus_one,
                                              sizeof(Scalar),
                                              (n_points - 1) * sizeof(std::array<Scalar, 4>),
                                              sizeof(std::array<Scalar, 4>),
                                              1,
                                              img_loaded_events.data(),
                                              &ev);
                if (ev) { offscreen_fill_event = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error setting the offscreen pixel values");


                // These events are required for our first convolution
                std::vector<cl::event> events({img_load_event, offscreen_fill_event, cl_neighbourhood_loaded});

                for (const auto& conv : conv_layers) {
                    cl_mem arg = nullptr;
                    arg        = cl_neighbourhood;
                    throw_cl_error(::clSetKernelArg(conv.first, 0, MEM_SIZE, &arg),
                                   "Error setting argument 0 for convolution kernel");
                    arg = cl_conv_input;
                    throw_cl_error(::clSetKernelArg(conv.first, 1, MEM_SIZE, &arg),
                                   "Error setting argument 1 for convolution kernel");
                    arg = cl_conv_output;
                    throw_cl_error(::clSetKernelArg(conv.first, 2, MEM_SIZE, &arg),
                                   "Error setting argument 2 for convolution kernel");

                    // When calculating global_size we round to the nearest workgroup size
                    size_t offset      = 0;
                    size_t global_size = (((n_points - 1) / workgroup_size) + 1) * workgroup_size;
                    cl::event event;
                    ev = nullptr;
                    std::vector<cl_event> cl_events(events.begin(), events.end());
                    error = ::clEnqueueNDRangeKernel(queue,
                                                     conv.first,
                                                     1,
                                                     &offset,
                                                     &global_size,
                                                     &workgroup_size,
                                                     cl_events.size(),
                                                     cl_events.data(),
                                                     &ev);
                    if (ev) { event = cl::event(ev, ::clReleaseEvent); }
                    throw_cl_error(error, "Error queueing convolution kernel");

                    // Convert our events into a vector of events and ping pong our buffers
                    events           = std::vector<cl::event>({event});
                    network_complete = event;
                    std::swap(cl_conv_input, cl_conv_output);
                }

                // Read the pixel coordinates off the device
                cl::event pixels_read;
                ev = nullptr;
                std::vector<std::array<Scalar, 2>> pixels(neighbourhood.size() - 1);
                cl_event iev = cl_pixels_loaded;
                error        = ::clEnqueueReadBuffer(queue,
                                              cl_pixels,
                                              false,
                                              0,
                                              pixels.size() * sizeof(std::array<Scalar, 2>),
                                              pixels.data(),
                                              1,
                                              &iev,
                                              &ev);
                if (ev) { pixels_read = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error reading projected pixels");

                // Read the classifications off the device (they'll be in input)
                cl::event classes_read;
                ev  = nullptr;
                iev = network_complete;
                std::vector<Scalar> classifications(neighbourhood.size() * conv_layers.back().second);
                error = ::clEnqueueReadBuffer(queue,
                                              cl_conv_input,
                                              false,
                                              0,
                                              classifications.size() * sizeof(Scalar),
                                              classifications.data(),
                                              1,
                                              &iev,
                                              &ev);
                if (ev) { classes_read = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error reading classified values");

                // Flush the queue to ensure all the commands have been issued
                ::clFlush(queue);

                // Wait for the chain to finish up to where we care about it
                std::array<cl_event, 2> end_events = {pixels_read, classes_read};
                ::clWaitForEvents(2, end_events.data());

                return ClassifiedMesh<Scalar, N_NEIGHBOURS>{
                  std::move(pixels), std::move(neighbourhood), std::move(indices), std::move(classifications)};
            }

            /**
             * @brief Project and classify a mesh using the neural network that is loaded into this engine.
             * This version takes an aggregate VisualMesh object
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh    the mesh table that we are projecting to pixel coordinates
             * @param Hoc     the homogenous transformation matrix from the camera to the observation plane
             * @param lens    the lens parameters that describe the optics of the camera
             * @param image   the data that represents the image the network will run from
             * @param format  the pixel format of this image as a fourcc code
             *
             * @return a classified mesh for the provided arguments
             */
            template <template <typename> class Model>
            ClassifiedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const VisualMesh<Scalar, Model>& mesh,
                                                                           const mat4<Scalar>& Hoc,
                                                                           const Lens<Scalar>& lens,
                                                                           const void* image,
                                                                           const uint32_t& format) const {
                return operator()(mesh.height(Hoc[2][3]), Hoc, lens, image, format);
            }

            void clear_cache() {
                device_points_cache.clear();
                indices_map_memory.memory         = nullptr;
                indices_map_memory.n_points       = 0;
                pixel_coordinates_memory.memory   = nullptr;
                pixel_coordinates_memory.n_points = 0;
                neighbourhood_memory.memory       = nullptr;
                neighbourhood_memory.n_points     = 0;
                network_memory.memory             = {nullptr, nullptr};
                network_memory.n_points           = 0;
                image_memory.memory               = nullptr;
                image_memory.dimensions           = {0, 0};
                image_memory.format               = 0;
            }

        private:
            template <template <typename> class Model>
            std::tuple<std::vector<std::array<int, Model<Scalar>::N_NEIGHBOURS>>, std::vector<int>, cl::mem, cl::event>
              do_project(const Mesh<Scalar, Model>& mesh, const mat4<Scalar>& Hoc, const Lens<Scalar>& lens) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                // Lookup the on screen ranges
                auto ranges = mesh.lookup(Hoc, lens);

                // Reused variables
                cl_int error = 0;
                cl_event ev  = nullptr;

                // Pack Rco into a Scalar16
                // clang-format off
                std::array<Scalar, 16> Rco{{
                    Hoc[0][0],     Hoc[1][0],   Hoc[2][0], Scalar(0.0),
                    Hoc[0][1],     Hoc[1][1],   Hoc[2][1], Scalar(0.0),
                    Hoc[0][2],     Hoc[1][2],   Hoc[2][2], Scalar(0.0),
                    Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(1.0)
                }};
                // clang-format on

                // Convenience variables
                const auto& nodes = mesh.nodes;

                // Upload our visual mesh unit vectors if we have to
                cl::mem cl_points;

                auto device_mesh = device_points_cache.find(&mesh);
                if (device_mesh == device_points_cache.end()) {
                    cl_points =
                      cl::mem(::clCreateBuffer(
                                context, CL_MEM_READ_ONLY, sizeof(vec4<Scalar>) * mesh.nodes.size(), nullptr, &error),
                              ::clReleaseMemObject);

                    // Flatten our rays
                    std::vector<vec4<Scalar>> rays;
                    rays.reserve(mesh.nodes.size());
                    for (const auto& n : mesh.nodes) {
                        rays.emplace_back(vec4<Scalar>{n.ray[0], n.ray[1], n.ray[2], 0});
                    }

                    // Write the points buffer to the device and cache it
                    error = ::clEnqueueWriteBuffer(
                      queue, cl_points, true, 0, rays.size() * sizeof(vec4<Scalar>), rays.data(), 0, nullptr, nullptr);
                    throw_cl_error(error, "Error writing points to the device buffer");

                    // Cache for future runs
                    device_points_cache[&mesh] = cl_points;
                }
                else { cl_points = device_mesh->second; }

                // First count the size of the buffer we will need to allocate
                int n_points = 0;
                for (const auto& range : ranges) {
                    n_points += range.second - range.first;
                }

                // No point processing if we have no points, return an empty mesh
                if (n_points == 0) {
                    return std::make_tuple(
                      std::vector<std::array<int, N_NEIGHBOURS>>(), std::vector<int>(), cl::mem(), cl::event());
                }

                // Build up our list of indices for OpenCL
                // Use iota to fill in the numbers
                std::vector<int> indices(n_points);
                auto it = indices.begin();
                for (const auto& range : ranges) {
                    auto n = std::next(it, range.second - range.first);
                    std::iota(it, n, range.first);
                    it = n;
                }

                // Create buffers for indices map
                cl::mem indices_map       = get_indices_map_memory(n_points);
                cl::mem pixel_coordinates = get_pixel_coordinates_memory(n_points);

                // Upload our indices map
                cl::event indices_event;
                ev    = nullptr;
                error = ::clEnqueueWriteBuffer(
                  queue, indices_map, false, 0, indices.size() * sizeof(cl_int), indices.data(), 0, nullptr, &ev);
                if (ev) { indices_event = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error uploading indices_map to device");

                // When everything is uploaded, we can run our projection kernel to get the pixel coordinates
                cl::event projected;

                cl::kernel projection_kernel;

                // Select a projection kernel
                switch (lens.projection) {
                    case RECTILINEAR: projection_kernel = project_rectilinear; break;
                    case EQUIDISTANT: projection_kernel = project_equidistant; break;
                    case EQUISOLID: projection_kernel = project_equisolid; break;
                }

                // Calculate the coefficients for performing a distortion to give to the engine
                vec4<Scalar> ik = inverse_coefficients(lens.k);

                // Load the arguments
                cl_mem arg = nullptr;
                arg        = cl_points;
                throw_cl_error(::clSetKernelArg(projection_kernel, 0, MEM_SIZE, &arg),
                               "Error setting kernel argument 0 for projection kernel");
                arg = indices_map;
                throw_cl_error(::clSetKernelArg(projection_kernel, 1, MEM_SIZE, &arg),
                               "Error setting kernel argument 1 for projection kernel");
                throw_cl_error(::clSetKernelArg(projection_kernel, 2, sizeof(Rco), Rco.data()),
                               "Error setting kernel argument 2 for projection kernel");
                throw_cl_error(::clSetKernelArg(projection_kernel, 3, sizeof(lens.focal_length), &lens.focal_length),
                               "Error setting kernel argument 3 for projection kernel");
                throw_cl_error(::clSetKernelArg(projection_kernel, 4, sizeof(lens.centre), lens.centre.data()),
                               "Error setting kernel argument 4 for projection kernel");
                throw_cl_error(::clSetKernelArg(projection_kernel, 5, sizeof(ik), ik.data()),
                               "Error setting kernel argument 5 for projection kernel");
                throw_cl_error(::clSetKernelArg(projection_kernel, 6, sizeof(lens.dimensions), lens.dimensions.data()),
                               "Error setting kernel argument 6 for projection kernel");
                arg = pixel_coordinates;
                throw_cl_error(::clSetKernelArg(projection_kernel, 7, MEM_SIZE, &arg),
                               "Error setting kernel argument 7 for projection kernel");

                // Project!
                // When calculating global_size we round to the nearest workgroup size
                size_t offset      = 0;
                size_t global_size = (((n_points - 1) / workgroup_size) + 1) * workgroup_size;
                ev                 = nullptr;
                cl_event iev       = indices_event;
                error              = ::clEnqueueNDRangeKernel(
                  queue, projection_kernel, 1, &offset, &global_size, &workgroup_size, 1, &iev, &ev);
                if (ev) { projected = cl::event(ev, ::clReleaseEvent); }
                throw_cl_error(error, "Error queueing the projection kernel");

                // This can happen on the CPU while the OpenCL device is busy
                // Build the reverse lookup map where the offscreen point is one past the end
                std::vector<int> r_indices(nodes.size() + 1, n_points);
                for (unsigned int i = 0; i < indices.size(); ++i) {
                    r_indices[indices[i]] = i;
                }

                // Build the packed neighbourhood map with an extra offscreen point at the end
                std::vector<std::array<int, N_NEIGHBOURS>> local_neighbourhood(n_points + 1);
                for (unsigned int i = 0; i < indices.size(); ++i) {
                    const auto& node = nodes[indices[i]];
                    for (unsigned int j = 0; j < node.neighbours.size(); ++j) {
                        const auto& n             = node.neighbours[j];
                        local_neighbourhood[i][j] = r_indices[n];
                    }
                }
                // Fill in the final offscreen point which connects only to itself
                local_neighbourhood[n_points].fill(n_points);

                // This ensures that all elements in the queue have been issued to the device NOT that they are all
                // finished If we don't do this here, some of our buffers can go out of scope before the queue picks
                // them up causing errors
                ::clFlush(queue);

                // Return what we calculated
                return std::make_tuple(std::move(local_neighbourhood),  // CPU buffer
                                       std::move(indices),              // CPU buffer
                                       pixel_coordinates,               // GPU buffer
                                       projected);                      // GPU event
            }

            cl::mem get_indices_map_memory(const int& n_points) const {

                if (indices_map_memory.n_points < n_points) {
                    // Align the size to the nearest workgroup size
                    size_t size               = ((n_points - 1) / workgroup_size + 1) * workgroup_size * sizeof(int);
                    cl_int error              = 0;
                    indices_map_memory.memory = cl::mem(
                      ::clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &error), ::clReleaseMemObject);
                    throw_cl_error(error, "Error allocating indices map buffer on device");
                    indices_map_memory.n_points = n_points;
                }
                return indices_map_memory.memory;
            }

            cl::mem get_pixel_coordinates_memory(const int& n_points) const {

                if (pixel_coordinates_memory.n_points < n_points) {
                    // Align the size to the nearest workgroup size
                    size_t size  = ((n_points - 1) / workgroup_size + 1) * workgroup_size * sizeof(Scalar) * 2;
                    cl_int error = 0;
                    pixel_coordinates_memory.memory = cl::mem(
                      ::clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &error), ::clReleaseMemObject);
                    throw_cl_error(error, "Error allocating pixel coordinates buffer on device");
                    pixel_coordinates_memory.n_points = n_points;
                }
                return pixel_coordinates_memory.memory;
            }

            std::array<cl::mem, 2> get_network_memory(const int& n_points) const {
                if (network_memory.n_points < n_points) {
                    // Align the size to the nearest workgroup size
                    size_t size  = ((n_points - 1) / workgroup_size + 1) * workgroup_size * sizeof(Scalar) * max_width;
                    cl_int error = 0;
                    network_memory.memory[0] = cl::mem(
                      ::clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &error), ::clReleaseMemObject);
                    throw_cl_error(error, "Error allocating ping pong buffer 1 on device");
                    network_memory.memory[1] = cl::mem(
                      ::clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &error), ::clReleaseMemObject);
                    network_memory.n_points = n_points;
                    throw_cl_error(error, "Error allocating ping pong buffer 2 on device");
                }
                return network_memory.memory;
            }

            cl::mem get_neighbourhood_memory(const int& n_points, int n_neighbours) const {

                if (neighbourhood_memory.n_points < n_points) {
                    // Align the size to the nearest workgroup size
                    size_t size  = ((n_points - 1) / workgroup_size + 1) * workgroup_size * sizeof(int) * n_neighbours;
                    cl_int error = 0;
                    neighbourhood_memory.memory = cl::mem(
                      ::clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &error), ::clReleaseMemObject);
                    throw_cl_error(error, "Error allocating neighbourhood buffer on device");
                    neighbourhood_memory.n_points = n_points;
                }
                return neighbourhood_memory.memory;
            }

            cl::mem get_image_memory(vec2<int> dimensions, uint32_t format) const {

                // If our dimensions and format haven't changed from last time we can reuse the same memory location
                if (dimensions != image_memory.dimensions || format != image_memory.format) {
                    cl_image_format fmt;
                    switch (format) {
                        // Bayer
                        case fourcc("GRBG"):
                        case fourcc("RGGB"):
                        case fourcc("GBRG"):
                        case fourcc("BGGR"): fmt = cl_image_format{CL_R, CL_UNORM_INT8}; break;
                        case fourcc("GRAY"):
                        case fourcc("GREY"):
                        case fourcc("Y8  "): fmt = cl_image_format{CL_LUMINANCE, CL_UNORM_INT8}; break;
                        case fourcc("BGRA"): fmt = cl_image_format{CL_BGRA, CL_UNORM_INT8}; break;
                        case fourcc("RGBA"): fmt = cl_image_format{CL_RGBA, CL_UNORM_INT8}; break;
                        // Oh no...
                        default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
                    }

                    cl_image_desc desc = {
                      CL_MEM_OBJECT_IMAGE2D, size_t(dimensions[0]), size_t(dimensions[1]), 1, 1, 0, 0, 0, 0, nullptr};

                    // Create a buffer for our image
                    cl_int error = 0;
                    cl::mem memory(::clCreateImage(context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &error),
                                   ::clReleaseMemObject);
                    throw_cl_error(error, "Error creating image on device");

                    // Update what we are caching
                    image_memory.dimensions = dimensions;
                    image_memory.format     = format;
                    image_memory.memory     = memory;
                }

                // Return the cache
                return image_memory.memory;
            }

            /// OpenCL context
            cl::context context;

            /// OpenCL command queue
            cl::command_queue queue;

            /// OpenCL program
            cl::program program;
            /// Kernel for projecting rays to pixels using an equidistant projection
            cl::kernel project_equidistant;
            /// Kernel for projecting rays to pixels using an equisolid projection
            cl::kernel project_equisolid;
            /// Kernel for projecting rays to pixels using a rectilinear projection
            cl::kernel project_rectilinear;
            /// Kernel for reading projected pixel coordinates from an image into the network input layer
            cl::kernel load_image;
            /// A list of kernels to run in sequence to run the network
            std::vector<std::pair<cl::kernel, size_t>> conv_layers;

            /// A location to cache the GPU memory allocated for indices map so we don't reallocate between runs
            mutable struct {
                int n_points = 0;
                cl::mem memory;
            } indices_map_memory;

            /// A location to cache the GPU memory allocated for pixel coordinates so we don't reallocate between runs
            mutable struct {
                int n_points = 0;
                cl::mem memory;
            } pixel_coordinates_memory;

            /// A location to cache the GPU memory allocated for the ping pong network buffers so we don't reallocate
            /// between runs
            mutable struct {
                int n_points = 0;
                std::array<cl::mem, 2> memory;
            } network_memory;

            /// A location to cache the GPU memory allocated for the local graph so we don't reallocate between runs
            mutable struct {
                int n_points = 0;
                cl::mem memory;
            } neighbourhood_memory;

            /// A location to cache the GPU memory allocated for the image so we don't reallocate between runs
            mutable struct {
                vec2<int> dimensions = {0, 0};
                uint32_t format      = 0;
                cl::mem memory;
            } image_memory;

            /// The width of the maximumally wide layer in the network (always at least 4 because of the input)
            size_t max_width = 4;

            /// The largest preferred workgroup size so we can overallocate memory
            size_t workgroup_size;

            /// Cache of opencl buffers from mesh objects
            mutable std::map<const void*, cl::mem> device_points_cache;
        };

    }  // namespace opencl
}  // namespace engine
}  // namespace visualmesh

#endif  // !defined(VISUALMESH_DISABLE_OPENCL)
#endif  // VISUALMESH_ENGINE_OPENCL_ENGINE_HPP
