/*
 * Copyright (C) 2017-2019 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_OPENCL_CLASSIFIER_H
#define VISUALMESH_OPENCL_CLASSIFIER_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#if defined(__APPLE__) || defined(__MACOSX)
#  include <OpenCL/opencl.h>
#else
#  include <CL/opencl.h>
#endif  // !__APPLE__

#include <iomanip>
#include <memory>
#include <mutex>
#include <tuple>

#include "engine/opencl/kernels/read_image_to_network.cl.hpp"
#include "engine/opencl/opencl_error_category.hpp"
#include "engine/opencl/util.hpp"
#include "engine/opencl/wrapper.hpp"
#include "mesh/classified_mesh.hpp"
#include "mesh/mesh.hpp"
#include "mesh/network_structure.hpp"
#include "util/fourcc.hpp"

namespace visualmesh {
namespace engine {
  namespace opencl {

    template <typename Scalar>
    class Engine;

    template <typename Scalar, template <typename> class Generator>
    class Classifier {
    private:
      static constexpr size_t N_NEIGHBOURS = Generator<Scalar>::N_NEIGHBOURS;

    public:
      Classifier(Engine<Scalar>* engine, const network_structure_t<Scalar>& structure)
        : engine(engine), conv_mutex(std::make_shared<std::mutex>()) {

        // Keep our own copies
        context = engine->context;

        // Get the device from the context
        cl_device_id device;
        ::clGetContextInfo(engine->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, nullptr);

        // Make our three OpenCL command queues
        command_queue = make_queue(context, device);
        read_queue    = make_queue(context, device);
        write_queue   = make_queue(context, device);

        // Build using a string stream
        std::stringstream code;

        // Set our precision for how many digits our scalar has
        code << std::setprecision(std::numeric_limits<Scalar>::digits10 + 2);

        // Keep track of the input and output size of each layer for building the network
        // The first layer input is always 4 from the image
        unsigned int input_dimensions  = 4;
        unsigned int output_dimensions = 0;

        for (unsigned int conv_no = 0; conv_no < structure.size(); ++conv_no) {
          auto& conv = structure[conv_no];

          // Write our OpenCL kernel definition
          code << "kernel void conv" << conv_no
               << "(global const int* neighbourhood, global const Scalar* input, global Scalar* output) {" << std::endl
               << std::endl;

          code << "  // Get our kernel index" << std::endl;
          code << "  const int idx = get_global_id(0);" << std::endl << std::endl;

          /*************************************************
           *                    GATHER                     *
           *************************************************/

          code << "  // Gather from our neighbourhood " << std::endl;
          code << "  Scalar in0[" << (input_dimensions * (N_NEIGHBOURS + 1)) << "] = {" << std::endl;

          // Read the ones for our own index
          for (unsigned int j = 0; j < input_dimensions; ++j) {
            code << "    input[idx * " << input_dimensions << " + " << j << "]," << std::endl;
          }

          // Read our neighbourhood
          for (unsigned int i = 0; i < N_NEIGHBOURS; ++i) {
            for (unsigned int j = 0; j < input_dimensions; ++j) {
              code << "    input[neighbourhood[idx * " << N_NEIGHBOURS << " + " << i << "] * " << input_dimensions
                   << " + " << j << "]";

              // Comma separated except for the end
              if (i < N_NEIGHBOURS || j + 1 < input_dimensions) { code << ","; }
              code << std::endl;
            }
          }
          code << "  };";


          // We have gathered which increased the size of the input
          input_dimensions = input_dimensions * (N_NEIGHBOURS + 1);

          code << std::endl << std::endl;

          /*************************************************
           *                WEIGHTS + BIAS                 *
           *************************************************/

          // Now we have to do our layer operations
          for (unsigned int layer_no = 0; layer_no < conv.size(); ++layer_no) {
            const auto& weights = conv[layer_no].first;
            const auto& biases  = conv[layer_no].second;

            // Update our output dimensions
            output_dimensions = biases.size();

            // Perform the matrix multiplication
            code << "  // Perform our matrix multiplication for weights and add bias for layer " << layer_no
                 << std::endl;
            code << "  Scalar in" << (layer_no + 1) << "[" << output_dimensions << "] = {" << std::endl;
            for (unsigned int i = 0; i < output_dimensions; ++i) {
              code << "    ";
              for (unsigned int j = 0; j < input_dimensions; ++j) {
                code << "in" << layer_no << "[" << j << "] * " << weights[j][i] << " + ";
              }
              code << biases[i];
              if (i + 1 < output_dimensions) { code << ","; }
              code << std::endl;
            }
            code << "  };" << std::endl << std::endl;


            /*************************************************
             *                  ACTIVATION.                  *
             *************************************************/

            // Apply our activation function
            code << "  // Apply the activation function" << std::endl;

            // selu constants
            constexpr const Scalar lambda = 1.0507009873554804934193349852946;
            constexpr const Scalar alpha  = 1.6732632423543772848170429916717;

            // Apply selu
            if (conv_no + 1 < structure.size() || layer_no + 1 < conv.size()) {
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  " << e << " = " << lambda << "f * (" << e << " > 0 ? " << e << " : " << alpha << "f * exp("
                     << e << ") - " << alpha << "f);" << std::endl;
              }
            }
            else {  // If this is our last layer, apply softmax
              code << "  // Apply softmax to our final output" << std::endl;

              // Apply exp to each of the elements
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  " << e << " = exp(" << e << ");" << std::endl;
              }

              // Sum up all the values
              code << "Scalar exp_sum = 0;" << std::endl;
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  exp_sum += " << e << ";" << std::endl;
              }

              // Divide all the values
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  " << e << " /= exp_sum;" << std::endl;
              }
            }
            code << std::endl;

            // Update our input size for the next loop
            input_dimensions = output_dimensions;
          }

          /*************************************************
           *                    OUTPUT                     *
           *************************************************/
          code << "  // Save our value to the output" << std::endl;
          for (unsigned int i = 0; i < input_dimensions; ++i) {
            code << "  output[idx * " << input_dimensions << " + " << i << "] = in" << conv.size() << "[" << i << "];"
                 << std::endl;
          }

          code << "}" << std::endl << std::endl;

          // Update our input dimensions for the next round
          input_dimensions = output_dimensions;
        }

        // Create our OpenCL program, compile it and get our kernels
        cl_int error;
        std::string source = std::string(get_scalar_defines(Scalar())) + READ_IMAGE_TO_NETWORK_CL + code.str();

        const char* cstr = source.c_str();
        size_t csize     = source.size();

        program =
          cl::program(::clCreateProgramWithSource(engine->context, 1, &cstr, &csize, &error), ::clReleaseProgram);

        if (error != CL_SUCCESS) {
          throw std::system_error(error, opencl_error_category(), "Error adding sources to classifier program");
        }

        // Compile the program
        error = ::clBuildProgram(
          program, 0, nullptr, "-cl-single-precision-constant -cl-fast-relaxed-math", nullptr, nullptr);
        if (error != CL_SUCCESS) {

          // Get the first device
          cl_device_id device;
          ::clGetContextInfo(engine->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, nullptr);

          // Get program build log
          size_t used = 0;
          ::clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &used);
          std::vector<char> log(used);
          ::clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &used);

          // Throw an error with the build log
          throw std::system_error(error,
                                  opencl_error_category(),
                                  "Error building classifier program\n" + std::string(log.begin(), log.begin() + used));
        }

        for (unsigned int i = 0; i < structure.size(); ++i) {
          std::string kernel       = "conv" + std::to_string(i);
          unsigned int output_size = structure[i].back().second.size();

          cl::kernel k(::clCreateKernel(program, kernel.c_str(), &error), ::clReleaseKernel);
          throw_cl_error(error, "Failed to create kernel " + kernel);
          conv_layers.emplace_back(k, output_size);
        }

        // Load our image reader kernel
        read_image_to_network =
          cl::kernel(::clCreateKernel(program, "read_image_to_network", &error), ::clReleaseKernel);
        throw_cl_error(error, "Failed to create kernel read_image_to_network");

        // Work out what the widest network layer is
        max_width = 4;
        for (const auto& k : conv_layers) {
          max_width = std::max(max_width, k.second);
        }
      }

      ClassifiedMesh<Scalar, Generator<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Generator>& mesh,
                                                                         const void* image,
                                                                         const uint32_t& format,
                                                                         const mat4<Scalar>& Hoc,
                                                                         const Lens<Scalar>& lens) const {

        cl_image_format fmt;

        switch (format) {
          // Bayer
          case fourcc("GRBG"):
          case fourcc("RGGB"):
          case fourcc("GBRG"):
          case fourcc("BGGR"): fmt = cl_image_format{CL_R, CL_UNORM_INT8}; break;
          case fourcc("BGRA"): fmt = cl_image_format{CL_BGRA, CL_UNORM_INT8}; break;
          case fourcc("RGBA"): fmt = cl_image_format{CL_RGBA, CL_UNORM_INT8}; break;
          // Oh no...
          default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
        }

        cl_image_desc desc = {
          CL_MEM_OBJECT_IMAGE2D, size_t(lens.dimensions[0]), size_t(lens.dimensions[1]), 1, 1, 0, 0, 0, 0, nullptr};

        // Create a buffer for our image
        cl_int error;
        cl::mem cl_image(
          ::clCreateImage(
            context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &fmt, &desc, const_cast<void*>(image), &error),
          ::clReleaseMemObject);
        if (error != CL_SUCCESS) {
          throw std::system_error(error, opencl_error_category(), "Error creating image on device");
        }

        // Map our image into device memory
        std::array<size_t, 3> origin = {{0, 0, 0}};
        std::array<size_t, 3> region = {{size_t(lens.dimensions[0]), size_t(lens.dimensions[1]), 1}};

        cl::event cl_image_loaded;
        cl_event ev           = nullptr;
        std::size_t row_pitch = 0;
        ::clEnqueueMapImage(write_queue,
                            cl_image,
                            false,
                            CL_MAP_READ,
                            origin.data(),
                            region.data(),
                            &row_pitch,
                            nullptr,
                            0,
                            nullptr,
                            &ev,
                            &error);
        if (ev) cl_image_loaded = cl::event(ev, ::clReleaseEvent);
        throw_cl_error(error, "Error mapping image onto device");

        // Project our visual mesh
        std::vector<std::array<int, Generator<Scalar>::N_NEIGHBOURS>> neighbourhood;
        std::vector<int> indices;
        cl::mem cl_pixels;
        cl::event cl_pixels_loaded;
        auto ranges                                                   = mesh.lookup(Hoc, lens);
        std::tie(neighbourhood, indices, cl_pixels, cl_pixels_loaded) = engine->do_project(mesh, ranges, Hoc, lens);

        // This includes the offscreen point at the end
        int n_points = neighbourhood.size();

        // Allocate the neighbourhood buffer
        cl::mem cl_neighbourhood(::clCreateBuffer(engine->context,
                                                  CL_MEM_READ_WRITE,
                                                  n_points * sizeof(std::array<int, Generator<Scalar>::N_NEIGHBOURS>),
                                                  nullptr,
                                                  &error),
                                 ::clReleaseMemObject);
        throw_cl_error(error, "Error allocating neighbourhood buffer on device");

        // Upload the neighbourhood buffer
        cl::event cl_neighbourhood_loaded;
        ev    = nullptr;
        error = ::clEnqueueWriteBuffer(write_queue,
                                       cl_neighbourhood,
                                       false,
                                       0,
                                       n_points * sizeof(std::array<int, Generator<Scalar>::N_NEIGHBOURS>),
                                       neighbourhood.data(),
                                       0,
                                       nullptr,
                                       &ev);
        if (ev) cl_neighbourhood_loaded = cl::event(ev, ::clReleaseEvent);
        throw_cl_error(error, "Error writing neighbourhood points to the device");

        // Allocate two buffers device buffers that we can ping pong
        cl::mem cl_conv_input(
          ::clCreateBuffer(engine->context, CL_MEM_READ_WRITE, sizeof(Scalar) * max_width * n_points, nullptr, &error),
          ::clReleaseMemObject);
        throw_cl_error(error, "Error allocating ping pong buffer 1 on device");
        cl::mem cl_conv_output(
          ::clCreateBuffer(engine->context, CL_MEM_READ_WRITE, sizeof(Scalar) * max_width * n_points, nullptr, &error),
          ::clReleaseMemObject);
        throw_cl_error(error, "Error allocating ping pong buffer 2 on device");

        // The offscreen point gets a value of -1.0 to make it easy to distinguish
        cl::event offscreen_fill_event;
        ev               = nullptr;
        Scalar minus_one = static_cast<Scalar>(-1.0);
        error            = ::clEnqueueFillBuffer(write_queue,
                                      cl_conv_input,
                                      &minus_one,
                                      sizeof(Scalar),
                                      (n_points - 1) * sizeof(std::array<Scalar, 4>),
                                      sizeof(std::array<Scalar, 4>),
                                      0,
                                      nullptr,
                                      &ev);
        if (ev) offscreen_fill_event = cl::event(ev, ::clReleaseEvent);
        throw_cl_error(error, "Error setting the offscreen pixel values");

        // Read the pixels into the buffer
        cl::event img_load_event;
        cl::event network_complete;
        /* Mutex scope */ {
          std::lock_guard<std::mutex> lock(*conv_mutex);

          cl_mem arg;
          arg   = cl_image;
          error = ::clSetKernelArg(read_image_to_network, 0, sizeof(arg), &arg);
          throw_cl_error(error, "Error setting kernel argument 0 for image load kernel");
          error = ::clSetKernelArg(read_image_to_network, 1, sizeof(format), &format);
          throw_cl_error(error, "Error setting kernel argument 1 for image load kernel");
          arg   = cl_pixels;
          error = ::clSetKernelArg(read_image_to_network, 2, sizeof(arg), &arg);
          throw_cl_error(error, "Error setting kernel argument 2 for image load kernel");
          arg   = cl_conv_input;
          error = ::clSetKernelArg(read_image_to_network, 3, sizeof(arg), &arg);
          throw_cl_error(error, "Error setting kernel argument 3 for image load kernel");

          size_t offset[1]       = {0};
          size_t global_size[1]  = {size_t(n_points - 1)};  // -1 as we don't project the offscreen point
          cl_event event_list[2] = {cl_pixels_loaded, cl_image_loaded};
          ev                     = nullptr;
          error                  = ::clEnqueueNDRangeKernel(
            command_queue, read_image_to_network, 1, offset, global_size, nullptr, 2, event_list, &ev);
          if (ev) img_load_event = cl::event(ev, ::clReleaseEvent);
          throw_cl_error(error, "Error queueing the image load kernel");

          // These events are required for our first convolution
          std::vector<cl::event> events({img_load_event, offscreen_fill_event, cl_neighbourhood_loaded});

          for (auto& conv : conv_layers) {
            cl_mem arg;
            arg   = cl_neighbourhood;
            error = ::clSetKernelArg(conv.first, 0, sizeof(arg), &arg);
            throw_cl_error(error, "Error setting argument 0 for convolution kernel");
            arg   = cl_conv_input;
            error = ::clSetKernelArg(conv.first, 1, sizeof(arg), &arg);
            throw_cl_error(error, "Error setting argument 1 for convolution kernel");
            arg   = cl_conv_output;
            error = ::clSetKernelArg(conv.first, 2, sizeof(arg), &arg);
            throw_cl_error(error, "Error setting argument 2 for convolution kernel");

            size_t offset[1]      = {0};
            size_t global_size[1] = {size_t(n_points)};
            cl::event event;
            ev = nullptr;
            std::vector<cl_event> cl_events(events.begin(), events.end());
            error = ::clEnqueueNDRangeKernel(
              command_queue, conv.first, 1, offset, global_size, nullptr, cl_events.size(), cl_events.data(), &ev);
            if (ev) event = cl::event(ev, ::clReleaseEvent);
            throw_cl_error(error, "Error queueing convolution kernel");

            // Convert our events into a vector of events and ping pong our buffers
            events           = std::vector<cl::event>({event});
            network_complete = event;
            std::swap(cl_conv_input, cl_conv_output);
          }
        }

        // Read the pixel coordinates off the device
        cl::event pixels_read;
        ev = nullptr;
        std::vector<std::array<Scalar, 2>> pixels(neighbourhood.size() - 1);
        cl_event iev = cl_pixels_loaded;
        error        = ::clEnqueueReadBuffer(
          read_queue, cl_pixels, false, 0, pixels.size() * sizeof(std::array<Scalar, 2>), pixels.data(), 1, &iev, &ev);
        if (ev) pixels_read = cl::event(ev, ::clReleaseEvent);
        throw_cl_error(error, "Error reading projected pixels");

        // Read the classifications off the device (they'll be in input)
        cl::event classes_read;
        ev  = nullptr;
        iev = network_complete;
        std::vector<Scalar> classifications(neighbourhood.size() * conv_layers.back().second);
        error = ::clEnqueueReadBuffer(read_queue,
                                      cl_conv_input,
                                      false,
                                      0,
                                      classifications.size() * sizeof(Scalar),
                                      classifications.data(),
                                      1,
                                      &iev,
                                      &ev);
        if (ev) classes_read = cl::event(ev, ::clReleaseEvent);
        throw_cl_error(error, "Error reading classified values");

        // Flush the queue to ensure it has executed
        ::clFlush(command_queue);
        ::clFlush(read_queue);
        ::clFlush(write_queue);

        // Wait for the chain to finish up to where we care about it
        cl_event end_events[2] = {pixels_read, classes_read};
        ::clWaitForEvents(2, end_events);

        return ClassifiedMesh<Scalar, Generator<Scalar>::N_NEIGHBOURS>{
          std::move(pixels), std::move(neighbourhood), std::move(indices), std::move(classifications)};
      }

    private:
      Engine<Scalar>* engine;
      cl::context context;
      cl::command_queue command_queue;
      cl::command_queue read_queue;
      cl::command_queue write_queue;
      cl::program program;
      cl::kernel read_image_to_network;
      std::vector<std::pair<cl::kernel, int>> conv_layers;
      std::shared_ptr<std::mutex> conv_mutex;
      int max_width;
    };  // namespace opencl

  }  // namespace opencl
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_OPENCL_CLASSIFIER_H
