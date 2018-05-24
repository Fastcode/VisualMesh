/*
 * Copyright (C) 2017-2018 Trent Houliston <trent@houliston.me>
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

#ifndef MESH_OPENCL_CLASSIFIER_H
#define MESH_OPENCL_CLASSIFIER_H

class Classifier {
private:
  using weights_t = std::vector<std::vector<Scalar>>;
  using biases_t  = std::vector<Scalar>;

  using layer_t             = std::pair<weights_t, biases_t>;
  using conv_layer_t        = std::vector<layer_t>;
  using network_structure_t = std::vector<conv_layer_t>;

public:
  Classifier() : mesh(nullptr) {

    // TODO move all the OpenCL Context creation code into this guy
  }

  Classifier(VisualMesh* mesh, const network_structure_t& structure)
    : mesh(mesh), conv_mutex(std::make_shared<std::mutex>()) {

    // Build using a string stream
    std::stringstream code;

    // Set our precision for how many digits a float has
    code << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1);

    auto vector_type = [](const int& size) {
      return (size == 1 || size == 2 || size == 4 || size == 8 || size == 16) ? size : 0;
    };

    for (uint conv_no = 0; conv_no < structure.size(); ++conv_no) {
      auto& conv = structure[conv_no];

      // We need to work out the input and output sizes for our convolution
      int conv_in_size;
      int conv_out_size;

      // On the first convolution we assume an input size of 4
      if (conv_no == 0) { conv_in_size = 4; }
      else {
        // The output dimension of our previous bias vector
        conv_in_size = structure[conv_no - 1].back().second.size();
      }

      // The output dimension of our last bias vector
      conv_out_size = conv.back().second.size();

      // Work out our input and output types
      std::string in_type("float");
      if (vector_type(conv_in_size)) { in_type.append(std::to_string(conv_in_size)); }
      std::string out_type("float");
      if (vector_type(conv_out_size)) { out_type.append(std::to_string(conv_out_size)); }

      // Write our OpenCL kernel definition
      code << "kernel void conv" << conv_no << "(global const int* neighbourhood, global const " << in_type
           << "* input, global " << out_type << "* output) {" << std::endl
           << std::endl;

      code << "    // Get our kernel index" << std::endl;
      code << "    const int idx = get_global_id(0);" << std::endl << std::endl;

      /*************************************************
       *                    GATHER                     *
       *************************************************/

      code << "    // Gather from our neighbourhood " << std::endl;
      if (vector_type(conv_in_size)) {
        code << "    " << in_type << " in0[7] = {" << std::endl;
        code << "        input[idx]," << std::endl;
        for (int i = 0; i < 6; ++i) {
          code << "        input[neighbourhood[idx * 6 + " << i << "]]";
          if (i != 5) { code << ","; }
          code << std::endl;
        }
        code << "    };";
      }
      // Perform our gather step for non vectorized data
      else {
        code << "    float in0[" << (conv_in_size * 7) << "] = {" << std::endl;

        // Read the ones for our own index
        for (int j = 0; j < conv_in_size; ++j) {
          code << "        input[idx * " << conv_in_size << " + " << j << "]";
        }

        // Read our neighbourhood
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < conv_in_size; ++j) {
            code << "        input[neighbourhood[idx * 6 + " << i << "] * " << conv_in_size << " + " << j << "]";

            if (i < 6 || j + 1 < conv_in_size) { code << ","; }
            code << std::endl;
          }
        }
        code << "    };";
      }

      code << std::endl << std::endl;

      /*************************************************
       *                WEIGHTS + BIAS                 *
       *************************************************/

      // Now we have to do our layer operations
      int in_size = conv_in_size;
      for (uint layer_no = 0; layer_no < conv.size(); ++layer_no) {
        const auto& weights = conv[layer_no].first;
        const auto& biases  = conv[layer_no].second;

        const int vector_in  = vector_type(in_size);
        const int vector_out = vector_type(biases.size());

        code << "    // Perform our matrix multiplication for weights and add bias for layer " << layer_no << std::endl;

        // Open our next input (either vector or not)
        if (vector_out) {
          code << "    float" << vector_out << " in" << (layer_no + 1) << " = (float" << vector_out << ")("
               << std::endl;
        }
        else {
          code << "    float in" << (layer_no + 1) << "[" << biases.size() << "] = {" << std::endl;
        }

        // Matrix multiplication + bias
        if (vector_in) {
          for (uint i = 0; i < biases.size(); ++i) {
            code << "        ";
            for (uint j = 0; j < weights.size(); j += vector_in) {

              // If our data is gathered, we need to get our gathered index
              std::string gathered_index = layer_no == 0 ? "[" + std::to_string(j / vector_in) + "]" : "";

              // Dot our element with our fixed data
              code << "dot(in" << layer_no << gathered_index << ", (float" << vector_in << ")(";

              // Write our fixed data
              for (uint k = j; k < j + vector_in; ++k) {
                code << weights[k][i];
                if (k + 1 < j + vector_in) { code << ", "; }
              }

              // End
              code << ")) + ";
            }
            code << biases[i];
            if (i + 1 < biases.size()) { code << ","; }
            code << std::endl;
          }
        }
        else {
          for (uint i = 0; i < biases.size(); ++i) {
            code << "        ";
            for (uint j = 0; j < weights.size(); ++j) {
              code << "in" << layer_no << "[" << j << "] * " << weights[j][i] << " + ";
            }
            code << biases[i];
            if (i + 1 < biases.size()) { code << ","; }
            code << std::endl;
          }
        }

        // Close our output
        if (vector_out) { code << "    );"; }
        else {
          code << "    };";
        }
        code << std::endl << std::endl;


        /*************************************************
         *                  ACTIVATION.                  *
         *************************************************/

        // Apply our activation function
        code << "    // Apply the activation function" << std::endl;

        // selu constants
        constexpr const float lambda = 1.0507009873554804934193349852946;
        constexpr const float alpha  = 1.6732632423543772848170429916717;

        // Apply selu
        if (vector_out) {
          std::string e = "in" + std::to_string(layer_no + 1);

          code << "    " << e << " = " << lambda << "f * select(" << alpha << "f * exp(" << e << ") - " << alpha
               << "f, in" << (layer_no + 1) << ", " << e << " > 0);" << std::endl;  // select(a, b, c) == c ? b : a
        }
        else {
          for (uint i = 0; i < biases.size(); ++i) {
            std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
            code << "    " << e << " = " << lambda << "f * (" << e << " > 0 ? " << e << " : " << alpha << "f * exp("
                 << e << ") - " << alpha << "f);" << std::endl;
          }
        }
        code << std::endl;

        // If this is our last layer, apply softmax
        if (conv_no + 1 == structure.size() && layer_no + 1 == conv.size()) {
          code << "    // Apply softmax to our final output" << std::endl;

          if (vector_out) {
            std::string e = "in" + std::to_string(layer_no + 1);
            code << "    " << e << " = exp(" << e << ");" << std::endl;
            code << "    " << e << " = " << e << " / dot(" << e << ", (float" << vector_out << ")(1));" << std::endl;
          }
          else {

            // Apply exp to each of the elements
            for (uint i = 0; i < biases.size(); ++i) {
              std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
              code << "    " << e << " = exp(" << e << ");" << std::endl;
            }

            // Sum up all the values
            code << "float exp_sum = 0";
            for (uint i = 0; i < biases.size(); ++i) {
              std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
              code << "    exp_sum += " << e << ";" << std::endl;
            }

            // Divide all the values
            for (uint i = 0; i < biases.size(); ++i) {
              std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
              code << "    " << e << " /= exp_sum" << std::endl;
            }
          }

          code << std::endl;
        }

        // Update our input size for the next loop
        in_size = biases.size();
      }

      /*************************************************
       *                    OUTPUT                     *
       *************************************************/
      code << "    // Save our value to the output" << std::endl;
      if (vector_type(conv_out_size)) {
        code << "    output[idx] = "
             << "in" << conv.size() << ";" << std::endl;
      }
      else {
        for (int i = 0; i < conv_out_size; ++i) {
          code << "    output[idx * " << conv_out_size << " + " << i << "] = in" << conv.size() << "[" << i << "];"
               << std::endl;
        }
      }

      code << "}" << std::endl << std::endl;
    }

    // Create our OpenCL program, compile it and get our kernels
    cl_int error;
    std::string source = code.str();
    const char* cstr   = source.c_str();
    size_t csize       = source.size();

    program = cl::program(::clCreateProgramWithSource(mesh->context, 1, &cstr, &csize, &error), ::clReleaseProgram);

    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error adding sources to classifier program");
    }

    // Compile the program
    error =
      ::clBuildProgram(program, 0, nullptr, "-cl-single-precision-constant -cl-fast-relaxed-math", nullptr, nullptr);
    if (error != CL_SUCCESS) {

      // Get the first device
      cl_device_id device;
      ::clGetContextInfo(mesh->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, nullptr);

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

    for (uint i = 0; i < structure.size(); ++i) {
      std::string kernel = "conv" + std::to_string(i);
      uint output_size   = structure[i].back().second.size();

      cl_int error;
      cl::kernel k(::clCreateKernel(program, kernel.c_str(), &error), ::clReleaseKernel);
      if (error != CL_SUCCESS) {
        throw std::system_error(error, opencl_error_category(), "Failed to create kernel " + kernel);
      }
      else {
        conv_layers.emplace_back(k, output_size);
      }
    }
  }

  ClassifiedMesh operator()(const void* image, const FOURCC& format, const mat4& Hoc, const Lens& lens) {


    cl_image_format fmt;

    switch (format) {
      // Bayer
      case GRBG:
      case RGGB:
      case GBRG:
      case BGGR: fmt = cl_image_format{CL_R, CL_UNORM_INT8}; break;
      case BGRA: fmt = cl_image_format{CL_BGRA, CL_UNORM_INT8}; break;
      case RGBA: fmt = cl_image_format{CL_RGBA, CL_UNORM_INT8}; break;
      // Oh no...
      default: throw std::runtime_error("Unsupported image format");
    }

    cl_image_desc desc = {
      CL_MEM_OBJECT_IMAGE2D, size_t(lens.dimensions[0]), size_t(lens.dimensions[1]), 1, 1, 0, 0, 0, 0, nullptr};

    // Create a buffer for our image
    cl_int error;
    cl::mem img(::clCreateImage(
                  mesh->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &fmt, &desc, const_cast<void*>(image), &error),
                ::clReleaseMemObject);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error creating image on device");
    }

    // Map our image into device memory
    std::array<size_t, 3> origin = {{0, 0, 0}};
    std::array<size_t, 3> region = {{size_t(lens.dimensions[0]), size_t(lens.dimensions[1]), 1}};

    cl::event img_event;
    cl_event ev           = nullptr;
    std::size_t row_pitch = 0;
    ::clEnqueueMapImage(
      mesh->queue, img, false, CL_MAP_READ, origin.data(), region.data(), &row_pitch, nullptr, 0, nullptr, &ev, &error);
    if (ev) img_event = cl::event(ev, ::clReleaseEvent);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error mapping image onto device");
    }

    // Project our visual mesh
    auto projection = mesh->project(Hoc, lens);

    // This includes the offscreen point at the end
    int points = projection.neighbourhood.size();


    // First layer, output from the image
    cl::mem img_load_buffer(
      ::clCreateBuffer(mesh->context, CL_MEM_READ_WRITE, sizeof(cl_float4) * points, nullptr, &error),
      ::clReleaseMemObject);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error allocating buffer on device");
    }

    // Zero out the final value in the buffer
    cl::event offscreen_fill_event;
    ev         = nullptr;
    float zero = 0.0f;
    error      = ::clEnqueueFillBuffer(mesh->queue,
                                  img_load_buffer,
                                  &zero,
                                  sizeof(float),
                                  (points - 1) * sizeof(cl_float4),
                                  sizeof(cl_float4),
                                  0,
                                  nullptr,
                                  &ev);
    if (ev) offscreen_fill_event = cl::event(ev, ::clReleaseEvent);
    if (error != CL_SUCCESS) {
      throw std::system_error(error, opencl_error_category(), "Error setting the offscreen pixel values");
    }

    // Read the pixels into the buffer
    cl::event img_load_event;
    ev = nullptr;
    /* Mutex scope */ {
      std::lock_guard<std::mutex> lock(mesh->read_image_to_network_mutex);

      error = ::clSetKernelArg(mesh->read_image_to_network, 0, img.size(), &img);
      if (error != CL_SUCCESS) {
        throw std::system_error(
          error, opencl_error_category(), "Error setting kernel argument 0 for image load kernel");
      }
      error = ::clSetKernelArg(mesh->read_image_to_network, 1, sizeof(format), &format);
      if (error != CL_SUCCESS) {
        throw std::system_error(
          error, opencl_error_category(), "Error setting kernel argument 1 for image load kernel");
      }
      error = ::clSetKernelArg(
        mesh->read_image_to_network, 2, projection.cl_pixel_coordinates.size(), &projection.cl_pixel_coordinates);
      if (error != CL_SUCCESS) {
        throw std::system_error(
          error, opencl_error_category(), "Error setting kernel argument 2 for image load kernel");
      }
      error = ::clSetKernelArg(mesh->read_image_to_network, 3, img_load_buffer.size(), &img_load_buffer);
      if (error != CL_SUCCESS) {
        throw std::system_error(
          error, opencl_error_category(), "Error setting kernel argument 3 for image load kernel");
      }

      size_t offset[1]       = {0};
      size_t global_size[1]  = {size_t(points - 1)};
      cl_event event_list[2] = {projection.cl_pixel_coordinates_event, img_event};
      error                  = ::clEnqueueNDRangeKernel(mesh->queue,
                                       mesh->read_image_to_network,
                                       1,
                                       offset,
                                       global_size,  // -1 as we don't project the offscreen point
                                       nullptr,
                                       2,
                                       event_list,
                                       &ev);
      if (ev) img_load_event = cl::event(ev, ::clReleaseEvent);
      if (error != CL_SUCCESS) {
        throw std::system_error(error, opencl_error_category(), "Error queueing the image load kernel");
      }
    }

    // Our buffers for each layer
    std::vector<std::pair<cl::mem, std::vector<cl::event>>> layer_buffers;

    // These make up our first buffers
    layer_buffers.emplace_back(
      img_load_buffer,
      std::vector<cl::event>({img_load_event, offscreen_fill_event, projection.cl_neighbourhood_event}));

    // Run each of our conv layers
    /* Mutex Scope */ {
      std::lock_guard<std::mutex> lock(*conv_mutex);

      for (auto& conv : conv_layers) {

        // Create an output buffer
        cl::mem out_buffer(
          ::clCreateBuffer(
            mesh->context, CL_MEM_READ_WRITE, size_t(conv.second * points * sizeof(float)), nullptr, &error),
          ::clReleaseMemObject);
        if (error) {
          throw std::system_error(
            error, opencl_error_category(), "Error creating output buffer for the convolution kernel");
        }

        error = ::clSetKernelArg(conv.first, 0, projection.cl_neighbourhood.size(), &projection.cl_neighbourhood);
        if (error) {
          throw std::system_error(error, opencl_error_category(), "Error setting argument 0 for convolution kernel");
        }
        error = ::clSetKernelArg(conv.first, 1, layer_buffers.back().first.size(), &layer_buffers.back().first);
        if (error) {
          throw std::system_error(error, opencl_error_category(), "Error setting argument 1 for convolution kernel");
        }
        error = ::clSetKernelArg(conv.first, 2, out_buffer.size(), &out_buffer);
        if (error) {
          throw std::system_error(error, opencl_error_category(), "Error setting argument 2 for convolution kernel");
        }


        // Convert our events into
        std::vector<cl_event> events(layer_buffers.back().second.begin(), layer_buffers.back().second.end());

        size_t offset[1]      = {0};
        size_t global_size[1] = {size_t(points)};
        cl::event event;
        ev    = nullptr;
        error = ::clEnqueueNDRangeKernel(
          mesh->queue, conv.first, 1, offset, global_size, nullptr, events.size(), events.data(), &ev);
        if (ev) event = cl::event(ev, ::clReleaseEvent);
        if (error) { throw std::system_error(error, opencl_error_category(), "Error queueing convolution kernel"); }

        layer_buffers.emplace_back(out_buffer, std::vector<cl::event>({event}));
      }
    }

    // Flush the queue to ensure it has executed
    ::clFlush(mesh->queue);

    std::vector<std::pair<int, LazyBufferReader<Scalar>>> outputs;
    for (uint i = 0; i < layer_buffers.size(); ++i) {

      uint dims = i == 0 ? 4 : conv_layers[i - 1].second;

      outputs.emplace_back(
        dims, LazyBufferReader<Scalar>(mesh->queue, layer_buffers[i].first, layer_buffers[i].second, points * dims));
    }

    return ClassifiedMesh{projection.pixel_coordinates,
                          std::move(projection.neighbourhood),
                          std::move(projection.global_indices),
                          std::move(outputs)};
  }

private:
  VisualMesh* mesh;
  cl::program program;
  std::vector<std::pair<cl::kernel, int>> conv_layers;
  std::shared_ptr<std::mutex> conv_mutex;
};

#endif  // MESH_OPENCL_CLASSIFIER_H
