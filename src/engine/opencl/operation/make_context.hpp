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

#ifndef VISUALMESH_OPENCL_OPERATION_MAKE_CONTEXT_HPP
#define VISUALMESH_OPENCL_OPERATION_MAKE_CONTEXT_HPP

#include <utility>
#include <vector>

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
  namespace opencl {
    namespace operation {

      /**
       * @brief Find and create an OpenCL command context for a specific device. Or the best device if a specific device
       * is not provided.
       *
       */
      inline std::pair<cl::context, cl_device_id> make_context() {

        // Get our platforms
        cl_uint platform_count = 0;
        ::clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms(platform_count);
        ::clGetPlatformIDs(platforms.size(), platforms.data(), nullptr);

        // Which device/platform we are going to use
        cl_device_id best_device   = nullptr;
        cl_uint best_compute_units = 0;

        // Go through our platforms
        for (const auto& platform : platforms) {

          cl_uint device_count = 0;
          ::clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
          std::vector<cl_device_id> devices(device_count);
          ::clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);

          // Go through our devices on the platform
          for (const auto& device : devices) {

            cl_uint max_compute_units = 0;
            ::clGetDeviceInfo(
              device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, nullptr);

            if (max_compute_units > best_compute_units) {
              best_compute_units = max_compute_units;
              best_device        = device;
            }
          }
        }

        // Print information about our selected device
        if (!best_device) {
          throw std::system_error(CL_INVALID_DEVICE, opencl_error_category(), "Error selecting an OpenCL device");
        }

        // Make context
        cl_int error;
        cl::context context =
          cl::context(::clCreateContext(nullptr, 1, &best_device, nullptr, nullptr, &error), ::clReleaseContext);
        if (error) { throw std::system_error(error, opencl_error_category(), "Error creating the OpenCL context"); }
        return std::make_pair(context, best_device);
      }

    }  // namespace operation
  }    // namespace opencl
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_OPENCL_OPERATION_MAKE_CONTEXT_HPP
