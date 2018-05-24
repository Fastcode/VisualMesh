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

#ifndef VISUALMESH_ENGINE_OPENCL_LAZY_BUFFER_READER_HPP
#define VISUALMESH_ENGINE_OPENCL_LAZY_BUFFER_READER_HPP

namespace visualmesh {
namespace engine {
  namespace opencl {

    template <typename T>
    struct LazyBufferReader {

      LazyBufferReader() = default;

      LazyBufferReader(const cl::command_queue& queue,
                       const cl::mem& buffer,
                       const std::vector<cl::event>& ready,
                       uint n_elements)
        : queue(queue), buffer(buffer), ready(ready), n_elements(n_elements) {}

      LazyBufferReader(const cl::command_queue& queue, const cl::mem& buffer, const cl::event& ready, uint n_elements)
        : queue(queue), buffer(buffer), ready(std::vector<cl::event>({ready})), n_elements(n_elements) {}

      template <typename U>
      std::vector<U> as() const {
        // Number of output elements will change if the sizes are different
        std::vector<U> output(n_elements * sizeof(T) / sizeof(U));

        std::vector<cl_event> events(ready.begin(), ready.end());
        cl_int error = ::clEnqueueReadBuffer(
          queue, buffer, true, 0, n_elements * sizeof(T), output.data(), events.size(), events.data(), nullptr);
        if (error != CL_SUCCESS) {
          throw std::system_error(error, opencl_error_category(), "Error reading vector buffer for lazy evaluation");
        }
        return output;
      }

      operator std::vector<T>() const {
        std::vector<T> output(n_elements);
        std::vector<cl_event> events(ready.begin(), ready.end());
        cl_int error = ::clEnqueueReadBuffer(
          queue, buffer, true, 0, n_elements * sizeof(T), output.data(), events.size(), events.data(), nullptr);
        if (error != CL_SUCCESS) {
          throw std::system_error(error, opencl_error_category(), "Error reading vector buffer for lazy evaluation");
        }
        return output;
      }

      operator T() const {
        T output;
        std::vector<cl_event> events(ready.begin(), ready.end());
        cl_int error =
          ::clEnqueueReadBuffer(queue, buffer, true, 0, sizeof(T), &output, events.size(), events.data(), nullptr);
        if (error != CL_SUCCESS) {
          throw std::system_error(error, opencl_error_category(), "Error reading buffer for lazy evaluation");
        }
        return output;
      }

      cl::command_queue queue;
      cl::mem buffer;
      std::vector<cl::event> ready;
      uint n_elements;
    };

  }  // namespace opencl
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_OPENCL_LAZY_BUFFER_READER_HPP
