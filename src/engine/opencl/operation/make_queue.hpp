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

#ifndef VISUALMESH_OPENCL_OPERATION_MAKE_QUEUE_HPP
#define VISUALMESH_OPENCL_OPERATION_MAKE_QUEUE_HPP

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace opencl {
        namespace operation {

            /**
             * @brief Make an OpenCL command queue
             *
             * @param context the context to make the queue for
             * @param device  the device to make the queue for
             *
             * @return a reference counted tracker of a command queue
             */
            inline cl::command_queue make_queue(cl_context context, cl_device_id device) {
                cl_command_queue queue;
                cl_int error;
                // Use out of order execution if we can
                queue = ::clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
                if (error == CL_INVALID_VALUE) { queue = ::clCreateCommandQueue(context, device, 0, &error); }
                throw_cl_error(error, "Error creating the OpenCL command queue");
                return cl::command_queue(queue, ::clReleaseCommandQueue);
            }

        }  // namespace operation
    }      // namespace opencl
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_OPENCL_OPERATION_MAKE_QUEUE_HPP
