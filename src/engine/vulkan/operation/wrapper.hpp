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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_WRAPPER_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_WRAPPER_HPP

#include <string>
#include <vulkan/vulkan.hpp>

#include "vulkan_error_category.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {

        /**
         * @brief A shorthand function to throw an Vulkan system error if the error code is not success
         *
         * @param code  the error code to check and throw
         * @param msg   the message to attach to the exception if it is thrown
         */
        void throw_vk_error(const vk::Result& code, const std::string& msg) {
            if (code != vk::Result::eSuccess) {
                throw std::system_error(int(code), operation::vulkan_error_category(), msg);
            }
        }

    }  // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_WRAPPER_HPP
