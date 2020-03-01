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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_WRAPPER_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_WRAPPER_HPP

extern "C" {
#include <vulkan/vulkan.h>
}

#include <memory>
#include <string>

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
        void throw_vk_error(const VkResult& code, const std::string& msg) {
            if (code != VK_SUCCESS) { throw std::system_error(code, operation::vulkan_error_category(), msg); }
        }

        namespace vk {
            template <typename T>
            struct vulkan_wrapper : public std::shared_ptr<std::remove_pointer_t<T>> {
                using std::shared_ptr<std::remove_pointer_t<T>>::shared_ptr;
                using std::shared_ptr<std::remove_pointer_t<T>>::reset;

                operator T() const {
                    return this->get();
                }
            };

            using instance               = vulkan_wrapper<::VkInstance>;
            using device                 = vulkan_wrapper<::VkDevice>;
            using buffer                 = vulkan_wrapper<::VkBuffer>;
            using device_memory          = vulkan_wrapper<::VkDeviceMemory>;
            using sampler                = vulkan_wrapper<::VkSampler>;
            using shader_module          = vulkan_wrapper<::VkShaderModule>;
            using image                  = vulkan_wrapper<::VkImage>;
            using descriptor_pool        = vulkan_wrapper<::VkDescriptorPool>;
            using command_pool           = vulkan_wrapper<::VkCommandPool>;
            using command_buffer         = vulkan_wrapper<::VkCommandBuffer>;
            using descriptor_buffer_info = vulkan_wrapper<::VkDescriptorBufferInfo>;
        }  // namespace vk

        enum class DeviceType { CPU, GPU, INTEGRATED_GPU, DISCRETE_GPU, VIRTUAL_GPU, ANY };

        struct VulkanContext {
            vk::instance instance;
            VkPhysicalDevice phys_device;
            vk::device device;
            uint32_t compute_queue_family;
            uint32_t transfer_queue_family;
            VkQueue compute_queue;
            VkQueue transfer_queue;
            vk::command_pool compute_command_pool;
            vk::command_pool transfer_command_pool;
        };
    }  // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_WRAPPER_HPP
