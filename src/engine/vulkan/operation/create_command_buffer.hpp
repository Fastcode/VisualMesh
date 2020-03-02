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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_COMMAND_BUFFER_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_COMMAND_BUFFER_HPP

extern "C" {
#include <vulkan/vulkan.h>
}

#include <array>
#include <vector>

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            inline vk::command_pool create_command_pool(const VulkanContext& context, const uint32_t& queue_family) {
                VkCommandPoolCreateInfo create_info = {
                  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, queue_family};
                VkCommandPool command_pool;
                throw_vk_error(vkCreateCommandPool(context.device, &create_info, 0, &command_pool),
                               "Failed to create a command pool");
                return vk::command_pool(command_pool,
                                        [&context](auto p) { vkDestroyCommandPool(context.device, p, nullptr); });
            }

            inline vk::command_buffer create_command_buffer(const VulkanContext& context,
                                                            const vk::command_pool& command_pool,
                                                            const bool& primary) {
                // Allocate a command buffer from the command pool
                VkCommandBufferAllocateInfo alloc_info = {
                  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                  0,
                  command_pool,
                  primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                  1};

                VkCommandBuffer cmd_buf;
                throw_vk_error(vkAllocateCommandBuffers(context.device, &alloc_info, &cmd_buf),
                               "Failed to allocate a command buffer");
                vk::command_buffer command_buffer(cmd_buf, [&context, &command_pool](auto p) {
                    std::array<VkCommandBuffer, 1> p_arr = {p};
                    vkFreeCommandBuffers(context.device, command_pool, p_arr.size(), p_arr.data());
                });

                VkCommandBufferBeginInfo begin_info = {
                  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};

                throw_vk_error(vkBeginCommandBuffer(command_buffer, &begin_info), "Failed to begin a command buffer");

                return command_buffer;
            }

            inline void submit_command_buffer(
              const VkQueue& queue,
              const vk::command_buffer& command_buffer,
              const std::vector<std::pair<vk::semaphore, VkPipelineStageFlags>>& wait_semaphores,
              const std::vector<vk::semaphore>& signal_semaphores) {
                throw_vk_error(vkEndCommandBuffer(command_buffer), "Failed to end command buffer");

                std::vector<VkSemaphore> waits;
                std::vector<VkPipelineStageFlags> wait_stages;
                for (const auto& wait : wait_semaphores) {
                    waits.push_back(wait.first);
                    wait_stages.push_back(wait.second);
                }
                std::vector<VkSemaphore> signals;
                for (const auto& semaphore : signal_semaphores) {
                    signals.push_back(semaphore);
                }
                std::array<VkCommandBuffer, 1> buf = {command_buffer};
                VkSubmitInfo submit_info           = {VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                            nullptr,
                                            static_cast<uint32_t>(waits.size()),
                                            waits.data(),
                                            wait_stages.data(),
                                            buf.size(),
                                            buf.data(),
                                            static_cast<uint32_t>(signals.size()),
                                            signals.data()};

                throw_vk_error(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE),
                               "Failed to submit command buffer to queue");
            }

            inline void submit_command_buffer(const VkQueue& queue,
                                              const vk::command_buffer& command_buffer,
                                              const VkFence& fence) {
                throw_vk_error(vkEndCommandBuffer(command_buffer), "Failed to end command buffer");

                std::array<VkCommandBuffer, 1> buf = {command_buffer};
                VkSubmitInfo submit_info           = {
                  VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, 0, 0, buf.size(), buf.data(), 0, nullptr};

                throw_vk_error(vkQueueSubmit(queue, 1, &submit_info, fence),
                               "Failed to submit command buffer to queue");
            }
        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_COMMAND_BUFFER_HPP
