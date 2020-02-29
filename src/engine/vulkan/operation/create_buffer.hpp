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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_BUFFER_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_BUFFER_HPP

#include <vulkan/vulkan.h>

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            template <typename Scalar, typename MapFunc>
            inline void map_memory(const VulkanContext& context,
                                   const size_t& size,
                                   const vk::device_memory& mem,
                                   MapFunc&& map) {
                Scalar* payload;
                throw_vk_error(vkMapMemory(context.device, mem, 0, size, 0, (void**) &payload),
                               "Failed to host map rco device memory");
                map(payload);
                vkUnmapMemory(context.device, mem);
            }

            inline std::pair<vk::buffer, vk::device_memory> create_buffer(const VulkanContext& context,
                                                                          const VkDeviceSize& size,
                                                                          const VkBufferUsageFlags& usage,
                                                                          const VkSharingMode& sharing,
                                                                          const std::vector<uint32_t>& queues,
                                                                          const VkMemoryPropertyFlags& properties) {

                // Create the buffer
                VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  size,
                                                  usage,
                                                  sharing,
                                                  static_cast<uint32_t>(queues.size()),
                                                  queues.data()};
                VkBuffer buf;
                throw_vk_error(vkCreateBuffer(context.device, &buffer_info, 0, &buf), "Failed to create buffer");
                vk::buffer buffer(buf, [&context](auto p) { vkDestroyBuffer(context.device, p, nullptr); });

                VkPhysicalDeviceMemoryProperties mem_props;
                vkGetPhysicalDeviceMemoryProperties(context.phys_device, &mem_props);

                VkMemoryRequirements mem_requirements;
                vkGetBufferMemoryRequirements(context.device, buffer, &mem_requirements);

                // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
                uint32_t heap_index = VK_MAX_MEMORY_TYPES;

                for (uint32_t k = 0; k < mem_props.memoryTypeCount; k++) {
                    if ((mem_requirements.memoryTypeBits & (1 << k))
                        && ((mem_props.memoryTypes[k].propertyFlags & properties) == properties)
                        && (size < mem_props.memoryHeaps[mem_props.memoryTypes[k].heapIndex].size)) {
                        heap_index = k;
                        break;
                    }
                }

                throw_vk_error(heap_index == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS,
                               "Failed to find enough allocatable device memory");

                VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, size, heap_index};

                VkDeviceMemory mem;
                throw_vk_error(vkAllocateMemory(context.device, &memoryAllocateInfo, 0, &mem),
                               "Failed to allocate memory on the device");
                vk::device_memory memory(mem, [&context](auto p) { vkFreeMemory(context.device, p, nullptr); });

                return std::make_pair(buffer, memory);
            }

            inline void bind_buffer(const VulkanContext& context,
                                    const vk::buffer& buffer,
                                    const vk::device_memory& memory,
                                    const uint32_t offset) {
                // Attach the allocated device memory to it
                throw_vk_error(vkBindBufferMemory(context.device, buffer, memory, offset),
                               "Failed to bind buffer to device memory");
            }
        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_BUFFER_HPP
