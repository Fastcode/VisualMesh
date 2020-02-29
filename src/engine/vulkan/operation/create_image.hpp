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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_IMAGE_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_IMAGE_HPP

extern "C" {
#include <vulkan/vulkan.h>
}

#include "create_command_buffer.hpp"
#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            inline std::pair<vk::image, vk::device_memory> create_image(const VulkanContext& context,
                                                                        const VkExtent3D& dimensions,
                                                                        const VkFormat& format,
                                                                        const VkBufferUsageFlags& usage,
                                                                        const VkSharingMode& sharing,
                                                                        const std::vector<uint32_t>& queues,
                                                                        const VkMemoryPropertyFlags& properties) {

                // Create the image
                VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                                nullptr,
                                                0,
                                                VK_IMAGE_TYPE_2D,
                                                format,
                                                dimensions,
                                                1,
                                                1,
                                                VK_SAMPLE_COUNT_1_BIT,  // 1 sample per pixel
                                                VK_IMAGE_TILING_OPTIMAL,
                                                usage,
                                                sharing,
                                                static_cast<uint32_t>(queues.size()),
                                                queues.data(),
                                                VK_IMAGE_LAYOUT_GENERAL};

                VkImage img;
                throw_vk_error(vkCreateImage(context.device, &image_info, nullptr, &img), "Failed to create image");
                vk::image image(img, [&context](auto p) { vkDestroyImage(context.device, p, nullptr); });

                // Get the memory requirements for the image
                VkMemoryRequirements mem_requirements;
                vkGetImageMemoryRequirements(context.device, image, &mem_requirements);

                VkPhysicalDeviceMemoryProperties mem_props;
                vkGetPhysicalDeviceMemoryProperties(context.phys_device, &mem_props);

                // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
                uint32_t heap_index = VK_MAX_MEMORY_TYPES;

                for (uint32_t k = 0; k < mem_props.memoryTypeCount; k++) {
                    if ((mem_requirements.memoryTypeBits & (1 << k))
                        && ((mem_props.memoryTypes[k].propertyFlags & properties) == properties)
                        && (mem_requirements.size < mem_props.memoryHeaps[mem_props.memoryTypes[k].heapIndex].size)) {
                        heap_index = k;
                        break;
                    }
                }

                throw_vk_error(heap_index == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS,
                               "Failed to find enough allocatable device memory");

                VkMemoryAllocateInfo memory_alloc_info = {
                  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mem_requirements.size, heap_index};

                VkDeviceMemory mem;
                throw_vk_error(vkAllocateMemory(context.device, &memory_alloc_info, 0, &mem),
                               "Failed to allocate memory on the device");
                vk::device_memory memory(mem, [&context](auto p) { vkFreeMemory(context.device, p, nullptr); });

                return std::make_pair(image, memory);
            }

            inline void bind_image(const VulkanContext& context,
                                   const vk::image& image,
                                   const vk::device_memory& memory,
                                   const uint32_t offset) {
                // Attach the allocated device memory to it
                throw_vk_error(vkBindImageMemory(context.device, image, memory, offset),
                               "Failed to bind image to device memory");
            }

        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_IMAGE_HPP
