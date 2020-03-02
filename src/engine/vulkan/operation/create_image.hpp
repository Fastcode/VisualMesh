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

#include <tuple>
#include <utility>
#include <vector>

#include "create_buffer.hpp"
#include "create_command_buffer.hpp"
#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            inline std::pair<vk::image, vk::device_memory> create_image(const VulkanContext& context,
                                                                        const VkExtent3D& extent,
                                                                        const VkFormat& format,
                                                                        const VkImageUsageFlags& usage,
                                                                        const VkSharingMode& sharing,
                                                                        const std::vector<uint32_t>& queues,
                                                                        const VkMemoryPropertyFlags& properties) {

                // Create the image
                VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                                nullptr,
                                                0,
                                                VK_IMAGE_TYPE_2D,
                                                format,
                                                extent,
                                                1,
                                                1,
                                                VK_SAMPLE_COUNT_1_BIT,  // 1 sample per pixel
                                                VK_IMAGE_TILING_OPTIMAL,
                                                usage,
                                                sharing,
                                                static_cast<uint32_t>(queues.size()),
                                                queues.data(),
                                                VK_IMAGE_LAYOUT_UNDEFINED};

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

            inline void transition_image_layout(const VulkanContext& context,
                                                const vk::image& image,
                                                const VkImageLayout& old_layout,
                                                const VkImageLayout& new_layout) {

                vk::command_buffer command_buffer =
                  operation::create_command_buffer(context, context.transfer_command_pool, true);

                VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                nullptr,
                                                0,
                                                0,
                                                old_layout,
                                                new_layout,
                                                VK_QUEUE_FAMILY_IGNORED,
                                                VK_QUEUE_FAMILY_IGNORED,
                                                image,
                                                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

                VkPipelineStageFlags source_stage;
                VkPipelineStageFlags destination_stage;

                if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                    barrier.srcAccessMask = 0;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                    source_stage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                    destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                }
                else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                         && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                    source_stage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
                    destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                }
                else {
                    throw_vk_error(VK_ERROR_FORMAT_NOT_SUPPORTED, "Unsupported layout transition");
                }

                vkCmdPipelineBarrier(
                  command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                submit_command_buffer(context.transfer_queue, command_buffer, {}, {});
                vkQueueWaitIdle(context.transfer_queue);
            }

            inline void copy_image_to_device(const VulkanContext& context,
                                             const void* image,
                                             const vec2<int>& dimensions,
                                             const std::pair<vk::image, vk::device_memory>& vk_image,
                                             const VkFormat& format) {

                // Create a staging buffer for the image and copy the image to the device
                uint32_t image_size = dimensions[0] * dimensions[1] * (format == VK_FORMAT_R8_UNORM ? 1 : 4);
                std::pair<vk::buffer, vk::device_memory> vk_image_buffer =
                  operation::create_buffer(context,
                                           image_size,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                operation::map_memory<void>(
                  context, 0, VK_WHOLE_SIZE, vk_image_buffer.second, [&image, &image_size](void* payload) {
                      std::memcpy(payload, image, image_size);
                  });
                operation::bind_buffer(context, vk_image_buffer.first, vk_image_buffer.second, 0);

                // Transistion the image layout from undefined to transfer destination
                transition_image_layout(
                  context, vk_image.first, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                vk::command_buffer command_buffer =
                  operation::create_command_buffer(context, context.transfer_command_pool, true);

                VkBufferImageCopy region = {
                  0,
                  0,
                  0,
                  {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                  {0, 0, 0},
                  {static_cast<uint32_t>(dimensions[0]), static_cast<uint32_t>(dimensions[1]), 1}};

                vkCmdCopyBufferToImage(command_buffer,
                                       vk_image_buffer.first,
                                       vk_image.first,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       1,
                                       &region);

                submit_command_buffer(context.transfer_queue, command_buffer, {}, {});
                vkQueueWaitIdle(context.transfer_queue);

                // Transistion the image layout from transfer destination to read only
                transition_image_layout(context,
                                        vk_image.first,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }

        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_IMAGE_HPP
