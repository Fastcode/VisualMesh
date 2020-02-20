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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_BUFFER_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_BUFFER_HPP

#include <vulkan/vulkan.hpp>

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {
            inline std::pair<vk::Buffer, vk::DeviceMemory> create_buffer(const vk::PhysicalDevice& phyiscal_device,
                                                                         const vk::Device& device,
                                                                         const vk::DeviceSize& size,
                                                                         const vk::BufferUsageFlags& usage,
                                                                         const vk::SharingMode& sharing,
                                                                         const std::vector<uint32_t>& queues,
                                                                         const vk::MemoryPropertyFlags& properties) {

                // Create the buffer
                vk::Buffer buffer = device.createBuffer(
                  vk::BufferCreateInfo(vk::BufferCreateFlags(), size, usage, sharing, queues.size(), queues.data()));

                // Get properties of the physical device memory
                vk::PhysicalDeviceMemoryProperties memory_properties = phyiscal_device.getMemoryProperties();
                vk::MemoryRequirements memory_requirements           = device.getBufferMemoryRequirements(buffer);
                vk::MemoryAllocateInfo allocInfo(memory_requirements.size);

                // Vulkan devices can have multiple different types of memory
                // Each memory type can have different properties and sizes, so we need to find one that suits our
                // purposes
                for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
                    if ((memory_requirements.memoryTypeBits & (1 << i))
                        && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                        allocInfo.memoryTypeIndex = i;
                        break;
                    }
                }

                // Allocate memory for the buffer on the device
                vk::DeviceMemory memory = device.allocateMemory(allocInfo);

                // Attach the allocated device memory to it
                device.bindBufferMemory(buffer, memory, 0);

                return std::make_pair(buffer, memory);
            }
        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_BUFFER_HPP
