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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_DESCRIPTOR_SET_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_DESCRIPTOR_SET_HPP

extern "C" {
#include <vulkan/vulkan.h>
}

#include <vector>

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            inline vk::descriptor_pool create_descriptor_pool(const VulkanContext& context,
                                                              const std::vector<VkDescriptorPoolSize>& pool_sizes,
                                                              const uint32_t& max_allocations = 1) {

                std::vector<VkDescriptorPoolSize> pools;
                for (uint32_t i = 0; i < max_allocations; ++i) {
                    for (const auto& pool_size : pool_sizes) {
                        pools.push_back(pool_size);
                    }
                }
                VkDescriptorPoolCreateInfo info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   max_allocations,
                                                   static_cast<uint32_t>(pools.size()),
                                                   pools.data()};

                VkDescriptorPool pool;
                throw_vk_error(vkCreateDescriptorPool(context.device, &info, 0, &pool),
                               "Failed to create descriptor pool");

                return vk::descriptor_pool(pool,
                                           [&context](auto p) { vkDestroyDescriptorPool(context.device, p, nullptr); });
            }

            inline std::vector<VkDescriptorSet> create_descriptor_set(
              const VulkanContext& context,
              const vk::descriptor_pool& pool,
              const std::vector<VkDescriptorSetLayout>& layouts) {

                VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                          nullptr,
                                                          pool,
                                                          static_cast<uint32_t>(layouts.size()),
                                                          layouts.data()};

                std::vector<VkDescriptorSet> descriptor_sets(layouts.size());
                throw_vk_error(vkAllocateDescriptorSets(context.device, &alloc_info, descriptor_sets.data()),
                               "Failed to allocate descriptor set");

                return descriptor_sets;
            }

        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_CREATE_DESCRIPTOR_SET_HPP
