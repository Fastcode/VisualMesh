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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_FIND_DEVICE_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_FIND_DEVICE_HPP

extern "C" {
#include <vulkan/vulkan.h>
}

#include <iomanip>
#include <iostream>
#include <vector>

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {
            VkResult get_best_transfer_queue(const VkPhysicalDevice& physical_device, uint32_t& queueFamilyIndex) {
                uint32_t queueFamilyPropertiesCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyPropertiesCount, 0);

                std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);

                vkGetPhysicalDeviceQueueFamilyProperties(
                  physical_device, &queueFamilyPropertiesCount, queueFamilyProperties.data());

                // first try and find a queue that has just the transfer bit set
                for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!)
                    const VkQueueFlags maskedFlags =
                      (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

                    if (!((VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) & maskedFlags)
                        && (VK_QUEUE_TRANSFER_BIT & maskedFlags)) {
                        queueFamilyIndex = i;
                        return VK_SUCCESS;
                    }
                }

                // otherwise we'll prefer using a compute-only queue,
                // remember that having compute on the queue implicitly enables transfer!
                for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!)
                    const VkQueueFlags maskedFlags =
                      (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

                    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
                        queueFamilyIndex = i;
                        return VK_SUCCESS;
                    }
                }

                // lastly get any queue that'll work for us (graphics, compute or transfer bit set)
                for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!)
                    const VkQueueFlags maskedFlags =
                      (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

                    if ((VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT) & maskedFlags) {
                        queueFamilyIndex = i;
                        return VK_SUCCESS;
                    }
                }

                return VK_ERROR_INITIALIZATION_FAILED;
            }

            VkResult get_best_compute_queue(const VkPhysicalDevice& physical_device, uint32_t& queueFamilyIndex) {
                uint32_t queueFamilyPropertiesCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyPropertiesCount, 0);

                std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);

                vkGetPhysicalDeviceQueueFamilyProperties(
                  physical_device, &queueFamilyPropertiesCount, queueFamilyProperties.data());

                // first try and find a queue that has just the compute bit set
                for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
                    const VkQueueFlags maskedFlags =
                      (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) & queueFamilyProperties[i].queueFlags);

                    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
                        queueFamilyIndex = i;
                        return VK_SUCCESS;
                    }
                }

                // lastly get any queue that'll work for us
                for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
                    const VkQueueFlags maskedFlags =
                      (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) & queueFamilyProperties[i].queueFlags);

                    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
                        queueFamilyIndex = i;
                        return VK_SUCCESS;
                    }
                }

                return VK_ERROR_INITIALIZATION_FAILED;
            }

            void create_device(const DeviceType& wanted_device_type, VulkanContext& context, bool verbose = false) {
                auto compare_device_type = [wanted_device_type](const VkPhysicalDeviceType& device_type) {
                    switch (wanted_device_type) {
                        case DeviceType::CPU: return device_type == VK_PHYSICAL_DEVICE_TYPE_CPU;
                        case DeviceType::GPU:
                            return (device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                                    || device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                                    || device_type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU);
                        case DeviceType::INTEGRATED_GPU: return device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                        case DeviceType::DISCRETE_GPU: return device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                        case DeviceType::VIRTUAL_GPU: return device_type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
                        case DeviceType::ANY: return true;
                        default: return false;
                    }
                };

                // Find a suitable device
                uint32_t physical_device_count = 0;
                throw_vk_error(vkEnumeratePhysicalDevices(context.instance, &physical_device_count, 0),
                               "Failed to get physical device count");
                std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
                throw_vk_error(
                  vkEnumeratePhysicalDevices(context.instance, &physical_device_count, physical_devices.data()),
                  "Failed to enumerate physical devices");

                uint32_t best_compute_queue_family    = std::numeric_limits<uint32_t>::max();
                uint32_t best_transfer_queue_family   = std::numeric_limits<uint32_t>::max();
                VkDeviceSize max_heap_size            = 0;
                VkPhysicalDevice best_physical_device = VK_NULL_HANDLE;

                for (const auto& physical_device : physical_devices) {
                    // Only consider GPU devices
                    VkPhysicalDeviceProperties props;
                    vkGetPhysicalDeviceProperties(physical_device, &props);
                    if (compare_device_type(props.deviceType)) {
                        uint32_t compute_queue_family;
                        uint32_t transfer_queue_family;
                        VkResult compute_result  = get_best_compute_queue(physical_device, compute_queue_family);
                        VkResult transfer_result = get_best_transfer_queue(physical_device, transfer_queue_family);

                        if ((compute_result == VK_SUCCESS) && (transfer_result == VK_SUCCESS)) {
                            // We have a device with the right capabilites, now make sure we have the one with the most
                            // local memory Doesn't seem to be possible to find one with the most number of parallel
                            // compute units? Therefore, more memory == better device?
                            VkPhysicalDeviceMemoryProperties mem_props;
                            vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
                            std::vector<VkMemoryHeap> heaps(mem_props.memoryHeaps,
                                                            mem_props.memoryHeaps + mem_props.memoryHeapCount);

                            VkDeviceSize heap_size = 0;
                            for (const auto& heap : heaps) {
                                if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                                    // Device local heap, should be size of total GPU VRAM.
                                    heap_size = std::max(heap.size, heap_size);
                                }
                            }

                            if (heap_size > max_heap_size) {
                                max_heap_size              = heap_size;
                                best_physical_device       = physical_device;
                                best_compute_queue_family  = compute_queue_family;
                                best_transfer_queue_family = transfer_queue_family;
                            }
                        }
                    }
                }

                // Make sure we found a device
                if ((best_compute_queue_family == std::numeric_limits<uint32_t>::max())
                    || (best_transfer_queue_family == std::numeric_limits<uint32_t>::max())) {
                    throw_vk_error(VK_ERROR_INITIALIZATION_FAILED, "Failed to find a suitable physical device");
                }

                // Display device information
                if (verbose) {
                    VkPhysicalDeviceProperties props;
                    vkGetPhysicalDeviceProperties(best_physical_device, &props);
                    std::cout << "Device Properties:" << std::endl;
                    std::cout << "------------------" << std::endl;
                    std::cout << "    Device Name......................: " << props.deviceName << std::endl;
                    std::cout << "    API Version......................: " << VK_VERSION_MAJOR(props.apiVersion) << "."
                              << VK_VERSION_MINOR(props.apiVersion) << "." << VK_VERSION_PATCH(props.apiVersion)
                              << std::endl;
                    std::cout << "    Driver Version...................: " << VK_VERSION_MAJOR(props.driverVersion)
                              << "." << VK_VERSION_MINOR(props.driverVersion) << "."
                              << VK_VERSION_PATCH(props.driverVersion) << std::endl;
                    std::cout << "    Vendor ID........................: 0x" << std::hex << props.vendorID << std::dec
                              << std::endl;
                    std::cout << "    Device ID........................: 0x" << std::hex << props.deviceID << std::dec
                              << std::endl;
                    std::cout << "    Device Type......................: ";
                    switch (props.deviceType) {
                        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: std::cout << "Integrated GPU" << std::endl; break;
                        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: std::cout << "Discrete GPU" << std::endl; break;
                        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: std::cout << "Virtual GPU" << std::endl; break;
                        case VK_PHYSICAL_DEVICE_TYPE_CPU: std::cout << "CPU" << std::endl; break;
                        case VK_PHYSICAL_DEVICE_TYPE_OTHER: std::cout << "Other device" << std::endl; break;
                        default: std::cout << "Unknown device" << std::endl; break;
                    }
                    std::cout << "    Max Heap Size....................: ";
                    // TB
                    if (max_heap_size > VkDeviceSize(1e12)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e12)) << " TB"
                                  << std::endl;
                    }
                    // GB
                    else if (max_heap_size > VkDeviceSize(1e9)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e9)) << " GB"
                                  << std::endl;
                    }
                    // MB
                    else if (max_heap_size > VkDeviceSize(1e6)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e6)) << " MB"
                                  << std::endl;
                    }
                    // KB
                    else if (max_heap_size > VkDeviceSize(1e3)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e3)) << " KB"
                                  << std::endl;
                    }
                    else {
                        std::cout << std::setprecision(2) << max_heap_size << " B" << std::endl;
                    }
                    std::cout << "    Max Compute Workgroup Count......: " << std::endl;
                    std::cout << "        X: " << props.limits.maxComputeWorkGroupCount[0] << std::endl;
                    std::cout << "        Y: " << props.limits.maxComputeWorkGroupCount[1] << std::endl;
                    std::cout << "        Z: " << props.limits.maxComputeWorkGroupCount[2] << std::endl;
                    std::cout << "    Max Compute Workgroup Size.......: " << std::endl;
                    std::cout << "        X: " << props.limits.maxComputeWorkGroupSize[0] << std::endl;
                    std::cout << "        Y: " << props.limits.maxComputeWorkGroupSize[1] << std::endl;
                    std::cout << "        Z: " << props.limits.maxComputeWorkGroupSize[2] << std::endl;
                    std::cout << "    Max Compute Workgroup Invocations: "
                              << props.limits.maxComputeWorkGroupInvocations << std::endl;
                    std::cout << "    Max Compute Shared Memory........: ";
                    // TB
                    VkDeviceSize shared_mem_size = props.limits.maxComputeSharedMemorySize;
                    if (shared_mem_size > VkDeviceSize(1e12)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e12)) << " TB"
                                  << std::endl;
                    }
                    // GB
                    else if (shared_mem_size > VkDeviceSize(1e9)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e9)) << " GB"
                                  << std::endl;
                    }
                    // MB
                    else if (shared_mem_size > VkDeviceSize(1e6)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e6)) << " MB"
                                  << std::endl;
                    }
                    // KB
                    else if (shared_mem_size > VkDeviceSize(1e3)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e3)) << " KB"
                                  << std::endl;
                    }
                    else {
                        std::cout << std::setprecision(2) << shared_mem_size << " B" << std::endl;
                    }
                }

                context.phys_device           = best_physical_device;
                context.compute_queue_family  = best_compute_queue_family;
                context.transfer_queue_family = best_transfer_queue_family;

                // Create device and queues
                float queue_priority = 1.0f;
                std::array<VkDeviceQueueCreateInfo, 2> queue_create_info;
                queue_create_info[0] = {
                  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, context.compute_queue_family, 1, &queue_priority};
                queue_create_info[1] = {
                  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, context.transfer_queue_family, 1, &queue_priority};
                VkDeviceCreateInfo device_create_info;

                if (context.compute_queue_family == context.transfer_queue_family) {
                    device_create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                          0,
                                          0,
                                          1,
                                          queue_create_info.data(),
                                          0,
                                          nullptr,
                                          0,
                                          nullptr,
                                          nullptr};
                }
                else {
                    device_create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                          0,
                                          0,
                                          2,
                                          queue_create_info.data(),
                                          0,
                                          nullptr,
                                          0,
                                          nullptr,
                                          nullptr};
                }

                VkDevice device;
                throw_vk_error(vkCreateDevice(context.phys_device, &device_create_info, nullptr, &device),
                               "Failed to create device");
                context.device = vk::device(device, [](auto p) { vkDestroyDevice(p, nullptr); });
            }
        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_FIND_DEVICE_HPP
