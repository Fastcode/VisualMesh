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

#ifndef VISUALMESH_ENGINE_VULKAN_OPERATION_FIND_DEVICE_HPP
#define VISUALMESH_ENGINE_VULKAN_OPERATION_FIND_DEVICE_HPP

#include <iomanip>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {
            vk::Result get_best_transfer_queue(const vk::PhysicalDevice& device, uint32_t& queue_family) {
                std::vector<vk::QueueFamilyProperties> queue_props = device.getQueueFamilyProperties();

                // first try and find a queue that has just the transfer bit set
                for (queue_family = 0; queue_family < queue_props.size(); queue_family++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!)
                    const vk::QueueFlags masked_flags =
                      ~vk::QueueFlagBits::eSparseBinding & queue_props[queue_family].queueFlags;

                    if (!((vk::QueueFlags(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute) & masked_flags))
                        && (vk::QueueFlagBits::eTransfer & masked_flags)) {
                        return vk::Result::eSuccess;
                    }
                }

                // otherwise we'll prefer using a compute-only queue,
                // remember that having compute on the queue implicitly enables transfer!
                for (queue_family = 0; queue_family < queue_props.size(); queue_family++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!)
                    const vk::QueueFlags masked_flags =
                      ~vk::QueueFlagBits::eSparseBinding & queue_props[queue_family].queueFlags;

                    if (!(vk::QueueFlagBits::eGraphics & masked_flags)
                        && (vk::QueueFlagBits::eCompute & masked_flags)) {
                        return vk::Result::eSuccess;
                    }
                }

                // lastly get any queue that'll work for us (graphics, compute or transfer bit set)
                for (queue_family = 0; queue_family < queue_props.size(); queue_family++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!)
                    const vk::QueueFlags masked_flags =
                      ~vk::QueueFlagBits::eSparseBinding & queue_props[queue_family].queueFlags;

                    if (vk::QueueFlags(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute
                                       | vk::QueueFlagBits::eTransfer)
                        & masked_flags) {
                        return vk::Result::eSuccess;
                    }
                }

                queue_family = 0;
                return vk::Result::eErrorInitializationFailed;
            }

            vk::Result get_best_compute_queue(const vk::PhysicalDevice& device, uint32_t& queue_family) {
                std::vector<vk::QueueFamilyProperties> queue_props = device.getQueueFamilyProperties();

                // first try and find a queue that has just the compute bit set
                for (queue_family = 0; queue_family < queue_props.size(); queue_family++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
                    const vk::QueueFlags masked_flags =
                      vk::QueueFlags(~vk::QueueFlagBits::eSparseBinding | ~vk::QueueFlagBits::eTransfer)
                      & queue_props[queue_family].queueFlags;

                    if (!(vk::QueueFlagBits::eGraphics & masked_flags)) { return vk::Result::eSuccess; }
                }

                // lastly get any queue that'll work for us
                for (queue_family = 0; queue_family < queue_props.size(); queue_family++) {
                    // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
                    const vk::QueueFlags masked_flags =
                      vk::QueueFlags(~vk::QueueFlagBits::eSparseBinding | ~vk::QueueFlagBits::eTransfer)
                      & queue_props[queue_family].queueFlags;

                    if (vk::QueueFlagBits::eCompute & masked_flags) { return vk::Result::eSuccess; }
                }

                queue_family = 0;
                return vk::Result::eErrorInitializationFailed;
            }

            enum class DeviceType { CPU, GPU, INTEGRATED_GPU, DISCRETE_GPU, VIRTUAL_GPU, ANY };

            struct VulkanInstance {
                vk::Instance instance;
                vk::PhysicalDevice phys_device;
                vk::Device device;
                uint32_t compute_queue_family;
                uint32_t transfer_queue_family;
                vk::Queue compute_queue;
                vk::Queue transfer_queue;
            };

            void create_device(const DeviceType& wanted_device_type, VulkanInstance& instance, bool verbose = false) {
                auto compare_device_type = [wanted_device_type](const vk::PhysicalDeviceType& device_type) {
                    switch (wanted_device_type) {
                        case DeviceType::CPU: return device_type == vk::PhysicalDeviceType::eCpu;
                        case DeviceType::GPU:
                            return (device_type == vk::PhysicalDeviceType::eIntegratedGpu
                                    || device_type == vk::PhysicalDeviceType::eDiscreteGpu
                                    || device_type == vk::PhysicalDeviceType::eVirtualGpu);
                        case DeviceType::INTEGRATED_GPU: return device_type == vk::PhysicalDeviceType::eIntegratedGpu;
                        case DeviceType::DISCRETE_GPU: return device_type == vk::PhysicalDeviceType::eDiscreteGpu;
                        case DeviceType::VIRTUAL_GPU: return device_type == vk::PhysicalDeviceType::eVirtualGpu;
                        case DeviceType::ANY: return true;
                        default: return false;
                    }
                };

                // Find a suitable device
                std::vector<vk::PhysicalDevice> physical_devices = instance.instance.enumeratePhysicalDevices();
                uint32_t best_compute_queue_family               = std::numeric_limits<uint32_t>::max();
                uint32_t best_transfer_queue_family              = std::numeric_limits<uint32_t>::max();
                uint32_t max_heap_size                           = 0;
                vk::PhysicalDevice best_physical_device;

                for (const auto& physical_device : physical_devices) {
                    // Only consider GPU devices
                    vk::PhysicalDeviceProperties props = physical_device.getProperties();
                    if (compare_device_type(props.deviceType)) {
                        uint32_t compute_queue_family;
                        uint32_t transfer_queue_family;
                        vk::Result compute_result  = get_best_compute_queue(physical_device, compute_queue_family);
                        vk::Result transfer_result = get_best_transfer_queue(physical_device, transfer_queue_family);

                        if ((compute_result == vk::Result::eSuccess) && (transfer_result == vk::Result::eSuccess)) {
                            // We have a device with the right capabilites, now make sure we have the one with the most
                            // local memory Doesn't seem to be possible to find one with the most number of parallel
                            // compute units? Therefore, more memory == better device?
                            vk::PhysicalDeviceMemoryProperties mem_props = physical_device.getMemoryProperties();
                            std::vector<vk::MemoryHeap> heaps(mem_props.memoryHeaps,
                                                              mem_props.memoryHeaps + mem_props.memoryHeapCount);

                            uint32_t heap_size = 0;
                            for (const auto& heap : heaps) {
                                if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                                    // Device local heap, should be size of total GPU VRAM.
                                    if (heap.size > heap_size) { heap_size = heap.size; }
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
                    throw_vk_error(vk::Result::eErrorInitializationFailed, "Failed to find a suitable Vulkan device");
                }

                // Display device information
                if (verbose) {
                    vk::PhysicalDeviceProperties props = best_physical_device.getProperties();
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
                        case vk::PhysicalDeviceType::eIntegratedGpu: std::cout << "Integrated GPU" << std::endl; break;
                        case vk::PhysicalDeviceType::eDiscreteGpu: std::cout << "Discrete GPU" << std::endl; break;
                        case vk::PhysicalDeviceType::eVirtualGpu: std::cout << "Virtual GPU" << std::endl; break;
                        case vk::PhysicalDeviceType::eCpu: std::cout << "CPU" << std::endl; break;
                        case vk::PhysicalDeviceType::eOther: std::cout << "Other device" << std::endl; break;
                        default: std::cout << "Unknown device" << std::endl; break;
                    }
                    std::cout << "    Max Heap Size....................: ";
                    // TB
                    if (max_heap_size > uint32_t(1e12)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e12)) << " TB"
                                  << std::endl;
                    }
                    // GB
                    else if (max_heap_size > uint32_t(1e9)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e9)) << " GB"
                                  << std::endl;
                    }
                    // MB
                    else if (max_heap_size > uint32_t(1e6)) {
                        std::cout << std::setprecision(2) << (double(max_heap_size) / double(1e6)) << " MB"
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
                    uint32_t shared_mem_size = props.limits.maxComputeSharedMemorySize;
                    if (shared_mem_size > uint32_t(1e12)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e12)) << " TB"
                                  << std::endl;
                    }
                    // GB
                    else if (shared_mem_size > uint32_t(1e9)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e9)) << " GB"
                                  << std::endl;
                    }
                    // MB
                    else if (shared_mem_size > uint32_t(1e6)) {
                        std::cout << std::setprecision(2) << (double(shared_mem_size) / double(1e6)) << " MB"
                                  << std::endl;
                    }
                    else {
                        std::cout << std::setprecision(2) << max_heap_size << " B" << std::endl;
                    }
                }

                instance.phys_device           = best_physical_device;
                instance.compute_queue_family  = best_compute_queue_family;
                instance.transfer_queue_family = best_transfer_queue_family;

                // Create device and queues
                float queue_priority = 1.0f;
                if (instance.compute_queue_family == instance.transfer_queue_family) {
                    vk::DeviceQueueCreateInfo queue_create_infos(
                      vk::DeviceQueueCreateFlags(), instance.compute_queue_family, 1, &queue_priority);
                    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(), 1, &queue_create_infos);
                    instance.device = instance.phys_device.createDevice(device_create_info);
                }
                else {
                    std::array<vk::DeviceQueueCreateInfo, 2> queue_create_infos = {
                      vk::DeviceQueueCreateInfo(
                        vk::DeviceQueueCreateFlags(), instance.compute_queue_family, 1, &queue_priority),
                      vk::DeviceQueueCreateInfo(
                        vk::DeviceQueueCreateFlags(), instance.transfer_queue_family, 1, &queue_priority)};
                    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(),
                                                            queue_create_infos.size(),
                                                            queue_create_infos.data(),
                                                            0,
                                                            nullptr,
                                                            0,
                                                            nullptr,
                                                            nullptr);
                    instance.device = instance.phys_device.createDevice(device_create_info);
                }
            }
        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh


#endif  // VISUALMESH_ENGINE_VULKAN_OPERATION_FIND_DEVICE_HPP
