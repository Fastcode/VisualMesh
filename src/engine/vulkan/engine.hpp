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

#ifndef VISUALMESH_ENGINE_VULKAN_ENGINE_HPP
#define VISUALMESH_ENGINE_VULKAN_ENGINE_HPP

#include <iomanip>
#include <numeric>
#include <sstream>
#include <tuple>

#include "engine/opencl/operation/make_network.hpp"
#include "engine/opencl/operation/wrapper.hpp"
#include "engine/vulkan/kernels/load_image.hpp"
#include "engine/vulkan/kernels/reprojection.hpp"
#include "engine/vulkan/operation/create_buffer.hpp"
#include "engine/vulkan/operation/create_device.hpp"
#include "engine/vulkan/operation/vulkan_error_category.hpp"

// #include "engine/opencl/operation/make_context.hpp"
// #include "engine/opencl/operation/make_queue.hpp"
#include "mesh/mesh.hpp"
#include "mesh/network_structure.hpp"
#include "mesh/projected_mesh.hpp"
#include "utility/math.hpp"
#include "utility/projection.hpp"

namespace visualmesh {
namespace engine {
  namespace vulkan {

    /**
     * @brief An Vulkan implementation of the visual mesh inference engine
     *
     * @details
     *  The Vulkan implementation is designed to be used for high performance inference. It is able to take advantage of
     *  either GPUs from Intel, AMD, ARM, NVIDIA etc as well as multithreaded CPU implementations. This allows it to be
     *  very flexible with its deployment on devices.
     *
     * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
     */
    template <typename Scalar>
    class Engine {
    public:
      /**
       * @brief Construct a new Vulkan Engine object
       *
       * @param structure the network structure to use classification
       */
      Engine(const network_structure_t<Scalar>& structure = {}) : max_width(4) {

        // Create the dynamic loader
        VULKAN_HPP_DEFAULT_DISPATCHER.init(dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

        // Get a Vulkan instance
        const vk::ApplicationInfo app_info("VisualMesh", 0, "", 0, VK_MAKE_VERSION(1, 0, 0));
        const vk::InstanceCreateInfo instance_info(vk::InstanceCreateFlags(), &app_info);

        throw_vk_error(vk::createInstance(&instance_info, nullptr, &instance.instance),
                       "Error creating the Vulkan instance");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.instance);

        // Create the Vulkan instance and find the best devices and queues
        operation::create_device(operation::DeviceType::GPU, instance, true);

        // Create device and queues
        float queue_priority                                        = 1.0f;
        std::array<vk::DeviceQueueCreateInfo, 2> queue_create_infos = {
          vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), instance.compute_queue_family, 1, &queue_priority),
          vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), instance.transfer_queue_family, 1, &queue_priority)};
        vk::DeviceCreateInfo device_create_info(
          vk::DeviceCreateFlags(), queue_create_infos.size(), queue_create_infos.data());
        instance.device = instance.phys_device.createDevice(device_create_info);

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.device);

        instance.compute_queue  = instance.device.getQueue(instance.compute_queue_family, 0);
        instance.transfer_queue = instance.device.getQueue(instance.transfer_queue_family, 0);

        // Created the projection kernel sources and programs
        std::vector<uint32_t> equidistant_reprojection_source =
          kernels::build_reprojection<Scalar>("project_equidistant", kernels::equidistant_reprojection<Scalar>);
        std::vector<uint32_t> equisolid_reprojection_source =
          kernels::build_reprojection<Scalar>("project_equisolid", kernels::equisolid_reprojection<Scalar>);
        std::vector<uint32_t> rectilinear_reprojection_source =
          kernels::build_reprojection<Scalar>("project_rectilinear", kernels::rectilinear_reprojection<Scalar>);

        vk::ShaderModule equidistant_reprojection_program = instance.device.createShaderModule(
          vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
                                     equidistant_reprojection_source.size() * sizeof(uint32_t),
                                     equidistant_reprojection_source.data()));
        vk::ShaderModule equisolid_reprojection_program = instance.device.createShaderModule(
          vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
                                     equisolid_reprojection_source.size() * sizeof(uint32_t),
                                     equisolid_reprojection_source.data()));
        vk::ShaderModule rectilinear_reprojection_program = instance.device.createShaderModule(
          vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
                                     rectilinear_reprojection_source.size() * sizeof(uint32_t),
                                     rectilinear_reprojection_source.data()));

        // Create the descriptor set for the reprojection programs
        // All reprojection programs have the same descriptor set
        // Descriptor Set 0: {points_ptr, indices_ptr, Rco_ptr, f_ptr, centre_ptr, k_ptr, dimensions_ptr, out_ptr}
        std::vector<vk::DescriptorSetLayoutBinding> reprojection_bindings;
        reprojection_bindings.emplace_back(
          0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        reprojection_bindings.emplace_back(
          7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);

        reprojection_descriptor_layout = instance.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(
          vk::DescriptorSetLayoutCreateFlags(), reprojection_bindings.size(), reprojection_bindings.data()));

        reprojection_pipeline_layout = instance.device.createPipelineLayout(vk::PipelineLayoutCreateInfo(
          vk::PipelineLayoutCreateFlags(), 1, &reprojection_descriptor_layout, 0, nullptr));

        project_equidistant = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          equidistant_reprojection_program,
                                                                          "project_equidistant"),
                                        reprojection_pipeline_layout));

        project_equisolid = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          equisolid_reprojection_program,
                                                                          "project_equisolid"),
                                        reprojection_pipeline_layout));

        project_rectilinear = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          rectilinear_reprojection_program,
                                                                          "project_rectilinear"),
                                        reprojection_pipeline_layout));

        // Created the load_image kernel source and program
        std::vector<uint32_t> load_image_source = kernels::load_image<Scalar>();

        vk::ShaderModule load_image_program = instance.device.createShaderModule(vk::ShaderModuleCreateInfo(
          vk::ShaderModuleCreateFlags(), load_image_source.size() * sizeof(uint32_t), load_image_source.data()));

        // Create the descriptor sets for the load_image program
        // Descriptor Set 0: {bayer_sampler, interp_sampler}
        std::vector<std::vector<vk::DescriptorSetLayoutBinding>> load_image_bindings;
        vk::Sampler bayer_sampler =
          instance.device.createSampler(vk::SamplerCreateInfo(vk::SamplerCreateFlags(),
                                                              vk::Filter::eNearest,
                                                              vk::Filter::eNearest,
                                                              vk::SamplerMipmapMode::eNearest,
                                                              vk::SamplerAddressMode::eClampToBorder,
                                                              vk::SamplerAddressMode::eClampToBorder,
                                                              vk::SamplerAddressMode::eClampToBorder,
                                                              0.0f,
                                                              false,
                                                              0.0f,
                                                              false,
                                                              vk::CompareOp::eNever,
                                                              0.0f,
                                                              0.0f,
                                                              vk::BorderColor::eFloatTransparentBlack,
                                                              false));
        vk::Sampler interp_sampler =
          instance.device.createSampler(vk::SamplerCreateInfo(vk::SamplerCreateFlags(),
                                                              vk::Filter::eLinear,
                                                              vk::Filter::eLinear,
                                                              vk::SamplerMipmapMode::eLinear,
                                                              vk::SamplerAddressMode::eClampToBorder,
                                                              vk::SamplerAddressMode::eClampToBorder,
                                                              vk::SamplerAddressMode::eClampToBorder,
                                                              0.0f,
                                                              false,
                                                              0.0f,
                                                              false,
                                                              vk::CompareOp::eNever,
                                                              0.0f,
                                                              0.0f,
                                                              vk::BorderColor::eFloatTransparentBlack,
                                                              false));
        load_image_bindings.emplace_back();
        load_image_bindings.back().emplace_back(
          0, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eAll, &bayer_sampler);
        load_image_bindings.back().emplace_back(
          1, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eAll, &interp_sampler);

        // Descriptor Set 1: {image, coordinates, network}
        load_image_bindings.emplace_back();
        load_image_bindings.back().emplace_back(
          0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        load_image_bindings.back().emplace_back(
          1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);
        load_image_bindings.back().emplace_back(
          2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr);

        load_image_descriptor_layout = {
          instance.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(), load_image_bindings[0].size(), load_image_bindings[0].data())),
          instance.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(), load_image_bindings[1].size(), load_image_bindings[1].data()))};

        load_image_pipeline_layout = instance.device.createPipelineLayout(vk::PipelineLayoutCreateInfo(
          vk::PipelineLayoutCreateFlags(), 2, load_image_descriptor_layout.data(), 0, nullptr));

        load_GRBG_image = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          load_image_program,
                                                                          "load_GRBG_image"),
                                        load_image_pipeline_layout));
        load_RGGB_image = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          load_image_program,
                                                                          "load_RGGB_image"),
                                        load_image_pipeline_layout));
        load_GBRG_image = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          load_image_program,
                                                                          "load_GBRG_image"),
                                        load_image_pipeline_layout));
        load_BGGR_image = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          load_image_program,
                                                                          "load_BGGR_image"),
                                        load_image_pipeline_layout));
        load_RGBA_image = instance.device.createComputePipeline(
          vk::PipelineCache(),
          vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                          vk::ShaderStageFlagBits::eCompute,
                                                                          load_image_program,
                                                                          "load_RGBA_image"),
                                        load_image_pipeline_layout));

        // Work out what the widest network layer is
        max_width = 4;
        for (const auto& k : conv_layers) {
          max_width = std::max(max_width, k.second);
        }
      }

      /**
       * @brief Projects a provided mesh to pixel coordinates
       *
       * @tparam Model the mesh model that we are projecting
       *
       * @param mesh the mesh table that we are projecting to pixel coordinates
       * @param Hoc  the homogenous transformation matrix from the camera to the observation plane
       * @param lens the lens parameters that describe the optics of the camera
       *
       * @return a projected mesh for the provided arguments
       */
      template <template <typename> class Model>
      inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> project(const Mesh<Scalar, Model>& mesh,
                                                                        const mat4<Scalar>& Hoc,
                                                                        const Lens<Scalar>& lens) const {
        static constexpr size_t N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

        std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
        std::vector<int> indices;
        std::pair<vk::Buffer, vk::DeviceMemory> vk_pixels;
        vk::Fence projected;

        std::tie(neighbourhood, indices, vk_pixels, projected) = do_project(mesh, Hoc, lens);

        // Wait for computation to finish
        // waitForFences can block for up to some number of nanoseconds (we have chosen 1e9 === 1s)
        vk::Result res = instance.device.waitForFences(1, &projected, true, static_cast<uint64_t>(1e9));
        while (res != vk::Result::eSuccess) {
          if (res == vk::Result::eErrorDeviceLost) {
            throw_vk_error(res, "Lost device while waiting for reprojection to finish");
          }
          res = instance.device.waitForFences(1, &projected, true, static_cast<uint64_t>(1e9));
        }

        // Read the pixels off the buffer
        std::vector<vec2<Scalar>> pixels(indices.size());
        Scalar* pixels_payload = reinterpret_cast<Scalar*>(
          instance.device.mapMemory(vk_pixels.second, 0, sizeof(vec2<Scalar>) * indices.size()));
        size_t index = 0;
        for (auto& pixel : pixels) {
          pixel = {pixels_payload[index + 0], pixels_payload[index + 1]};
          index += 2;
        }
        instance.device.unmapMemory(vk_pixels.second);

        return ProjectedMesh<Scalar, N_NEIGHBOURS>{std::move(pixels), std::move(neighbourhood), std::move(indices)};
      }

      /**
       * @brief Projects a provided mesh to pixel coordinates from an aggregate VisualMesh object
       *
       * @tparam Model the mesh model that we are projecting
       *
       * @param mesh the mesh table that we are projecting to pixel coordinates
       * @param Hoc  the homogenous transformation matrix from the camera to the observation plane
       * @param lens the lens parameters that describe the optics of the camera
       *
       * @return a projected mesh for the provided arguments
       */
      template <template <typename> class Model>
      inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> project(const VisualMesh<Scalar, Model>& mesh,
                                                                        const mat4<Scalar>& Hoc,
                                                                        const Lens<Scalar>& lens) const {
        return project(mesh.height(Hoc[2][3]), Hoc, lens);
      }

      /**
       * @brief Project and classify a mesh using the neural network that is loaded into this engine
       *
       * @tparam Model the mesh model that we are projecting
       *
       * @param mesh    the mesh table that we are projecting to pixel coordinates
       * @param Hoc     the homogenous transformation matrix from the camera to the observation plane
       * @param lens    the lens parameters that describe the optics of the camera
       * @param image   the data that represents the image the network will run from
       * @param format  the pixel format of this image as a fourcc code
       *
       * @return a classified mesh for the provided arguments
       */
      template <template <typename> class Model>
      ClassifiedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Model>& mesh,
                                                                     const mat4<Scalar>& Hoc,
                                                                     const Lens<Scalar>& lens,
                                                                     const void* image,
                                                                     const uint32_t& format) const {
        static constexpr size_t N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;
        return ClassifiedMesh<Scalar, N_NEIGHBOURS>();

        // cl_int error                         = CL_SUCCESS;

        //// Grab the image memory from the cache
        // cl::mem cl_image = get_image_memory(lens.dimensions, format);

        //// Map our image into device memory
        // std::array<size_t, 3> origin = {{0, 0, 0}};
        // std::array<size_t, 3> region = {{size_t(lens.dimensions[0]), size_t(lens.dimensions[1]), 1}};

        // cl::event cl_image_loaded;
        // cl_event ev = nullptr;
        // error = clEnqueueWriteImage(queue, cl_image, false, origin.data(), region.data(), 0, 0, image, 0, nullptr,
        // &ev); if (ev) cl_image_loaded = cl::event(ev, ::clReleaseEvent); throw_cl_error(error, "Error mapping image
        // onto device");

        //// Project our visual mesh
        // std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
        // std::vector<int> indices;
        // cl::mem cl_pixels;
        // cl::event cl_pixels_loaded;
        // std::tie(neighbourhood, indices, cl_pixels, cl_pixels_loaded) = do_project(mesh, Hoc, lens);

        //// This includes the offscreen point at the end
        // int n_points = neighbourhood.size();

        //// Get the neighbourhood memory from cache
        // cl::mem cl_neighbourhood = get_neighbourhood_memory(n_points * N_NEIGHBOURS);

        //// Upload the neighbourhood buffer
        // cl::event cl_neighbourhood_loaded;
        // ev    = nullptr;
        // error = ::clEnqueueWriteBuffer(queue,
        //                               cl_neighbourhood,
        //                               false,
        //                               0,
        //                               n_points * sizeof(std::array<int, N_NEIGHBOURS>),
        //                               neighbourhood.data(),
        //                               0,
        //                               nullptr,
        //                               &ev);
        // if (ev) cl_neighbourhood_loaded = cl::event(ev, ::clReleaseEvent);
        // throw_cl_error(error, "Error writing neighbourhood points to the device");

        //// Grab our ping pong buffers from the cache
        // auto cl_conv_buffers   = get_network_memory(max_width * n_points);
        // cl::mem cl_conv_input  = cl_conv_buffers[0];
        // cl::mem cl_conv_output = cl_conv_buffers[1];

        //// The offscreen point gets a value of -1.0 to make it easy to distinguish
        // cl::event offscreen_fill_event;
        // Scalar minus_one(-1.0);
        // ev    = nullptr;
        // error = ::clEnqueueFillBuffer(queue,
        //                              cl_conv_input,
        //                              &minus_one,
        //                              sizeof(Scalar),
        //                              (n_points - 1) * sizeof(std::array<Scalar, 4>),
        //                              sizeof(std::array<Scalar, 4>),
        //                              0,
        //                              nullptr,
        //                              &ev);
        // if (ev) offscreen_fill_event = cl::event(ev, ::clReleaseEvent);
        // throw_cl_error(error, "Error setting the offscreen pixel values");

        //// Read the pixels into the buffer
        // cl::event img_load_event;
        // cl::event network_complete;

        // cl_mem arg;
        // arg   = cl_image;
        // error = ::clSetKernelArg(load_image, 0, sizeof(arg), &arg);
        // throw_cl_error(error, "Error setting kernel argument 0 for image load kernel");
        // error = ::clSetKernelArg(load_image, 1, sizeof(format), &format);
        // throw_cl_error(error, "Error setting kernel argument 1 for image load kernel");
        // arg   = cl_pixels;
        // error = ::clSetKernelArg(load_image, 2, sizeof(arg), &arg);
        // throw_cl_error(error, "Error setting kernel argument 2 for image load kernel");
        // arg   = cl_conv_input;
        // error = ::clSetKernelArg(load_image, 3, sizeof(arg), &arg);
        // throw_cl_error(error, "Error setting kernel argument 3 for image load kernel");

        // size_t offset[1]       = {0};
        // size_t global_size[1]  = {size_t(n_points - 1)};  // -1 as we don't project the offscreen point
        // cl_event event_list[2] = {cl_pixels_loaded, cl_image_loaded};
        // ev                     = nullptr;
        // error = ::clEnqueueNDRangeKernel(queue, load_image, 1, offset, global_size, nullptr, 2, event_list, &ev);
        // if (ev) img_load_event = cl::event(ev, ::clReleaseEvent);
        // throw_cl_error(error, "Error queueing the image load kernel");

        //// These events are required for our first convolution
        // std::vector<cl::event> events({img_load_event, offscreen_fill_event, cl_neighbourhood_loaded});

        // for (auto& conv : conv_layers) {
        //  cl_mem arg;
        //  arg   = cl_neighbourhood;
        //  error = ::clSetKernelArg(conv.first, 0, sizeof(arg), &arg);
        //  throw_cl_error(error, "Error setting argument 0 for convolution kernel");
        //  arg   = cl_conv_input;
        //  error = ::clSetKernelArg(conv.first, 1, sizeof(arg), &arg);
        //  throw_cl_error(error, "Error setting argument 1 for convolution kernel");
        //  arg   = cl_conv_output;
        //  error = ::clSetKernelArg(conv.first, 2, sizeof(arg), &arg);
        //  throw_cl_error(error, "Error setting argument 2 for convolution kernel");

        //  size_t offset[1]      = {0};
        //  size_t global_size[1] = {size_t(n_points)};
        //  cl::event event;
        //  ev = nullptr;
        //  std::vector<cl_event> cl_events(events.begin(), events.end());
        //  error = ::clEnqueueNDRangeKernel(
        //    queue, conv.first, 1, offset, global_size, nullptr, cl_events.size(), cl_events.data(), &ev);
        //  if (ev) event = cl::event(ev, ::clReleaseEvent);
        //  throw_cl_error(error, "Error queueing convolution kernel");

        //  // Convert our events into a vector of events and ping pong our buffers
        //  events           = std::vector<cl::event>({event});
        //  network_complete = event;
        //  std::swap(cl_conv_input, cl_conv_output);
        //}

        //// Read the pixel coordinates off the device
        // cl::event pixels_read;
        // ev = nullptr;
        // std::vector<std::array<Scalar, 2>> pixels(neighbourhood.size() - 1);
        // cl_event iev = cl_pixels_loaded;
        // error        = ::clEnqueueReadBuffer(
        //  queue, cl_pixels, false, 0, pixels.size() * sizeof(std::array<Scalar, 2>), pixels.data(), 1, &iev, &ev);
        // if (ev) pixels_read = cl::event(ev, ::clReleaseEvent);
        // throw_cl_error(error, "Error reading projected pixels");

        //// Read the classifications off the device (they'll be in input)
        // cl::event classes_read;
        // ev  = nullptr;
        // iev = network_complete;
        // std::vector<Scalar> classifications(neighbourhood.size() * conv_layers.back().second);
        // error = ::clEnqueueReadBuffer(queue,
        //                              cl_conv_input,
        //                              false,
        //                              0,
        //                              classifications.size() * sizeof(Scalar),
        //                              classifications.data(),
        //                              1,
        //                              &iev,
        //                              &ev);
        // if (ev) classes_read = cl::event(ev, ::clReleaseEvent);
        // throw_cl_error(error, "Error reading classified values");

        //// Flush the queue to ensure all the commands have been issued
        //::clFlush(queue);

        //// Wait for the chain to finish up to where we care about it
        // cl_event end_events[2] = {pixels_read, classes_read};
        //::clWaitForEvents(2, end_events);

        // return ClassifiedMesh<Scalar, N_NEIGHBOURS>{
        //  std::move(pixels), std::move(neighbourhood), std::move(indices), std::move(classifications)};
      }

      /**
       * @brief Project and classify a mesh using the neural network that is loaded into this engine.
       * This version takes an aggregate VisualMesh object
       *
       * @tparam Model the mesh model that we are projecting
       *
       * @param mesh    the mesh table that we are projecting to pixel coordinates
       * @param Hoc     the homogenous transformation matrix from the camera to the observation plane
       * @param lens    the lens parameters that describe the optics of the camera
       * @param image   the data that represents the image the network will run from
       * @param format  the pixel format of this image as a fourcc code
       *
       * @return a classified mesh for the provided arguments
       */
      template <template <typename> class Model>
      ClassifiedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const VisualMesh<Scalar, Model>& mesh,
                                                                     const mat4<Scalar>& Hoc,
                                                                     const Lens<Scalar>& lens,
                                                                     const void* image,
                                                                     const uint32_t& format) const {
        return operator()(mesh.height(Hoc[2][3]), Hoc, lens, image, format);
      }

      void clear_cache() {
        device_points_cache.clear();
        // image_memory.memory           = nullptr;
        // image_memory.dimensions       = {0, 0};
        // image_memory.format           = 0;
        // neighbourhood_memory.memory   = nullptr;
        // neighbourhood_memory.max_size = 0;
        // network_memory.memory         = {nullptr, nullptr};
        // network_memory.max_size       = 0;
      }

    private:
      template <template <typename> class Model>
      std::tuple<std::vector<std::array<int, Model<Scalar>::N_NEIGHBOURS>>,
                 std::vector<int>,
                 std::pair<vk::Buffer, vk::DeviceMemory>,
                 vk::Fence>
        do_project(const Mesh<Scalar, Model>& mesh, const mat4<Scalar>& Hoc, const Lens<Scalar>& lens) const {
        static constexpr size_t N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

        // Lookup the on screen ranges
        auto ranges = mesh.lookup(Hoc, lens);

        // Transfer Rco to the device
        std::pair<vk::Buffer, vk::DeviceMemory> vk_rco =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(vec4<Scalar>) * 4,
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        Scalar* rco_payload =
          reinterpret_cast<Scalar*>(instance.device.mapMemory(vk_rco.second, 0, sizeof(vec4<Scalar>) * 4));
        for (size_t i = 0, index = 0; i < 3; ++i, index += 4) {
          rco_payload[index + 0] = Hoc[0][i];
          rco_payload[index + 1] = Hoc[1][i];
          rco_payload[index + 2] = Hoc[2][i];
          rco_payload[index + 3] = Scalar(0);
        }
        rco_payload[12] = rco_payload[13] = rco_payload[14] = Scalar(0);
        rco_payload[15]                                     = Scalar(1);
        instance.device.unmapMemory(vk_rco.second);

        // Transfer f to the device
        std::pair<vk::Buffer, vk::DeviceMemory> vk_f =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(Scalar),
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        Scalar* f_payload = reinterpret_cast<Scalar*>(instance.device.mapMemory(vk_f.second, 0, sizeof(Scalar)));
        f_payload[0]      = lens.focal_length;
        instance.device.unmapMemory(vk_f.second);

        // Transfer centre to the device
        std::pair<vk::Buffer, vk::DeviceMemory> vk_centre =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(vec2<Scalar>),
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        Scalar* centre_payload =
          reinterpret_cast<Scalar*>(instance.device.mapMemory(vk_centre.second, 0, sizeof(vec2<Scalar>)));
        centre_payload[0] = lens.centre[0];
        centre_payload[1] = lens.centre[1];
        instance.device.unmapMemory(vk_centre.second);

        // Calculate the coefficients for performing a distortion to give to the engine
        vec4<Scalar> ik = inverse_coefficients(lens.k);

        // Transfer k to the device
        std::pair<vk::Buffer, vk::DeviceMemory> vk_k =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(vec4<Scalar>),
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        Scalar* k_payload = reinterpret_cast<Scalar*>(instance.device.mapMemory(vk_k.second, 0, sizeof(vec4<Scalar>)));
        k_payload[0]      = ik[0];
        k_payload[1]      = ik[1];
        k_payload[2]      = ik[2];
        k_payload[3]      = ik[3];
        instance.device.unmapMemory(vk_k.second);

        // Transfer dimensions to the device
        std::pair<vk::Buffer, vk::DeviceMemory> vk_dimensions =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(vec2<int>),
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        int* dimensions_payload =
          reinterpret_cast<int*>(instance.device.mapMemory(vk_dimensions.second, 0, sizeof(vec2<int>)));
        dimensions_payload[0] = lens.dimensions[0];
        dimensions_payload[1] = lens.dimensions[1];
        instance.device.unmapMemory(vk_dimensions.second);

        // Convenience variables
        const auto& nodes = mesh.nodes;

        // Upload our visual mesh unit vectors if we have to
        std::pair<vk::Buffer, vk::DeviceMemory> vk_points;

        auto device_mesh = device_points_cache.find(&mesh);
        if (device_mesh == device_points_cache.end()) {
          vk_points =
            operation::create_buffer(instance.phys_device,
                                     instance.device,
                                     sizeof(vec4<Scalar>) * mesh.nodes.size(),
                                     vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                     vk::SharingMode::eExclusive,
                                     {instance.transfer_queue_family},
                                     vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                       | vk::MemoryPropertyFlagBits::eHostCoherent);

          // Write the points buffer to the device and cache it
          Scalar* points_payload = reinterpret_cast<Scalar*>(
            instance.device.mapMemory(vk_points.second, 0, sizeof(vec4<Scalar>) * mesh.nodes.size()));
          size_t index = 0;
          for (const auto& n : mesh.nodes) {
            points_payload[index + 0] = Scalar(n.ray[0]);
            points_payload[index + 1] = Scalar(n.ray[1]);
            points_payload[index + 2] = Scalar(n.ray[2]);
            points_payload[index + 3] = Scalar(0);
            index += 4;
          }
          instance.device.unmapMemory(vk_points.second);

          // Cache for future runs
          device_points_cache[&mesh] = vk_points;
        }
        else {
          vk_points = device_mesh->second;
        }

        // First count the size of the buffer we will need to allocate
        int points = 0;
        for (const auto& range : ranges) {
          points += range.second - range.first;
        }

        // No point processing if we have no points, return an empty mesh
        if (points == 0) {
          return std::make_tuple(std::vector<std::array<int, N_NEIGHBOURS>>(),
                                 std::vector<int>(),
                                 std::pair<vk::Buffer, vk::DeviceMemory>(),
                                 vk::Fence());
        }

        // Build up our list of indices for OpenCL
        // Use iota to fill in the numbers
        std::vector<int> indices(points);
        auto it = indices.begin();
        for (const auto& range : ranges) {
          auto n = std::next(it, range.second - range.first);
          std::iota(it, n, range.first);
          it = n;
        }

        // Create buffer for indices map
        std::pair<vk::Buffer, vk::DeviceMemory> vk_indices_map =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(int) * points,
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        // Create output buffer for pixel_coordinates
        std::pair<vk::Buffer, vk::DeviceMemory> vk_pixel_coordinates =
          operation::create_buffer(instance.phys_device,
                                   instance.device,
                                   sizeof(vec2<Scalar>) * points,
                                   vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::SharingMode::eExclusive,
                                   {instance.transfer_queue_family},
                                   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
                                     | vk::MemoryPropertyFlagBits::eHostCoherent);

        // Upload our indices map
        int* indices_payload =
          reinterpret_cast<int*>(instance.device.mapMemory(vk_indices_map.second, 0, sizeof(int) * indices.size()));
        size_t count = 0;
        for (const auto& index : indices) {
          indices_payload[count] = index;
          count++;
        }
        instance.device.unmapMemory(vk_indices_map.second);

        // --------------------------------------------------
        // At this point the point and the indices should be
        // uploaded since both device memories are coherent
        // --------------------------------------------------

        vk::Pipeline reprojection_pipeline;

        // Select a projection kernel
        switch (lens.projection) {
          case RECTILINEAR: reprojection_pipeline = project_rectilinear; break;
          case EQUIDISTANT: reprojection_pipeline = project_equidistant; break;
          case EQUISOLID: reprojection_pipeline = project_equisolid; break;
          default:
            throw_vk_error(vk::Result::eErrorFormatNotSupported,
                           "Requested lens projection is not currently supported.");
            return std::make_tuple(std::vector<std::array<int, N_NEIGHBOURS>>(),
                                   std::vector<int>(),
                                   std::pair<vk::Buffer, vk::DeviceMemory>(),
                                   vk::Fence());
        }

        // Create a descriptor pool
        // Descriptor Set 0: {points_ptr, indices_ptr, Rco_ptr, f_ptr, centre_ptr, k_ptr, dimensions_ptr, out_ptr}
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 8);
        vk::DescriptorPool descriptor_pool = instance.device.createDescriptorPool(
          vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, 1, &descriptor_pool_size));

        // Allocate the descriptor set
        vk::DescriptorSet descriptor_set =
          instance.device
            .allocateDescriptorSets(vk::DescriptorSetAllocateInfo(descriptor_pool, 1, &reprojection_descriptor_layout))
            .back();

        // Load the arguments
        std::vector<vk::WriteDescriptorSet> write_descriptors;
        vk::DescriptorBufferInfo buffer_info(vk_points.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_indices_map.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_rco.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_f.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_centre.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_k.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_dimensions.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);
        buffer_info = vk::DescriptorBufferInfo(vk_pixel_coordinates.first, 0, VK_WHOLE_SIZE);
        write_descriptors.emplace_back(
          descriptor_set, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info, nullptr);

        instance.device.updateDescriptorSets(write_descriptors.size(), write_descriptors.data(), 0, nullptr);

        // Project!
        vk::CommandPool command_pool = instance.device.createCommandPool(
          vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, instance.compute_queue_family));

        vk::CommandBuffer command_buffer =
          instance.device
            .allocateCommandBuffers(vk::CommandBufferAllocateInfo(command_pool, vk::CommandBufferLevel::ePrimary, 1))
            .back();

        command_buffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit, nullptr));
        command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, reprojection_pipeline);
        command_buffer.bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, reprojection_pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        command_buffer.dispatch(static_cast<uint32_t>(points), 1, 1);
        command_buffer.end();

        vk::Fence reprojection_complete = instance.device.createFence(vk::FenceCreateInfo());
        vk::SubmitInfo submit_info(0, nullptr, nullptr, 1, &command_buffer, 0, nullptr);
        instance.compute_queue.submit(1, &submit_info, reprojection_complete);

        // This can happen on the CPU while the OpenCL device is busy
        // Build the reverse lookup map where the offscreen point is one past the end
        std::vector<int> r_indices(nodes.size() + 1, points);
        for (unsigned int i = 0; i < indices.size(); ++i) {
          r_indices[indices[i]] = i;
        }

        // Build the packed neighbourhood map with an extra offscreen point at the end
        std::vector<std::array<int, N_NEIGHBOURS>> local_neighbourhood(points + 1);
        for (unsigned int i = 0; i < indices.size(); ++i) {
          const auto& node = nodes[indices[i]];
          for (unsigned int j = 0; j < node.neighbours.size(); ++j) {
            const auto& n             = node.neighbours[j];
            local_neighbourhood[i][j] = r_indices[n];
          }
        }
        // Fill in the final offscreen point which connects only to itself
        local_neighbourhood[points].fill(points);

        // Return what we calculated
        return std::make_tuple(std::move(local_neighbourhood),  // CPU buffer
                               std::move(indices),              // CPU buffer
                               vk_pixel_coordinates,            // GPU buffer
                               reprojection_complete);          // GPU event
      }

      // cl::mem get_image_memory(vec2<int> dimensions, uint32_t format) const {

      //   // If our dimensions and format haven't changed from last time we can reuse the same memory location
      //   if (dimensions != image_memory.dimensions || format != image_memory.format) {
      //     cl_image_format fmt;
      //     switch (format) {
      //       // Bayer
      //       case fourcc("GRBG"):
      //       case fourcc("RGGB"):
      //       case fourcc("GBRG"):
      //       case fourcc("BGGR"): fmt = cl_image_format{CL_R, CL_UNORM_INT8}; break;
      //       case fourcc("BGRA"): fmt = cl_image_format{CL_BGRA, CL_UNORM_INT8}; break;
      //       case fourcc("RGBA"): fmt = cl_image_format{CL_RGBA, CL_UNORM_INT8}; break;
      //       // Oh no...
      //       default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
      //     }

      //     cl_image_desc desc = {
      //       CL_MEM_OBJECT_IMAGE2D, size_t(dimensions[0]), size_t(dimensions[1]), 1, 1, 0, 0, 0, 0, nullptr};

      //     // Create a buffer for our image
      //     cl_int error;
      //     cl::mem memory(::clCreateImage(context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &error),
      //                    ::clReleaseMemObject);
      //     throw_cl_error(error, "Error creating image on device");

      //     // Update what we are caching
      //     image_memory.dimensions = dimensions;
      //     image_memory.format     = format;
      //     image_memory.memory     = memory;
      //   }

      //   // Return the cache
      //   return image_memory.memory;
      // }

      // std::array<cl::mem, 2> get_network_memory(const int& max_size) const {
      //   if (network_memory.max_size < max_size) {
      //     cl_int error;
      //     network_memory.memory[0] =
      //       cl::mem(::clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Scalar) * max_size, nullptr, &error),
      //               ::clReleaseMemObject);
      //     throw_cl_error(error, "Error allocating ping pong buffer 1 on device");
      //     network_memory.memory[1] =
      //       cl::mem(::clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Scalar) * max_size, nullptr, &error),
      //               ::clReleaseMemObject);
      //     network_memory.max_size = max_size;
      //     throw_cl_error(error, "Error allocating ping pong buffer 2 on device");
      //   }
      //   return network_memory.memory;
      // }

      // cl::mem get_neighbourhood_memory(const int& max_size) const {

      //   if (neighbourhood_memory.max_size < max_size) {
      //     cl_int error;
      //     neighbourhood_memory.memory =
      //       cl::mem(::clCreateBuffer(context, CL_MEM_READ_WRITE, max_size * sizeof(int), nullptr, &error),
      //               ::clReleaseMemObject);
      //     throw_cl_error(error, "Error allocating neighbourhood buffer on device");
      //     neighbourhood_memory.max_size = max_size;
      //   }
      //   return neighbourhood_memory.memory;
      // }

      /// Vulkan dynamic loader
      vk::DynamicLoader dl;

      /// Vulkan instance
      operation::VulkanInstance instance;

      /// DescriptorSetLayout for the reprojection kernels
      vk::DescriptorSetLayout reprojection_descriptor_layout;
      /// PipelineLayout for the reprojection kernels
      vk::PipelineLayout reprojection_pipeline_layout;
      /// Kernel for projecting rays to pixels using an equidistant projection
      vk::Pipeline project_equidistant;
      /// Kernel for projecting rays to pixels using an equisolid projection
      vk::Pipeline project_equisolid;
      /// Kernel for projecting rays to pixels using a rectilinear projection
      vk::Pipeline project_rectilinear;

      /// DescriptorSetLayouts for the load_image kernel
      std::array<vk::DescriptorSetLayout, 2> load_image_descriptor_layout;
      /// PipelineLayout for the load_image kernel
      vk::PipelineLayout load_image_pipeline_layout;
      /// Kernel for reading projected pixel coordinates from an image into the network input layer
      vk::Pipeline load_GRBG_image;
      vk::Pipeline load_RGGB_image;
      vk::Pipeline load_GBRG_image;
      vk::Pipeline load_BGGR_image;
      vk::Pipeline load_RGBA_image;
      /// A list of kernels to run in sequence to run the network
      std::vector<std::pair<vk::Pipeline, size_t>> conv_layers;

      // mutable struct {
      //   vec2<int> dimensions = {0, 0};
      //   uint32_t format      = 0;
      //   cl::mem memory;
      // } image_memory;

      // mutable struct {
      //   int max_size = 0;
      //   std::array<cl::mem, 2> memory;
      // } network_memory;

      // mutable struct {
      //   int max_size = 0;
      //   cl::mem memory;
      // } neighbourhood_memory;

      // The width of the maximumally wide layer in the network
      size_t max_width;

      // Cache of opencl buffers from mesh objects
      mutable std::map<const void*, std::pair<vk::Buffer, vk::DeviceMemory>> device_points_cache;
    };  // namespace vulkan

  }  // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_ENGINE_HPP
