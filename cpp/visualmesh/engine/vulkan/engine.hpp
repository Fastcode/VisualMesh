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

#ifndef VISUALMESH_ENGINE_VULKAN_ENGINE_HPP
#define VISUALMESH_ENGINE_VULKAN_ENGINE_HPP

// If OpenCL is disabled then don't provide this file
#if !defined(VISUALMESH_DISABLE_VULKAN)

#include <fstream>
#include <iomanip>
#include <numeric>
#include <spirv/unified1/spirv.hpp11>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "visualmesh/engine/vulkan/kernels/load_image.hpp"
#include "visualmesh/engine/vulkan/kernels/make_network.hpp"
#include "visualmesh/engine/vulkan/kernels/reprojection.hpp"
#include "visualmesh/engine/vulkan/operation/create_buffer.hpp"
#include "visualmesh/engine/vulkan/operation/create_command_buffer.hpp"
#include "visualmesh/engine/vulkan/operation/create_descriptor_set.hpp"
#include "visualmesh/engine/vulkan/operation/create_device.hpp"
#include "visualmesh/engine/vulkan/operation/create_image.hpp"
#include "visualmesh/engine/vulkan/operation/vulkan_error_category.hpp"
#include "visualmesh/engine/vulkan/operation/wrapper.hpp"
#include "visualmesh/mesh.hpp"
#include "visualmesh/network_structure.hpp"
#include "visualmesh/projected_mesh.hpp"
#include "visualmesh/utility/math.hpp"
#include "visualmesh/utility/projection.hpp"
#include "visualmesh/utility/static_if.hpp"
#include "visualmesh/visualmesh.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {

        /**
         * @brief An Vulkan implementation of the visual mesh inference engine
         *
         * @details
         *  The Vulkan implementation is designed to be used for high performance inference. It is able to take
         * advantage of either GPUs from Intel, AMD, ARM, NVIDIA etc as well as multithreaded CPU implementations. This
         * allows it to be very flexible with its deployment on devices.
         *
         * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
         */
        template <typename Scalar, bool debug = false>
        class Engine {
        public:
            /**
             * @brief Construct a new Vulkan Engine object
             *
             * @param structure the network structure to use classification
             */
            Engine(const NetworkStructure<Scalar>& structure = {}) : max_width(4) {
                // Get a Vulkan instance
                const VkApplicationInfo app_info = {
                  VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "VisualMesh", 0, "", 0, VK_MAKE_VERSION(1, 1, 0)};

                const VkInstanceCreateInfo instance_info = {
                  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app_info, 0, 0, 0, 0};

                VkInstance instance;
                throw_vk_error(vkCreateInstance(&instance_info, 0, &instance), "Failed to create instance");
                context.instance = vk::instance(instance, [](auto p) { vkDestroyInstance(p, nullptr); });

                // Create the Vulkan instance and find the best devices and queues
                operation::create_device(DeviceType::GPU, context, debug);

                // Create queues and command pools
                vkGetDeviceQueue(context.device, context.compute_queue_family, 0, &context.compute_queue);
                context.compute_command_pool = operation::create_command_pool(context, context.compute_queue_family);
                if (context.compute_queue_family == context.transfer_queue_family) {
                    context.transfer_queue        = context.compute_queue;
                    context.transfer_command_pool = context.compute_command_pool;
                }
                else {
                    vkGetDeviceQueue(context.device, context.transfer_queue_family, 0, &context.transfer_queue);
                    context.transfer_command_pool =
                      operation::create_command_pool(context, context.transfer_queue_family);
                }

                // Created the projection kernel sources and programs
                std::vector<uint32_t> equidistant_reprojection_source =
                  kernels::build_reprojection<Scalar>("project_equidistant", kernels::equidistant_reprojection<Scalar>);
                std::vector<uint32_t> equisolid_reprojection_source =
                  kernels::build_reprojection<Scalar>("project_equisolid", kernels::equisolid_reprojection<Scalar>);
                std::vector<uint32_t> rectilinear_reprojection_source =
                  kernels::build_reprojection<Scalar>("project_rectilinear", kernels::rectilinear_reprojection<Scalar>);

                if (debug) {
                    std::ofstream ofs;
                    ofs.open("project_equidistant.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(equidistant_reprojection_source.data()),
                              equidistant_reprojection_source.size() * sizeof(uint32_t));
                    ofs.close();
                    ofs.open("project_equisolid.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(equisolid_reprojection_source.data()),
                              equisolid_reprojection_source.size() * sizeof(uint32_t));
                    ofs.close();
                    ofs.open("project_rectilinear.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(rectilinear_reprojection_source.data()),
                              rectilinear_reprojection_source.size() * sizeof(uint32_t));
                    ofs.close();
                }

                VkShaderModuleCreateInfo equidistant_reprojection_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(equidistant_reprojection_source.size()) * sizeof(uint32_t),
                  equidistant_reprojection_source.data()};
                VkShaderModuleCreateInfo equisolid_reprojection_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(equisolid_reprojection_source.size()) * sizeof(uint32_t),
                  equisolid_reprojection_source.data()};
                VkShaderModuleCreateInfo rectilinear_reprojection_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(rectilinear_reprojection_source.size()) * sizeof(uint32_t),
                  rectilinear_reprojection_source.data()};

                VkShaderModule equidistant_reprojection_shader;
                VkShaderModule equisolid_reprojection_shader;
                VkShaderModule rectilinear_reprojection_shader;

                throw_vk_error(
                  vkCreateShaderModule(
                    context.device, &equidistant_reprojection_info, nullptr, &equidistant_reprojection_shader),
                  "Failed to create equidistant_reprojection shader module");
                throw_vk_error(vkCreateShaderModule(
                                 context.device, &equisolid_reprojection_info, nullptr, &equisolid_reprojection_shader),
                               "Failed to create equisolid_reprojection shader module");
                throw_vk_error(
                  vkCreateShaderModule(
                    context.device, &rectilinear_reprojection_info, nullptr, &rectilinear_reprojection_shader),
                  "Failed to create rectilinear_reprojection shader module");
                equidistant_reprojection_program = vk::shader_module(equidistant_reprojection_shader, [this](auto p) {
                    vkDestroyShaderModule(context.device, p, nullptr);
                });
                equisolid_reprojection_program   = vk::shader_module(
                  equisolid_reprojection_shader, [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });
                rectilinear_reprojection_program = vk::shader_module(rectilinear_reprojection_shader, [this](auto p) {
                    vkDestroyShaderModule(context.device, p, nullptr);
                });

                // Create the descriptor set for the reprojection programs
                // All reprojection programs have the same descriptor set
                // Descriptor Set 0: {points_ptr, indices_ptr, Rco_ptr, f_ptr, centre_ptr, k_ptr, dimensions_ptr,
                // out_ptr}
                std::array<VkDescriptorSetLayoutBinding, 8> reprojection_bindings;
                reprojection_bindings[0] = {
                  0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[1] = {
                  1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[2] = {
                  2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[3] = {
                  3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[4] = {
                  4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[5] = {
                  5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[6] = {
                  6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                reprojection_bindings[7] = {
                  7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

                VkDescriptorSetLayoutCreateInfo reprojection_descriptor_set_layout_create_info = {
                  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(reprojection_bindings.size()),
                  reprojection_bindings.data()};

                throw_vk_error(vkCreateDescriptorSetLayout(context.device,
                                                           &reprojection_descriptor_set_layout_create_info,
                                                           0,
                                                           &reprojection_descriptor_layout),
                               "Failed to create reprojection descriptor set layout");

                VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &reprojection_descriptor_layout, 0, nullptr};

                throw_vk_error(vkCreatePipelineLayout(
                                 context.device, &pipeline_layout_create_info, 0, &reprojection_pipeline_layout),
                               "Failed to create reprojection pipeline layout");

                VkComputePipelineCreateInfo equidistant_pipeline_create_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  0,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   0,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   equidistant_reprojection_program,
                   "project_equidistant",
                   0},
                  reprojection_pipeline_layout,
                  0,
                  0};
                throw_vk_error(vkCreateComputePipelines(
                                 context.device, 0, 1, &equidistant_pipeline_create_info, 0, &project_equidistant),
                               "Failed to create equidistant reprojection pipeline");

                VkComputePipelineCreateInfo equisolid_pipeline_create_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  0,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   0,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   equisolid_reprojection_program,
                   "project_equisolid",
                   0},
                  reprojection_pipeline_layout,
                  0,
                  0};
                throw_vk_error(vkCreateComputePipelines(
                                 context.device, 0, 1, &equisolid_pipeline_create_info, 0, &project_equisolid),
                               "Failed to create equisolid reprojection pipeline");

                VkComputePipelineCreateInfo rectilinear_pipeline_create_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  0,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   0,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   rectilinear_reprojection_program,
                   "project_rectilinear",
                   0},
                  reprojection_pipeline_layout,
                  0,
                  0};
                throw_vk_error(vkCreateComputePipelines(
                                 context.device, 0, 1, &rectilinear_pipeline_create_info, 0, &project_rectilinear),
                               "Failed to create rectilinear reprojection pipeline");

                // ******************
                // *** LOAD_IMAGE ***
                // ******************

                // Created the load_image kernel source and program
                std::vector<uint32_t> load_GRBG_image_source =
                  kernels::load_image<Scalar, debug>(kernels::load_GRBG_image<Scalar>);
                std::vector<uint32_t> load_RGGB_image_source =
                  kernels::load_image<Scalar, debug>(kernels::load_RGGB_image<Scalar>);
                std::vector<uint32_t> load_GBRG_image_source =
                  kernels::load_image<Scalar, debug>(kernels::load_GBRG_image<Scalar>);
                std::vector<uint32_t> load_BGGR_image_source =
                  kernels::load_image<Scalar, debug>(kernels::load_BGGR_image<Scalar>);
                std::vector<uint32_t> load_RGBA_image_source =
                  kernels::load_image<Scalar, debug>(kernels::load_RGBA_image<Scalar>);
                if (debug) {
                    std::ofstream ofs;
                    ofs.open("load_GRBG_image.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(load_GRBG_image_source.data()),
                              load_GRBG_image_source.size() * sizeof(uint32_t));
                    ofs.close();
                    ofs.open("load_RGGB_image.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(load_RGGB_image_source.data()),
                              load_RGGB_image_source.size() * sizeof(uint32_t));
                    ofs.close();
                    ofs.open("load_GBRG_image.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(load_GBRG_image_source.data()),
                              load_GBRG_image_source.size() * sizeof(uint32_t));
                    ofs.close();
                    ofs.open("load_BGGR_image.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(load_BGGR_image_source.data()),
                              load_BGGR_image_source.size() * sizeof(uint32_t));
                    ofs.close();
                    ofs.open("load_RGBA_image.spv", std::ios::binary | std::ios::out);
                    ofs.write(reinterpret_cast<const char*>(load_RGBA_image_source.data()),
                              load_RGBA_image_source.size() * sizeof(uint32_t));
                    ofs.close();
                }
                VkShaderModuleCreateInfo load_GRBG_image_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(load_GRBG_image_source.size()) * sizeof(uint32_t),
                  load_GRBG_image_source.data()};
                VkShaderModuleCreateInfo load_RGGB_image_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(load_RGGB_image_source.size()) * sizeof(uint32_t),
                  load_RGGB_image_source.data()};
                VkShaderModuleCreateInfo load_GBRG_image_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(load_GBRG_image_source.size()) * sizeof(uint32_t),
                  load_GBRG_image_source.data()};
                VkShaderModuleCreateInfo load_BGGR_image_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(load_BGGR_image_source.size()) * sizeof(uint32_t),
                  load_BGGR_image_source.data()};
                VkShaderModuleCreateInfo load_RGBA_image_info = {
                  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                  0,
                  0,
                  static_cast<uint32_t>(load_RGBA_image_source.size()) * sizeof(uint32_t),
                  load_RGBA_image_source.data()};

                VkShaderModule load_GRBG_image_shader;
                throw_vk_error(
                  vkCreateShaderModule(context.device, &load_GRBG_image_info, nullptr, &load_GRBG_image_shader),
                  "Failed to create load_image shader module");
                VkShaderModule load_RGGB_image_shader;
                throw_vk_error(
                  vkCreateShaderModule(context.device, &load_RGGB_image_info, nullptr, &load_RGGB_image_shader),
                  "Failed to create load_image shader module");
                VkShaderModule load_GBRG_image_shader;
                throw_vk_error(
                  vkCreateShaderModule(context.device, &load_GBRG_image_info, nullptr, &load_GBRG_image_shader),
                  "Failed to create load_image shader module");
                VkShaderModule load_BGGR_image_shader;
                throw_vk_error(
                  vkCreateShaderModule(context.device, &load_BGGR_image_info, nullptr, &load_BGGR_image_shader),
                  "Failed to create load_image shader module");
                VkShaderModule load_RGBA_image_shader;
                throw_vk_error(
                  vkCreateShaderModule(context.device, &load_RGBA_image_info, nullptr, &load_RGBA_image_shader),
                  "Failed to create load_image shader module");

                load_GRBG_image_program = vk::shader_module(
                  load_GRBG_image_shader, [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });
                load_RGGB_image_program = vk::shader_module(
                  load_RGGB_image_shader, [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });
                load_GBRG_image_program = vk::shader_module(
                  load_GBRG_image_shader, [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });
                load_BGGR_image_program = vk::shader_module(
                  load_BGGR_image_shader, [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });
                load_RGBA_image_program = vk::shader_module(
                  load_RGBA_image_shader, [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });

                // Create the descriptor sets for the load_image program
                // Descriptor Set 0: {bayer_sampler, interp_sampler, image, coordinates, network}
                VkSamplerCreateInfo bayer_sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                          nullptr,
                                                          0,
                                                          VK_FILTER_NEAREST,
                                                          VK_FILTER_NEAREST,
                                                          VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                                          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                          0.0f,
                                                          VK_FALSE,
                                                          0.0f,
                                                          VK_FALSE,
                                                          VK_COMPARE_OP_NEVER,
                                                          0.0f,
                                                          0.0f,
                                                          VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                                                          VK_TRUE};
                VkSampler bsampler;
                throw_vk_error(vkCreateSampler(context.device, &bayer_sampler_info, nullptr, &bsampler),
                               "Failed to create bayer sampler");
                VkSamplerCreateInfo interp_sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                           nullptr,
                                                           0,
                                                           VK_FILTER_LINEAR,
                                                           VK_FILTER_LINEAR,
                                                           VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                                           0.0f,
                                                           VK_FALSE,
                                                           0.0f,
                                                           VK_FALSE,
                                                           VK_COMPARE_OP_NEVER,
                                                           0.0f,
                                                           0.0f,
                                                           VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                                                           VK_TRUE};
                VkSampler isampler;
                throw_vk_error(vkCreateSampler(context.device, &interp_sampler_info, nullptr, &isampler),
                               "Failed to create interp sampler");

                bayer_sampler = vk::sampler(bsampler, [this](auto p) { vkDestroySampler(context.device, p, nullptr); });
                interp_sampler =
                  vk::sampler(isampler, [this](auto p) { vkDestroySampler(context.device, p, nullptr); });

                // Create the descriptor sets for the load_image program
                // Descriptor Set 0: {image+sampler, coordinates, network}
                std::array<VkDescriptorSetLayoutBinding, 3> load_image_bindings{
                  VkDescriptorSetLayoutBinding{
                    0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                  VkDescriptorSetLayoutBinding{
                    1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                  VkDescriptorSetLayoutBinding{
                    2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

                VkDescriptorSetLayoutCreateInfo load_image_descriptor_set_info = {
                  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                  nullptr,
                  0,
                  static_cast<uint32_t>(load_image_bindings.size()),
                  load_image_bindings.data()};
                throw_vk_error(vkCreateDescriptorSetLayout(
                                 context.device, &load_image_descriptor_set_info, 0, &load_image_descriptor_layout),
                               "Failed to create load_image descriptor set layout");

                VkPipelineLayoutCreateInfo load_image_pipeline_layout_info = {
                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &load_image_descriptor_layout, 0, 0};

                throw_vk_error(vkCreatePipelineLayout(
                                 context.device, &load_image_pipeline_layout_info, 0, &load_image_pipeline_layout),
                               "Failed to create load_image pipeline layout");

                VkComputePipelineCreateInfo load_GRBG_image_pipeline_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  nullptr,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   nullptr,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   load_GRBG_image_program,
                   "load_GRBG_image",
                   0},
                  load_image_pipeline_layout,
                  0,
                  0};
                throw_vk_error(
                  vkCreateComputePipelines(context.device, 0, 1, &load_GRBG_image_pipeline_info, 0, &load_GRBG_image),
                  "Failed to create load_GRBG_image pipeline");

                VkComputePipelineCreateInfo load_RGGB_image_pipeline_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  nullptr,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   nullptr,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   load_RGGB_image_program,
                   "load_RGGB_image",
                   0},
                  load_image_pipeline_layout,
                  0,
                  0};
                throw_vk_error(
                  vkCreateComputePipelines(context.device, 0, 1, &load_RGGB_image_pipeline_info, 0, &load_RGGB_image),
                  "Failed to create load_RGGB_image pipeline");

                VkComputePipelineCreateInfo load_GBRG_image_pipeline_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  nullptr,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   nullptr,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   load_GBRG_image_program,
                   "load_GBRG_image",
                   0},
                  load_image_pipeline_layout,
                  0,
                  0};
                throw_vk_error(
                  vkCreateComputePipelines(context.device, 0, 1, &load_GBRG_image_pipeline_info, 0, &load_GBRG_image),
                  "Failed to create load_GBRG_image pipeline");

                VkComputePipelineCreateInfo load_BGGR_image_pipeline_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  nullptr,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   nullptr,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   load_BGGR_image_program,
                   "load_BGGR_image",
                   0},
                  load_image_pipeline_layout,
                  0,
                  0};
                throw_vk_error(
                  vkCreateComputePipelines(context.device, 0, 1, &load_BGGR_image_pipeline_info, 0, &load_BGGR_image),
                  "Failed to create load_BGGR_image pipeline");

                VkComputePipelineCreateInfo load_RGBA_image_pipeline_info = {
                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                  nullptr,
                  0,
                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   nullptr,
                   0,
                   VK_SHADER_STAGE_COMPUTE_BIT,
                   load_RGBA_image_program,
                   "load_RGBA_image",
                   0},
                  load_image_pipeline_layout,
                  0,
                  0};
                throw_vk_error(
                  vkCreateComputePipelines(context.device, 0, 1, &load_RGBA_image_pipeline_info, 0, &load_RGBA_image),
                  "Failed to create load_RGBA_image pipeline");

                // ***************
                // *** Network ***
                // ***************

                // Descriptor Set 0: {neighbourhood_ptr, input_ptr, output_ptr}
                std::array<VkDescriptorSetLayoutBinding, 3> conv_bindings{
                  VkDescriptorSetLayoutBinding{
                    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                  VkDescriptorSetLayoutBinding{
                    1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                  VkDescriptorSetLayoutBinding{
                    2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

                VkDescriptorSetLayoutCreateInfo conv_descriptor_set_info = {
                  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                  nullptr,
                  0,
                  static_cast<uint32_t>(conv_bindings.size()),
                  conv_bindings.data()};
                throw_vk_error(
                  vkCreateDescriptorSetLayout(context.device, &conv_descriptor_set_info, 0, &conv_descriptor_layout),
                  "Failed to create conv descriptor set layout");

                VkPipelineLayoutCreateInfo conv_pipeline_layout_info = {
                  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &conv_descriptor_layout, 0, 0};

                throw_vk_error(
                  vkCreatePipelineLayout(context.device, &conv_pipeline_layout_info, 0, &conv_pipeline_layout),
                  "Failed to create conv pipeline layout");

                std::vector<std::pair<uint32_t, std::vector<uint32_t>>> conv_sources =
                  kernels::make_network<Scalar, debug>(structure);
                for (const auto& conv_source : conv_sources) {
                    std::string kernel = "conv" + std::to_string(conv_source.first);
                    if (debug) {
                        std::ofstream ofs;
                        ofs.open(kernel + ".spv", std::ios::binary | std::ios::out);
                        ofs.write(reinterpret_cast<const char*>(conv_source.second.data()),
                                  conv_source.second.size() * sizeof(uint32_t));
                        ofs.close();
                    }

                    VkShaderModuleCreateInfo conv_info = {
                      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                      nullptr,
                      0,
                      static_cast<uint32_t>(conv_source.second.size()) * sizeof(uint32_t),
                      conv_source.second.data()};
                    VkShaderModule conv_shader;
                    throw_vk_error(vkCreateShaderModule(context.device, &conv_info, nullptr, &conv_shader),
                                   "Failed to create conv shader module");
                    conv_program.emplace_back(conv_shader,
                                              [this](auto p) { vkDestroyShaderModule(context.device, p, nullptr); });

                    VkComputePipelineCreateInfo conv_pipeline_info = {
                      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                      nullptr,
                      0,
                      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       nullptr,
                       0,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       conv_program.back(),
                       kernel.c_str(),
                       0},
                      conv_pipeline_layout,
                      0,
                      0};
                    VkPipeline pipeline;
                    throw_vk_error(vkCreateComputePipelines(context.device, 0, 1, &conv_pipeline_info, 0, &pipeline),
                                   "Failed to create conv pipeline");
                    conv_layers.emplace_back(pipeline, structure[conv_source.first].back().biases.size());
                }

                // Work out what the widest network layer is
                max_width = 4;
                for (const auto& k : conv_layers) {
                    max_width = std::max(max_width, k.second);
                }
            }

            ~Engine() {
                vkDestroyDescriptorSetLayout(context.device, reprojection_descriptor_layout, nullptr);
                vkDestroyPipelineLayout(context.device, reprojection_pipeline_layout, nullptr);
                vkDestroyPipeline(context.device, project_equidistant, nullptr);
                vkDestroyPipeline(context.device, project_equisolid, nullptr);
                vkDestroyPipeline(context.device, project_rectilinear, nullptr);
                vkDestroyPipelineLayout(context.device, load_image_pipeline_layout, nullptr);
                vkDestroyPipeline(context.device, load_GRBG_image, nullptr);
                vkDestroyPipeline(context.device, load_RGGB_image, nullptr);
                vkDestroyPipeline(context.device, load_GBRG_image, nullptr);
                vkDestroyPipeline(context.device, load_BGGR_image, nullptr);
                vkDestroyPipeline(context.device, load_RGBA_image, nullptr);
                vkDestroyDescriptorSetLayout(context.device, load_image_descriptor_layout, nullptr);

                vkDestroyDescriptorSetLayout(context.device, conv_descriptor_layout, nullptr);
                vkDestroyPipelineLayout(context.device, conv_pipeline_layout, nullptr);
                for (const auto& layer : conv_layers) {
                    vkDestroyPipeline(context.device, layer.first, nullptr);
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
            inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Model>& mesh,
                                                                                 const mat4<Scalar>& Hoc,
                                                                                 const Lens<Scalar>& lens) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
                std::vector<int> indices;
                std::pair<vk::buffer, vk::device_memory> vk_pixels;
                vk::fence fence;

                std::tie(neighbourhood, indices, vk_pixels, fence) = do_project<Model, vk::fence>(mesh, Hoc, lens);

                // Wait 10,000 * 1,000 nanoseconds = 10,000 * 1us = 10ms
                VkResult res;
                VkFence vk_fence = fence;
                for (uint32_t timeout_count = 0; timeout_count < 10000; ++timeout_count) {
                    res = vkWaitForFences(context.device, 1, &vk_fence, VK_TRUE, static_cast<uint64_t>(1e3));
                    if (res == VK_SUCCESS) { break; }
                    else if (res == VK_ERROR_DEVICE_LOST) {
                        throw_vk_error(VK_ERROR_DEVICE_LOST, "Lost device while waiting for reprojection to complete");
                    }
                }
                if (res != VK_SUCCESS) { throw_vk_error(res, "Timed out waiting for reprojection to complete"); }

                // Read the pixels off the buffer
                std::vector<vec2<Scalar>> pixels(indices.size());
                operation::map_memory<void>(context, 0, VK_WHOLE_SIZE, vk_pixels.second, [&pixels](void* payload) {
                    std::memcpy(pixels.data(), payload, pixels.size() * sizeof(vec2<Scalar>));
                });

                // Perform cleanup
                reprojection_command_buffer.reset();
                reprojection_descriptor_pool.reset();
                reprojection_buffers.clear();

                return ProjectedMesh<Scalar, N_NEIGHBOURS>{
                  std::move(pixels), std::move(neighbourhood), std::move(indices)};
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
            inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const VisualMesh<Scalar, Model>& mesh,
                                                                                 const mat4<Scalar>& Hoc,
                                                                                 const Lens<Scalar>& lens) const {
                return operator()(mesh.height(Hoc[2][3]), Hoc, lens);
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
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                std::vector<std::pair<vk::semaphore, VkPipelineStageFlags>> wait_semaphores;

                // *******************************
                // *** PROJECT OUR VISUAL MESH ***
                // *******************************

                std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
                std::vector<int> indices;
                std::pair<vk::buffer, vk::device_memory> vk_pixels;
                vk::semaphore reprojection_semaphore;
                std::tie(neighbourhood, indices, vk_pixels, reprojection_semaphore) =
                  do_project<Model, vk::semaphore>(mesh, Hoc, lens);

                wait_semaphores.push_back(std::make_pair(reprojection_semaphore, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));

                // ****************************
                // *** LOAD IMAGE TO DEVICE ***
                // ****************************

                // Grab the image memory from the cache
                std::pair<vk::image, vk::device_memory> vk_image = get_image_memory(lens.dimensions, format);
                operation::copy_image_to_device(context, image, lens.dimensions, vk_image, get_image_format(format));

                // This includes the offscreen point at the end
                int n_points = neighbourhood.size();

                // Get the neighbourhood memory from cache
                std::pair<vk::buffer, vk::device_memory> vk_neighbourhood =
                  get_neighbourhood_memory(n_points * N_NEIGHBOURS);

                // Upload the neighbourhood buffer
                operation::map_memory<void>(
                  context, 0, VK_WHOLE_SIZE, vk_neighbourhood.second, [&neighbourhood](void* payload) {
                      std::memcpy(payload, neighbourhood.data(), neighbourhood.size() * N_NEIGHBOURS * sizeof(int));
                  });

                // Grab our ping pong buffers from the cache
                auto vk_conv_mem                                        = get_network_memory(max_width * n_points);
                std::pair<vk::buffer, vk::device_memory> vk_conv_input  = vk_conv_mem[0];
                std::pair<vk::buffer, vk::device_memory> vk_conv_output = vk_conv_mem[1];

                // The offscreen point gets a value of -1.0 to make it easy to distinguish
                operation::map_memory<Scalar>(context,
                                              (n_points - 1) * sizeof(Scalar),
                                              VK_WHOLE_SIZE,
                                              vk_conv_input.second,
                                              [&n_points](Scalar* payload) { payload[0] = Scalar(-1); });

                // Read the pixels into the buffer
                // Create a descriptor pool
                // Descriptor Set 0: {image+sampler, coordinates, network}
                std::vector<VkDescriptorPoolSize> load_image_pool_size = {
                  VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                  VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
                };
                vk::descriptor_pool load_image_pool = operation::create_descriptor_pool(context, load_image_pool_size);

                // Allocate the descriptor set
                VkDescriptorSet descriptor_set =
                  operation::create_descriptor_set(context, load_image_pool, {load_image_descriptor_layout}).back();

                // Load the arguments
                std::array<VkDescriptorBufferInfo, 2> buffer_infos = {
                  VkDescriptorBufferInfo{vk_pixels.first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{vk_conv_input.first, 0, VK_WHOLE_SIZE},
                };
                vk::image_view image_view        = get_image_view(vk_image.first, format);
                VkDescriptorImageInfo image_info = {
                  get_image_sampler(format), image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

                std::array<VkWriteDescriptorSet, 3> write_descriptors;
                write_descriptors[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        nullptr,
                                        descriptor_set,
                                        0,
                                        0,
                                        1,
                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        &image_info,
                                        nullptr,
                                        nullptr};
                for (size_t i = 0; i < buffer_infos.size(); ++i) {
                    write_descriptors[i + 1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                nullptr,
                                                descriptor_set,
                                                static_cast<uint32_t>(i + 1),
                                                0,
                                                1,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                nullptr,
                                                &buffer_infos[i],
                                                nullptr};
                }

                vkUpdateDescriptorSets(context.device, write_descriptors.size(), write_descriptors.data(), 0, nullptr);

                vk::command_buffer command_buffer =
                  operation::create_command_buffer(context, context.compute_command_pool, true);

                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, get_image_pipeline(format));

                vkCmdBindDescriptorSets(command_buffer,
                                        VK_PIPELINE_BIND_POINT_COMPUTE,
                                        load_image_pipeline_layout,
                                        0,
                                        1,
                                        &descriptor_set,
                                        0,
                                        nullptr);

                vkCmdDispatch(command_buffer, static_cast<uint32_t>(n_points - 1), 1, 1);

                VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
                VkSemaphore semaphore;
                throw_vk_error(vkCreateSemaphore(context.device, &semaphore_info, nullptr, &semaphore),
                               "Failed to create reprojection semaphore");
                vk::semaphore load_image_semaphore =
                  vk::semaphore(semaphore, [this](auto p) { vkDestroySemaphore(context.device, p, nullptr); });
                operation::submit_command_buffer(
                  context.compute_queue, command_buffer, {wait_semaphores.back()}, {load_image_semaphore});
                wait_semaphores.push_back(std::make_pair(load_image_semaphore, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));

                // *******************
                // *** RUN NETWORK ***
                // *******************

                // Create a descriptor pool
                // Descriptor Set 0: {neighbourhood_ptr, input_ptr, output_ptr}
                std::vector<VkDescriptorPoolSize> conv_pool_size = {
                  VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
                };
                vk::descriptor_pool conv_layer_pool =
                  operation::create_descriptor_pool(context, conv_pool_size, conv_layers.size());

                // Allocate the descriptor set
                std::vector<VkDescriptorSet> conv_descriptor_sets = operation::create_descriptor_set(
                  context,
                  conv_layer_pool,
                  std::vector<VkDescriptorSetLayout>(conv_layers.size(), conv_descriptor_layout));

                std::vector<std::array<VkDescriptorBufferInfo, 3>> conv_buffer_infos;
                std::vector<std::array<VkWriteDescriptorSet, 3>> conv_write_descriptors;
                std::vector<vk::command_buffer> conv_command_buffers;
                vk::fence fence;
                for (size_t conv_no = 0; conv_no < conv_layers.size(); ++conv_no) {
                    auto& conv = conv_layers[conv_no];

                    // Load the arguments
                    std::array<VkDescriptorBufferInfo, 3> buffer_infos = {
                      VkDescriptorBufferInfo{vk_neighbourhood.first, 0, VK_WHOLE_SIZE},
                      VkDescriptorBufferInfo{vk_conv_input.first, 0, VK_WHOLE_SIZE},
                      VkDescriptorBufferInfo{vk_conv_output.first, 0, VK_WHOLE_SIZE}};
                    conv_buffer_infos.push_back(buffer_infos);

                    std::array<VkWriteDescriptorSet, 3> write_descriptors;
                    for (size_t i = 0; i < buffer_infos.size(); ++i) {
                        write_descriptors[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                nullptr,
                                                conv_descriptor_sets[conv_no],
                                                static_cast<uint32_t>(i),
                                                0,
                                                1,
                                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                nullptr,
                                                &conv_buffer_infos.back()[i],
                                                nullptr};
                    }
                    conv_write_descriptors.push_back(write_descriptors);

                    vkUpdateDescriptorSets(
                      context.device, write_descriptors.size(), write_descriptors.data(), 0, nullptr);

                    // Project!
                    conv_command_buffers.push_back(
                      operation::create_command_buffer(context, context.compute_command_pool, true));

                    vkCmdBindPipeline(conv_command_buffers.back(), VK_PIPELINE_BIND_POINT_COMPUTE, conv.first);

                    vkCmdBindDescriptorSets(conv_command_buffers.back(),
                                            VK_PIPELINE_BIND_POINT_COMPUTE,
                                            conv_pipeline_layout,
                                            0,
                                            1,
                                            &conv_descriptor_sets[conv_no],
                                            0,
                                            nullptr);

                    vkCmdDispatch(conv_command_buffers.back(), static_cast<uint32_t>(n_points), 1, 1);

                    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
                    VkSemaphore semaphore;
                    throw_vk_error(vkCreateSemaphore(context.device, &semaphore_info, nullptr, &semaphore),
                                   "Failed to create conv layer semaphore");
                    vk::semaphore conv_layer_semaphore =
                      vk::semaphore(semaphore, [this](auto p) { vkDestroySemaphore(context.device, p, nullptr); });

                    if (conv_no + 1 == conv_layers.size()) {
                        VkFence f;
                        VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
                        throw_vk_error(vkCreateFence(context.device, &fence_info, nullptr, &f),
                                       "Failed to create reprojection semaphore");
                        fence = vk::fence(f, [this](auto p) { vkDestroyFence(context.device, p, nullptr); });
                        operation::submit_command_buffer(
                          context.compute_queue, conv_command_buffers.back(), fence, {wait_semaphores.back()});
                    }
                    else {
                        operation::submit_command_buffer(context.compute_queue,
                                                         conv_command_buffers.back(),
                                                         {wait_semaphores.back()},
                                                         {conv_layer_semaphore});
                    }

                    wait_semaphores.push_back(std::make_pair(conv_layer_semaphore, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));

                    // Convert our events into a vector of events and ping pong our buffers
                    std::swap(vk_conv_input, vk_conv_output);
                }

                // ***************************
                // *** WAIT FOR COMPLETION ***
                // ***************************

                // Wait 10,000 * 1,000 nanoseconds = 10,000 * 1us = 10ms
                VkResult res;
                VkFence vk_fence = fence;
                for (uint32_t timeout_count = 0; timeout_count < 10000; ++timeout_count) {
                    res = vkWaitForFences(context.device, 1, &vk_fence, VK_TRUE, static_cast<uint64_t>(1e3));
                    if (res == VK_SUCCESS) { break; }
                    else if (res == VK_ERROR_DEVICE_LOST) {
                        throw_vk_error(VK_ERROR_DEVICE_LOST, "Lost device while waiting for network to complete");
                    }
                }
                if (res != VK_SUCCESS) { throw_vk_error(res, "Timed out waiting for network to complete"); }

                // ************************
                // *** RETRIEVE RESULTS ***
                // ************************

                // Read the pixels off the buffer
                std::vector<vec2<Scalar>> pixels(neighbourhood.size() - 1);
                operation::map_memory<void>(context, 0, VK_WHOLE_SIZE, vk_pixels.second, [&pixels](void* payload) {
                    std::memcpy(pixels.data(), payload, pixels.size() * sizeof(vec2<Scalar>));
                });

                // Read the classifications off the device (they'll be in input)
                std::vector<Scalar> classifications(neighbourhood.size() * conv_layers.back().second);
                operation::map_memory<void>(
                  context, 0, VK_WHOLE_SIZE, vk_conv_input.second, [&classifications](void* payload) {
                      std::memcpy(classifications.data(), payload, classifications.size() * sizeof(Scalar));
                  });

                return ClassifiedMesh<Scalar, N_NEIGHBOURS>{
                  std::move(pixels), std::move(neighbourhood), std::move(indices), std::move(classifications)};
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
                image_memory.memory           = std::make_pair(nullptr, nullptr);
                image_memory.dimensions       = {0, 0};
                image_memory.format           = VK_FORMAT_UNDEFINED;
                neighbourhood_memory.memory   = std::make_pair(nullptr, nullptr);
                neighbourhood_memory.max_size = 0;
                network_memory.memory         = {std::make_pair(nullptr, nullptr), std::make_pair(nullptr, nullptr)};
                network_memory.max_size       = 0;
                indices_memory.max_size       = 0;
                indices_memory.memory         = std::make_pair(nullptr, nullptr);
            }

        private:
            template <template <typename> class Model, typename CheckpointType>
            std::tuple<std::vector<std::array<int, Model<Scalar>::N_NEIGHBOURS>>,
                       std::vector<int>,
                       std::pair<vk::buffer, vk::device_memory>,
                       CheckpointType>
              do_project(const Mesh<Scalar, Model>& mesh, const mat4<Scalar>& Hoc, const Lens<Scalar>& lens) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                // We only support VkSemaphore and VkFence here.
                // Use vk::fence if the CPU will be waiting for the signal.
                // Use vk::semaphore if the GPU will be waiting for the signal
                static_assert(
                  std::is_same<CheckpointType, vk::semaphore>::value || std::is_same<CheckpointType, vk::fence>::value,
                  "Unknown checkpoint type. Must be one of VkFence or VkSemaphore");

                // Lookup the on screen ranges
                auto ranges = mesh.lookup(Hoc, lens);

                // Transfer Rco to the device
                reprojection_buffers["vk_rco"] =
                  operation::create_buffer(context,
                                           sizeof(vec4<Scalar>) * sizeof(Scalar),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                operation::map_memory<vec4<Scalar>>(
                  context, 0, VK_WHOLE_SIZE, reprojection_buffers["vk_rco"].second, [&Hoc](vec4<Scalar>* payload) {
                      for (size_t index = 0; index < 3; ++index) {
                          payload[index] = vec4<Scalar>{Hoc[0][index], Hoc[1][index], Hoc[2][index], Scalar(0)};
                      }
                      payload[4] = vec4<Scalar>{Scalar(0), Scalar(0), Scalar(0), Scalar(1)};
                  });
                operation::bind_buffer(
                  context, reprojection_buffers["vk_rco"].first, reprojection_buffers["vk_rco"].second, 0);

                // Transfer f to the device
                reprojection_buffers["vk_f"] =
                  operation::create_buffer(context,
                                           sizeof(Scalar),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                operation::map_memory<Scalar>(
                  context, 0, VK_WHOLE_SIZE, reprojection_buffers["vk_f"].second, [&lens](Scalar* payload) {
                      payload[0] = lens.focal_length;
                  });
                operation::bind_buffer(
                  context, reprojection_buffers["vk_f"].first, reprojection_buffers["vk_f"].second, 0);

                // Transfer centre to the device
                reprojection_buffers["vk_centre"] =
                  operation::create_buffer(context,
                                           sizeof(vec2<Scalar>),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                operation::map_memory<vec2<Scalar>>(
                  context, 0, VK_WHOLE_SIZE, reprojection_buffers["vk_centre"].second, [&lens](vec2<Scalar>* payload) {
                      payload[0] = lens.centre;
                  });
                operation::bind_buffer(
                  context, reprojection_buffers["vk_centre"].first, reprojection_buffers["vk_centre"].second, 0);

                // Calculate the coefficients for performing a distortion to give to the engine
                vec4<Scalar> ik = inverse_coefficients(lens.k);

                // Transfer k to the device
                reprojection_buffers["vk_k"] =
                  operation::create_buffer(context,
                                           sizeof(vec4<Scalar>),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                operation::map_memory<vec4<Scalar>>(
                  context, 0, VK_WHOLE_SIZE, reprojection_buffers["vk_k"].second, [&ik](vec4<Scalar>* payload) {
                      payload[0] = ik;
                  });
                operation::bind_buffer(
                  context, reprojection_buffers["vk_k"].first, reprojection_buffers["vk_k"].second, 0);

                // Transfer dimensions to the device
                reprojection_buffers["vk_dimensions"] =
                  operation::create_buffer(context,
                                           sizeof(vec2<int>),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                operation::map_memory<vec2<int>>(
                  context, 0, VK_WHOLE_SIZE, reprojection_buffers["vk_dimensions"].second, [&lens](vec2<int>* payload) {
                      payload[0] = lens.dimensions;
                  });
                operation::bind_buffer(context,
                                       reprojection_buffers["vk_dimensions"].first,
                                       reprojection_buffers["vk_dimensions"].second,
                                       0);

                // Convenience variables
                const auto& nodes = mesh.nodes;

                // Upload our visual mesh unit vectors if we have to
                std::pair<vk::buffer, vk::device_memory> vk_points;

                auto device_mesh = device_points_cache.find(&mesh);
                if (device_mesh == device_points_cache.end()) {
                    vk_points = operation::create_buffer(
                      context,
                      sizeof(vec4<Scalar>) * mesh.nodes.size(),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_SHARING_MODE_EXCLUSIVE,
                      {context.transfer_queue_family},
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                    // Write the points buffer to the device and cache it
                    operation::map_memory<vec4<Scalar>>(
                      context, 0, VK_WHOLE_SIZE, vk_points.second, [&mesh](vec4<Scalar>* payload) {
                          size_t index = 0;
                          for (const auto& n : mesh.nodes) {
                              payload[index] =
                                vec4<Scalar>{Scalar(n.ray[0]), Scalar(n.ray[1]), Scalar(n.ray[2]), Scalar(0)};
                              index++;
                          }
                      });
                    operation::bind_buffer(context, vk_points.first, vk_points.second, 0);

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
                                           std::pair<vk::buffer, vk::device_memory>(),
                                           CheckpointType());
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
                std::pair<vk::buffer, vk::device_memory> vk_indices = get_indices_memory(points);

                // Upload our indices map
                operation::map_memory<void>(
                  context, 0, points * sizeof(int), vk_indices.second, [&indices](void* payload) {
                      std::memcpy(payload, indices.data(), indices.size() * sizeof(int));
                  });

                // Create output buffer for pixel_coordinates
                std::pair<vk::buffer, vk::device_memory> vk_pixels =
                  operation::create_buffer(context,
                                           sizeof(vec2<Scalar>) * points,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE,
                                           {context.transfer_queue_family},
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                operation::bind_buffer(context, vk_pixels.first, vk_pixels.second, 0);

                // --------------------------------------------------
                // At this point the point and the indices should be
                // uploaded since both device memories are coherent
                // --------------------------------------------------

                VkPipeline reprojection_pipeline;

                // Select a projection kernel
                switch (lens.projection) {
                    case RECTILINEAR: reprojection_pipeline = project_rectilinear; break;
                    case EQUIDISTANT: reprojection_pipeline = project_equidistant; break;
                    case EQUISOLID: reprojection_pipeline = project_equisolid; break;
                    default:
                        throw_vk_error(VK_ERROR_FORMAT_NOT_SUPPORTED,
                                       "Requested lens projection is not currently supported.");
                        return std::make_tuple(std::vector<std::array<int, N_NEIGHBOURS>>(),
                                               std::vector<int>(),
                                               std::pair<vk::buffer, vk::device_memory>(),
                                               CheckpointType());
                }

                // Create a descriptor pool
                // Descriptor Set 0: {points_ptr, indices_ptr, Rco_ptr, f_ptr, centre_ptr, k_ptr, dimensions_ptr,
                // out_ptr}
                std::vector<VkDescriptorPoolSize> descriptor_pool_size = {
                  VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}};
                reprojection_descriptor_pool = operation::create_descriptor_pool(context, descriptor_pool_size);

                // Allocate the descriptor set
                VkDescriptorSet descriptor_set =
                  operation::create_descriptor_set(
                    context, reprojection_descriptor_pool, {reprojection_descriptor_layout})
                    .back();

                // Load the arguments
                std::array<VkDescriptorBufferInfo, 8> buffer_infos = {
                  VkDescriptorBufferInfo{vk_points.first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{vk_indices.first, 0, points * sizeof(int)},
                  VkDescriptorBufferInfo{reprojection_buffers["vk_rco"].first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{reprojection_buffers["vk_f"].first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{reprojection_buffers["vk_centre"].first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{reprojection_buffers["vk_k"].first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{reprojection_buffers["vk_dimensions"].first, 0, VK_WHOLE_SIZE},
                  VkDescriptorBufferInfo{vk_pixels.first, 0, VK_WHOLE_SIZE},
                };
                std::array<VkWriteDescriptorSet, 8> write_descriptors;
                for (size_t i = 0; i < buffer_infos.size(); ++i) {
                    write_descriptors[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                            nullptr,
                                            descriptor_set,
                                            static_cast<uint32_t>(i),
                                            0,
                                            1,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            nullptr,
                                            &buffer_infos[i],
                                            nullptr};
                }

                vkUpdateDescriptorSets(context.device, write_descriptors.size(), write_descriptors.data(), 0, nullptr);

                // Project!
                reprojection_command_buffer =
                  operation::create_command_buffer(context, context.compute_command_pool, true);

                vkCmdBindPipeline(reprojection_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, reprojection_pipeline);

                vkCmdBindDescriptorSets(reprojection_command_buffer,
                                        VK_PIPELINE_BIND_POINT_COMPUTE,
                                        reprojection_pipeline_layout,
                                        0,
                                        1,
                                        &descriptor_set,
                                        0,
                                        nullptr);

                vkCmdDispatch(reprojection_command_buffer, static_cast<int32_t>(points), 1, 1);

                CheckpointType checkpoint;
                static_if<std::is_same<CheckpointType, vk::semaphore>::value>([&](auto f) {
                    VkSemaphore semaphore;
                    VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
                    throw_vk_error(vkCreateSemaphore(context.device, &semaphore_info, nullptr, &semaphore),
                                   "Failed to create reprojection semaphore");
                    f(checkpoint) =
                      vk::semaphore(semaphore, [this](auto p) { vkDestroySemaphore(context.device, p, nullptr); });
                    operation::submit_command_buffer(
                      context.compute_queue, reprojection_command_buffer, {}, {f(checkpoint)});
                }).else_([&](auto f) {
                    VkFence fence;
                    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
                    throw_vk_error(vkCreateFence(context.device, &fence_info, nullptr, &fence),
                                   "Failed to create reprojection semaphore");
                    f(checkpoint) = vk::fence(fence, [this](auto p) { vkDestroyFence(context.device, p, nullptr); });
                    operation::submit_command_buffer(context.compute_queue, reprojection_command_buffer, f(checkpoint));
                });

                // This can happen on the CPU while the Vulkan device is busy
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
                                       std::move(vk_pixels),            // Projected pixels
                                       std::move(checkpoint));          // Checkpoint to wait on
            }

            uint32_t get_image_depth(const uint32_t& format) const {
                switch (format) {
                    // Bayer
                    case fourcc("GRBG"):
                    case fourcc("RGGB"):
                    case fourcc("GBRG"):
                    case fourcc("BGGR"): return 1;
                    case fourcc("BGRA"):
                    case fourcc("RGBA"): return 4;
                    // Oh no...
                    default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
                }
                return 0;
            }

            VkFormat get_image_format(const uint32_t& format) const {
                switch (format) {
                    // Bayer
                    case fourcc("GRBG"):
                    case fourcc("RGGB"):
                    case fourcc("GBRG"):
                    case fourcc("BGGR"): return VK_FORMAT_R8_UNORM;
                    case fourcc("BGRA"): return VK_FORMAT_B8G8R8A8_UNORM;
                    case fourcc("RGBA"): return VK_FORMAT_R8G8B8A8_UNORM;
                    // Oh no...
                    default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
                }
                return VK_FORMAT_UNDEFINED;
            }

            vk::sampler get_image_sampler(const uint32_t& format) const {
                switch (format) {
                    // Bayer
                    case fourcc("GRBG"):
                    case fourcc("RGGB"):
                    case fourcc("GBRG"):
                    case fourcc("BGGR"): return bayer_sampler;
                    case fourcc("BGRA"):
                    case fourcc("RGBA"): return interp_sampler;
                    // Oh no...
                    default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
                }
                return vk::sampler();
            }

            vk::image_view get_image_view(const vk::image& image, const uint32_t& format) const {
                // TODO: Can we use the swizzling here to map RGBA <-> BGRA?
                VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   image,
                                                   VK_IMAGE_VIEW_TYPE_2D,
                                                   get_image_format(format),
                                                   {VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY,
                                                    VK_COMPONENT_SWIZZLE_IDENTITY},
                                                   {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

                if (format == fourcc("BGRA")) {
                    view_info.components.r = VK_COMPONENT_SWIZZLE_B;
                    view_info.components.b = VK_COMPONENT_SWIZZLE_R;
                }

                VkImageView img_view;
                throw_vk_error(vkCreateImageView(context.device, &view_info, nullptr, &img_view),
                               "Failed to create image view");
                return vk::image_view(img_view, [this](auto p) { vkDestroyImageView(context.device, p, nullptr); });
            }

            VkPipeline get_image_pipeline(const uint32_t& format) const {
                switch (format) {
                    // Bayer
                    case fourcc("GRBG"): return load_GRBG_image;
                    case fourcc("RGGB"): return load_RGGB_image;
                    case fourcc("GBRG"): return load_GBRG_image;
                    case fourcc("BGGR"): return load_BGGR_image;
                    case fourcc("BGRA"):
                    case fourcc("RGBA"): return load_RGBA_image;
                    // Oh no...
                    default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
                }
                return VkPipeline();
            }

            std::pair<vk::image, vk::device_memory> get_image_memory(const vec2<int>& dimensions,
                                                                     const uint32_t& format) const {

                // If our dimensions and format haven't changed from last time we can reuse the same memory location
                if (dimensions != image_memory.dimensions || format != image_memory.format) {
                    VkFormat vk_format   = get_image_format(format);
                    VkExtent3D vk_extent = {
                      static_cast<uint32_t>(dimensions[0]), static_cast<uint32_t>(dimensions[1]), 1};

                    image_memory.memory =
                      operation::create_image(context,
                                              vk_extent,
                                              vk_format,
                                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                              VK_SHARING_MODE_EXCLUSIVE,
                                              {context.transfer_queue_family},
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    operation::bind_image(context, image_memory.memory.first, image_memory.memory.second, 0);

                    // Update what we are caching
                    image_memory.dimensions = dimensions;
                    image_memory.format     = vk_format;
                }

                // Return the cache
                return image_memory.memory;
            }

            std::array<std::pair<vk::buffer, vk::device_memory>, 2> get_network_memory(const int& max_size) const {
                if (network_memory.max_size < max_size) {
                    network_memory.memory[0] = operation::create_buffer(
                      context,
                      max_size * sizeof(Scalar),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_SHARING_MODE_EXCLUSIVE,
                      {context.transfer_queue_family},
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    network_memory.memory[1] = operation::create_buffer(
                      context,
                      max_size * sizeof(Scalar),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_SHARING_MODE_EXCLUSIVE,
                      {context.transfer_queue_family},
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                    network_memory.max_size = max_size;
                    operation::bind_buffer(context, network_memory.memory[0].first, network_memory.memory[0].second, 0);
                    operation::bind_buffer(context, network_memory.memory[1].first, network_memory.memory[1].second, 0);
                }
                return network_memory.memory;
            }

            std::pair<vk::buffer, vk::device_memory> get_neighbourhood_memory(const int& max_size) const {
                if (neighbourhood_memory.max_size < max_size) {
                    neighbourhood_memory.memory = operation::create_buffer(
                      context,
                      max_size * sizeof(int),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_SHARING_MODE_EXCLUSIVE,
                      {context.transfer_queue_family},
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    neighbourhood_memory.max_size = max_size;
                    operation::bind_buffer(
                      context, neighbourhood_memory.memory.first, neighbourhood_memory.memory.second, 0);
                }
                return neighbourhood_memory.memory;
            }

            std::pair<vk::buffer, vk::device_memory> get_indices_memory(const int& max_size) const {
                if (indices_memory.max_size < max_size) {
                    indices_memory.memory = operation::create_buffer(
                      context,
                      max_size * sizeof(int),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_SHARING_MODE_EXCLUSIVE,
                      {context.transfer_queue_family},
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    indices_memory.max_size = max_size;
                    operation::bind_buffer(context, indices_memory.memory.first, indices_memory.memory.second, 0);
                }
                return indices_memory.memory;
            }

            /// Vulkan instance
            VulkanContext context;

            /// DescriptorSetLayout for the reprojection kernels
            VkDescriptorSetLayout reprojection_descriptor_layout;
            /// PipelineLayout for the reprojection kernels
            VkPipelineLayout reprojection_pipeline_layout;
            /// Shader module for projecting rays to pixels using an equidistant projection
            vk::shader_module equidistant_reprojection_program;
            /// Shader module for projecting rays to pixels using an equisolid projection
            vk::shader_module equisolid_reprojection_program;
            /// Shader module for projecting rays to pixels using a rectilinear projection
            vk::shader_module rectilinear_reprojection_program;
            /// Kernel for projecting rays to pixels using an equidistant projection
            VkPipeline project_equidistant;
            /// Kernel for projecting rays to pixels using an equisolid projection
            VkPipeline project_equisolid;
            /// Kernel for projecting rays to pixels using a rectilinear projection
            VkPipeline project_rectilinear;
            /// Reusable descriptor pool for the reprojection pipelines
            mutable vk::descriptor_pool reprojection_descriptor_pool;
            /// Reusable command buffer for the reprojection pipelines
            mutable vk::command_buffer reprojection_command_buffer;
            /// Memory buffers for the reprojection pipelines
            mutable std::map<std::string, std::pair<vk::buffer, vk::device_memory>> reprojection_buffers;

            /// DescriptorSetLayouts for the load_image kernel
            VkDescriptorSetLayout load_image_descriptor_layout;
            /// PipelineLayout for the load_image kernel
            VkPipelineLayout load_image_pipeline_layout;
            /// Samplers for the load image kernels
            vk::sampler bayer_sampler;
            vk::sampler interp_sampler;
            /// Shader modules for reading projected pixel coordinates from an image into the network input layer
            vk::shader_module load_GRBG_image_program;
            vk::shader_module load_RGGB_image_program;
            vk::shader_module load_GBRG_image_program;
            vk::shader_module load_BGGR_image_program;
            vk::shader_module load_RGBA_image_program;
            /// Kernel for reading projected pixel coordinates from an image into the network input layer
            VkPipeline load_GRBG_image;
            VkPipeline load_RGGB_image;
            VkPipeline load_GBRG_image;
            VkPipeline load_BGGR_image;
            VkPipeline load_RGBA_image;

            /// DescriptorSetLayout for the conv kernels
            VkDescriptorSetLayout conv_descriptor_layout;
            /// PipelineLayout for the conv kernels
            VkPipelineLayout conv_pipeline_layout;
            /// Shader module for the network
            std::vector<vk::shader_module> conv_program;
            /// A list of kernels to run in sequence to run the network
            std::vector<std::pair<VkPipeline, size_t>> conv_layers;

            mutable struct {
                vec2<int> dimensions = {0, 0};
                VkFormat format      = VK_FORMAT_UNDEFINED;
                std::pair<vk::image, vk::device_memory> memory;
            } image_memory;

            mutable struct {
                int max_size = 0;
                std::array<std::pair<vk::buffer, vk::device_memory>, 2> memory;
            } network_memory;

            mutable struct {
                int max_size = 0;
                std::pair<vk::buffer, vk::device_memory> memory;
            } neighbourhood_memory;

            mutable struct {
                int max_size = 0;
                std::pair<vk::buffer, vk::device_memory> memory;
            } indices_memory;

            // The width of the maximumally wide layer in the network
            size_t max_width;

            // Cache of Vulkan buffers from mesh objects
            mutable std::map<const void*, std::pair<vk::buffer, vk::device_memory>> device_points_cache;
        };

    }  // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // !defined(VISUALMESH_DISABLE_VULKAN)
#endif  // VISUALMESH_ENGINE_VULKAN_ENGINE_HPP
