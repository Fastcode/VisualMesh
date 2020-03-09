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

#ifndef VISUALMESH_VULKAN_KERNELS_MAKE_NETWORK_HPP
#define VISUALMESH_VULKAN_KERNELS_MAKE_NETWORK_HPP

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// #include "engine/vulkan/operation/wrapper.hpp"
#include "mesh/network_structure.hpp"
#include "utility/vulkan_compute.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace kernels {

            /**
             * @brief Given a network structure object generate the SPIRV source code for the kernels needed to execute
             * it
             *
             * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
             *
             * @param structure the network structure to generate the kernels from
             *
             * @return the SPIRV source code for the kernels to be built
             */
            template <typename Scalar>
            std::vector<std::pair<uint32_t, std::vector<uint32_t>>> make_network(
              const network_structure_t<Scalar>& structure) {
                std::vector<std::pair<uint32_t, std::vector<uint32_t>>> programs;

                // If our structure has no layers, return empty code
                if (structure.empty() || structure.front().empty()) { return programs; }

                // Keep track of the input and output size of each layer for building the network
                // The first layer input is always 4 from the image
                uint32_t input_dimensions  = 4;
                uint32_t output_dimensions = 0;

                // First layer has 4 inputs, so that tells us how many neighbours we have (minus ourself)
                const uint32_t n_neighbours = (structure.front().front().first.size() / 4) - 1;

                for (uint32_t conv_no = 0; conv_no < structure.size(); ++conv_no) {
                    auto& conv = structure[conv_no];

                    // Initialise the program.
                    Program::Config config;
                    config.enable_glsl_extensions = true;
                    config.enable_float64         = ((sizeof(Scalar) == 8) && std::is_floating_point<Scalar>::value);
                    config.address_model          = spv::AddressingModel::Logical;
                    config.memory_model           = spv::MemoryModel::GLSL450;
                    config.enable_debug           = true;
                    Program program(config);

                    uint32_t uint_type  = program.add_type(spv::Op::OpTypeInt, {32, 0});
                    uint32_t float_type = program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)});
                    uint32_t uvec3      = program.add_vec_type(spv::Op::OpTypeInt, {32, 0}, 3);

                    uint32_t uint_ptr       = program.add_pointer(uint_type, spv::StorageClass::Input);
                    uint32_t uvec3_ptr      = program.add_pointer(uvec3, spv::StorageClass::Input);
                    uint32_t uint_ptr_sb    = program.add_pointer(uint_type, spv::StorageClass::StorageBuffer);
                    uint32_t float_ptr      = program.add_pointer(float_type, spv::StorageClass::StorageBuffer);
                    uint32_t float_ptr_func = program.add_pointer(float_type, spv::StorageClass::Function);

                    // Define the GlobalInvocationID (for get_global_id(0))
                    uint32_t global_id = program.add_variable(uvec3_ptr, spv::StorageClass::Input);
                    program.add_builtin_decoration(global_id, spv::BuiltIn::GlobalInvocationId);

                    // Prepare the input/output/descriptor set variables
                    uint32_t neighbourhood_array  = program.add_array_type(uint_type);
                    uint32_t neighbourhood_struct = program.add_struct({neighbourhood_array});
                    uint32_t neighbourhood_ptr =
                      program.add_variable(program.add_pointer(neighbourhood_struct, spv::StorageClass::StorageBuffer),
                                           spv::StorageClass::StorageBuffer);

                    // The input layer is coming from the image, so the input is an fvec4
                    uint32_t input_array  = program.add_array_type(float_type);
                    uint32_t input_struct = program.add_struct({input_array});
                    uint32_t input_ptr =
                      program.add_variable(program.add_pointer(input_struct, spv::StorageClass::StorageBuffer),
                                           spv::StorageClass::StorageBuffer);

                    uint32_t output_array  = program.add_array_type(float_type);
                    uint32_t output_struct = program.add_struct({output_array});
                    uint32_t output_ptr =
                      program.add_variable(program.add_pointer(output_struct, spv::StorageClass::StorageBuffer),
                                           spv::StorageClass::StorageBuffer);

                    // Decorate the structs and their members.
                    uint32_t block_decoration = program.add_decoration_group(spv::Decoration::Block);
                    program.add_group_decoration(block_decoration, {neighbourhood_struct, input_struct, output_struct});
                    program.add_member_decoration(neighbourhood_struct, 0, spv::Decoration::Offset, {0});
                    program.add_member_decoration(input_struct, 0, spv::Decoration::Offset, {0});
                    program.add_member_decoration(output_struct, 0, spv::Decoration::Offset, {0});

                    program.add_decoration(neighbourhood_array, spv::Decoration::ArrayStride, {4});
                    program.add_decoration(input_array, spv::Decoration::ArrayStride, {sizeof(Scalar)});
                    program.add_decoration(output_array, spv::Decoration::ArrayStride, {sizeof(Scalar)});

                    // Set up the descriptor set for all convolutional kernels
                    // Descriptor Set 0: {neighbourhood_ptr, input_ptr, output_ptr}
                    program.create_descriptor_set({neighbourhood_ptr, input_ptr, output_ptr});

                    // Index 0 is used in every member_access call
                    uint32_t idx0 = program.add_constant(uint_type, {0u});

                    // Write our Vulkan kernel definition
                    program.begin_entry_point("conv" + std::to_string(conv_no), {global_id});
                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // Pre-allocate all arrays
                    // These array variables must be the first things in the function
                    std::vector<uint32_t> layers;
                    layers.push_back(program.add_variable(
                      program.add_pointer(
                        program.add_array_type(
                          float_type, program.add_constant(uint_type, {input_dimensions * (n_neighbours + 1u)})),
                        spv::StorageClass::Function),
                      spv::StorageClass::Function));

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    for (uint32_t layer_no = 0; layer_no < conv.size(); ++layer_no) {
                        layers.push_back(program.add_variable(
                          program.add_pointer(
                            program.add_array_type(
                              float_type,
                              program.add_constant(uint_type, {static_cast<uint32_t>(conv[layer_no].second.size())})),
                            spv::StorageClass::Function),
                          spv::StorageClass::Function));
                    }

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // Get our kernel index
                    // idx = get_global_id(0);
                    program.add_source_line(__FILE__, __LINE__, conv_no);
                    uint32_t idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                    program.add_name(idx, "get_global_id");

                    /*************************************************
                     *                    GATHER                     *
                     *************************************************/

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // idx * n_neighbours is used a lot, precalculate it
                    program.add_source_line(__FILE__, __LINE__, conv_no);
                    uint32_t idx_dim = program.add_name(
                      program.imul(idx, program.add_constant(uint_type, {input_dimensions}), uint_type),
                      "idx_times_n_neighbours");
                    uint32_t idx_neighbours =
                      program.add_name(program.imul(idx, program.add_constant(uint_type, {n_neighbours}), uint_type),
                                       "idx_times_n_neighbours");

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // Read the ones for our own index
                    for (uint32_t j = 0; j < input_dimensions; ++j) {
                        // input[idx * input_dimensions + j]
                        uint32_t input_val = program.load_variable(
                          program.member_access(
                            input_ptr,
                            // idx0 = offset to struct member (only one member in struct, its at offset 0)
                            {idx0,
                             // idx * input_dimensions + j
                             program.iadd(idx_dim, program.add_constant(uint_type, {j}), uint_type)},
                            float_ptr),
                          float_type);

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        program.store_variable(
                          program.member_access(layers[0], {program.add_constant(uint_type, {j})}, float_ptr_func),
                          input_val);

                        program.add_source_line(__FILE__, __LINE__, conv_no);
                    }

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // Read our neighbourhood
                    for (uint32_t i = 0; i < n_neighbours; ++i) {
                        // neighbour_idx is used in every iteration of the sub loop
                        // neighbourhood[idx * n_neighbours + i]
                        uint32_t neighbour_idx = program.load_variable(
                          program.member_access(
                            neighbourhood_ptr,
                            // idx0 = offset to struct member (only one member in struct, its at offset 0)
                            // idx * n_neighbours + i
                            {idx0, program.iadd(idx_neighbours, program.add_constant(uint_type, {i}), uint_type)},
                            uint_ptr_sb),
                          uint_type);

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        // neighbour_input_idx is used in every iteration of the sub loop
                        // neighbourhood[idx * 6 + i] * conv_in_size
                        uint32_t neighbour_input_idx =
                          program.imul(neighbour_idx, program.add_constant(uint_type, {input_dimensions}), uint_type);

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        for (uint32_t j = 0; j < input_dimensions; ++j) {
                            // input[neighbourhood[idx * n_neighbours + i] * input_dimensions + j]
                            uint32_t neighbour_val = program.load_variable(
                              program.member_access(
                                input_ptr,
                                // idx0 = offset to struct member (only one member in struct, its at offset 0)
                                {idx0,
                                 // neighbourhood[idx * 6 + i] * conv_in_size + j
                                 program.iadd(neighbour_input_idx, program.add_constant(uint_type, {j}), uint_type)},
                                float_ptr),
                              float_type);

                            program.add_source_line(__FILE__, __LINE__, conv_no);

                            program.store_variable(
                              program.member_access(
                                layers[0],
                                {program.add_constant(uint_type, {(i + 1u) * input_dimensions + j})},
                                float_ptr_func),
                              neighbour_val);

                            program.add_source_line(__FILE__, __LINE__, conv_no);
                        }

                        program.add_source_line(__FILE__, __LINE__, conv_no);
                    }

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // We have gathered which increased the size of the input
                    input_dimensions = input_dimensions * (n_neighbours + 1);

                    // selu constants
                    uint32_t lambda = program.add_name(
                      program.add_constant(float_type, {Scalar(1.0507009873554804934193349852946)}), "lambda");
                    uint32_t alpha = program.add_name(
                      program.add_constant(float_type, {Scalar(1.6732632423543772848170429916717)}), "alpha");

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // Now we have to do our layer operations
                    for (uint32_t layer_no = 0; layer_no < conv.size(); ++layer_no) {
                        const auto& weights = conv[layer_no].first;
                        const auto& biases  = conv[layer_no].second;

                        output_dimensions = biases.size();

                        /*************************************************
                         *                WEIGHTS + BIAS                 *
                         *************************************************/

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        // Perform our matrix multiplication for weights and add bias for layer
                        for (uint32_t i = 0; i < output_dimensions; ++i) {
                            uint32_t total_val = program.add_constant(float_type, {Scalar(0)});
                            for (uint32_t j = 0; j < input_dimensions; ++j) {
                                uint32_t current_val = program.load_variable(
                                  program.member_access(
                                      layers[layer_no], {program.add_constant(uint_type, {j})}, float_ptr_func),
                                  float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                current_val = program.fmul(
                                  current_val, program.add_constant(float_type, {weights[j][i]}), float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                total_val = program.fadd(total_val, current_val, float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                program.store_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  total_val);

                                program.add_source_line(__FILE__, __LINE__, conv_no);
                            }
                        }

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        /*************************************************
                         *                  ACTIVATION.                  *
                         *************************************************/

                        // Apply selu
                        if (conv_no + 1 < structure.size() || layer_no + 1 < conv.size()) {
                            program.add_source_line(__FILE__, __LINE__, conv_no);

                            for (uint32_t i = 0; i < output_dimensions; ++i) {
                                // in1[i] = lambda * (in1[i] > 0 ? in1[i] : alpha * exp(in1[i]) - alpha;
                                uint32_t current_val = program.load_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                // selu = alpha * exp(in1[i]) - alpha
                                uint32_t selu =
                                  program.fsub(program.fmul(alpha, program.exp(current_val, float_type), float_type),
                                               alpha,
                                               float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                // in1[i] = lambda * (in1[i] > 0 ? in1[i] : selu)
                                uint32_t condition =
                                  program.fgeq(current_val, program.add_constant(float_type, {Scalar(0)}));
                                current_val = program.fmul(
                                  lambda, program.select(float_type, condition, current_val, selu), float_type);
                                program.store_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  current_val);

                                program.add_source_line(__FILE__, __LINE__, conv_no);
                            }

                            program.add_source_line(__FILE__, __LINE__, conv_no);
                        }
                        // If this is our last layer, apply softmax
                        else {
                            program.add_source_line(__FILE__, __LINE__, conv_no);

                            // Apply exp to each of the elements
                            for (uint32_t i = 0; i < output_dimensions; ++i) {
                                // in1[i] = exp(in1[i])
                                uint32_t current_val = program.load_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                current_val = program.exp(current_val, float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                program.store_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  current_val);

                                program.add_source_line(__FILE__, __LINE__, conv_no);
                            }

                            program.add_source_line(__FILE__, __LINE__, conv_no);

                            // Sum up all the values
                            uint32_t exp_sum = program.add_constant(float_type, {Scalar(0)});
                            for (uint32_t i = 0; i < output_dimensions; ++i) {
                                // exp_sum += in1[i]
                                uint32_t current_val = program.load_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                exp_sum = program.fadd(exp_sum, current_val, float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);
                            }

                            program.add_source_line(__FILE__, __LINE__, conv_no);

                            // Divide all the values
                            for (uint32_t i = 0; i < output_dimensions; ++i) {
                                // in1[i] /= exp_sum
                                uint32_t current_val = program.load_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                current_val = program.fdiv(current_val, exp_sum, float_type);

                                program.add_source_line(__FILE__, __LINE__, conv_no);

                                program.store_variable(
                                  program.member_access(
                                    layers[layer_no + 1], {program.add_constant(uint_type, {i})}, float_ptr_func),
                                  current_val);

                                program.add_source_line(__FILE__, __LINE__, conv_no);
                            }
                        }

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        // Update our input size for the next loop
                        input_dimensions = output_dimensions;
                    }

                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    /*************************************************
                     *                    OUTPUT                     *
                     *************************************************/
                    // Save our value to the output
                    uint32_t output_idx = program.add_constant(uint_type, {idx * input_dimensions});
                    for (unsigned int i = 0; i < input_dimensions; ++i) {
                        // output[idx * input_dimensions + i] = inN[i]
                        uint32_t current_val = program.load_variable(
                          program.member_access(layers.back(), {program.add_constant(uint_type, {i})}, float_ptr_func),
                          float_type);

                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        program.store_variable(
                          program.member_access(
                            output_ptr,
                            {idx0, program.iadd(output_idx, program.add_constant(uint_type, {i}), uint_type)},
                            float_ptr),
                          current_val);

                        program.add_source_line(__FILE__, __LINE__, conv_no);
                    }

                    // Update our input dimensions for the next round
                    input_dimensions = output_dimensions;

                    program.return_function();
                    program.end_function();
                    programs.emplace_back(conv_no, program.build());
                }

                return programs;
            }  // namespace kernels

        }  // namespace kernels
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMEVULKANNCL_KERNELS_MAKE_NETWORK_HPP
