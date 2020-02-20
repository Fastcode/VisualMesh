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

#ifndef VISUALMESH_VULKAN_OPERATION_MAKE_NETWORK_HPP
#define VISUALMESH_VULKAN_OPERATION_MAKE_NETWORK_HPP

#include <iostream>
#include <utility>
#include <vector>

#include "mesh/network_structure.hpp"
#include "utility/vulkan_compute.hpp"
#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

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
            std::vector<uint32_t> make_network(const network_structure_t<Scalar>& structure) {
                // Initialise the program.
                Program::Config config;
                config.enable_glsl_extensions = true;
                config.enable_float64         = ((sizeof(Scalar) == 8) && std::is_floating_point<Scalar>::value);
                config.address_model          = spv::AddressingModel::Logical;
                config.memory_model           = spv::MemoryModel::GLSL450;
                config.enable_debug           = false;
                Program program(config);

                // Small utility to figure out if we are using a defined vector size
                auto vector_type = [](const uint32_t& size) -> bool { return (size == 2 || size == 3 || size == 4); };

                uint32_t void_type  = program.add_type(spv::Op::OpTypeVoid, {});
                uint32_t uint_type  = program.add_type(spv::Op::OpTypeInt, {32, 0});
                uint32_t float_type = program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)});
                uint32_t uvec3      = program.add_vec_type(spv::Op::OpTypeInt, {32, 0}, 3);
                uint32_t fvec4      = program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, 4);

                uint32_t uint_ptr  = program.add_pointer(uint_type, spv::StorageClass::Input);
                uint32_t uvec3_ptr = program.add_pointer(uvec3, spv::StorageClass::Input);
                uint32_t fvec4_ptr = program.add_pointer(fvec4, spv::StorageClass::StorageBuffer);

                for (uint32_t conv_no = 0; conv_no < network.size(); ++conv_no) {
                    auto& conv = network[conv_no];

                    // We need to work out the input and output sizes for our convolution
                    uint32_t conv_in_size;
                    uint32_t conv_out_size;

                    // On the first convolution we assume an input size of 4
                    if (conv_no == 0) { conv_in_size = 4; }
                    else {
                        // The output dimension of our previous bias vector
                        conv_in_size = network[conv_no - 1].back().second.size();
                    }

                    // The output dimension of our last bias vector
                    conv_out_size = conv.back().second.size();

                    // Define the GlobalInvocationID (for get_global_id(0))
                    uint32_t global_id = program.add_variable(uvec3_ptr, spv::StorageClass::Input);
                    program.add_builtin_decoration(global_id, spv::BuiltIn::GlobalInvocationId);

                    // Prepare the points, indices, Rco, f, dimensions, centre, and out variables
                    uint32_t neighbourhood_array  = program.add_array_type(uint_type);
                    uint32_t neighbourhood_struct = program.add_struct({neighbourhood_array});
                    uint32_t neighbourhood_ptr =
                      program.add_variable(program.add_pointer(neighbourhood_struct, spv::StorageClass::StorageBuffer),
                                           spv::StorageClass::StorageBuffer);

                    // The input layer is coming from the image, so the input is an fvec4
                    uint32_t input_layer_type = fvec4;
                    uint32_t input_array      = program.add_array_type(input_layer_type);
                    uint32_t input_struct     = program.add_struct({input_array});
                    uint32_t input_ptr =
                      program.add_variable(program.add_pointer(input_struct, spv::StorageClass::StorageBuffer),
                                           spv::StorageClass::StorageBuffer);

                    uint32_t output_layer_type;
                    if (vector_type(conv_out_size)) {
                        output_layer_type = program.add_name(
                          program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, conv_out_size),
                          fmt::format("float{}", conv_out_size));
                    }
                    else {
                        output_layer_type =
                          program.add_name(program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}),
                                           fmt::format("float{}", conv_out_size));
                    }
                    uint32_t output_array  = program.add_name(program.add_array_type(output_layer_type),
                                                             fmt::format("float{}_array", conv_out_size));
                    uint32_t output_struct = program.add_name(program.add_struct({output_array}),
                                                              fmt::format("struct_float{}_array", conv_out_size));
                    uint32_t output_ptr    = program.add_name(
                      program.add_variable(program.add_pointer(output_struct, spv::StorageClass::StorageBuffer),
                                           spv::StorageClass::StorageBuffer),
                      fmt::format("struct_float{}_array_ptr", conv_out_size));

                    // Decorate the structs and their members.
                    uint32_t block_decoration = program.add_decoration_group(spv::Decoration::Block);
                    program.add_group_decoration(block_decoration, {neighbourhood_struct, input_struct, output_struct});

                    program.add_member_decoration(neighbourhood_struct, 0, spv::Decoration::Offset, {0});
                    program.add_member_decoration(input_struct, 0, spv::Decoration::Offset, {0});
                    program.add_member_decoration(output_struct, 0, spv::Decoration::Offset, {0});

                    uint32_t stride16_decoration = program.add_decoration_group(spv::Decoration::ArrayStride, {16});
                    program.add_group_decoration(stride16_decoration, {input_array, output_array});
                    program.add_decoration(neighbourhood_array, spv::Decoration::ArrayStride, {4});

                    // Set up the descriptor set for all convolutional kernels
                    // Descriptor Set 0: {neighbourhood_ptr, input_ptr, output_ptr}
                    program.create_descriptor_set({neighbourhood_ptr, input_ptr, output_ptr});

                    // Index 0 is used in every member_access call
                    uint32_t idx0 = program.add_constant(uint_type, {0u});

                    // Write our OpenCL kernel definition
                    program.begin_entry_point(fmt::format("conv{}", conv_no), {global_id});
                    program.add_source_line(__FILE__, __LINE__, conv_no);

                    // Array storing the gathered data
                    // This need to be one of the first things in the function
                    program.add_source_line(__FILE__, __LINE__, conv_no);
                    uint32_t in0 = program.add_variable(
                      program.add_pointer(
                        program.add_array_type(input_layer_type, program.add_constant(uint_type, {7u})),
                        spv::StorageClass::Function),
                      spv::StorageClass::Function);

                    // Get our kernel index
                    // idx = get_global_id(0);
                    program.add_source_line(__FILE__, __LINE__, conv_no);
                    uint32_t idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                    program.add_name(idx, "get_global_id");

                    // idx * 6 is used a lot, precalculate it
                    program.add_source_line(__FILE__, __LINE__, conv_no);
                    uint32_t idx6 = program.imul(idx, program.add_constant(uint_type, {6u}), uint_type);
                    program.add_name(idx6, "idx6");

                    /*************************************************
                     *                    GATHER                     *
                     *************************************************/

                    // Gather from our neighbourhood
                    program.add_source_line(__FILE__, __LINE__, conv_no);
                    program.add_name(in0, "in_layer_type_array_7_in0");
                    if (vector_type(conv_in_size)) {
                        // float4 in0[7] = {
                        //     input[idx],
                        //     input[neighbourhood[idx * 6 + 0]],
                        //     input[neighbourhood[idx * 6 + 1]],
                        //     input[neighbourhood[idx * 6 + 2]],
                        //     input[neighbourhood[idx * 6 + 3]],
                        //     input[neighbourhood[idx * 6 + 4]],
                        //     input[neighbourhood[idx * 6 + 5]]
                        // };
                        // The current point
                        // input[idx]
                        program.add_source_line(__FILE__, __LINE__, conv_no);
                        program.store_variable(
                          program.member_access(in0,
                                                {program.add_constant(uint_type, {0u})},
                                                program.add_pointer(input_layer_type, spv::StorageClass::Function)),
                          program.load_variable(
                            program.member_access(
                              input_ptr,
                              {idx0, idx},
                              program.add_pointer(input_layer_type, spv::StorageClass::StorageBuffer)),
                            input_layer_type));

                        // The current neighbourhood
                        for (uint32_t i = 1; i < 7; ++i) {
                            // input[neighbourhood[idx * 6 + i]]
                            // input[neighbourhood[idx6 + i]]
                            program.add_source_line(__FILE__, __LINE__, conv_no);
                            program.store_variable(
                              program.member_access(in0,
                                                    {program.add_constant(uint_type, {i})},
                                                    program.add_pointer(input_layer_type, spv::StorageClass::Function)),
                              program.load_variable(
                                program.member_access(
                                  input_ptr,
                                  {idx0,
                                   program.load_variable(
                                     program.member_access(
                                       neighbourhood_ptr,
                                       // idx0 = offset to struct member (only one member in struct, its at offset 0)
                                       // idx6 + i
                                       {idx0, program.iadd(idx6, program.add_constant(uint_type, {i}), uint_type)},
                                       program.add_pointer(uint_type, spv::StorageClass::StorageBuffer)),
                                     uint_type)},
                                  program.add_pointer(input_layer_type, spv::StorageClass::StorageBuffer)),
                                input_layer_type));
                        }
                    }

                    // Perform our gather step for non vectorized data
                    else {
                        program.add_source_line(__FILE__, __LINE__, conv_no);
                        // idx * conv_in_size is used a lot, precalculate it
                        uint32_t idx_conv =
                          program.imul(idx, program.add_constant(uint_type, {conv_in_size}), uint_type);

                        // The current point
                        // Read the ones for our own index
                        for (uint32_t j = 0; j < conv_in_size; ++j) {
                            // input[idx * conv_in_size + j]
                            program.store_variable(
                              program.member_access(in0,
                                                    {program.add_constant(uint_type, {j})},
                                                    program.add_pointer(input_layer_type, spv::StorageClass::Function)),
                              program.load_variable(
                                program.member_access(
                                  input_ptr,
                                  // idx0 = offset to struct member (only one member in struct, its at offset 0)
                                  {idx0,
                                   // idx6 + j
                                   program.iadd(idx_conv, program.add_constant(uint_type, {j}), uint_type)},
                                  program.add_pointer(input_layer_type, spv::StorageClass::StorageBuffer)),
                                input_layer_type));
                        }

                        // The current points neighbourhood
                        for (uint32_t i = 0; i < 6; ++i) {
                            // neighbour_idx is used in every iteration of the sub loop
                            // neighbourhood[idx * 6 + i]
                            uint32_t neighbour_idx = program.load_variable(
                              program.member_access(
                                neighbourhood_ptr,
                                // idx0 = offset to struct member (only one member in struct, its at offset 0)
                                // idx6 + i
                                {idx0, program.iadd(idx6, program.add_constant(uint_type, {i}), uint_type)},
                                input_layer_type),
                              fvec4);

                            // neighbour_conv_idx is used in every iteration of the sub loop
                            // neighbourhood[idx * 6 + i] * conv_in_size
                            uint32_t neighbour_conv_idx =
                              program.imul(neighbour_idx, program.add_constant(uint_type, {conv_in_size}), uint_type);

                            for (uint32_t j = 0; j < conv_in_size; ++j) {
                                // input[neighbourhood[idx * 6 * + i] * conv_in_size + j]
                                program.store_variable(
                                  program.member_access(
                                    in0,
                                    {program.add_constant(uint_type, {j})},
                                    program.add_pointer(input_layer_type, spv::StorageClass::Function)),
                                  program.load_variable(
                                    program.member_access(
                                      input_ptr,
                                      // idx0 = offset to struct member (only one member in struct, its at offset 0)
                                      {idx0,
                                       // neighbourhood[idx * 6 + i] * conv_in_size + j
                                       program.iadd(
                                         neighbour_conv_idx, program.add_constant(uint_type, {j}), uint_type)},
                                      program.add_pointer(input_layer_type, spv::StorageClass::StorageBuffer)),
                                    input_layer_type));
                            }
                        }
                    }

                    // Stores result IDs for the output of the previous layer
                    std::vector<uint32_t> layers = {in0};

                    /*************************************************
                     *                WEIGHTS + BIAS                 *
                     *************************************************/

                    // Now we have to do our layer operations
                    uint32_t in_size = conv_in_size;
                    for (uint32_t layer_no = 0; layer_no < conv.size(); ++layer_no) {
                        const auto& weights = conv[layer_no].first;
                        const auto& biases  = conv[layer_no].second;

                        const uint32_t vector_in_size  = in_size;
                        const uint32_t vector_out_size = biases.size();

                        // Perform our matrix multiplication for weights and add bias for layer
                        // Open our next input (either vector or not)
                        uint32_t layer_in_type;
                        if (vector_type(vector_in_size)) {
                            layer_in_type =
                              program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, vector_in_size);
                            program.add_name(layer_in_type, fmt::format("float{}", vector_in_size));
                        }
                        else {
                            layer_in_type = program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)});
                            program.add_name(layer_in_type, "float");
                        }
                        uint32_t layer_out_type;
                        if (vector_type(vector_out_size)) {
                            layer_out_type =
                              program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, vector_out_size);
                            program.add_name(layer_out_type, fmt::format("float{}", vector_out_size));
                        }
                        else {
                            layer_out_type = program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)});
                            program.add_name(layer_out_type, "float");
                        }

                        // Transposes a vector of vectors
                        auto transpose = [](const std::vector<std::vector<Scalar>>& weights) {
                            // This assumes that all inner vectors have the same size and allocates space for the
                            // complete result in advance
                            std::vector<std::vector<Scalar>> result(weights[0].size(),
                                                                    std::vector<Scalar>(weights.size()));
                            for (std::vector<Scalar>::size_type i = 0; i < weights[0].size(); i++) {
                                for (std::vector<Scalar>::size_type j = 0; j < weights.size(); j++) {
                                    result[i][j] = weights[j][i];
                                }
                            }
                            return result;
                        };

                        // If layer_no is 0 then we need to take our input from in0
                        // If in_size is a vector type then in0 is a 7 x vecX "matrix"
                        // In this case we can use the built in dot-products to multiply by the weights matrix
                        // Otherwise, in0 is a 7 * in_size (= X) array of Scalars
                        // In this  case we need to roll our own dot products

                        // To make matters worse, our weights matrix will have 7 * X rows with vector_out_size columns!
                        // Need to flatten in0 to a 1 x (7 * X) "vector" to multiply our (7 * X) x (vector_out_size)
                        // "matrix" This will result in a 1 x (vector_out_size) "vector" result
                        //
                        // Equivalently, transpose the weights to a (vector_out_size) x (7 * X) "matrix"
                        // Then extract 1 x X blocks from rows in the transposed weights and perfrom the matrix
                        // multiplication that way Matrix multiplication + bias
                        if (layer_no == 0) {
                            if (vector_type(vector_in_size)) {
                                // Dot product each row of the weights matrix with each of the elements in the input
                                // layer and add in the bias
                                program.add_source_line(__FILE__, __LINE__, conv_no);
                                std::vector<uint32_t> layer;

                                for (const auto& bias : biases) {
                                    uint32_t result = program.add_constant(float_type, {bias});
                                    for (uint32_t row = 0; row < weights.size(); row += vector_in_size) {
                                        // Create a vector_in_size vector
                                        std::vector<uint32_t> vec;
                                        for (uint32_t i = row; i < row + vector_in_size; ++i) {
                                            vec.push_back(program.add_constant(float_type, {weights[i][row]}));
                                        }
                                        uint32_t w = program.create_vector(layer_in_type, vec);
                                        result     = program.fadd(
                                          result,
                                          program.dot(
                                            w,
                                            program.load_variable(
                                              program.member_access(
                                                layers.back(),
                                                {program.add_constant(uint_type, {row / vector_in_size})},
                                                program.add_pointer(layer_in_type, spv::StorageClass::Function)),
                                              layer_in_type),
                                            float_type),
                                          float_type);
                                    }
                                    layer.push_back(result);
                                }
                                layers.push_back(program.create_vector(layer_out_type, layer));
                            }
                            else {
                                program.add_source_line(__FILE__, __LINE__, conv_no);
                                uint32_t vector_out_type = program.add_array_type(
                                  layer_out_type, program.add_constant(uint_type, {vector_out_size}));
                                uint32_t vector_out_ptr =
                                  program.add_pointer(vector_out_type, spv::StorageClass::Function);
                                uint32_t vector_out = program.add_variable(vector_out_ptr, spv::StorageClass::Function);

                                for (uint32_t i = 0; i < biases.size(); ++i) {
                                    uint32_t result = program.add_constant(float_type, {biases[i]});
                                    for (uint32_t j = 0; j < weights.size(); ++j) {
                                        result = program.fadd(
                                          result,
                                          program.fmul(
                                            program.load_variable(
                                              program.member_access(
                                                layers.back(),
                                                {program.add_constant(uint_type, {j})},
                                                program.add_pointer(layer_in_type, spv::StorageClass::Function)),
                                              float_type),
                                            program.add_constant(float_type, {weights[j][i]}),
                                            float_type),
                                          float_type);
                                    }
                                    program.store_variable(
                                      program.member_access(
                                        vector_out,
                                        {program.add_constant(uint_type, {i})},
                                        program.add_pointer(float_type, spv::StorageClass::Function)),
                                      result);
                                }
                                layers.push_back(vector_out);
                            }
                        }
                        else if (vector_type(vector_in_size) && vector_type(vector_out_size)) {
                            program.add_source_line(__FILE__, __LINE__, conv_no);
                            // vector_in x vector_out matrix
                            // vector_out_size colunms of layer_in_type
                            uint32_t mat_type = program.add_mat_type(layer_in_type, vector_out_size);
                            program.add_name(mat_type, fmt::format("float{}x{}", vector_in_size, vector_out_size));

                            // Weights is a vector of rows, we need to extract the columns to populate our matrix
                            // Transpose our weights so that they are a vector of columns
                            auto transposed_weights = transpose(weights);

                            // Now generate spirv vecs and store them as columns for our matrix
                            std::vector<uint32_t> columns;
                            for (const auto& column : transposed_weights) {
                                columns.push_back(program.add_constant(layer_in_type, column));
                            }

                            // Now populate the matrix
                            uint32_t mat_weights = program.add_constant(mat_type, columns);

                            // Now multiply the matrix by the layer input and add the bias
                            layers.push_back(program.fadd(program.mvmul(mat_weights, layers.back(), layer_out_type),
                                                          program.add_constant(layer_out_type, biases),
                                                          layer_out_type));
                        }

                        // Vector type in, non-vector type out (OpArray)
                        else if (vector_type(vector_in_size)) {
                            program.add_source_line(__FILE__, __LINE__, conv_no);
                            // Proceed similar to above, but we have the wrong number of columns to make a matrix (1, or
                            // > 4) This also means the output cant be a vector type

                            // Weights is a vector of rows
                            // TODO: Confirm this assumption

                            // We can now dot-product each row with the input vector and add the bias
                            // We can't create a spirv vec for the output as it doesn't have the right number of
                            // elements so we will need to create an array
                            uint32_t array_out_type = program.add_array_type(
                              layer_out_type, program.add_constant(uint_type, {vector_out_size}));
                            uint32_t array_out_ptr = program.add_pointer(array_out_type, spv::StorageClass::Function);
                            uint32_t array_out     = program.add_variable(array_out_ptr, spv::StorageClass::Function);

                            for (uint32_t i = 0; i < vector_out_size; ++i) {
                                program.store_variable(
                                  program.member_access(
                                    array_out, {program.add_constant(uint_type, {i})}, array_out_ptr),
                                  program.fadd(
                                    program.dot(
                                      program.add_constant(layer_in_type, weights[i]), layers.back(), layer_out_type),
                                    program.add_constant(layer_out_type, {biases[i]}),
                                    layer_out_type));
                            }

                            layers.push_back(array_out);
                        }

                        // Non-vector type in (OpArray), vector type out
                        else if (vector_type(vector_out_size)) {
                            program.add_source_line(__FILE__, __LINE__, conv_no);
                            // We need to do the full matrix multiply manually
                            std::vector<uint32_t> layer;
                            for (uint32_t i = 0; i < biases.size(); ++i) {
                                uint32_t result = program.add_constant(float_type, {biases[i]});
                                for (uint32_t j = 0; j < weights.size(); ++j) {
                                    result = program.fadd(
                                      result,
                                      program.fmul(program.load_variable(
                                                     program.member_access(
                                                       layers.back(),
                                                       {program.add_constant(uint_type, {j})},
                                                       program.add_pointer(float_type, spv::StorageClass::Function)),
                                                     float_type),
                                                   program.add_constant(float_type, {weights[j][i]}),
                                                   float_type),
                                      float_type);
                                }
                                layer.push_back(result);
                            }
                            layers.push_back(program.create_vector(layer_out_type, layer));
                        }

                        // Non-vector types everywhere (OpArray in, OpArray out)
                        else {
                            program.add_source_line(__FILE__, __LINE__, conv_no);
                            // We need to do the full matrix multiply manually
                            uint32_t array_out_type = program.add_array_type(
                              layer_out_type, program.add_constant(uint_type, {vector_out_size}));
                            uint32_t array_out_ptr = program.add_pointer(array_out_type, spv::StorageClass::Function);
                            uint32_t array_out     = program.add_variable(array_out_ptr, spv::StorageClass::Function);

                            for (uint32_t i = 0; i < biases.size(); ++i) {
                                uint32_t result = program.add_constant(float_type, {biases[i]});
                                for (uint32_t j = 0; j < weights.size(); ++j) {
                                    result = program.fadd(
                                      result,
                                      program.fmul(program.load_variable(
                                                     program.member_access(
                                                       layers.back(),
                                                       {program.add_constant(uint_type, {j})},
                                                       program.add_pointer(float_type, spv::StorageClass::Function)),
                                                     float_type),
                                                   program.add_constant(float_type, {weights[j][i]}),
                                                   float_type),
                                      float_type);
                                }
                                program.store_variable(
                                  program.member_access(array_out,
                                                        {program.add_constant(uint_type, {i})},
                                                        program.add_pointer(float_type, spv::StorageClass::Function)),
                                  result);
                            }
                            layers.push_back(array_out);
                        }

                        /*************************************************
                         *                  ACTIVATION.                  *
                         *************************************************/
                        program.add_source_line(__FILE__, __LINE__, conv_no);

                        // Apply our activation function
                        // selu constants
                        uint32_t lambda = program.add_constant(
                          layer_out_type,
                          std::vector<Scalar>(vector_out_size, Scalar(1.0507009873554804934193349852946)));
                        uint32_t alpha = program.add_constant(
                          layer_out_type,
                          std::vector<Scalar>(vector_out_size, Scalar(1.6732632423543772848170429916717)));

                        // If this is not our last layer, apply selu
                        if (conv_no + 1 < network.size() || layer_no + 1 < conv.size()) {
                            if (vector_type(vector_out_size)) {
                                program.add_source_line(__FILE__, __LINE__, conv_no);
                                // inX = lambda * select(alpha * exp(inX) - alpha, inX, inX > 0)
                                // inX = lambda * ((inX > 0) ? inX : alpha * exp(inX) - alpha)
                                uint32_t condition =
                                  program.fgeq(layers.back(),
                                               program.add_constant(layer_out_type,
                                                                    std::vector<Scalar>(vector_out_size, Scalar(0.0))),
                                               vector_out_size);
                                uint32_t selu = program.fsub(
                                  program.fmul(alpha, program.exp(layers.back(), layer_out_type), layer_out_type),
                                  alpha,
                                  layer_out_type);
                                layers.back() =
                                  program.fmul(lambda,
                                               program.select(layer_out_type, condition, layers.back(), selu),
                                               layer_out_type);
                            }

                            else {
                                program.add_source_line(__FILE__, __LINE__, conv_no);
                                layer_out_type = program.add_array_type(
                                  layer_out_type, program.add_constant(uint_type, {vector_out_size}));
                                uint32_t vector_out_ptr =
                                  program.add_pointer(layer_out_type, spv::StorageClass::Function);
                                for (uint32_t i = 0; i < biases.size(); ++i) {
                                    uint32_t result = program.load_variable(
                                      program.member_access(
                                        layers.back(),
                                        {program.add_constant(uint_type, {i})},
                                        program.add_pointer(float_type, spv::StorageClass::Function)),
                                      float_type);
                                    uint32_t selu =
                                      program.fsub(program.fmul(alpha, program.exp(result, float_type), float_type),
                                                   alpha,
                                                   float_type);
                                    uint32_t condition =
                                      program.fgeq(result, program.add_constant(float_type, {Scalar(0.0)}));
                                    result = program.fmul(
                                      lambda, program.select(float_type, condition, result, selu), float_type);
                                    program.store_variable(
                                      program.member_access(
                                        layers.back(),
                                        {program.add_constant(uint_type, {i})},
                                        program.add_pointer(float_type, spv::StorageClass::Function)),
                                      result);
                                }
                            }
                        }
                        // If this is our last layer, apply softmax
                        else {
                            if (vector_type(vector_out_size)) {
                                program.add_source_line(__FILE__, __LINE__, conv_no);
                                // e = exp(inX)
                                // sum = dot(inX, 1)
                                // inX = e / sum
                                uint32_t e = program.exp(layers.back(), layer_out_type);
                                uint32_t sum =
                                  program.dot(e,
                                              program.add_constant(layer_out_type,
                                                                   std::vector<Scalar>(vector_out_size, Scalar(1.0))),
                                              float_type);
                                layers.back() = program.fdiv(
                                  e,
                                  program.create_vector(layer_out_type, std::vector<uint32_t>(vector_out_size, sum)),
                                  layer_out_type);
                            }
                            else {
                                program.add_source_line(__FILE__, __LINE__, conv_no);
                                std::vector<uint32_t> exps;
                                uint32_t sum = program.add_constant(float_type, {Scalar(0.0)});
                                for (uint32_t i = 0; i < biases.size(); ++i) {
                                    uint32_t exp =
                                      program.exp(program.load_variable(
                                                    program.member_access(
                                                      layers.back(),
                                                      {program.add_constant(uint_type, {i})},
                                                      program.add_pointer(float_type, spv::StorageClass::Function)),
                                                    float_type),
                                                  float_type);
                                    exps.push_back(exp);
                                    sum = program.fadd(sum, exp, float_type);
                                }

                                layer_out_type = program.add_array_type(
                                  layer_out_type, program.add_constant(uint_type, {vector_out_size}));
                                uint32_t vector_out_ptr =
                                  program.add_pointer(layer_out_type, spv::StorageClass::Function);
                                for (uint32_t i = 0; i < biases.size(); ++i) {
                                    program.store_variable(
                                      program.member_access(
                                        layers.back(),
                                        {program.add_constant(uint_type, {i})},
                                        program.add_pointer(float_type, spv::StorageClass::Function)),
                                      program.fdiv(exps[i], sum, float_type));
                                }
                            }
                        }

                        // Update our input size for the next loop
                        in_size = biases.size();
                    }

                    /*************************************************
                     *                    OUTPUT                     *
                     *************************************************/
                    // Save our value to the output
                    if (vector_type(conv_out_size)) {
                        program.add_source_line(__FILE__, __LINE__, conv_no);
                        program.store_variable(
                          program.member_access(
                            output_ptr,
                            {idx0, idx},
                            program.add_pointer(output_layer_type, spv::StorageClass::StorageBuffer)),
                          layers.back());
                    }
                    else {
                        program.add_source_line(__FILE__, __LINE__, conv_no);
                        uint32_t idx_conv =
                          program.imul(idx, program.add_constant(uint_type, {conv_out_size}), uint_type);
                        for (uint32_t i = 0; i < conv_out_size; ++i) {
                            program.store_variable(
                              program.member_access(
                                output_ptr,
                                {idx0, program.iadd(idx_conv, program.add_constant(uint_type, {i}), uint_type)},
                                program.add_pointer(output_layer_type, spv::StorageClass::StorageBuffer)),
                              program.load_variable(
                                program.member_access(layers.back(),
                                                      {program.add_constant(uint_type, {i})},
                                                      program.add_pointer(float_type, spv::StorageClass::Function)),
                                float_type));
                        }
                    }

                    program.return_function();
                    program.end_function();
                }

                return program.build();
            }

        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMEVULKANNCL_OPERATION_MAKE_NETWORK_HPP
