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

#ifndef VISUALMESH_ENGINE_VULKAN_KERNELS_LOAD_IMAGE_HPP
#define VISUALMESH_ENGINE_VULKAN_KERNELS_LOAD_IMAGE_HPP

#include <vector>

#include "utility/vulkan_compute.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace kernels {

            template <typename Scalar>
            inline std::vector<uint32_t> load_image() {
                // Initialise the program.
                Program::Config config;
                config.enable_glsl_extensions = true;
                config.enable_float64         = ((sizeof(Scalar) == 8) && std::is_floating_point<Scalar>::value);
                config.address_model          = spv::AddressingModel::Logical;
                config.memory_model           = spv::MemoryModel::GLSL450;
                Program program(config);

                // Add the standard types.
                // uint32_t void_type  = program.add_type(spv::Op::OpTypeVoid, {});
                uint32_t uint_type  = program.add_type(spv::Op::OpTypeInt, {32, 0});
                uint32_t float_type = program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)});
                uint32_t uvec2      = program.add_vec_type(spv::Op::OpTypeInt, {32, 0}, 2);
                uint32_t uvec3      = program.add_vec_type(spv::Op::OpTypeInt, {32, 0}, 3);
                uint32_t fvec2      = program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, 2);
                uint32_t fvec3      = program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, 3);
                uint32_t fvec4      = program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, 4);

                uint32_t uint_ptr  = program.add_pointer(uint_type, spv::StorageClass::Input);
                uint32_t uvec3_ptr = program.add_pointer(uvec3, spv::StorageClass::Input);
                uint32_t fvec2_ptr = program.add_pointer(fvec2, spv::StorageClass::StorageBuffer);
                uint32_t fvec4_ptr = program.add_pointer(fvec4, spv::StorageClass::StorageBuffer);

                // Define image and sampler types.
                uint32_t image_id = program.add_image_type(
                  float_type, spv::Dim::Dim2D, false, false, false, true, spv::ImageFormat::Rgba8);
                uint32_t sampler_id       = program.add_sampler_type();
                uint32_t sampled_image_id = program.add_sampled_image_type(image_id);

                // Create the "fetch" function.
                // Sample image "params[1]" using the sampler "params[2]" at position "params[3]" and return the first
                // component
                auto params = program.begin_function("fetch",
                                                     {float_type, image_id, sampler_id, fvec2},
                                                     spv::FunctionControlMask::Pure | spv::FunctionControlMask::Inline);
                program.return_function(program.vector_component(
                  float_type, program.sample_image(params[1], params[2], sampled_image_id, params[3], fvec4), 0));
                program.end_function();

                // Keep a handle to the function ID so we can call the function.
                uint32_t fetch_func = params[0];

                // Create the "bayer_to_rgb" function.
                // http://graphics.cs.williams.edu/papers/BayerJGT09/
                params = program.begin_function(
                  "bayer_to_rgb", {fvec4, image_id, sampler_id, fvec2, fvec2}, spv::FunctionControlMask::Pure);

                // Define some constants.
                uint32_t offsets = program.add_constant(fvec4, {Scalar(-2.0), Scalar(-1.0), Scalar(1.0), Scalar(2.0)});
                uint32_t kA =
                  program.add_constant(fvec4, {Scalar(-0.125), Scalar(-0.1875), Scalar(0.0625), Scalar(-0.125)});
                uint32_t kB = program.add_constant(fvec4, {Scalar(0.25), Scalar(0.0), Scalar(0.0), Scalar(0.5)});
                uint32_t kC = program.add_constant(fvec4, {Scalar(0.5), Scalar(0.75), Scalar(0.625), Scalar(0.625)});
                uint32_t kD = program.add_constant(fvec4, {Scalar(0.0), Scalar(0.25), Scalar(-0.125), Scalar(-0.125)});
                uint32_t kE =
                  program.add_constant(fvec4, {Scalar(-0.125), Scalar(-0.1875), Scalar(-0.125), Scalar(0.0625)});
                uint32_t kF = program.add_constant(fvec4, {Scalar(0.25), Scalar(0.0), Scalar(0.5), Scalar(0.0)});

                // centre.xy = coord
                // centre.zw = coord + first_red
                uint32_t centre = program.concatenate_fvec2(params[3], program.fadd(params[3], params[4], fvec2));

                // Extract components from centre.
                uint32_t centre_x  = program.vector_component(float_type, centre, 0);
                uint32_t centre_y  = program.vector_component(float_type, centre, 0);
                uint32_t centre_xy = program.swizzle(centre, fvec2, {0, 1});
                uint32_t centre_zw = program.swizzle(centre, fvec2, {2, 3});

                // Sample image at centre.xy.
                uint32_t C = program.call_function(fetch_func, float_type, {params[1], params[2], centre_xy});

                // float4 x_coord = centre.x + (float4){-2.0f, -1.0f, 1.0f, 2.0f};
                uint32_t x_coord =
                  program.fadd(program.create_vector(fvec4, {centre_x, centre_x, centre_x, centre_x}), offsets, fvec4);

                // float4 y_coord = centre.y + (float4){-2.0f, -1.0f, 1.0f, 2.0f};
                uint32_t y_coord =
                  program.fadd(program.create_vector(fvec4, {centre_y, centre_y, centre_y, centre_y}), offsets, fvec4);

                // Determine which of four types of pixels we are on.
                // float2 alternate = fmod(floor(centre.zw), 2.0f);
                uint32_t alternate = program.fmod(program.floor(centre_zw, fvec2, uvec2),
                                                  program.add_constant(fvec2, {Scalar(2.0), Scalar(2.0)}),
                                                  fvec2);

                // Extract components from alternate, x_coord, and y_coord.
                uint32_t alternate_x = program.vector_component(float_type, alternate, 0);
                uint32_t alternate_y = program.vector_component(float_type, alternate, 1);
                uint32_t x_coord_x   = program.vector_component(float_type, x_coord, 0);
                uint32_t x_coord_y   = program.vector_component(float_type, x_coord, 1);
                uint32_t x_coord_z   = program.vector_component(float_type, x_coord, 2);
                uint32_t x_coord_w   = program.vector_component(float_type, x_coord, 3);
                uint32_t y_coord_x   = program.vector_component(float_type, y_coord, 0);
                uint32_t y_coord_y   = program.vector_component(float_type, y_coord, 1);
                uint32_t y_coord_z   = program.vector_component(float_type, y_coord, 2);
                uint32_t y_coord_w   = program.vector_component(float_type, y_coord, 3);

                // float4 Dvec = (float4){
                //     fetch(raw_image, sampler, (float2){x_coord.y, y_coord.y}), // (-1,-1)
                //     fetch(raw_image, sampler, (float2){x_coord.y, y_coord.z}), // (-1, 1)
                //     fetch(raw_image, sampler, (float2){x_coord.z, y_coord.y}), // ( 1,-1)
                //     fetch(raw_image, sampler, (float2){x_coord.z, y_coord.z})  // ( 1,  1)
                // };
                uint32_t Dvec = program.create_vector(
                  fvec4,
                  {
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_y, y_coord_y})}),  // (-1, -1)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_y, y_coord_z})}),  // (-1,  1)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_z, y_coord_y})}),  // ( 1, -1)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_z, y_coord_z})})  // ( 1,  1)

                  });

                // Can also be a dot product with (1,1,1,1) on hardware where that is specially optimized.
                // Equivalent to: D = Dvec.x + Dvec.y + Dvec.z + Dvec.w;
                // Dvec.xy += Dvec.zw;
                // Dvec.x += Dvec.y;
                uint32_t D = program.dot(
                  Dvec, program.add_constant(fvec4, {Scalar(1.0), Scalar(1.0), Scalar(1.0), Scalar(1.0)}), float_type);

                // float4 value = (float4){
                //     fetch(raw_image, sampler, (float2){centre.x,  y_coord.x}), // ( 0,-2)
                //     fetch(raw_image, sampler, (float2){centre.x,  y_coord.y}), // ( 0,-1)
                //     fetch(raw_image, sampler, (float2){x_coord.x, centre.y}),  // (-1, 0)
                //     fetch(raw_image, sampler, (float2){x_coord.y, centre.y})   // (-2, 0)
                // };
                uint32_t value = program.create_vector(
                  fvec4,
                  {
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {centre_x, y_coord_x})}),  // ( 0, -2)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {centre_x, y_coord_y})}),  // ( 0, -1)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_x, centre_y})}),  // (-1,  0)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_y, centre_y})})  // (-2, 0)

                  });

                // float4 temp = (float4){
                //     fetch(raw_image, sampler, (float2){centre.x, y_coord.w}), // ( 0, 2)
                //     fetch(raw_image, sampler, (float2){centre.x, y_coord.z}), // ( 0, 1)
                //     fetch(raw_image, sampler, (float2){x_coord.w, centre.y}), // ( 2, 0)
                //     fetch(raw_image, sampler, (float2){x_coord.z, centre.y})  // ( 1, 0)
                // };
                uint32_t temp = program.create_vector(
                  fvec4,
                  {
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {centre_x, y_coord_w})}),  // ( 0,  2)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {centre_x, y_coord_z})}),  // ( 0,  1)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_w, centre_y})}),  // ( 2,  0)
                    program.call_function(
                      fetch_func,
                      float_type,
                      {params[1], params[2], program.create_vector(fvec2, {x_coord_z, centre_y})})  // ( 1, 0)

                  });

                // value += temp;
                value = program.fadd(value, temp, fvec4);

                // float4 PATTERN = (kC.xyz * C).xyzz;
                uint32_t PATTERN =
                  program.swizzle(program.vmul(program.swizzle(kC, fvec3, {0, 1, 2}), C, fvec3), fvec4, {0, 1, 2, 2});

                // There are five filter patterns (identity, cross, checker,
                // theta, phi).  Precompute the terms from all of them and then
                // use swizzles to assign to color channels.
                //
                // Channel   Matches
                //   x       cross   (e.g., EE G)
                //   y       checker (e.g., EE B)
                //   z       theta   (e.g., EO R)
                //   w       phi     (e.g., EO R)
                // const float A = value.x;
                // const float B = value.y;
                // const float D = Dvec.x;
                // const float E = value.z;
                // const float F = value.w;

                // Avoid zero elements. On a scalar processor this saves two MADDs and it has no
                // effect on a vector processor.

                // PATTERN.yzw += (kD.yz * D).xyy; ==> PATTERN = PATTERN + (0, (kD.yz * D).xyy);
                PATTERN = program.fadd(
                  PATTERN,
                  program.swizzle(
                    kD,  // We need the 0 that is in the first component of kD here (any vector type with a 0 in
                         // it would do, but this one is convenient)
                    program.swizzle(program.vmul(program.swizzle(kD, fvec2, {1, 2}), D, fvec2), fvec3, {0, 1, 1}),
                    fvec4,
                    {0, 4, 5, 6}),
                  fvec4);

                // PATTERN += (kA.xyz * A).xyzx + (kE.xyw * E).xyxz;
                PATTERN =
                  program.fadd(program.fadd(program.swizzle(program.vmul(program.swizzle(kA, fvec3, {0, 1, 2}),
                                                                         program.vector_component(float_type, value, 0),
                                                                         fvec3),
                                                            fvec4,
                                                            {0, 1, 2, 0}),
                                            program.swizzle(program.vmul(program.swizzle(kE, fvec3, {0, 1, 3}),
                                                                         program.vector_component(float_type, value, 2),
                                                                         fvec3),
                                                            fvec4,
                                                            {0, 1, 2, 0}),
                                            fvec4),
                               PATTERN,
                               fvec4);

                // PATTERN.xw += kB.xw * B; ==> PATTERN = PATTERN + ((kB.xw * B).x, 0, 0, (kB.xw * B).y)
                PATTERN = program.fadd(
                  PATTERN,
                  program.swizzle(
                    kB,  // We need the 0s that are in the second and third components of kB here
                         // (any vector type with a 0 in it would do, but this one is convenient)
                    program.vmul(
                      program.swizzle(kB, fvec2, {0, 3}), program.vector_component(float_type, value, 1), fvec2),
                    fvec4,
                    {4, 1, 2, 5}),
                  fvec4);

                // PATTERN.xz += kF.xz * F; ==> PATTERN = PATTERN + ((kF.xz * F).x, 0, (kF.xz * F).y, 0)
                PATTERN = program.fadd(
                  PATTERN,
                  program.swizzle(
                    kF,  // We need the 0s that are in the second and third components of kB here
                         // (any vector type with a 0 in it would do, but this one is convenient)
                    program.vmul(
                      program.swizzle(kF, fvec2, {0, 2}), program.vector_component(float_type, value, 3), fvec2),
                    fvec4,
                    {4, 1, 5, 3}),
                  fvec4);

                // Extract components from PATTERN.
                uint32_t PATTERN_x = program.vector_component(float_type, PATTERN, 0);
                uint32_t PATTERN_y = program.vector_component(float_type, PATTERN, 1);
                uint32_t PATTERN_z = program.vector_component(float_type, PATTERN, 2);
                uint32_t PATTERN_w = program.vector_component(float_type, PATTERN, 3);

                // alternate.x == 0.0f
                // alternate.y == 0.0f
                uint32_t condition[] = {program.feq(alternate_x, program.add_constant(float_type, {Scalar(0.0)})),
                                        program.feq(alternate_y, program.add_constant(float_type, {Scalar(0.0)}))};

                // Select needs the condition to a vector of the same size as the result and the operands.
                uint32_t vec_condition[] = {
                  program.create_vector(program.add_vec_type(spv::Op::OpTypeBool, {}, 4),
                                        {condition[0], condition[0], condition[0], condition[0]}),
                  program.create_vector(program.add_vec_type(spv::Op::OpTypeBool, {}, 4),
                                        {condition[1], condition[1], condition[1], condition[1]})};

                // Select the appropriate output based on the above defined conditions.
                program.return_function(program.select(
                  fvec4,
                  vec_condition[1],
                  program.select(fvec4,
                                 vec_condition[0],
                                 program.create_vector(
                                   fvec4, {C, PATTERN_x, PATTERN_y, program.add_constant(float_type, {Scalar(1.0)})}),
                                 program.create_vector(
                                   fvec4, {PATTERN_z, C, PATTERN_w, program.add_constant(float_type, {Scalar(1.0)})})),
                  program.select(
                    fvec4,
                    vec_condition[0],
                    program.create_vector(
                      fvec4, {PATTERN_w, C, PATTERN_z, program.add_constant(float_type, {Scalar(1.0)})}),
                    program.create_vector(
                      fvec4, {PATTERN_y, PATTERN_x, C, program.add_constant(float_type, {Scalar(1.0)})}))));
                program.end_function();

                // Keep a handle to the function ID so we can call the function.
                uint32_t bayer_to_rgb_func = params[0];

                // Define the GlobalInvocationID (for get_global_id(0))
                uint32_t global_id = program.add_variable(uvec3_ptr, spv::StorageClass::Input);
                program.add_builtin_decoration(global_id, spv::BuiltIn::GlobalInvocationId);

                // Index 0 is used in every member_access call
                uint32_t idx0 = program.add_constant(uint_type, {0u});

                // Define the WorkgroupSize constant
                uint32_t workgroup_size = program.add_constant(uvec3, {idx0, idx0, idx0}, 1, true);
                program.add_builtin_decoration(workgroup_size, spv::BuiltIn::WorkgroupSize);

                // Prepare the descriptor set variables.
                // Prepare the image sampler variables.
                uint32_t sampler_ptr    = program.add_pointer(sampler_id, spv::StorageClass::UniformConstant);
                uint32_t bayer_sampler  = program.add_variable(sampler_ptr, spv::StorageClass::UniformConstant);
                uint32_t interp_sampler = program.add_variable(sampler_ptr, spv::StorageClass::UniformConstant);

                // Prepare the image, coordinates array, and network array variables.
                uint32_t image_ptr = program.add_variable(program.add_pointer(image_id, spv::StorageClass::Image),
                                                          spv::StorageClass::Image);

                uint32_t coords_array  = program.add_array_type(fvec2);
                uint32_t coords_struct = program.add_struct({coords_array});
                uint32_t coords_ptr =
                  program.add_variable(program.add_pointer(coords_struct, spv::StorageClass::StorageBuffer),
                                       spv::StorageClass::StorageBuffer);

                uint32_t network_array  = program.add_array_type(fvec4);
                uint32_t network_struct = program.add_struct({network_array});
                uint32_t network_ptr =
                  program.add_variable(program.add_pointer(network_struct, spv::StorageClass::StorageBuffer),
                                       spv::StorageClass::StorageBuffer);

                // Decorate the structs and their members.
                uint32_t block_decoration = program.add_decoration_group(spv::Decoration::Block);
                program.add_group_decoration(block_decoration, {coords_struct, network_struct});
                program.add_decoration(coords_array, spv::Decoration::ArrayStride, {8});
                program.add_member_decoration(coords_struct, 0, spv::Decoration::Offset, {0});
                program.add_decoration(network_array, spv::Decoration::ArrayStride, {16});
                program.add_member_decoration(network_struct, 0, spv::Decoration::Offset, {0});

                // Create the descriptor sets.
                // Descriptor Set 0: {bayer_sampler, interp_sampler}
                program.create_descriptor_set({bayer_sampler, interp_sampler});

                // Descriptor Set 1: {image, coordinates, network}
                program.create_descriptor_set({image_ptr, coords_ptr, network_ptr});

                // Create the "read_GRBG_image_to_network" entry point.
                program.begin_entry_point("load_GRBG_image", {global_id});
                uint32_t idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                program.store_variable(
                  program.member_access(network_ptr, {idx0, idx}, fvec4_ptr),
                  program.call_function(
                    bayer_to_rgb_func,
                    fvec4,
                    {program.load_variable(image_ptr, image_id),
                     program.load_variable(bayer_sampler, sampler_id),
                     program.load_variable(program.member_access(coords_ptr, {idx0, idx}, fvec2_ptr), fvec2),
                     program.add_constant(fvec2, {Scalar(1.0), Scalar(0.0)})}));
                program.return_function();
                program.end_function();

                // Create the "read_RGGB_image_to_network" entry point.
                program.begin_entry_point("load_RGGB_image", {global_id});
                idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                program.store_variable(
                  program.member_access(network_ptr, {idx0, idx}, fvec4_ptr),
                  program.call_function(
                    bayer_to_rgb_func,
                    fvec4,
                    {program.load_variable(image_ptr, image_id),
                     program.load_variable(bayer_sampler, sampler_id),
                     program.load_variable(program.member_access(coords_ptr, {idx0, idx}, fvec2_ptr), fvec2),
                     program.add_constant(fvec2, {Scalar(0.0), Scalar(0.0)})}));
                program.return_function();
                program.end_function();

                // Create the "read_GBRG_image_to_network" entry point.
                program.begin_entry_point("load_GBRG_image", {global_id});
                idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                program.store_variable(
                  program.member_access(network_ptr, {idx0, idx}, fvec4_ptr),
                  program.call_function(
                    bayer_to_rgb_func,
                    fvec4,
                    {program.load_variable(image_ptr, image_id),
                     program.load_variable(bayer_sampler, sampler_id),
                     program.load_variable(program.member_access(coords_ptr, {idx0, idx}, fvec2_ptr), fvec2),
                     program.add_constant(fvec2, {Scalar(0.0), Scalar(1.0)})}));
                program.return_function();
                program.end_function();

                // Create the "read_BGGR_image_to_network" entry point.
                program.begin_entry_point("load_BGGR_image", {global_id});
                idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                program.store_variable(
                  program.member_access(network_ptr, {idx0, idx}, fvec4_ptr),
                  program.call_function(
                    bayer_to_rgb_func,
                    fvec4,
                    {program.load_variable(image_ptr, image_id),
                     program.load_variable(bayer_sampler, sampler_id),
                     program.load_variable(program.member_access(coords_ptr, {idx0, idx}, fvec2_ptr), fvec2),
                     program.add_constant(fvec2, {Scalar(1.0), Scalar(1.0)})}));
                program.return_function();
                program.end_function();

                // Create the "read_RGBA_image_to_network" entry point.
                program.begin_entry_point("load_RGBA_image", {global_id});
                idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);
                program.store_variable(
                  program.member_access(network_ptr, {idx0, idx}, fvec4_ptr),
                  program.sample_image(
                    program.load_variable(image_ptr, image_id),
                    program.load_variable(interp_sampler, sampler_id),
                    sampled_image_id,
                    program.load_variable(program.member_access(coords_ptr, {idx0, idx}, fvec2_ptr), fvec2),
                    fvec4));
                program.return_function();
                program.end_function();

                return program.build();
            }
        }  // namespace kernels
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_KERNELS_LOAD_IMAGE_HPP
