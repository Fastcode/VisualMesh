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

#ifndef VISUALMESH_ENGINE_VULKAN_KERNELS_REPROJECTION_HPP
#define VISUALMESH_ENGINE_VULKAN_KERNELS_REPROJECTION_HPP

#include <vector>

#include "visualmesh/engine/vulkan/vulkan_compute.hpp"

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace kernels {

            template <typename Scalar>
            inline uint32_t equidistant_reprojection(Program& program,
                                                     const uint32_t& f,
                                                     const uint32_t& theta,
                                                     const uint32_t& return_type) {
                return program.fmul(f, theta, return_type);
            }

            template <typename Scalar>
            inline uint32_t equisolid_reprojection(Program& program,
                                                   const uint32_t& f,
                                                   const uint32_t& theta,
                                                   const uint32_t& return_type) {
                return program.fmul(
                  program.fmul(program.add_constant(return_type, {Scalar(2)}), f, return_type),
                  program.sin(program.fmul(theta, program.add_constant(return_type, {Scalar(0.5)}), return_type),
                              return_type),
                  return_type);
            }

            template <typename Scalar>
            inline uint32_t rectilinear_reprojection(Program& program,
                                                     const uint32_t& f,
                                                     const uint32_t& theta,
                                                     const uint32_t& return_type) {
                return program.fmul(f, program.tan(theta, return_type), return_type);
            }

            template <typename Scalar, typename ReprojectionFunction>
            inline std::vector<uint32_t> build_reprojection(const std::string& kernel_name,
                                                            ReprojectionFunction&& r_u_func) {
                // Initialise the program.
                Program::Config config;
                config.enable_glsl_extensions = true;
                config.enable_float64         = ((sizeof(Scalar) == 8) && std::is_floating_point<Scalar>::value);
                config.address_model          = spv::AddressingModel::Logical;
                config.memory_model           = spv::MemoryModel::GLSL450;
                Program program(config);

                // uint32_t void_type  = program.add_type(spv::Op::OpTypeVoid, {});
                uint32_t uint_type  = program.add_type(spv::Op::OpTypeInt, {32, 0});
                uint32_t float_type = program.add_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)});
                uint32_t uvec2      = program.add_vec_type(spv::Op::OpTypeInt, {32, 0}, 2);
                uint32_t uvec3      = program.add_vec_type(spv::Op::OpTypeInt, {32, 0}, 3);
                uint32_t fvec2      = program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, 2);
                uint32_t fvec4      = program.add_vec_type(spv::Op::OpTypeFloat, {8 * sizeof(Scalar)}, 4);

                uint32_t uint_ptr    = program.add_pointer(uint_type, spv::StorageClass::Input);
                uint32_t uint_ptr_sb = program.add_pointer(uint_type, spv::StorageClass::StorageBuffer);
                uint32_t uvec3_ptr   = program.add_pointer(uvec3, spv::StorageClass::Input);
                uint32_t float_ptr   = program.add_pointer(float_type, spv::StorageClass::StorageBuffer);
                uint32_t fvec2_ptr   = program.add_pointer(fvec2, spv::StorageClass::StorageBuffer);
                uint32_t fvec4_ptr   = program.add_pointer(fvec4, spv::StorageClass::StorageBuffer);

                // Define the GlobalInvocationID (for get_global_id(0))
                uint32_t global_id = program.add_variable(uvec3_ptr, spv::StorageClass::Input);
                program.add_builtin_decoration(global_id, spv::BuiltIn::GlobalInvocationId);

                // Index 0 is used in every member_access call
                uint32_t idx0 = program.add_constant(uint_type, {0u});

                // Prepare the points, indices, Rco, f, dimensions, centre, and out variables
                uint32_t points_array  = program.add_array_type(fvec4);
                uint32_t points_struct = program.add_struct({points_array});
                uint32_t points_ptr =
                  program.add_variable(program.add_pointer(points_struct, spv::StorageClass::StorageBuffer),
                                       spv::StorageClass::StorageBuffer);

                uint32_t indices_array  = program.add_array_type(uint_type);
                uint32_t indices_struct = program.add_struct({indices_array});
                uint32_t indices_ptr =
                  program.add_variable(program.add_pointer(indices_struct, spv::StorageClass::StorageBuffer),
                                       spv::StorageClass::StorageBuffer);

                uint32_t Rco_array  = program.add_array_type(fvec4);
                uint32_t Rco_struct = program.add_struct({Rco_array});
                uint32_t Rco_ptr    = program.add_variable(
                  program.add_pointer(Rco_struct, spv::StorageClass::StorageBuffer), spv::StorageClass::StorageBuffer);

                uint32_t f_struct = program.add_struct({float_type});
                uint32_t f_ptr = program.add_variable(program.add_pointer(f_struct, spv::StorageClass::StorageBuffer),
                                                      spv::StorageClass::StorageBuffer);

                uint32_t dimensions_struct = program.add_struct({uvec2});
                uint32_t dimensions_ptr =
                  program.add_variable(program.add_pointer(dimensions_struct, spv::StorageClass::StorageBuffer),
                                       spv::StorageClass::StorageBuffer);

                uint32_t centre_struct = program.add_struct({fvec2});
                uint32_t centre_ptr =
                  program.add_variable(program.add_pointer(centre_struct, spv::StorageClass::StorageBuffer),
                                       spv::StorageClass::StorageBuffer);

                uint32_t k_struct = program.add_struct({fvec4});
                uint32_t k_ptr = program.add_variable(program.add_pointer(k_struct, spv::StorageClass::StorageBuffer),
                                                      spv::StorageClass::StorageBuffer);

                uint32_t out_array  = program.add_array_type(fvec2);
                uint32_t out_struct = program.add_struct({out_array});
                uint32_t out_ptr    = program.add_variable(
                  program.add_pointer(out_struct, spv::StorageClass::StorageBuffer), spv::StorageClass::StorageBuffer);

                // Decorate the structs and their members.
                uint32_t block_decoration = program.add_decoration_group(spv::Decoration::Block);
                program.add_group_decoration(
                  block_decoration,
                  {points_struct, indices_struct, f_struct, centre_struct, k_struct, dimensions_struct, out_struct});

                program.add_member_decoration(points_struct, 0, spv::Decoration::Offset, {0});
                program.add_member_decoration(indices_struct, 0, spv::Decoration::Offset, {0});
                program.add_member_decoration(f_struct, 0, spv::Decoration::Offset, {0});
                program.add_member_decoration(dimensions_struct, 0, spv::Decoration::Offset, {0});
                program.add_member_decoration(centre_struct, 0, spv::Decoration::Offset, {0});
                program.add_member_decoration(k_struct, 0, spv::Decoration::Offset, {0});
                program.add_member_decoration(out_struct, 0, spv::Decoration::Offset, {0});

                uint32_t stride16_decoration =
                  program.add_decoration_group(spv::Decoration::ArrayStride, {4 * sizeof(Scalar)});
                program.add_group_decoration(stride16_decoration, {points_array, Rco_array});
                program.add_decoration(indices_array, spv::Decoration::ArrayStride, {4});
                program.add_decoration(out_array, spv::Decoration::ArrayStride, {2 * sizeof(Scalar)});

                // Create the descriptor set.
                // Descriptor Set 0: {points_ptr, indices_ptr, Rco_ptr, f_ptr, centre_ptr, k_ptr, dimensions_ptr,
                // out_ptr}
                program.create_descriptor_set(
                  {points_ptr, indices_ptr, Rco_ptr, f_ptr, centre_ptr, k_ptr, dimensions_ptr, out_ptr});

                program.begin_entry_point(kernel_name, {global_id});
                // idx = get_global_id(0);
                uint32_t idx = program.load_variable(program.member_access(global_id, {idx0}, uint_ptr), uint_type);

                // Get our real index
                // id = indices[idx];
                uint32_t id =
                  program.load_variable(program.member_access(indices_ptr, {idx0, idx}, uint_ptr_sb), uint_type);

                // Get our LUT point
                // ray = points[id];
                uint32_t ray = program.load_variable(program.member_access(points_ptr, {idx0, id}, fvec4_ptr), fvec4);

                // Rotate our ray by our matrix to put it into camera space
                // ray = (Scalar4)(dot(Rco[0], ray), dot(Rco[1], ray), dot(Rco[2], ray), 0);
                uint32_t Rco0 = program.load_variable(program.member_access(Rco_ptr, {idx0, idx0}, fvec4_ptr), fvec4);
                uint32_t Rco1 = program.load_variable(
                  program.member_access(Rco_ptr, {idx0, program.add_constant(uint_type, {1u})}, fvec4_ptr), fvec4);
                uint32_t Rco2 = program.load_variable(
                  program.member_access(Rco_ptr, {idx0, program.add_constant(uint_type, {2u})}, fvec4_ptr), fvec4);
                uint32_t ray_x = program.dot(Rco0, ray, float_type);
                uint32_t ray_y = program.dot(Rco1, ray, float_type);
                uint32_t ray_z = program.dot(Rco2, ray, float_type);

                // Calculate some intermediates
                // theta     = acos(ray.x);
                uint32_t theta = program.acos(ray_x, float_type);

                // rsin_theta = rsqrt((Scalar)(1.0) - ray.x * ray.x);
                uint32_t rsin_theta = program.rsqrt(
                  program.fsub(
                    program.add_constant(float_type, {Scalar(1)}), program.fmul(ray_x, ray_x, float_type), float_type),
                  float_type);

                // r_u       = f * theta;
                uint32_t f   = program.load_variable(program.member_access(f_ptr, {idx0}, float_ptr), float_type);
                uint32_t r_u = r_u_func(program, f, theta, float_type);

                // r_d = r_u
                //        * (1.0                                                                 //
                //          + k.x * (r_u * r_u)                                                  //
                //          + k.y * ((r_u * r_u) * (r_u * r_u))                                  //
                //          + k.z * (((r_u * r_u) * (r_u * r_u)) * (r_u * r_u))                  //
                //          + k.w * (((r_u * r_u) * (r_u * r_u)) * ((r_u * r_u) * (r_u * r_u)))  //
                //        );
                uint32_t k   = program.load_variable(program.member_access(k_ptr, {idx0}, fvec4_ptr), fvec4);
                uint32_t k_x = program.vector_component(float_type, k, 0);
                uint32_t k_y = program.vector_component(float_type, k, 1);
                uint32_t k_z = program.vector_component(float_type, k, 2);
                uint32_t k_w = program.vector_component(float_type, k, 3);

                uint32_t r_u_2 = program.fmul(r_u, r_u, float_type);
                uint32_t r_u_4 = program.fmul(r_u_2, r_u_2, float_type);
                uint32_t r_u_6 = program.fmul(r_u_2, r_u_4, float_type);
                uint32_t r_u_8 = program.fmul(r_u_4, r_u_4, float_type);

                uint32_t r_d =
                  program.fmul(r_u,
                               program.fadd(program.add_constant(float_type, {Scalar(1)}),
                                            program.fadd(program.fmul(k_x, r_u_2, float_type),
                                                         program.fadd(program.fmul(k_y, r_u_4, float_type),
                                                                      program.fadd(program.fmul(k_z, r_u_6, float_type),
                                                                                   program.fmul(k_w, r_u_8, float_type),
                                                                                   float_type),
                                                                      float_type),
                                                         float_type),
                                            float_type),
                               float_type);

                // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
                // screen = (Scalar2)(r * ray.y * rsin_theta, r * ray.z * rsin_theta);
                uint32_t screen =
                  program.create_vector(fvec2,
                                        {program.fmul(r_d, program.fmul(ray_y, rsin_theta, float_type), float_type),
                                         program.fmul(r_d, program.fmul(ray_z, rsin_theta, float_type), float_type)});

                // When the pixel is at (1,0,0) lots of NaNs show up
                // screen = ray.x >= 1 ? (Scalar2)(0.0, 0.0) : screen;
                uint32_t condition = program.fgeq(ray_x, program.add_constant(float_type, {Scalar(1)}));
                uint32_t vec_condition =
                  program.create_vector(program.add_vec_type(spv::Op::OpTypeBool, {}, 2), {condition, condition});
                screen =
                  program.select(fvec2, vec_condition, program.add_constant(fvec2, {Scalar(0), Scalar(0)}), screen);

                // Apply our offset to move into image space (0 at top left, x to the right, y down)
                // Then apply the offset to the centre of our lens
                // image = (Scalar2)((Scalar)(dimensions.x - 1) * (Scalar)(0.5), (Scalar)(dimensions.y - 1) *
                // (Scalar)(0.5)) -
                //               screen - centre;
                // image_x = (Scalar)(dimensions.x - 1) * (Scalar)(0.5);
                // image_y = (Scalar)(dimensions.y - 1) * (Scalar)(0.5);
                // image = (Scalar2)(image0, image1) - screen - centre;
                uint32_t dimensions_x = program.cast_uint_to_float(
                  program.load_variable(program.member_access(dimensions_ptr, {idx0, idx0}, uint_ptr_sb), uint_type),
                  float_type);
                uint32_t dimensions_y = program.cast_uint_to_float(
                  program.load_variable(
                    program.member_access(dimensions_ptr, {idx0, program.add_constant(uint_type, {1u})}, uint_ptr_sb),
                    uint_type),
                  float_type);
                uint32_t image_x =
                  program.fmul(program.fsub(dimensions_x, program.add_constant(float_type, {Scalar(1)}), float_type),
                               program.add_constant(float_type, {Scalar(0.5)}),
                               float_type);
                uint32_t image_y =
                  program.fmul(program.fsub(dimensions_y, program.add_constant(float_type, {Scalar(1)}), float_type),
                               program.add_constant(float_type, {Scalar(0.5)}),
                               float_type);
                uint32_t image =
                  program.fsub(program.fsub(program.create_vector(fvec2, {image_x, image_y}), screen, fvec2),
                               program.load_variable(program.member_access(centre_ptr, {idx0}, fvec2_ptr), fvec2),
                               fvec2);

                // Store our output coordinates
                program.store_variable(program.member_access(out_ptr, {idx0, idx}, fvec2_ptr), image);
                program.return_function();
                program.end_function();

                return program.build();
            }
        }  // namespace kernels
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_KERNELS_REPROJECTION_HPP
