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

#ifndef VISUALMESH_ENGINE_CPU_BAYER_HPP
#define VISUALMESH_ENGINE_CPU_BAYER_HPP

#include <array>
#include <cstdint>

#include "visualmesh/utility/fourcc.hpp"
#include "visualmesh/utility/math.hpp"

namespace visualmesh {
namespace engine {
    namespace cpu {
        namespace bayer {

            enum BayerPixelType {
                R,   // Its red
                GR,  // Green on red row
                GB,  // Green on blue row
                B    // Its blue
            };

            // Implemented from http://www.ipol.im/pub/art/2011/g_mhcd/
            // Malvar-He-Cutler Linear Image Demosaicking
            // These coefficent values are scaled by 64 (divide by 64 after multiplying out)
            // clang-format off
            // G at red locations
            constexpr std::array<int32_t, 25> G_R =  {{  0,  0, -8,  0,  0,
                                                         0,  0, 16,  0,  0,
                                                        -8, 16, 32, 16, -8,
                                                         0,  0, 16,  0,  0,
                                                         0,  0, -8,  0,  0 }};
            // G at blue locations
            constexpr std::array<int32_t, 25> G_B = G_R;


            // R at blue locations
            constexpr std::array<int32_t, 25> R_B =  {{  0,  0, -12,  0,   0,
                                                         0, 16,   0, 16,   0,
                                                       -12,  0,  48,  0, -12,
                                                         0, 16,   0, 16,   0,
                                                         0,  0, -12,  0,   0 }};
            // B at red locations
            constexpr std::array<int32_t, 25> B_R = R_B;

            // R at green locations on red rows
            constexpr std::array<int32_t, 25> R_GR = {{  0,  0,  4,  0,  0,
                                                         0, -8,  0, -8,  0,
                                                        -8, 32, 40, 32, -8,
                                                         0, -8,  0, -8,  0,
                                                         0,  0,  4,  0,  0 }};
            // Blue at green locations on red rows
            constexpr std::array<int32_t, 25> B_GB = R_GR;

            // Red at green locations on blue rows
            constexpr std::array<int32_t, 25> R_GB = {{  0,  0, -8,  0,  0,
                                                         0, -8, 32, -8,  0,
                                                         4,  0, 40,  0,  4,
                                                         0, -8, 32, -8,  0,
                                                         0,  0, -8,  0,  0 }};
            // Blue at green locations on red rows
            constexpr std::array<int32_t, 25> B_GR = R_GB;
            // clang-format on

            template <typename Scalar>
            inline vec4<Scalar> demosaic(const std::array<int32_t, 5 * 5>& p, const BayerPixelType type) {
                constexpr Scalar factor = 1.0 / (64.0 * 255.0);
                switch (type) {
                    case R:
                        return multiply(vec4<Scalar>{Scalar(p[13]), Scalar(dot(p, G_R)), Scalar(dot(p, B_R)), 0.0},
                                        factor);
                    case GR:
                        return multiply(vec4<Scalar>{Scalar(dot(p, R_GR)), Scalar(p[13]), Scalar(dot(p, B_GR)), 0.0},
                                        factor);
                    case GB:
                        return multiply(vec4<Scalar>{Scalar(dot(p, R_GB)), Scalar(p[13]), Scalar(dot(p, B_GB)), 0.0},
                                        factor);
                    case B:
                        return multiply(vec4<Scalar>{Scalar(dot(p, R_B)), Scalar(dot(p, G_B)), Scalar(p[13]), 0.0},
                                        factor);
                    default: throw std::runtime_error("Unknown bayer pixel type");
                }
            }

            template <typename Scalar>
            vec4<Scalar> get_pixel(const vec2<int>& px,
                                   const uint8_t* const image,
                                   const vec2<int>& dimensions,
                                   const uint32_t& format) {

                // Start coordinates for the patch
                int x_s = px[0] - 2;
                int y_s = px[1] - 2;

                // Read the image patch into a flat array
                std::array<int32_t, 5 * 5> patch;
                for (int y = 0; y < 5; ++y) {
                    int y_c = std::min(std::max(y + y_s, 0), dimensions[1] - 1);
                    for (int x = 0; x < 5; ++x) {
                        int x_c          = std::min(std::max(x + x_s, 0), dimensions[0] - 1);
                        patch[y * 5 + x] = image[y_c * dimensions[0] + x_c];
                    }
                }

                bool row = px[0] % 2 == 1;
                bool col = px[1] % 2 == 1;
                switch (format) {
                    case fourcc("GRBG"): return demosaic<Scalar>(patch, row ? col ? GB : B : col ? R : GR);
                    case fourcc("RGGB"): return demosaic<Scalar>(patch, row ? col ? B : GB : col ? GR : R);
                    case fourcc("GBRG"): return demosaic<Scalar>(patch, row ? col ? GR : R : col ? B : GB);
                    case fourcc("BGGR"): return demosaic<Scalar>(patch, row ? col ? R : GR : col ? GB : B);
                    default: throw std::runtime_error("The fourcc code provided is not a valid bayer pattern");
                }
            }

        }  // namespace bayer
    }      // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_BAYER_HPP
