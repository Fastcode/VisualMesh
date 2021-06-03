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
            // Note that this paper has an error in figure 3, the lower left and lower centre matrices should be swapped
            // These coefficent values are scaled by 64 (divide by 64 after multiplying out)
            // clang-format off
            // G at red locations
            constexpr std::array<int16_t, 25> G_R =  {{  0,  0, -8,  0,  0,
                                                         0,  0, 16,  0,  0,
                                                        -8, 16, 32, 16, -8,
                                                         0,  0, 16,  0,  0,
                                                         0,  0, -8,  0,  0 }};
            // G at blue locations
            constexpr std::array<int16_t, 25> G_B = G_R;


            // R at blue locations
            constexpr std::array<int16_t, 25> R_B =  {{  0,  0, -12,  0,   0,
                                                         0, 16,   0, 16,   0,
                                                       -12,  0,  48,  0, -12,
                                                         0, 16,   0, 16,   0,
                                                         0,  0, -12,  0,   0 }};
            // B at red locations
            constexpr std::array<int16_t, 25> B_R = R_B;

            // R at green locations on red rows
            constexpr std::array<int16_t, 25> R_GR = {{  0,  0,  4,  0,  0,
                                                         0, -8,  0, -8,  0,
                                                        -8, 32, 40, 32, -8,
                                                         0, -8,  0, -8,  0,
                                                         0,  0,  4,  0,  0 }};
            // Blue at green locations on red rows
            constexpr std::array<int16_t, 25> B_GB = R_GR;

            // Red at green locations on blue rows
            constexpr std::array<int16_t, 25> R_GB = {{  0,  0, -8,  0,  0,
                                                         0, -8, 32, -8,  0,
                                                         4,  0, 40,  0,  4,
                                                         0, -8, 32, -8,  0,
                                                         0,  0, -8,  0,  0 }};
            // Blue at green locations on red rows
            constexpr std::array<int16_t, 25> B_GR = R_GB;
            // clang-format on

            /**
             * @brief Demosaics the patch into an rgb pixel value
             *
             * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
             *
             * @param p     the image patch that we will apply the demosacing too
             * @param type  the bayer pixel type (which pixel in the pattern we are)
             *
             * @return an rgb value for the image patch
             */
            template <typename Scalar>
            inline vec4<Scalar> demosaic(const std::array<int16_t, 5 * 5>& p, const BayerPixelType type) {
                vec4<int16_t> output;
                switch (type) {
                    case R: output = vec4<int16_t>{{p[12], dot(p, G_R) / 64, dot(p, B_R) / 64, 255}}; break;
                    case GR: output = vec4<int16_t>{{dot(p, R_GR) / 64, p[12], dot(p, B_GR) / 64, 255}}; break;
                    case GB: output = vec4<int16_t>{{dot(p, R_GB) / 64, p[12], dot(p, B_GB) / 64, 255}}; break;
                    case B: output = vec4<int16_t>{{dot(p, R_B) / 64, dot(p, G_B) / 64, p[12], 255}}; break;
                    default: throw std::runtime_error("Unknown bayer pixel type"); break;
                }

                // Normalise to 0->1
                return multiply(cast<Scalar>(output), Scalar(1.0 / 255.0));
            }

            /**
             * @brief Get the pixel object
             *
             * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
             *
             * @param px            the pixel coordinates to get the rgb value for
             * @param image         the image bytes
             * @param dimensions    the dimensions of the input image
             * @param format        the pixel format of the input image
             *
             * @return an rgb value as three floats between 0.0 and 1.0
             */
            template <typename Scalar>
            vec4<Scalar> get_pixel(const vec2<int>& px,
                                   const uint8_t* const image,
                                   const vec2<int>& dimensions,
                                   const uint32_t& format) {

                // Start coordinates for the patch
                int x_s = px[0] - 2;
                int y_s = px[1] - 2;

                // Read the image patch into a flat array
                std::array<int16_t, 5 * 5> patch{};
                for (int y = 0; y < 5; ++y) {
                    int y_c = std::min(std::max(y_s + y, 0), dimensions[1] - 1);
                    for (int x = 0; x < 5; ++x) {
                        int x_c          = std::min(std::max(x_s + x, 0), dimensions[0] - 1);
                        patch[y * 5 + x] = image[y_c * dimensions[0] + x_c];
                    }
                }

                bool col = px[0] % 2 == 1;
                bool row = px[1] % 2 == 1;
                switch (format) {
                    // We can map row/col to get the colour from the character code
                    // e.g. RGGB becomes
                    //       | col=0 | col=1
                    // row=0 |   R   |   G
                    // row=1 |   G   |   B
                    //
                    // Therefore the mapping of letters in this boolean is         4th  3rd       2nd  1st
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
