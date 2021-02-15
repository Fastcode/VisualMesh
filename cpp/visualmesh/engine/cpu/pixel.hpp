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

#ifndef VISUALMESH_ENGINE_CPU_PIXEL_HPP
#define VISUALMESH_ENGINE_CPU_PIXEL_HPP

#include <cstdint>

#include "bayer.hpp"
#include "visualmesh/utility/fourcc.hpp"
#include "visualmesh/utility/math.hpp"

namespace visualmesh {
namespace engine {
    namespace cpu {

        /**
         * @brief Read the pixel value at a specific pixel coordinate
         *
         * @tparam Scalar the scalar type to use when calculating the pixel coordinates
         *
         * @param px            the pixel coordinates to sample
         * @param image         the image object we are sampling
         * @param dimensions    the dimensions of the image
         * @param format        the format of the image to sample
         *
         * @return a floating point image output with a value between 0.0 -> 1.0
         */
        template <typename Scalar>
        inline vec4<Scalar> get_pixel(const vec2<int>& px,
                                      const uint8_t* const image,
                                      const vec2<int>& dimensions,
                                      const uint32_t& format) {

            switch (format) {
                // Bayer
                case fourcc("GRBG"):
                case fourcc("RGGB"):
                case fourcc("GBRG"):
                case fourcc("BGGR"): {
                    return bayer::get_pixel<Scalar>(px, image, dimensions, format);
                }
                case fourcc("BGR3"):
                case fourcc("BGR8"): {
                    int c = (px[1] * dimensions[0] + px[0]) * 3;
                    return vec4<Scalar>{image[c + 2] * Scalar(1.0 / 255.0),
                                        image[c + 1] * Scalar(1.0 / 255.0),
                                        image[c + 0] * Scalar(1.0 / 255.0),
                                        1.0};
                }
                case fourcc("BGRA"): {
                    int c = (px[1] * dimensions[0] + px[0]) * 4;
                    return vec4<Scalar>{image[c + 2] * Scalar(1.0 / 255.0),
                                        image[c + 1] * Scalar(1.0 / 255.0),
                                        image[c + 0] * Scalar(1.0 / 255.0),
                                        image[c + 3] * Scalar(1.0 / 255.0)};
                }

                case fourcc("RGB3"):
                case fourcc("RGB8"): {
                    int c = (px[1] * dimensions[0] + px[0]) * 3;
                    return vec4<Scalar>{image[c + 0] * Scalar(1.0 / 255.0),
                                        image[c + 1] * Scalar(1.0 / 255.0),
                                        image[c + 2] * Scalar(1.0 / 255.0),
                                        1.0};
                }
                case fourcc("RGBA"): {
                    int c = (px[1] * dimensions[0] + px[0]) * 4;
                    return vec4<Scalar>{image[c + 0] * Scalar(1.0 / 255.0),
                                        image[c + 1] * Scalar(1.0 / 255.0),
                                        image[c + 2] * Scalar(1.0 / 255.0),
                                        image[c + 3] * Scalar(1.0 / 255.0)};
                }

                case fourcc("GRAY"):
                case fourcc("GREY"):
                case fourcc("Y8  "): {
                    int c = (px[1] * dimensions[0] + px[0]);
                    return vec4<Scalar>{image[c] * Scalar(1.0 / 255.0),
                                        image[c] * Scalar(1.0 / 255.0),
                                        image[c] * Scalar(1.0 / 255.0),
                                        1.0};
                }

                // Oh no...
                default: throw std::runtime_error("Unsupported image format " + fourcc_text(format));
            }
        }

        /**
         * @brief A version of the get_pixel function that allows for floating point pixel locations. It will perform a
         * linear interpolation on the surrounding pixels.
         *
         * @tparam Scalar
         * @param P
         * @param image
         * @param dimensions
         * @param format
         * @return vec4<Scalar>
         */
        template <typename Scalar>
        inline vec4<Scalar> interpolate(const vec2<Scalar>& P,
                                        const uint8_t* const image,
                                        const vec2<int>& dimensions,
                                        const uint32_t& format) {

            // (x1, y1) -------------- (x2, y1)
            //    |                       |
            //    |                       |
            //    |                       |
            //    |           P           |
            //    |                       |
            //    |                       |
            // (x1, y2) -------------- (x2, y2)
            const Scalar x        = P[0];
            const Scalar y        = P[1];
            const int x1          = std::max(int(std::floor(P[0])), 0);
            const int y1          = std::max(int(std::floor(P[1])), 0);
            const int x2          = std::max(x1 + 1, dimensions[0]);
            const int y2          = std::max(y1 + 1, dimensions[1]);
            const vec4<Scalar> Q1 = get_pixel<Scalar>(vec2<int>{x1, y1}, image, dimensions, format);
            const vec4<Scalar> Q2 = get_pixel<Scalar>(vec2<int>{x2, y1}, image, dimensions, format);
            const vec4<Scalar> Q3 = get_pixel<Scalar>(vec2<int>{x1, y2}, image, dimensions, format);
            const vec4<Scalar> Q4 = get_pixel<Scalar>(vec2<int>{x2, y2}, image, dimensions, format);

            const vec4<Scalar> R1 = add(multiply(Q1, ((x2 - x) / (x2 - x1))), multiply(Q2, ((x - x1) / (x2 - x1))));
            const vec4<Scalar> R2 = add(multiply(Q3, ((x2 - x) / (x2 - x1))), multiply(Q4, ((x - x1) / (x2 - x1))));

            return add(multiply(R1, ((y2 - y) / (y2 - y1))), multiply(R2, ((y - y1) / (y2 - y1))));
        }

    }  // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_PIXEL_HPP
