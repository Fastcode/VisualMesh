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

#ifndef VISUALMESH_MODEL_XMGRID_HPP
#define VISUALMESH_MODEL_XMGRID_HPP

#include "grid_base.hpp"
#include "utility/math.hpp"

namespace visualmesh {
namespace model {

    template <typename Scalar>
    struct XMGridMap {

        /**
         * @brief Takes a point in n, m space (jumps along x and jumps along y) and converts it into a vector to the
         * centre of the object using the x grid method
         *
         * @param shape the shape object used to calculate the angles
         * @param n     the coordinate in the n coordinate (object jumps along the x axis)
         * @param m     the coordinate in the m coordinate (object jumps along the y axis)
         * @param h     the height of the camera above the observation plane
         * @param c     the height of the object above the observation plane
         *
         * @return a vector <x, y, z> that points to the centre of the object at these coordinates
         */
        template <typename Shape>
        static vec3<Scalar> map(const Shape& shape, const vec2<Scalar>& nm, const Scalar& h, const Scalar& c) {
            const Scalar phi_x = shape.phi(nm[0], h);
            const Scalar x     = (h - c) * std::tan(phi_x);

            const Scalar h_p = (h - c) / std::cos(phi_x);
            const Scalar y   = h_p * std::tan(shape.phi(nm[1], h_p + c));

            return vec3<Scalar>{x, y, c - h};
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_XMGRID_HPP
