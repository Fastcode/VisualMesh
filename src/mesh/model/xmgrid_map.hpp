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

#ifndef VISUALMESH_MODEL_XMGRID_MAP_HPP
#define VISUALMESH_MODEL_XMGRID_MAP_HPP

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
         * @tparam Shape the shape of the object we are mapping for
         *
         * @param shape the shape object used to calculate the angles
         * @param nm    the coordinates in the nm space (object space)
         * @param h     the height of the camera above the observation plane
         *
         * @return a vector <x, y, z> that points to the centre of the object at these coordinates
         */
        template <typename Shape>
        static vec3<Scalar> map(const Shape& shape, const vec2<Scalar>& nm, const Scalar& h) {
            const Scalar phi_x = shape.phi(nm[0], h);
            const Scalar x     = (h - shape.c()) * std::tan(phi_x);

            const Scalar h_p = (h - shape.c()) / std::cos(phi_x);
            const Scalar y   = h_p * std::tan(shape.phi(nm[1], h_p + shape.c()));

            return vec3<Scalar>{x, y, shape.c() - h};
        }

        /**
         * @brief Takes a unit vector that points to a location and maps it to object coordinates as nm space
         *
         * @tparam Shape the shape of the object we are mapping for
         *
         * @param shape the shape object used to calculate the angles
         * @param u     the unit vector that points towards the centre of the object
         * @param h     the height of the camera above the observation plane
         *
         * @return a vector <x, y, z> that points to the centre of the object at these coordinates
         */
        template <typename Shape>
        static vec2<Scalar> unmap(const Shape& shape, const vec3<Scalar>& u, const Scalar& h) {

            // Height of the object above the observation plane so we can get planes from it's centre
            const Scalar& c = shape.c();

            // Extend out vec to the ground (divide by z and multiply by h-c)
            vec3<Scalar> v = multiply(u, (c - h) / u[2]);

            // Calculate what the h values must be for each direction
            const Scalar h_m = std::sqrt(v[0] * v[0] + v[2] * v[2]) + c;

            // The phi angle for the guess is the angle between <x, y, z> and <x, 0, z>
            const Scalar phi_n = std::atan(-std::abs(v[0]) / std::abs(c - h));
            const Scalar phi_m =
              std::acos(std::sqrt((v[0] * v[0] + v[2] * v[2]) / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])));

            vec2<Scalar> nm = {{shape.n(phi_n, h), shape.n(phi_m, h_m)}};
            nm[0] *= u[0] >= 0 ? -1 : 1;
            nm[1] *= u[1] >= 0 ? 1 : -1;

            return nm;
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_XMGRID_MAP_HPP
