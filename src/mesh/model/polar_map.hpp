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

#ifndef VISUALMESH_MODEL_POLAR_MAP_HPP
#define VISUALMESH_MODEL_POLAR_MAP_HPP


#include "grid_base.hpp"
#include "utility/math.hpp"

namespace visualmesh {
namespace model {

    template <typename Scalar>
    struct PolarMap {

        /**
         * @brief
         *  Converts angles into a unit vector where phi is defined as the angle from -z to the vector, and theta is
         *  measured around the z axis.
         *
         * @param phi   the phi angle (up from -z)
         * @param theta the theta angle (around the z axis)
         *
         * @return The unit vector that these angles represent
         */
        static inline vec3<Scalar> unit_vector(const Scalar& phi, const Scalar& theta) {
            return vec3<Scalar>{{std::cos(theta) * std::sin(phi), std::sin(theta) * std::sin(phi), -std::cos(phi)}};
        }

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

            // Work out the phi ring from the n value
            const Scalar phi = shape.phi(nm[0], h);

            // Work out the radial value from the m value
            const Scalar d_theta = shape.theta(nm[0], h);
            const Scalar theta   = d_theta * nm[1];

            return unit_vector(phi, theta);
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

            // Phi value measured from the -z axis
            const Scalar n = shape.n(std::acos(-u[2]), h);

            // Work out theta from x/y
            const Scalar d_theta = shape.theta(n, h);
            const Scalar theta   = std::fmod(2.0 * M_PI + std::atan2(u[1], u[0]), 2.0 * M_PI);

            return vec2<Scalar>{{n, theta / d_theta}};
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_POLAR_MAP_HPP
