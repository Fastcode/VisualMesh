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
    private:
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

    public:
        /**
         * @brief Takes a point in n, m space (jumps along x and jumps along y) and converts it into a vector to the
         * centre of the object using the x grid method
         *
         * @tparam Shape the shape of the object we are mapping for
         *
         * @param shape the shape object used to calculate the angles
         * @param h     the height of the camera above the observation plane
         * @param nm    the coordinates in the nm space (object space)
         *
         * @return a vector <x, y, z> that points to the centre of the object at these coordinates
         */
        template <typename Shape>
        static vec3<Scalar> map(const Shape& shape, const Scalar& h, const vec2<Scalar>& nm) {

            // If n is negative it will jump us over to the other side of the origin
            // This is the same as a positive n but with a +pi offset to theta

            // Work out the phi ring from the n value
            const Scalar phi = shape.phi(std::abs(nm[0]), h);

            // Work out the radial value from the m value
            const Scalar d_theta = shape.theta(std::abs(nm[0]), h);

            // If n is negative, then add a pi offset as we went through the origin to the other side
            const Scalar theta = d_theta * nm[1] + (nm[0] < 0 ? M_PI : 0.0);

            return unit_vector(phi, theta);
        }

        /**
         * @brief Takes a unit vector that points to a location and maps it to object coordinates as nm space
         *
         * @tparam Shape the shape of the object we are mapping for
         *
         * @param shape the shape object used to calculate the angles
         * @param h     the height of the camera above the observation plane
         * @param u     the unit vector that points towards the centre of the object
         *
         * @return a vector <x, y, z> that points to the centre of the object at these coordinates
         */
        template <typename Shape>
        static vec2<Scalar> unmap(const Shape& shape, const Scalar& h, const vec3<Scalar>& u) {

            // Phi value measured from the -z axis
            const Scalar n = shape.n(std::acos(-u[2]), h);

            // Work out theta from x/y
            const Scalar d_theta = shape.theta(n, h);
            const Scalar theta   = std::fmod(2.0 * M_PI + std::atan2(u[1], u[0]), 2.0 * M_PI);

            return vec2<Scalar>{{n, theta / d_theta}};
        }

        template <typename Shape>
        static vec2<Scalar> difference(const Shape& shape,
                                       const Scalar& h,
                                       const vec2<Scalar>& a,
                                       const vec2<Scalar>& b) {

            // The equation we are trying to get to work is b + diff = a;
            // Clearly that won't work properly on it's own with a weird polar coordinate system so we have to do our
            // transforms here so that will work. Note that once we get this answer, it will only work if you add b, any
            // other offset won't work. This means it's giving the m coordinate difference within the b coordinate
            // system

            // Difference in n value is just the simple difference
            const Scalar n_d = std::abs(a[0]) - std::abs(b[0]);

            // Calculate the angles for both m components
            const Scalar a_dtheta = shape.theta(std::abs(a[0]), h);
            const Scalar b_dtheta = shape.theta(std::abs(b[0]), h);

            // If n is negative we went down past the origin which gives us a π offset
            const Scalar a_theta = a[1] * a_dtheta + (a[0] < 0 ? M_PI : 0.0);
            const Scalar b_theta = b[1] * b_dtheta + (b[0] < 0 ? M_PI : 0.0);

            // Calculate the smallest distance between the two angles and convert into n_b m coordinates as a value
            // between -π and π
            Scalar theta_d = std::fmod(a_theta - b_theta, Scalar(2.0 * M_PI));
            theta_d        = theta_d > M_PI ? theta_d - Scalar(2.0 * M_PI) : theta_d;
            theta_d        = theta_d < -M_PI ? theta_d + Scalar(2.0 * M_PI) : theta_d;

            return vec2<Scalar>{{n_d, theta_d * b_dtheta}};
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_POLAR_MAP_HPP
