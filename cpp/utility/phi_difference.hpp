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

#ifndef VISUALMESH_UTILITY_PHI_DIFFERENCE_HPP
#define VISUALMESH_UTILITY_PHI_DIFFERENCE_HPP

#include "math.hpp"

namespace visualmesh {
namespace util {

    /**
     * @brief Represents an angular difference between two objects on a plane below the camera
     *
     * @details
     *  The phi difference represents the angular difference between two objects by creating a new observation plane
     * that passes under both of them, and has a plane that contains the z axis. This allows the use of the 2d equations
     * to calculate distances between objects. This plane has a new height for the camera of h_prime.
     *
     * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
     */
    template <typename Scalar>
    struct PhiDifference {
        /// The new height above the ground
        Scalar h_prime;
        /// The phi value for the first vector
        Scalar phi_0;
        /// The phi value for the second vector
        Scalar phi_1;
    };

    /**
     * @brief Calculates the components needed to find the difference between two points in terms of number of objects.
     *
     * @details
     *  This function takes two unit vectors and a heigh above the observation plane for the camera and calculates the
     *  observation plane that is perpendicular to the two objects. This plane has a new height h' and two new phi
     * angles which give a 2d representation of the objects. These values can be used to calculate the n difference
     * between them.
     *
     * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
     *
     * @param h   the height of the camera above the observation plane
     * @param c   the height of the centre of the object above the observation plane
     * @param a_u the unit vector a to calculate a h' and phi angle for
     * @param b_u the unit vector b to calculate a h' and phi angle for
     *
     * @return an array of three values, h' the height above the new observation plane, phi_a and phi_b, the angle in
     * this plane for the two objects
     */
    template <typename Scalar>
    constexpr PhiDifference<Scalar> phi_difference(const Scalar& h,
                                                   const Scalar& c,
                                                   const vec3<Scalar>& a_u,
                                                   const vec3<Scalar>& b_u) {

        // Project the vectors to the plane that passes through the centre of the object
        vec3<Scalar> a_c = multiply(a_u, (h - c) / a_u[2]);
        vec3<Scalar> b_c = multiply(b_u, (h - c) / b_u[2]);

        // Distance from point to line equation
        // We use this to get the distance from the camera to the line connecting the two objects (giving h')
        // This plane is actually not the true h' as it goes through the centre of the object rather than the ground
        // If a and b are the same vector there are infinite possible planes, so just select the one that gives h
        const Scalar h_prime = a_u == b_u ? (h - c) : norm(cross(a_c, b_c)) / norm(subtract(a_c, b_c));

        // Calculate phi_0 and phi_1 in this new observation plane
        const Scalar phi0 = std::acos(h_prime / norm(a_c));
        const Scalar phi1 = std::acos(h_prime / norm(b_c));

        // Actual angle between a and b so we can check if we crossed from negative to positive
        const Scalar theta = std::acos(dot(a_u, b_u));

        // Choose the combination of phi0 and phi1 that give the angle closest to the true angle of theta
        return PhiDifference<Scalar>{
          h_prime + c,  // Add c to push the plane down to the ground
          phi0,
          std::abs(phi0 - phi1 - theta) < std::abs(phi0 + phi1 - theta) ? phi1 : -phi1,
        };
    }

}  // namespace util
}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_PHI_DIFFERENCE_HPP
