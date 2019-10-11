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

#ifndef VISUALMESH_TEST_MESH_HPP
#define VISUALMESH_TEST_MESH_HPP

namespace visualmesh {
namespace util {

  /**
   * @brief Calculates the components needed to find the difference between two points in terms of number of objects.
   *
   * @details
   *  This function takes two unit vectors and a heigh above the observation plane for the camera and calculates the
   *  observation plane that is perpendicular to the two objects. This plane has a new height h' and two new phi angles
   *  which give a 2d representation of the objects. These values can be used to calculate the n difference between
   *  them.
   *
   * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
   *
   * @param h   the height of the camera above the observation plane
   * @param a_u the unit vector a to calculate a h' and phi angle for
   * @param b_u the unit vector b to calculate a h' and phi angle for
   *
   * @return an array of three values, h' the height above the new observation plane, phi_a and phi_b, the angle in this
   *         plane for the two objects
   */
  template <typename Scalar>
  constexpr vec3<Scalar> phi_difference(const Scalar& h, const vec3<Scalar>& a_u, const vec3<Scalar>& b_u) {

    // Project the vectors to the ground
    vec3<Scalar> a = multiply(a_u, h / a_u[2]);
    vec3<Scalar> b = multiply(b_u, h / b_u[2]);

    // Distance from point to line equation
    // We use this to get the distance from the camera to the line connecting the two objects (giving h')
    vec3<Scalar> u       = subtract(b, a);
    const Scalar h_prime = norm(cross(a, u)) / norm(u);

    // Calculate phi_0 and phi_1 in this new plane
    const Scalar phi0 = std::acos(h_prime / norm(a));
    const Scalar phi1 = std::acos(h_prime / norm(b));

    // Actual angle between a and b so we can check if we crossed from negative to positive
    const Scalar theta = std::acos(dot(a_u, b_u));

    // Choose the combination of phi0 and phi1 that give the angle closest to the true angle of theta
    return {h_prime, phi0, std::abs(phi0 - phi1 - theta) < std::abs(phi0 + phi1 - theta) ? phi1 : -phi1};
  }

}  // namespace util
}  // namespace visualmesh

#endif  // VISUALMESH_TEST_MESH_HPP
