/*
 * Copyright (C) 2017-2018 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_GEOMETRY_SPHERE_HPP
#define VISUALMESH_GEOMETRY_SPHERE_HPP

#include <cmath>
#include <limits>

namespace visualmesh {
namespace geometry {

  template <typename Scalar>
  struct Sphere {

    /**
     * @brief Construct a new Sphere object for building a visual mesh
     *
     * @param radius the radius of the sphere
     * @param intersections the number of intersections to ensure with this object
     * @param max_distance  the maximum distance we want to look for this object
     */
    Sphere(const Scalar& radius) : r(radius) {}

    /**
     * @brief Given a number of radial jumps (n) give the phi angle required
     *
     * @details
     *  To calculate the angle to the base of the object we can use the following equation
     *
     *                     ⎛         n⎞
     *         π        -1 ⎜⎛    2⋅r⎞ ⎟
     *  φ(n) = ─ - 2⋅tan   ⎜⎜1 - ───⎟ ⎟
     *         2           ⎝⎝     h ⎠ ⎠
     *
     * Then to get the equation for the centre of the sphere, we calculate given the average angle of n ± 0.5
     *
     * @param n the number of whole objects to jump from the origin to reach this point (from object centres)
     * @param h the height of the camera above the observation plane
     */
    Scalar phi(const Scalar& n, const Scalar& h) const {
      Scalar v = 1 - 2 * r / h;
      return M_PI_2 - std::atan(pow(v, n - 0.5)) - std::atan(std::pow(v, n + 0.5));
    }

    /**
     * @brief Given a phi angle calculate how many object jumps from the origin are required to reach this location
     *
     * @details
     *  This equation can also be used to calculate the n difference between any two objects by calculating the
     *  augmented height above the ground h' and the two φ' angles and using those in this equation instead of the real
     *  values
     *
     *       ⎛   ⎛φ   π⎞⎞
     *    log⎜cot⎜─ + ─⎟⎟
     *       ⎝   ⎝2   4⎠⎠
     *  ─────────────────────
     *  log(h - 2⋅r) - log(h)
     *
     * @param phi the phi angle measured from below the camera
     */
    Scalar n(const Scalar& phi, const Scalar& h) const {
      return std::log(1 / std::tan(phi * 0.5 + M_PI_4)) / (std::log(h - 2 * r) - log(h));
    }

    /**
     * @brief Given a value for phi and a camera height, return the angular width for an object
     *
     * @param phi the phi value to calculate our theta value for
     * @param h the height of the camera above the observation plane
     *
     * @return the angular width of the object around a phi circle
     */
    Scalar theta(const Scalar& phi, const Scalar& h) const {
      return 2 * std::asin(r / ((h - r) * std::tan(phi)));
    }

    // The radius of the sphere
    Scalar r;
  };

}  // namespace geometry
}  // namespace visualmesh

#endif  // VISUALMESH_GEOMETRY_SPHERE_HPP
