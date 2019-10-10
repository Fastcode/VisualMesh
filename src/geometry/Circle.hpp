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

#ifndef VISUALMESH_GEOMETRY_CIRCLE_HPP
#define VISUALMESH_GEOMETRY_CIRCLE_HPP

#include <cmath>
#include <limits>

namespace visualmesh {
namespace geometry {

  template <typename Scalar>
  struct Circle {

    /**
     * @brief Construct a new Circle object for building a visual mesh
     *
     * @param radius the radius of the circle
     * @param intersections the number of intersections to ensure with this object
     * @param max_distance  the maximum distance we want to look for this object
     */
    Circle(const Scalar& radius) : r(radius) {}

    /**
     * @brief Given a number of radial jumps (n) give the phi angle required
     *
     * @details
     *  This equation gives the phi angle that results from a number of radial jumps of circles on the observation
     *  plane. It measures from the centre of the circle on the ground, although this gives identical results to
     *  measuring to the tangents.
     *
     *             -1 ⎛2⋅n⋅r⎞
     *  φ(n) =  tan   ⎜─────⎟
     *                ⎝  h  ⎠
     *
     * @param n the number of whole objects to jump from the origin to reach this point (from object centres)
     * @param h the height of the camera above the observation plane
     */
    Scalar phi(const Scalar& n, const Scalar& h) const {
      return std::atan((2.0 * n * r) / h);
    }

    /**
     * @brief Given a phi angle calculate how many object jumps from the origin are required to reach this location
     *
     * @details
     *  This equation can also be used to calculate the n difference between any two objects by calculating the
     *  augmented height above the ground h' and the two φ' angles and using those in this equation instead of the real
     *  values
     *
     *
     *         h⋅tan(φ)
     *  n(φ) = ────────
     *           2⋅r
     *
     * @param phi the phi angle measured from below the camera
     */
    Scalar n(const Scalar& phi, const Scalar& h) const {
      return (h * std::tan(phi)) / (2.0 * r);
    }


    /**
     * @brief Gets the ratio of intersections using a Visual Mesh that was produced with a different height.
     *
     * @details
     *  This equation can be used to work out how many intersections will happen when you take a Visual Mesh that was
     *  created for one height and use it for another height. This can be used to check if an existing Visual Mesh will
     *  be appropriate for use when the height of the camera changes. The result of this function is a ratio which gives
     *  the number of intersections that will result from changing the height. We can use the tangent form of the
     *  equation here as it is simpler and does not change the result.
     *
     *              h₀
     *  k(h₀, h₁) = ──
     *              h₁
     *
     */
    Scalar k(const Scalar& h_0, const Scalar& h_1) const {
      return h_0 / h_1;
    }

    /**
     * @brief Given a value for phi and a camera height, return the angular width for an object
     *
     * @param phi  the phi value to calculate our theta value for
     * @param h    the height of the camera above the observation plane
     *
     * @return the angular width of the object around a phi circle
     */
    Scalar theta(const Scalar& phi, const Scalar& h) const {
      return 2.0 * std::asin(r / (h * std::tan(phi)));
    }

    // The radius of the circle
    Scalar r;
  };

}  // namespace geometry
}  // namespace visualmesh

#endif  // VISUALMESH_GEOMETRY_CIRCLE_HPP
