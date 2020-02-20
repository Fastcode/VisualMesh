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

#ifndef VISUALMESH_GEOMETRY_SPHERE_HPP
#define VISUALMESH_GEOMETRY_SPHERE_HPP

#include <cmath>
#include <limits>

namespace visualmesh {
namespace geometry {

    /**
     * @brief Represents a spherical object for the visual mesh
     *
     * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
     */
    template <typename Scalar>
    struct Sphere {

        /**
         * @brief Construct a new Sphere object for building a visual mesh
         *
         * @param radius the radius of the sphere
         */
        Sphere(const Scalar& radius) : r(radius) {}

        /**
         * @brief Given a number of radial jumps (n) give the phi angle required
         *
         * @details
         *  To calculate the angle to the base of the object we can use the following equation to find the next
         * tangential sphere. However for the visual mesh we would rather work in the sphere centres
         *
         *                      ⎛         n⎞
         *          π        -1 ⎜⎛    2 r⎞ ⎟
         *  φₜ(n) = ─ - 2 tan   ⎜⎜1 - ───⎟ ⎟
         *          2           ⎝⎝     h ⎠ ⎠
         *
         * Then to get the equation for the centre of the sphere, we calculate given the average angle of n ± 0.5
         *
         *
         *         φ(n - 0.5) + φ(n + 0.5)
         *  φ(n) = ───────────────────────
         *                   2
         *
         * @param n the number of whole objects to jump from the origin to reach this point (from object centres)
         * @param h the height of the camera above the observation plane
         *
         * @return the phi angle to the centre of the object that would result if n stacked objects were placed end on
         * end visually
         */
        Scalar phi(const Scalar& n, const Scalar& h) const {
            const Scalar v = 1 - 2 * r / h;
            return M_PI_2 - std::atan(pow(v, n - 0.5)) - std::atan(std::pow(v, n + 0.5));
        }

        /**
         * @brief Given a phi angle calculate how many object jumps from the origin are required to reach this location
         *
         * @details
         *  This equation can also be used to calculate the n difference between any two objects by calculating the
         *  augmented height above the ground h' and the two φ' angles and using those in this equation instead of the
         * real values. The following equation is for the tangent form of the equation found by inverting it
         *
         *               ⎛   ⎛φ   π⎞⎞
         *            log⎜cot⎜─ + ─⎟⎟
         *               ⎝   ⎝2   4⎠⎠
         *  n(φₜ) = ─────────────────────
         *          log(h - 2 r) - log(h)
         *
         *  However for inverting the centre form of the equation we must use its inversion
         *
         *            ⎛                         _________________________⎞
         *            ⎜                        ╱  2            2    2    ⎟
         *            ⎜r sin(φ) - h sin(φ) + ╲╱  h  - 2 h r + r  sin (φ) ⎟
         *         log⎜──────────────────────────────────────────────────⎟
         *            ⎝                     h cos(φ)                     ⎠    1
         *  n(φ) = ───────────────────────────────────────────────────────  - ─
         *                         log(h - 2 r) -log(h)                       2
         *
         * @param phi the phi angle measured from below the camera
         * @param h   the height that the camera is above the observation plane
         *
         * @return the number of object jumps that would be required to reach the angle to the centre of the object
         */
        Scalar n(const Scalar& phi, const Scalar& h) const {
            const Scalar sp = std::sin(phi);
            const Scalar cp = std::cos(phi);

            return std::log((r * sp - h * sp + std::sqrt(h * h - 2 * h * r + r * r * sp * sp)) / (h * cp))
                     / (std::log(h - 2 * r) - std::log(h))
                   - 0.5;
        }

        /**
         * @brief Gets the ratio of intersections using a Visual Mesh that was produced with a different height.
         *
         * @details
         *  This equation can be used to work out how many intersections will happen when you take a Visual Mesh that
         * was created for one height and use it for another height. This can be used to check if an existing Visual
         * Mesh will be appropriate for use when the height of the camera changes. The result of this function is a
         * ratio which gives the number of intersections that will result from changing the height.
         *
         *              log(h₁ - 2 r) - log(h₁)
         *  k(h₀, h₁) = ───────────────────────
         *              log(h₀ - 2 r) - log(h₀)
         *
         * @param h_0 the height above the observation plane that the original mesh was generated for
         * @param h_1 the height above the observation plane that we want to check to see how much k varies by
         *
         * @return the ratio of intersections between the mesh with h_0 and h_1. To get the actual difference in
         *         intersections multiply the output of this by k
         */
        Scalar k(const Scalar& h_0, const Scalar& h_1) const {
            return (std::log(h_1 - 2 * r) - std::log(h_1)) / (std::log(h_0 - 2 * r) - std::log(h_0));
        }

        /**
         * @brief Gets the height of the centre of the object above the observation plane
         *
         * @details For a sphere the height of the object above the observation plane is its radius
         *
         * @return Scalar the height of the centre of the object above the observation plane
         */
        Scalar c() const {
            return r;
        }

        /**
         * @brief Given a value for phi and a camera height, return the angular width for an object
         *
         * @details
         *  Calculates the angle in theta that an object would have if projected on the ground. This can be used for
         * radial slicing of objects so they fit around a circle.
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
