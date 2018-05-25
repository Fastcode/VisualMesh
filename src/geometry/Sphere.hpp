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
    Sphere(const Scalar& radius, const unsigned int& intersections, const Scalar& max_distance)
      : r(radius), k(intersections), d(max_distance) {}

    /**
     * @brief Given a value for phi and a camera height, return the value to the next phi in the sequence.
     *
     * @param phi_n the current phi value in the series
     * @param h     the height of the camera above the observation plane
     *
     * @return the next phi in the sequence (phi_{n+1})
     */
    Scalar phi(const Scalar& phi_n, const Scalar& h) const {

      // If we are beyond our max distance return nan
      if (std::abs(h) * tan(phi_n > M_PI_2 ? M_PI - phi_n : phi_n) > d) {
        return std::numeric_limits<Scalar>::quiet_NaN();
      }

      // Our effective radius for the number of intersections
      Scalar er = effective_radius(std::abs(h));

      // Valid below the horizon
      if (h > 0 && phi_n < M_PI_2) { return phi0(phi_n, er, h); }
      // Valid above the horizon
      else if (h < 0 && phi_n > M_PI_2) {
        return M_PI - phi0(M_PI - phi_n, er, -h);
      }
      // Other situations are invalid so return NaN
      else {
        return std::numeric_limits<Scalar>::quiet_NaN();
      }
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

      // If we are beyond our max distance return nan
      if (std::abs(h) * std::tan(phi > M_PI_2 ? M_PI - phi : phi) > d) {
        return std::numeric_limits<Scalar>::quiet_NaN();
      }

      // Valid below the horizon
      if (h > 0 && phi < M_PI_2) { return 2 * std::asin(r / (h * std::tan(phi) + r)) / k; }
      // Valid above the horizon
      else if (h < 0 && phi > M_PI_2) {
        return 2 * std::asin(r / (-h * std::tan(M_PI - phi) + r)) / k;
      }
      // Other situations are invalid so return NaN
      else {
        return std::numeric_limits<Scalar>::quiet_NaN();
      }
    }

    // The radius of the sphere
    Scalar r;
    // The number of intersections the mesh should have with this sphere
    unsigned int k;
    /// The maximum distance we want to see this object
    Scalar d;

  private:
    /**
     * @brief This function calculates the new phi ignoring number of intersections. It is used to find appropriate
     * radius for a number of intersections.
     *
     * @param phi_n  the phi we are starting at
     * @param er     the effective radius of the sphere
     * @param h      the height of the camera above the observation plane
     *
     * @return the new phi after moving the angular width of sphere at the old phi
     */
    static Scalar phi0(const Scalar& phi_n, const Scalar& er, const Scalar& h) {
      return 2 * std::atan(er / (std::cos(phi_n) * (h - er)) + std::tan(phi_n)) - phi_n;
    }

    /**
     * @brief Calculates the effective radius of a sphere that will have the correct number of intersections
     *
     * @param h the height of the camera above the observation plane
     *
     * @return the radius of a sphere that will intersect an appropriate number of times
     */
    Scalar effective_radius(const Scalar& h) const {

      // If 1 short circuit
      if (k == 1) { return r; }

      // The angle of a single intersection used as our target total phi
      Scalar p0 = phi0(0, r, h);

      // This function calculates the equation for optimising
      // It works out the difference between the target and obtained total phi
      auto f = [this, h, p0](const Scalar& er) {
        // Calculate phi0 once starting at 0
        Scalar p = phi0(0, er, h);

        // Update our phi the desired number of times
        for (unsigned int v = 1; v < k; ++v) {
          p = phi0(p, er, h);
        }

        // Return the difference between them
        return p - p0;
      };

      // We know that s > r/k and s < r which give us bounds for the method
      Scalar x1      = r;                                         // s < r
      Scalar x2      = r / k;                                     // s > r/i
      Scalar x3      = std::numeric_limits<Scalar>::quiet_NaN();  // The value of our best guess
      Scalar x3_prev = std::numeric_limits<Scalar>::quiet_NaN();  // Our previous best guess
      Scalar y       = std::numeric_limits<Scalar>::max();        // Our best guess error

      // Iterate until we reach the accuracy floor normally comparing floats like this is dangerous. However in this
      // situation we are relying on the determinism of floating point math, not the two values being equal. Since
      // this function is convex and almost linear this will never iterate too many times to solve.
      while (x3 != x3_prev) {

        // Feed our previous value through
        x3_prev = x3;

        // Perform our iteration
        Scalar y1 = f(x1);
        Scalar y2 = f(x2);
        x3        = (x1 * y2 - x2 * y1) / (y2 - y1);

        // Our new best guess quality
        y = f(x3);

        // Put the new guess into the right spot
        if (y > 0) { x1 = x3; }
        else if (y < 0) {
          x2 = x3;
        }
        // If y == 0 we have found the perfect solution
        else {
          return x3;
        }
      }

      return x3;
    }
  };

}  // namespace geometry
}  // namespace visualmesh

#endif  // VISUALMESH_GEOMETRY_SPHERE_HPP
