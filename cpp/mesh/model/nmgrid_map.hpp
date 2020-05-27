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

#ifndef VISUALMESH_MODEL_NMGRID_MAP_HPP
#define VISUALMESH_MODEL_NMGRID_MAP_HPP

#include "grid_base.hpp"
#include "utility/math.hpp"

namespace visualmesh {
namespace model {

    template <typename Scalar>
    struct NMGridMap {

    private:
        template <typename Shape>
        static vec3<Scalar> xyz(const Shape& shape, const Scalar& h, const Scalar& n, const Scalar& h_n) {

            // Height of the object above the observation plane so we can get planes from it's centre
            const Scalar& c = shape.c();

            // Calculate x via phi at the respective height
            const Scalar x = (h_n - c) * std::tan(shape.phi(n, h_n));
            // Calculate y via the expected value given the true height and x height
            const Scalar y = std::sqrt((h_n - c) * (h_n - c) - (h - c) * (h - c));

            return vec3<Scalar>{{x, y, c - h}};
        }

        template <typename Shape>
        static Scalar guess_m(const Shape& shape, const Scalar& h, const Scalar& n, const Scalar& h_n) {

            // Height of the object above the observation plane so we can get planes from it's centre
            const Scalar& c = shape.c();

            // Calculate what the vector must be given this height for the x axis
            const vec3<Scalar> v = xyz(shape, h, n, h_n);

            // The height this decides for the m coordinate is the distance along the x axis
            const Scalar h_m = std::sqrt(v[0] * v[0] + v[2] * v[2]) + c;

            // The phi angle for the guess is the angle between <x, y, z> and <x, 0, z>
            const Scalar phi_m =
              std::acos(std::sqrt((v[0] * v[0] + v[2] * v[2]) / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])));

            // Work out what m must have been given these parameters
            return shape.n(phi_m, h_m);
        }

    public:
        /**
         * @brief Takes a point in n, m space (jumps along x and jumps along y) and converts it into a vector to the
         * centre of the object using the x grid method
         *
         * @details
         * How this works is by optimising the solution until we find one that matches. We do this by optimising the
         * "height" that the plane must be for the first coordinate to give the correct value for the second
         * coordinate.
         *
         * We set the bounds for the optimisation by noting that if the second coordinate were 0, then the height
         * value for the first coordinate must be the height of the camera. We then note that as our second coordinate
         * increases, the height of the first coordinates plane must increase as it must be further away. The furthest
         * we could possibly be away however, is if the first coordinate were 0, and then we could calculate the
         * height using the phi equation on the second of the nm pair. This gives us an upper and lower bound for what
         * the height could be for the first coordinate.
         *
         * Having the height value for the first coordinate implicitly gives us a y coordinate on our ground plane and
         * using the phi equation we can calculate an x value on the ground plane too.
         * From this x value we can calculate a height for the second axis. Then using the y value on the ground we
         * can calculate a phi value along this orthogonal direction plane. We can feed this into the shape.n function
         * to calculate what the second coordinate would be given the provided angle. This can be compared to the true
         * m value and if it is low or high we can adjust the height to find the actual height for the first equation
         * that gives the appropriate value for the second equation.
         *
         * @param shape the shape object used to calculate the angles
         * @param h     the height of the camera above the observation plane
         * @param nm    the coordinates in the nm space (object space)
         *
         * @return a vector <x, y, z> that points to the centre of the object at these coordinates
         */
        template <typename Shape>
        static vec3<Scalar> map(const Shape& shape, const Scalar& h, const vec2<Scalar>& nm) {
            // Abs first so we don't need to worry about quadrant, we will fix the signs at the end
            const Scalar& n = std::abs(nm[0]);
            const Scalar& m = std::abs(nm[1]);
            const Scalar& c = shape.c();

            // Set our bounds for the search
            // The smallest that h could be is the h we are provided (below the camera)
            // The largest h could be is the h on the axis using the 2nd coordinate because if h were larger than this
            // then the 2nd coordinate would be out of range as it cannot exceed this amount
            Scalar lo(h);
            Scalar hi((h - c) / std::cos(shape.phi(m, h)) + c);
            Scalar h_n((lo + hi) * 0.5);

            // Optimise as far as we can
            while (true) {
                if (h_n == lo || h_n == hi) { break; }

                // Work out what m would be if this was our h
                const Scalar& m_g   = guess_m(shape, h, n, h_n);
                (m_g > m ? hi : lo) = h_n;

                // If we didn't move the bisection finish
                if (h_n == (lo + hi) * 0.5) { break; }

                // Update the bisection
                h_n = (lo + hi) * 0.5;
            }

            // Make the vector and flip the vectors to point to the correct directions
            vec3<Scalar> vec = xyz(shape, h, n, h_n);
            vec[0] *= nm[0] >= 0 ? 1 : -1;
            vec[1] *= nm[1] >= 0 ? 1 : -1;

            return vec;
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

            // Height of the object above the observation plane so we can get planes from it's centre
            const Scalar& c = shape.c();

            // Extend out vec to the ground (divide by z and multiply by h-c)
            vec3<Scalar> v = multiply(u, (c - h) / u[2]);

            // Calculate what the h values must be for each direction
            const Scalar h_n = std::sqrt(v[1] * v[1] + v[2] * v[2]) + c;
            const Scalar h_m = std::sqrt(v[0] * v[0] + v[2] * v[2]) + c;

            // The phi angle for the guess is the angle between <x, y, z> and <x, 0, z>
            const Scalar phi_n =
              std::acos(std::sqrt((v[1] * v[1] + v[2] * v[2]) / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])));
            const Scalar phi_m =
              std::acos(std::sqrt((v[0] * v[0] + v[2] * v[2]) / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])));

            vec2<Scalar> nm = {{shape.n(phi_n, h_n), shape.n(phi_m, h_m)}};
            nm[0] *= u[0] >= 0 ? 1 : -1;
            nm[1] *= u[1] >= 0 ? 1 : -1;

            return nm;
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_NMGRID_MAP_HPP
