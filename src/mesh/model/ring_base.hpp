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

#ifndef VISUALMESH_MODEL_RING_BASE_HPP
#define VISUALMESH_MODEL_RING_BASE_HPP

#include <array>
#include <vector>

#include "mesh/node.hpp"

namespace visualmesh {
namespace model {

    /**
     * @brief Model utilising the Ring4 method
     *
     * @tparam Scalar   the scalar type used for calculations and storage (normally one of float or double)
     * @tparam Ring     the specific ring type that is used to generate the neighbour indices
     */
    template <typename Scalar, template <typename> class Ring, int N_NEIGHBOURS>
    struct RingBase {
    private:
        /**
         * @brief Calculates the details needed for a slice at a paticular n value
         *
         * @tparam Shape  the type of shape that this model will use to create the mesh
         *
         * @param shape the shape instance that is used for calculating details
         * @param n     the radial ring out from the origin in object space
         * @param h     the height of the camera above the observation plane
         */
        template <typename Shape>
        static std::pair<vec2<Scalar>, int> slice(const Shape& shape,
                                                  const Scalar& n,
                                                  const Scalar& h,
                                                  const Scalar& k) {

            // Get phi from the shape
            const Scalar& phi = shape.phi(n, h);

            // Specifically for the origin there is only one point
            if (n == 0) { return std::make_pair(vec2<Scalar>{0, 2.0 * M_PI}, 1); }

            // At n = 0.5 we will have a delta theta of pi (two objects touching would fit). At any shorter distance
            // than this then the theta angle does not make sense as the objects intersect with the origin. As an easy
            // method to adapt below this we will assume a linear interpolation between 1 sphere at the origin and 2
            // objects at 0.5n. Therefore once this is cut up by the k value it should give an approximation
            const Scalar dtheta = n <= 0.5 ? (2.0 * M_PI) / (1.0 + n * Scalar(2.0)) : shape.theta(phi, h);

            return std::make_pair(vec2<Scalar>{phi, dtheta}, int(std::ceil(k * 2 * M_PI / dtheta)));
        }

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
         * @brief Generates the visual mesh vectors and graph using the Ring4 method
         *
         * @tparam Shape  the type of shape that this model will use to create the mesh
         *
         * @param shape         the shape instance that is used for calculating details
         * @param h             the height of the camera above the observation plane
         * @param k             the number of radial intersections per object
         * @param max_distance  the maximum distance that this mesh will be targeted for
         *
         * @return the visual mesh graph that was generated
         */
        template <typename Shape>
        static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                                const Scalar& h,
                                                                const Scalar& k,
                                                                const Scalar& max_distance) {

            std::vector<Node<Scalar, N_NEIGHBOURS>> nodes;
            const Scalar jump = 1.0 / k;
            int start         = 1;

            // Create the origin node and connect it
            nodes.emplace_back(Node<Scalar, N_NEIGHBOURS>{
              vec3<Scalar>{{0.0, 0.0, -1.0}},
              std::array<int, N_NEIGHBOURS>{},
            });
            int n_first       = slice(shape, jump, h, k).second;
            Scalar first_jump = Scalar(n_first) / Scalar(N_NEIGHBOURS);
            for (unsigned int i = 0; i < N_NEIGHBOURS; ++i) {
                nodes.front().neighbours[i] = int(i * first_jump) + start;
            }

            // Loop through until we reach our max distance
            for (int i = 1; h * std::tan(shape.phi(i * jump, h)) < max_distance; ++i) {

                // Calculate the n values for our ring and adjacent rings
                const Scalar& p_n = (i - 1) * jump;
                const Scalar& c_n = i * jump;
                const Scalar& n_n = (i + 1) * jump;

                // Calculate the phi and delta theta angles
                const auto& prev = slice(shape, p_n, h, k);
                const auto& curr = slice(shape, c_n, h, k);
                const auto& next = slice(shape, n_n, h, k);

                // Recalculate delta theta for each of these slices based on a whole number of objects
                const Scalar d_theta = (M_PI * 2) / curr.second;

                // Loop through and generate all the theta slices
                start = int(nodes.size());
                for (int j = 0; j < curr.second; ++j) {

                    Scalar theta = d_theta * j;

                    Node<Scalar, N_NEIGHBOURS> n;
                    //  Calculate our unit vector with x facing forward and z up
                    n.ray = unit_vector(curr.first[0], theta);

                    // Get the neighbours using our specific class
                    n.neighbours.fill(start);
                    n.neighbours =
                      add(n.neighbours, Ring<Scalar>::neighbours(j, prev.second, curr.second, next.second));

                    // Add to the list
                    nodes.push_back(n);
                }
            }

            // Clip all neighbours that are past the end to one past the end
            for (int i = start; i < int(nodes.size()); ++i) {
                for (auto& n : nodes[i].neighbours) {
                    n = std::min(n, int(nodes.size()));
                }
            }

            return nodes;
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_RING_BASE_HPP
