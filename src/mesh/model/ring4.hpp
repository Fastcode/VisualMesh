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

#ifndef VISUALMESH_MODEL_RING4_HPP
#define VISUALMESH_MODEL_RING4_HPP

#include <array>
#include <vector>

#include "mesh/node.hpp"

namespace visualmesh {
namespace model {

    /**
     * @brief Model utilising the Ring4 method
     *
     * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
     */
    template <typename Scalar>
    struct Ring4 {
    private:
        /**
         * @brief
         *  Converts angles into a unit vector where phi is defined as the angle from -z to the vector, and theta is
         *  measured around the z axis.
         *
         * @details
         *  The reason that sin_phi and cos_phi are passed in rather than being calculated every time is that where this
         *  function is called both of those are precalcualted and available. Therefore we can improve the performance
         * by using these already calculated values.
         *
         * @param sin_phi sin of the phi angle (up from -z)
         * @param cos_phi cos of the phi angle (up from -z)
         * @param theta   the theta angle (around the z axis)
         *
         * @return The unit vector that these angles represent
         */
        static inline vec3<Scalar> unit_vector(const Scalar& sin_phi, const Scalar& cos_phi, const Scalar& theta) {
            return vec3<Scalar>{{std::cos(theta) * sin_phi, std::sin(theta) * sin_phi, -cos_phi}};
        }

    public:
        /// The number of neighbours that each node in the graph has
        static constexpr size_t N_NEIGHBOURS = 4;

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

            // Loop through until we reach our max distance
            std::vector<Node<Scalar, N_NEIGHBOURS>> nodes;

            // Create our first interconnected ring of 6 values that exist in a ring
            const Scalar start_n      = Scalar(1.0) / (2.0 * k);
            const Scalar phi_0        = shape.phi(start_n, h);
            const Scalar cos_phi_0    = std::cos(phi_0);
            const Scalar sin_phi_0    = std::sin(phi_0);
            const int first_ring_size = 4;

            // Around the origin are 6 equally spaced points
            for (int j = 0; j < first_ring_size; ++j) {
                Node<Scalar, N_NEIGHBOURS> n;
                n.ray = unit_vector(sin_phi_0, cos_phi_0, j * M_PI * 2.0 / 6.0);

                // Left and right is just our index += 1 with wraparound
                const int l = (j + 3) % first_ring_size;
                const int r = (j + 1) % first_ring_size;

                // Top left and top right are the next ring which we don't know about yet
                const int b = (j + 2) % first_ring_size;
                const int t = r;  // start + c_slices + ((static_cast<int>(f * n_slices) + 1) % n_slices);

                // The absolute indices of our neighbours presented in a clockwise arrangement
                n.neighbours = {{l, t, r, b}};

                nodes.push_back(n);
            }

            // We store the start out here, that way we can use it later to work out what the last ring was
            std::size_t start = nodes.size();

            // Loop through our n values until we exceed the max distance
            const Scalar jump = 1.0 / k;
            for (int i = 0; h * std::tan(shape.phi(i * jump + start_n, h)) < max_distance; ++i) {

                // Calculate phi phi for our ring, the previous ring, and the next ring
                const Scalar p_phi = shape.phi((i - 1) * jump + start_n, h);
                const Scalar c_phi = shape.phi(i * jump + start_n, h);
                const Scalar n_phi = shape.phi((i + 1) * jump + start_n, h);

                // Calculate delta theta for our ring, the previous ring and the next ring
                const Scalar p_raw_dtheta = shape.theta(p_phi, h);
                const Scalar c_raw_dtheta = shape.theta(c_phi, h);
                const Scalar n_raw_dtheta = shape.theta(n_phi, h);

                // Calculate the number of slices in our ring, the previous ring and the next ring
                const int p_slices = !std::isfinite(p_raw_dtheta)
                                       ? first_ring_size
                                       : static_cast<int>(std::ceil(k * 2 * M_PI / p_raw_dtheta));
                const int c_slices = static_cast<int>(std::ceil(k * 2 * M_PI / c_raw_dtheta));
                const int n_slices = static_cast<int>(std::ceil(k * 2 * M_PI / n_raw_dtheta));

                // Recalculate delta theta for each of these slices based on a whole number of spheres
                const Scalar c_dtheta = (M_PI * 2) / c_slices;

                // Optimisation since we use these a lot
                const Scalar sin_phi = std::sin(c_phi);
                const Scalar cos_phi = std::cos(c_phi);

                // Create this node slice, but first get the position the nodes list so we can work out absolute
                // coordinates
                start = nodes.size();

                // Check for nan theta jumps which happen near the origin where dtheta doesn't make sense
                if (std::isfinite(c_raw_dtheta) && std::isfinite(n_raw_dtheta)) {

                    // Loop through and generate all the slices
                    for (int j = 0; j < c_slices; ++j) {
                        Scalar theta = c_dtheta * j;

                        Node<Scalar, N_NEIGHBOURS> n;
                        //  Calculate our unit vector with x facing forward and z up
                        n.ray = unit_vector(sin_phi, cos_phi, theta);

                        // Get how far we are through this ring as a value between 0 and 1
                        const Scalar f = Scalar(j) / Scalar(c_slices);

                        // Left and right is just our index += 1 with wraparound
                        const int l = static_cast<int>(j > 0 ? start + j - 1 : start + c_slices - 1);
                        const int r = static_cast<int>(j + 1 < c_slices ? start + j + 1 : start);

                        // Top left and top right are the next ring around nearest left and right with wraparound
                        const int t = start + c_slices + (static_cast<int>(std::round(f * n_slices)) % n_slices);

                        // Bottom left and bottom right are the next ring around nearest left and right with wraparound
                        const int b = start - p_slices + (static_cast<int>(std::round(f * p_slices)) % p_slices);

                        // The absolute indices of our neighbours presented in a clockwise arrangement
                        n.neighbours = {{l, t, r, b}};

                        nodes.push_back(n);
                    }
                }
            }

            // Clip all neighbours that are past the end to one past the end
            for (unsigned int i = start; i < nodes.size(); ++i) {
                for (auto& n : nodes[i].neighbours) {
                    n = std::min(n, static_cast<int>(nodes.size()));
                }
            }

            return nodes;
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_RING4_HPP
