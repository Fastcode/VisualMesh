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
#include <cmath>
#include <vector>

#include "polar_map.hpp"
#include "visualmesh/node.hpp"
#include "visualmesh/utility/math.hpp"

namespace visualmesh {
namespace model {

    /**
     * @brief Model utilising the Ring4 method
     *
     * @tparam Scalar   the scalar type used for calculations and storage (normally one of float or double)
     * @tparam Ring     the specific ring type that is used to generate the neighbour indices
     */
    template <typename Scalar, template <typename> class Ring, int N_NEIGHBOURS>
    struct RingBase : public PolarMap<Scalar> {
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
            int n_first       = int(std::ceil((k * Scalar(2.0 * M_PI)) / shape.theta(jump, h)));
            Scalar first_jump = Scalar(n_first) / Scalar(N_NEIGHBOURS);
            for (unsigned int i = 0; i < N_NEIGHBOURS; ++i) {
                nodes.front().neighbours[i] = int(i * first_jump) + start;
            }

            // Loop through until we reach our max distance
            for (int i = 1; h * std::tan(shape.phi(i * jump, h)) < max_distance; ++i) {
                // Calculate the n values for our ring and adjacent rings
                const Scalar& n_p = (i - 1) * jump;
                const Scalar& n_c = i * jump;
                const Scalar& n_n = (i + 1) * jump;

                // Calculate the number of slices for each ring
                // Specifically for the case where n == 0 we have 1 point (origin ring)
                const Scalar s_p = i == 1 ? Scalar(1.0) : k * Scalar(2.0 * M_PI) / shape.theta(n_p, h);
                const Scalar s_c = k * Scalar(2.0 * M_PI) / shape.theta(n_c, h);
                const Scalar s_n = k * Scalar(2.0 * M_PI) / shape.theta(n_n, h);

                // Calculate how much we should jump in m space to make an even number of points by oversampling by one
                const Scalar m_jump = s_c / (k * std::ceil(s_c));

                // Loop through and generate all the theta slices
                start = int(nodes.size());
                for (int j = 0; j < int(std::ceil(s_c)); ++j) {
                    Node<Scalar, N_NEIGHBOURS> n;
                    n.ray = PolarMap<Scalar>::map(shape, h, vec2<Scalar>{{n_c, j * m_jump}});

                    // Get the neighbours using our specific class
                    n.neighbours.fill(start);
                    n.neighbours =
                      add(n.neighbours, Ring<Scalar>::neighbours(j, std::ceil(s_p), std::ceil(s_c), std::ceil(s_n)));

                    // Add to the list
                    nodes.push_back(n);
                }
            }

            // Clip all neighbours that are past the end to one past the end
            // We start at start - 1 as a hack so that if there is only a single point (the origin) it will be corrected
            for (int i = start - 1; i < int(nodes.size()); ++i) {
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
