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

#ifndef VISUALMESH_MODEL_GRID_BASE_HPP
#define VISUALMESH_MODEL_GRID_BASE_HPP

#include <array>
#include <map>
#include <set>
#include <stack>
#include <vector>

#include "mesh/node.hpp"

namespace visualmesh {
namespace model {

    template <int>
    struct GridOffsets;

    template <>
    struct GridOffsets<4> {
        static const std::array<vec2<int>, 4> offsets;
    };
    const std::array<vec2<int>, 4> GridOffsets<4>::offsets = {{
      vec2<int>{{-1, +0}},  // Left
      vec2<int>{{+0, +1}},  // Top
      vec2<int>{{+1, +0}},  // Right
      vec2<int>{{+0, -1}},  // Bottom
    }};
    template <>
    struct GridOffsets<6> {
        static const std::array<vec2<int>, 6> offsets;
    };
    const std::array<vec2<int>, 6> GridOffsets<6>::offsets = {{
      vec2<int>{{-1, +0}},  // Left
      vec2<int>{{-1, +1}},  // Top Left
      vec2<int>{{-0, +1}},  // Top Right
      vec2<int>{{+1, +0}},  // Right
      vec2<int>{{+1, -1}},  // Bottom Right
      vec2<int>{{+0, -1}},  // Bottom Left
    }};
    template <>
    struct GridOffsets<8> {
        static const std::array<vec2<int>, 8> offsets;
    };
    const std::array<vec2<int>, 8> GridOffsets<8>::offsets = {{
      vec2<int>{{-1, +0}},  // Left
      vec2<int>{{-1, +1}},  // Top Left
      vec2<int>{{+0, +1}},  // Top
      vec2<int>{{+1, +1}},  // Top Right
      vec2<int>{{+1, +0}},  // Right
      vec2<int>{{+1, -1}},  // Bottom Right
      vec2<int>{{+0, -1}},  // Bottom
      vec2<int>{{-1, -1}},  // Bottom Left
    }};

    template <typename Scalar, template <typename> class Map, int N_NEIGHBOURS>
    struct GridBase {
    public:
        template <typename Shape>
        static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                                const Scalar& h,
                                                                const Scalar& k,
                                                                const Scalar& max_distance) {

            // Our jumps are based on if we are hexagonal or a quad base
            const Scalar jump = 1.0 / k;

            // Hold the final nodes, as well as a map to create the link locations
            std::vector<Node<Scalar, N_NEIGHBOURS>> output;
            std::map<vec2<int>, int> locations;

            // Perform a flood fill to find all the points that are on the screen
            std::set<vec2<int>> seen;
            std::stack<vec2<int>> stack;
            seen.insert(vec2<int>{{0, 0}});
            stack.push(vec2<int>{{0, 0}});
            while (!stack.empty()) {
                // Get the next point to inspect on the stack
                auto e = stack.top();
                stack.pop();

                // 6 Neighbours are using hexagonal axial coordinates so need special calculations
                vec2<Scalar> nm =
                  N_NEIGHBOURS == 6 ? multiply(
                    vec2<Scalar>{{e[0] + Scalar(0.5) * e[1], std::sqrt(Scalar(3)) * Scalar(0.5) * e[1]}}, jump)
                                    : multiply(cast<Scalar>(e), jump);
                static_assert(N_NEIGHBOURS == 4 || N_NEIGHBOURS == 6 || N_NEIGHBOURS == 8,
                              "You must choose 4, 6 or 8 neighbours");

                // Map the point using our mapping function
                vec3<Scalar> vec = Map<Scalar>::map(shape, nm, h);
                Scalar distance  = norm(head<2>(vec));

                // We only work with this point if we didn't exceed our max distance
                if (distance <= max_distance) {
                    // Add the element to the output and store where this coordinate ended up so we can find it later
                    Node<Scalar, N_NEIGHBOURS> node;
                    node.ray = normalise(vec);
                    locations.emplace(e, output.size());
                    output.push_back(node);

                    // Add in each of our neighbours to be checked
                    for (const auto& o : GridOffsets<N_NEIGHBOURS>::offsets) {
                        vec2<int> n = add(e, o);
                        if (seen.count(n) == 0) {
                            seen.insert(n);
                            stack.push(n);
                        }
                    }
                }
            }

            // Set all the neighbours using the map
            for (const auto& loc : locations) {
                const vec2<int>& coord           = loc.first;
                Node<Scalar, N_NEIGHBOURS>& node = output[loc.second];

                for (int i = 0; i < N_NEIGHBOURS; ++i) {
                    vec2<int> target   = add(coord, GridOffsets<N_NEIGHBOURS>::offsets[i]);
                    node.neighbours[i] = locations.count(target) > 0 ? locations.find(target)->second : output.size();
                }
            }

            return output;
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_GRID_BASE_HPP
