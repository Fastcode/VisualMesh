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

#ifndef VISUALMESH_ENGINE_CPU_ENGINE_HPP
#define VISUALMESH_ENGINE_CPU_ENGINE_HPP

#include <numeric>

#include "classifier.hpp"
#include "mesh/mesh.hpp"
#include "mesh/network_structure.hpp"
#include "mesh/projected_mesh.hpp"
#include "util/math.hpp"
#include "util/projection.hpp"

namespace visualmesh {
namespace engine {
  namespace cpu {

    template <typename Scalar>
    class Engine {

    public:
      template <template <typename> class Model>
      ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> project(const Mesh<Scalar, Model>& mesh,
                                                                 const std::vector<std::pair<int, int>>& ranges,
                                                                 const mat4<Scalar>& Hoc,
                                                                 const Lens<Scalar>& lens) const {

        // Convenience variables
        const auto& nodes = mesh.nodes;
        const mat3<Scalar> Rco(block<3, 3>(transpose(Hoc)));

        // Work out how many points total there are in the ranges
        unsigned int n_points = 0;
        for (auto& r : ranges) {
          n_points += r.second - r.first;
        }

        // Output variables
        std::vector<int> global_indices;
        global_indices.reserve(n_points);
        std::vector<vec2<Scalar>> pixels;
        pixels.reserve(n_points);

        // Loop through adding global indices and pixel coordinates
        for (const auto& range : ranges) {
          for (int i = range.first; i < range.second; ++i) {
            // Even though we have already gone through a bsp to remove out of range points, sometimes it's not perfect
            // and misses by a few pixels. So as we are projecting the points here we also need to check that they are
            // on screen
            auto px = ::visualmesh::project(multiply(Rco, nodes[i].ray), lens);
            if (0 < px[0] && px[0] + 1 < lens.dimensions[0] && 0 < px[1] && px[1] + 1 < lens.dimensions[1]) {
              global_indices.emplace_back(i);
              pixels.emplace_back(px);
            }
          }
        }

        // Update the number of points to account for how many pixels we removed
        n_points = pixels.size();

        // Build our reverse lookup, the default point goes to the null point
        std::vector<int> r_lookup(nodes.size() + 1, n_points);
        for (unsigned int i = 0; i < n_points; ++i) {
          r_lookup[global_indices[i]] = i;
        }

        // Build our local neighbourhood map
        std::vector<std::array<int, Model<Scalar>::N_NEIGHBOURS>> neighbourhood(n_points + 1);  // +1 for the null point
        for (unsigned int i = 0; i < n_points; ++i) {
          const Node<Scalar, Model<Scalar>::N_NEIGHBOURS>& node = nodes[global_indices[i]];
          for (unsigned int j = 0; j < node.neighbours.size(); ++j) {
            const auto& n       = node.neighbours[j];
            neighbourhood[i][j] = r_lookup[n];
          }
        }
        // Last point is the null point
        neighbourhood[n_points].fill(n_points);

        return ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS>{
          std::move(pixels), std::move(neighbourhood), std::move(global_indices)};
      }

      template <template <typename> class Model>
      auto make_classifier(const network_structure_t<Scalar>& structure) {
        return Classifier<Scalar, Model>(this, structure);
      }
    };

  }  // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_ENGINE_HPP
