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
      ProjectedMesh<Scalar> project(const Mesh<Scalar>& mesh,
                                    const std::vector<std::pair<int, int>>& ranges,
                                    const mat4<Scalar>& Hoc,
                                    const Lens<Scalar>& lens) const {

        // Convenience variables
        const auto& nodes = mesh.nodes;
        const mat3<Scalar> Rco(block<3, 3>(transpose(Hoc)));

        // Work out how many points total there are
        int n_points = 0;
        for (auto& r : ranges) {
          n_points += r.second - r.first;
        }

        // Output variables
        std::vector<int> indices(n_points);
        std::vector<int> global_indices;
        global_indices.reserve(n_points);
        std::vector<vec2<Scalar>> pixels;
        pixels.reserve(n_points);

        // Get the indices for each point in the mesh on screen
        auto it = indices.begin();
        for (const auto& range : ranges) {
          auto n = std::next(it, range.second - range.first);
          std::iota(it, n, range.first);
          it = n;
        }

        // Project each of the nodes into pixel space
        for (unsigned int i = 0; i < indices.size(); ++i) {
          const Node<Scalar>& node = nodes[indices[i]];
          // Rotate the ray by the rotation matrix and project to pixel coordinates
          vec2<Scalar> px = ::visualmesh::project(multiply(Rco, node.ray), lens);

          // Check if the pixel is on the screen, this is needed as the cutoffs for some lenses aren't perfect yet
          if (0 < px[0] && px[0] < lens.dimensions[0] - 1 && 0 < px[1] && px[1] < lens.dimensions[1] - 1) {
            pixels.emplace_back(px);
            global_indices.emplace_back(indices[i]);
          }
        }

        // Update the number of points to account for how many pixels we removed
        n_points = pixels.size();

        // Build our reverse lookup, the default point goes to the null point
        std::vector<int> r_lookup(nodes.size(), n_points);
        for (unsigned int i = 0; i < n_points; ++i) {
          r_lookup[global_indices[i]] = i;
        }


        // Build our local neighbourhood map
        std::vector<std::array<int, 6>> neighbourhood(n_points + 1);  // +1 for the null point
        for (unsigned int i = 0; i < n_points; ++i) {
          const Node<Scalar>& node = nodes[global_indices[i]];
          for (unsigned int j = 0; j < 6; ++j) {
            const auto& n       = node.neighbours[j];
            neighbourhood[i][j] = r_lookup[n];
          }
        }
        // Last point is the null point
        neighbourhood[n_points].fill(n_points);

        return ProjectedMesh<Scalar>{std::move(pixels), std::move(neighbourhood), std::move(global_indices)};
      }

      auto make_classifier(const network_structure_t<Scalar>& structure) {
        return Classifier<Scalar>(this, structure);
      }
    };

  }  // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_ENGINE_HPP
