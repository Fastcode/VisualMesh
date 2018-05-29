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
#include "mesh/mesh.hpp"
#include "mesh/projected_mesh.hpp"
#include "util/math.hpp"

namespace visualmesh {
namespace engine {
  namespace cpu {

    template <typename Scalar>
    class Engine {
    private:
      vec2<Scalar> project_equidistant(const vec4<Scalar>& p, const Lens<Scalar>& lens) const {
        // Calculate some intermediates
        Scalar theta     = std::acos(p[0]);
        Scalar r         = lens.focal_length * theta;
        Scalar sin_theta = std::sin(theta);

        // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
        vec2<Scalar> screen = {{r * p[1] / sin_theta, r * p[2] / sin_theta}};

        // Apply our offset to move into image space (0 at top left, x to the right, y down)
        vec2<Scalar> image = {{static_cast<Scalar>(lens.dimensions[0] - 1) * static_cast<Scalar>(0.5) - screen[0],
                               static_cast<Scalar>(lens.dimensions[1] - 1) * static_cast<Scalar>(0.5) - screen[1]}};

        return image;
      }

      vec2<Scalar> project_equisolid(const vec4<Scalar>& p, const Lens<Scalar>& lens) const {
        // Calculate some intermediates
        Scalar theta     = std::acos(p[0]);
        Scalar r         = static_cast<Scalar>(2.0) * lens.focal_length * std::sin(theta * static_cast<Scalar>(0.5));
        Scalar sin_theta = std::sin(theta);

        // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
        vec2<Scalar> screen = {{r * p[1] / sin_theta, r * p[2] / sin_theta}};

        // Apply our offset to move into image space (0 at top left, x to the right, y down)
        vec2<Scalar> image = {{static_cast<Scalar>(lens.dimensions[0] - 1) * static_cast<Scalar>(0.5) - screen[0],
                               static_cast<Scalar>(lens.dimensions[1] - 1) * static_cast<Scalar>(0.5) - screen[1]}};

        return image;
      }

      vec2<Scalar> project_rectilinear(const vec4<Scalar>& p, const Lens<Scalar>& lens) const {
        // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
        vec2<Scalar> screen = {{lens.focal_length * p[1] / p[0], lens.focal_length * p[2] / p[0]}};

        // Apply our offset to move into image space (0 at top left, x to the right, y down)
        vec2<Scalar> image = {{static_cast<Scalar>(lens.dimensions[0] - 1) * static_cast<Scalar>(0.5) - screen[0],
                               static_cast<Scalar>(lens.dimensions[1] - 1) * static_cast<Scalar>(0.5) - screen[1]}};

        return image;
      }

    public:
      ProjectedMesh<Scalar> project(const Mesh<Scalar>& mesh,
                                    const std::vector<std::pair<unsigned int, unsigned int>>& ranges,
                                    const mat4<Scalar>& Hoc,
                                    const Lens<Scalar>& lens) const {

        // Convenience variables
        const auto& nodes = mesh.nodes;
        const auto Hco    = transpose(Hoc);

        // Work out how many points total there are
        int n_points = 0;
        for (auto& r : ranges) {
          n_points += r.second - r.first;
        }

        // Output variables
        std::vector<vec2<Scalar>> pixels(n_points);
        std::vector<int> indices(n_points);
        std::vector<std::array<int, 6>> neighbourhood(n_points + 1);  // +1 for the null point

        // Get the indices for each point in the mesh on screen
        auto it = indices.begin();
        for (const auto& range : ranges) {
          auto n = std::next(it, range.second - range.first);
          std::iota(it, n, range.first);
          it = n;
        }

        // Build our reverse lookup, the default point goes to the null point
        std::vector<int> r_lookup(nodes.size(), n_points);
        for (unsigned int i = 0; i < indices.size(); ++i) {
          r_lookup[indices[i]] = i;
        }

        // Build our local neighbourhood map
        for (unsigned int i = 0; i < indices.size(); ++i) {
          const Node<Scalar>& node = nodes[indices[i]];
          for (unsigned int j = 0; j < 6; ++j) {
            const auto& n       = node.neighbours[j];
            neighbourhood[i][j] = r_lookup[n];
          }
        }
        // Last point is the null point
        neighbourhood[n_points].fill(n_points);

        // Project each of the nodes into pixel space
        for (unsigned int i = 0; i < indices.size(); ++i) {
          const Node<Scalar>& node = nodes[indices[i]];

          // Rotate point by matrix (since we are doing this rowwise, it's like we are transposing at the same time)
          vec4<Scalar> p = {{dot(node.ray, Hco[0]), dot(node.ray, Hco[1]), dot(node.ray, Hco[2]), 0}};

          // Any compiler worth it's salt should move this switch outside the for
          switch (lens.projection) {
            case EQUISOLID: pixels[i] = project_equisolid(p, lens); break;
            case EQUIDISTANT: pixels[i] = project_equidistant(p, lens); break;
            case RECTILINEAR: pixels[i] = project_rectilinear(p, lens); break;
          }
        }

        return ProjectedMesh<Scalar>{std::move(pixels), std::move(neighbourhood), std::move(indices)};
      }
    };

  }  // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_ENGINE_HPP
