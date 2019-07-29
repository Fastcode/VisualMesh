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

#ifndef VISUALMESH_GENERATOR_HEXAPIZZA_HPP
#define VISUALMESH_GENERATOR_HEXAPIZZA_HPP

#include <array>
#include <vector>
#include "mesh/node.hpp"

namespace visualmesh {
namespace generator {

  template <typename Scalar>
  struct HexaPizza {

    template <typename Shape>
    static std::vector<Node<Scalar>> generate(const Shape& shape,
                                              const Scalar& h,
                                              const Scalar& k,
                                              const Scalar& max_distance) {

      // Loop through until we reach our max distance
      std::vector<Node<Scalar>> nodes;

      nodes.push_back(Node<Scalar>{{{0, 0, -1}}, {{0, 0, 0, 0, 0, 0}}});

      // TODO Patch in the bottom of the mesh!

      // We store the start out here, that way we can use it later to work out what the last ring was
      std::size_t start = nodes.size();

      // Loop through our n values until we exceed the max distance
      const Scalar jump = 1.0 / k;
      for (int i = 0; h * std::tan(shape.phi(i * jump, h)) < max_distance; ++i) {

        // Calculate phi phi for our ring, the previous ring, and the next ring
        const Scalar p_phi = shape.phi((i - 1) * jump, h);
        const Scalar c_phi = shape.phi(i * jump, h);
        const Scalar n_phi = shape.phi((i + 1) * jump, h);

        // Calculate delta theta for our ring, the previous ring and the next ring
        const Scalar p_raw_dtheta = shape.theta(p_phi, h);
        const Scalar c_raw_dtheta = shape.theta(c_phi, h);
        const Scalar n_raw_dtheta = shape.theta(n_phi, h);

        // Calculate the number of slices in our ring, the previous ring and the next ring
        // TODO once the centre patch is done replace the 1 here with the correct size
        const int p_slices =
          !std::isfinite(p_raw_dtheta) ? 1 : static_cast<int>(std::ceil(k * 2 * M_PI / p_raw_dtheta));
        const int c_slices = static_cast<int>(std::ceil(k * 2 * M_PI / c_raw_dtheta));
        const int n_slices = static_cast<int>(std::ceil(k * 2 * M_PI / n_raw_dtheta));

        // Recalculate delta theta for each of these slices based on a whole number of spheres
        const Scalar p_dtheta = (M_PI * 2) / p_slices;
        const Scalar c_dtheta = (M_PI * 2) / c_slices;
        const Scalar n_dtheta = (M_PI * 2) / n_slices;

        // Optimisation since we use these a lot
        const Scalar sin_phi = std::sin(c_phi);
        const Scalar cos_phi = std::cos(c_phi);

        // Create this node slice, but first get the position the nodes list so we can work out absolute coordinates
        start = nodes.size();

        // Check for nan theta jumps which happen near the origin where dtheta doesn't make sense
        if (std::isfinite(c_raw_dtheta) && std::isfinite(n_raw_dtheta)) {

          // Loop through and generate all the slices
          for (int j = 0; j < c_slices; ++j) {
            Scalar theta = c_dtheta * j;

            Node<Scalar> n;
            //  Calculate our unit vector with x facing forward and z up
            n.ray = {{
              std::cos(theta) * sin_phi,  //
              std::sin(theta) * sin_phi,  //
              -cos_phi,                   //
              Scalar(0.0)                 //
            }};

            // Get how far we are through this ring as a value between 0 and 1
            const Scalar f = static_cast<Scalar>(j) / static_cast<Scalar>(c_slices);

            // Left and right is just our index += 1 with wraparound
            const int l = static_cast<int>(j > 0 ? start + j - 1 : start + c_slices - 1);
            const int r = static_cast<int>(j + 1 < c_slices ? start + j + 1 : start);

            // Top left and top right are the next ring around nearest left and right with wraparound
            const int tl = start + c_slices + (static_cast<int>(f * n_slices) % n_slices);
            const int tr = start + c_slices + ((static_cast<int>(f * n_slices) + 1) % n_slices);

            // Bottom left and bottom right are the next ring around nearest left and right with wraparound
            const int bl = start - p_slices + (static_cast<int>(f * p_slices) % p_slices);
            const int br = start - p_slices + ((static_cast<int>(f * p_slices) + 1) % p_slices);

            // The absolute indices of our neighbours presented in a clockwise arrangement
            n.neighbours = {{l, tl, tr, r, br, bl}};

            nodes.push_back(n);
          }
        }
      }

      // Clip all neighbours that are past the end to one past the end
      for (int i = start; i < nodes.size(); ++i) {
        for (auto& n : nodes[i].neighbours) {
          n = std::min(n, static_cast<int>(nodes.size()));
        }
      }

      return nodes;
    }
  };

}  // namespace generator
}  // namespace visualmesh

#endif  // VISUALMESH_GENERATOR_HEXAPIZZA_HPP
