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
      for (Scalar n = 0;; n += 1.0 / k) {

        // Calculate phi and theta for this shape and calculate how many slices ensures we have at least enough
        Scalar phi    = shape.phi(n, h);
        Scalar dtheta = shape.theta(phi, h);
        int slices    = static_cast<int>(std::ceil(k * 2 * M_PI / dtheta));
        dtheta        = (M_PI * 2) / slices;

        // Push back this new slice
        std::size_t start = nodes.size();
        if (!std::isnan(dtheta)) {
          for (int i = 0; i < slices; ++i) {
            Scalar theta = dtheta * i;

            Node<Scalar> n;
            //  Calculate our unit vector with x facing forward and z up
            Scalar sin_phi = std::sin(phi);
            n.ray          = {{
              std::cos(theta) * sin_phi,  //
              std::sin(theta) * sin_phi,  //
              -std::cos(phi),             //
              Scalar(0.0)                 //
            }};
            // Calculate the absolute indices of our 6 neighbours presented in a clockwise fashion
            n.neighbours = {{
              static_cast<int>(i > 0 ? start + i - 1 : start + slices - 1),  // l
              0,                                                             // tl
              0,                                                             // tr
              static_cast<int>(i + 1 < slices ? start + i + 1 : start),      // r
              0,                                                             // br
              0,                                                             // bl
            }};

            nodes.push_back(n);
          }
        }

        // End when our distance just went over our max distance
        if (h * std::tan(phi) > max_distance) break;
      }

      // TODO all the neighbours that are out of range need to be clipped back

      return nodes;
    }
  };

}  // namespace generator
}  // namespace visualmesh

#endif  // VISUALMESH_GENERATOR_HEXAPIZZA_HPP
