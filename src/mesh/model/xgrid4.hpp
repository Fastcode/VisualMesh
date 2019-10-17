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

#ifndef VISUALMESH_MODEL_XGRID_HPP
#define VISUALMESH_MODEL_XGRID_HPP

#include <array>
#include <list>
#include <vector>

#include "mesh/node.hpp"

namespace visualmesh {
namespace model {

  template <typename Scalar>
  struct XGrid4 {
  public:
    static constexpr size_t N_NEIGHBOURS = 4;

    template <typename Shape>
    static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                            const Scalar& h,
                                                            const Scalar& k,
                                                            const Scalar& max_distance) {

      // How much n we jump every iteration
      const Scalar jump = 1.0 / k;

      // The height difference for the shape between the ground plane and object centre
      const Scalar c = shape.c();

      // Loop through the values for phi
      std::list<Scalar> phi_xs(1, shape.phi(0, h));
      for (int i = 1; h * std::tan(shape.phi(i * jump, h)) < max_distance; ++i) {

        // Add the phis in both directions
        phi_xs.push_back(shape.phi(i * jump, h));
        phi_xs.push_front(shape.phi(-i * jump, h));
      }

      // For each plane phi we need to work out our h' and calculate our rays
      std::vector<std::vector<vec3<Scalar>>> grid;
      for (const auto& phi_x : phi_xs) {
        // Calculate h' and the distance to our object centre in x
        // In this case h' is to the plane in the centre of the object
        const Scalar h_p = (h - c) / std::cos(phi_x);
        const Scalar x   = (h - c) * std::tan(phi_x);

        // Calculate our y values in the plane
        std::list<vec3<Scalar>> row(1, normalise(vec3<Scalar>{x, h_p * std::tan(shape.phi(0, h_p + c)), c - h}));
        for (int i = 0; i < static_cast<int>(phi_xs.size()) / 2; ++i) {

          const Scalar y1 = h_p * std::tan(shape.phi(i * jump, h_p + c));
          const Scalar y2 = h_p * std::tan(shape.phi(-i * jump, h_p + c));

          row.push_back(normalise(vec3<Scalar>{x, y1, c - h}));
          row.push_front(normalise(vec3<Scalar>{x, y2, c - h}));
        }

        grid.emplace_back(row.begin(), row.end());
      }

      // Calculate the unit vectors from the two phi values and the coordinates of the neighbours as they are in a grid
      std::vector<Node<Scalar, N_NEIGHBOURS>> output(phi_xs.size() * phi_xs.size());
      for (unsigned int i = 0; i < grid.size(); ++i) {
        const auto& row = grid[i];
        for (unsigned int j = 0; j < row.size(); ++j) {

          // Calculate our index in the nodes list
          int idx = i * grid.size() + j;

          // Put in our ray
          output[idx].ray = row[j];

          // Calculate the indicies of our four neighbours accounting for off screen
          output[idx].neighbours[0] = j == 0 ? output.size() : (i * grid.size() + j - 1);
          output[idx].neighbours[1] = i + 1 == grid.size() ? output.size() : ((i + 1) * grid.size() + j);
          output[idx].neighbours[2] = j + 1 == grid.size() ? output.size() : (i * grid.size() + j + 1);
          output[idx].neighbours[3] = i == 0 ? output.size() : ((i - 1) * grid.size() + j);
        }
      }

      return output;
    }
  };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_XGRID_HPP
