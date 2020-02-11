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

#ifndef VISUALMESH_MODEL_XGRID4_HPP
#define VISUALMESH_MODEL_XGRID4_HPP

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

    /**
     * @brief Takes a point in n, m space (jumps along x and jumps along y) and converts it into a vector to the centre
     *        of the object using the x grid method
     *
     * @param shape the shape object used to calculate the angles
     * @param n     the coordinate in the n coordinate (object jumps along the x axis)
     * @param m     the coordinate in the m coordinate (object jumps along the y axis)
     * @param h     the height of the camera above the observation plane
     * @param c     the height of the object above the observation plane
     *
     * @return a vector <x, y, z> that points to the centre of the object at these coordinates
     */
    template <typename Shape>
    static vec3<Scalar> map(const Shape& shape, const Scalar& n, const Scalar& m, const Scalar& h, const Scalar& c) {
      const Scalar phi_x = shape.phi(n, h);
      const Scalar x     = (h - c) * std::tan(phi_x);

      const Scalar h_p = (h - c) / std::cos(phi_x);
      const Scalar y   = h_p * std::tan(shape.phi(m, h_p + c));

      return vec3<Scalar>{x, y, c - h};
    }

    template <typename Shape>
    static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                            const Scalar& h,
                                                            const Scalar& k,
                                                            const Scalar& max_distance) {

      // How much n we jump every iteration
      const Scalar jump = 1.0 / k;

      // The height difference for the shape between the ground plane and object centre
      const Scalar c = shape.c();

      // Function to build a row that fits on the screen located at n
      auto add_row = [&](const Scalar& n) {
        std::list<vec3<Scalar>> out;
        out.push_back(map(shape, n, 0, h, c));
        for (int j = 1; norm(head<2>(map(shape, n, j * jump, h, c))) < max_distance; ++j) {
          out.push_back(map(shape, n, j * jump, h, c));
          out.push_front(map(shape, n, -j * jump, h, c));
        }
        return std::vector<vec3<Scalar>>(out.begin(), out.end());
      };

      // Build the grid that covers the area by generating each row until the length exceeds the max distance
      std::list<std::vector<vec3<Scalar>>> vecs;
      unsigned int n_nodes = 0;
      vecs.push_back(add_row(0));
      n_nodes += vecs.back().size();
      for (int i = 1; norm(head<2>(map(shape, i * jump, 0, h, c))) < max_distance; ++i) {
        vecs.push_back(add_row(i * jump));
        vecs.push_front(add_row(-i * jump));
        n_nodes += vecs.back().size();
        n_nodes += vecs.front().size();
      }

      // Go through the lists we built up to work out coordinates
      std::vector<Node<Scalar, N_NEIGHBOURS>> output;
      for (auto it = vecs.begin(); it != vecs.end(); ++it) {
        // Get relevant sizes for calculations
        int start     = static_cast<int>(output.size());
        int prev_size = it == vecs.begin() ? 0 : static_cast<int>(std::prev(it)->size());
        int curr_size = static_cast<int>(it->size());
        int next_size = std::next(it) == vecs.end() ? 0 : static_cast<int>(std::next(it)->size());

        for (int j = 0; j < static_cast<int>(it->size()); ++j) {
          Node<Scalar, N_NEIGHBOURS> node;
          node.ray = normalise((*it)[j]);

          // Adjacent neighbours
          node.neighbours[0] = j == 0 ? n_nodes : start + j - 1;
          node.neighbours[2] = j + 1 == static_cast<int>(it->size()) ? n_nodes : start + j + 1;

          // Our point offset from the centre of our row
          int c = j - curr_size / 2;

          // Our point position on other rows (relative to the start)
          int n = c + next_size / 2;
          int p = c + prev_size / 2;

          // Up/down neighbours
          node.neighbours[1] = 0 <= p && p < prev_size ? start - prev_size + p : n_nodes;
          node.neighbours[3] = 0 <= n && n < next_size ? start + curr_size + n : n_nodes;

          output.push_back(node);
        }
      }

      return output;
    }
  };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_XGRID4_HPP
