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

#ifndef VISUALMESH_TEST_MESH_HPP
#define VISUALMESH_TEST_MESH_HPP

#include "mesh.hpp"

namespace visualmesh {

template <typename Scalar>
constexpr vec3<Scalar> phi_difference(const Scalar& h, const vec3<Scalar>& a_u, const vec3<Scalar>& b_u) {

  // Project the vectors to the ground
  vec3<Scalar> a = multiply(a_u, h / a_u[2]);
  vec3<Scalar> b = multiply(b_u, h / b_u[2]);

  // Distance from point to line equation
  vec3<Scalar> u       = subtract(b, a);
  const Scalar h_prime = norm(cross(a, u)) / norm(u);

  // Calculate phi0 and phi1
  const Scalar phi0 = std::acos(h_prime / norm(a));
  const Scalar phi1 = std::acos(h_prime / norm(b));

  // Actual angle between a and b
  const Scalar theta = std::acos(dot(a_u, b_u));

  // Choose the combination of phi0 and phi1 that give the angle closest to the true angle of theta
  return {h_prime, phi0, std::abs(phi0 - phi1 - theta) < std::abs(phi0 + phi1 - theta) ? phi1 : -phi1};
}

template <typename Scalar, typename Shape>
void test_mesh(const Mesh<Scalar>& mesh, const Shape& shape) {

  // Loop through all the nodes in the mesh
  for (const auto& node : mesh.nodes) {

    // Our ray pointing in the centre of the cluster
    const auto& ray0 = node.ray;

    // We look through each of our neighbours to see how good we are
    for (unsigned int i = 0; i < node.neighbours.size(); ++i) {

      // We get our next two neighbours in a clockwise direction
      int n0 = node.neighbours[i];
      int n1 = node.neighbours[(i + 1) % node.neighbours.size()];

      // Ignore points that go off the screen
      std::cout << "n" << i << " d: " << mesh.h * std::sqrt(1 - ray0[2] * ray0[2]) / -ray0[2];
      if (n0 < int(mesh.nodes.size())) {
        // The neighbours ray
        const auto& ray1 = mesh.nodes[n0].ray;
        // Difference between us and our neighbour ray
        vec3<Scalar> diff_01 = phi_difference<Scalar>(mesh.h, {ray0[0], ray0[1], ray0[2]}, {ray1[0], ray1[1], ray1[2]});
        Scalar n_01          = std::abs(shape.n(diff_01[1], diff_01[0]) - shape.n(diff_01[2], diff_01[0]));
        std::cout << " *: " << n_01;

        if (n1 < int(mesh.nodes.size())) {
          const auto& ray2 = mesh.nodes[n1].ray;

          // The difference between the two neighbour rays
          vec3<Scalar> diff_12 =
            phi_difference<Scalar>(mesh.h, {ray1[0], ray1[1], ray1[2]}, {ray2[0], ray2[1], ray2[2]});
          Scalar n_12 = std::abs(shape.n(diff_12[1], diff_12[0]) - shape.n(diff_12[2], diff_12[0]));

          std::cout << " o: " << n_12;
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
}  // namespace visualmesh

#endif  // VISUALMESH_TEST_MESH_HPP
