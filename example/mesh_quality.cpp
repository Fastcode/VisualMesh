
#include <iostream>

#include "ArrayPrint.hpp"
#include "Timer.hpp"
//
#include "geometry/Circle.hpp"
#include "geometry/Sphere.hpp"
#include "util/math.hpp"
#include "util/phi_difference.hpp"
#include "visualmesh.hpp"

template <typename Scalar>
using vec3 = visualmesh::vec3<Scalar>;

int main() {

  visualmesh::geometry::Sphere<float> shape(0.0949996);
  visualmesh::Mesh<float, visualmesh::generator::QuadPizza> mesh(shape, 1.0, 1, 20);

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
        vec3<float> diff_01 =
          visualmesh::util::phi_difference(mesh.h, {ray0[0], ray0[1], ray0[2]}, {ray1[0], ray1[1], ray1[2]});
        float n_01 = std::abs(shape.n(diff_01[1], diff_01[0]) - shape.n(diff_01[2], diff_01[0]));
        std::cout << " *: " << n_01;

        if (n1 < int(mesh.nodes.size())) {
          const auto& ray2 = mesh.nodes[n1].ray;

          // The difference between the two neighbour rays
          vec3<float> diff_12 =
            visualmesh::util::phi_difference<float>(mesh.h, {ray1[0], ray1[1], ray1[2]}, {ray2[0], ray2[1], ray2[2]});
          float n_12 = std::abs(shape.n(diff_12[1], diff_12[0]) - shape.n(diff_12[2], diff_12[0]));

          std::cout << " o: " << n_12;
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
