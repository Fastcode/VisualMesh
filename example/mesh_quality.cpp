
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

template <typename Scalar, size_t Neighbours>
struct NodeQuality {
  /// The distance this node is from the origin
  Scalar distance;
  /// The angle this node is around the z axis
  Scalar angle;
  /// The number of object jumps between this node and the nodes around it
  std::array<Scalar, Neighbours> radial;
  /// The number of object jumps between each neighbour and the subsequent neighbour
  std::array<Scalar, Neighbours> cyclical;
  /// The angle between each neighbour and the subsequent neighbour
  std::array<Scalar, Neighbours> angular;
};

template <typename Scalar, template <typename> class Shape, template <typename> class Generator>
std::vector<NodeQuality<Scalar, Generator<Scalar>::N_NEIGHBOURS>> check_quality(
  const Shape<Scalar>& shape, const visualmesh::Mesh<Scalar, Generator>& mesh) {

  constexpr size_t N_NEIGHBOURS = Generator<Scalar>::N_NEIGHBOURS;

  // Loop through all the nodes in the mesh
  std::vector<NodeQuality<Scalar, N_NEIGHBOURS>> nodes;
  for (const auto& node : mesh.nodes) {
    NodeQuality<Scalar, N_NEIGHBOURS> quality;

    // Our ray pointing in the centre of the cluster
    const auto& r0 = node.ray;

    // The rays location in the space
    quality.distance = (mesh.h - shape.c()) * std::sqrt(1 - r0[2] * r0[2]) / -r0[2];
    quality.angle    = std::atan2(r0[1], r0[0]);

    // By default set things to nan as a "this node did not exist"
    quality.radial.fill(std::numeric_limits<Scalar>::quiet_NaN());
    quality.cyclical.fill(std::numeric_limits<Scalar>::quiet_NaN());
    quality.angular.fill(std::numeric_limits<Scalar>::quiet_NaN());

    // We look through each of our neighbours to see how good we are
    for (unsigned int i = 0; i < node.neighbours.size(); ++i) {

      // We get our next two neighbours in a clockwise direction
      int n1 = node.neighbours[i];
      int n2 = node.neighbours[(i + 1) % node.neighbours.size()];

      // Ignore points that go off the screen
      if (n1 < int(mesh.nodes.size())) {
        // The neighbours ray
        const auto& r1 = mesh.nodes[n1].ray;

        // Radial difference to our neighbour
        auto r_d          = visualmesh::util::phi_difference(mesh.h, shape.c(), r0, r1);
        quality.radial[i] = std::abs(shape.n(r_d.phi_0, r_d.h_prime) - shape.n(r_d.phi_1, r_d.h_prime));

        // Ignore points that go off the screen
        if (n2 < int(mesh.nodes.size())) {
          const auto& r2 = mesh.nodes[n2].ray;

          // The distance difference between the two neighbour rays
          auto c_d            = visualmesh::util::phi_difference<Scalar>(mesh.h, shape.c(), r1, r2);
          quality.cyclical[i] = std::abs(shape.n(c_d.phi_0, c_d.h_prime) - shape.n(c_d.phi_1, c_d.h_prime));

          // The angular difference between two neighbourhood rays
          quality.angular[i] = std::acos(visualmesh::dot(visualmesh::normalise(visualmesh::cross(r0, r1)),
                                                         visualmesh::normalise(visualmesh::cross(r0, r2))));
        }
      }
    }

    nodes.push_back(quality);
  }

  return nodes;
}

template <typename Scalar, size_t Neighbours>
struct Statistics {
  Statistics() {
    means.fill(0);
    sums.fill(0);
    counts.fill(0);
  }

  void update(const std::array<Scalar, Neighbours> input) {

    for (unsigned int i = 0; i < Neighbours; ++i) {
      if (!std::isnan(input[i]) && input[i] != 0) {
        sums[i] += input[i];
        counts[i]++;
        means[i] = sums[i] / counts[i];
      }
    }
  }

  std::array<Scalar, Neighbours> means;
  std::array<Scalar, Neighbours> sums;
  std::array<uint32_t, Neighbours> counts;
};

template <typename Scalar, size_t Neighbours>
void print_quality(const std::vector<NodeQuality<Scalar, Neighbours>>& nodes, const Scalar& k) {

  // Storage for the statistics
  Statistics<Scalar, Neighbours> radial;
  Statistics<Scalar, Neighbours> radial_var;
  Statistics<Scalar, Neighbours> cyclical;
  Statistics<Scalar, Neighbours> cyclical_var;
  Statistics<Scalar, Neighbours> angular;
  Statistics<Scalar, Neighbours> angular_var;

  // Go through all the nodes and accumulate for the mean value
  for (const auto& node : nodes) {
    radial.update(visualmesh::multiply(node.radial, k));
    cyclical.update(visualmesh::multiply(node.cyclical, k));
    angular.update(visualmesh::multiply(node.angular, static_cast<Scalar>(Neighbours * (M_PI * 2.0))));
  }

  // Sum up the variance
  for (const auto& node : nodes) {
    auto v = visualmesh::subtract(node.radial, radial.means);
    radial_var.update(visualmesh::multiply(v, v));
    auto c = visualmesh::subtract(node.cyclical, cyclical.means);
    cyclical_var.update(visualmesh::multiply(c, c));
    auto a = visualmesh::subtract(node.angular, angular.means);
    angular_var.update(visualmesh::multiply(a, a));
  }

  std::cout << std::setprecision(4);
  for (unsigned int i = 0; i < Neighbours; ++i) {
    std::cout << "* " << (radial.means[i]) << "±" << (std::sqrt(radial_var.means[i]));
    std::cout << " o " << (cyclical.means[i]) << "±" << (std::sqrt(cyclical_var.means[i]));
    std::cout << " a " << (angular.means[i]) << "±" << (std::sqrt(angular_var.means[i]));
    std::cout << std::endl;
  }
}

int main(int argc, const char* argv[]) {

  const float h            = argc > 1 ? std::stof(argv[1]) : 1;
  const float r            = argc > 2 ? std::stof(argv[2]) : 0.0949996;
  const float k            = argc > 3 ? std::stof(argv[3]) : 1;
  const float max_distance = argc > 4 ? std::stof(argv[4]) : 20;

  visualmesh::geometry::Sphere<float> shape(r);

  {
    std::cout << "Hexapizza Quality:" << std::endl;
    visualmesh::Mesh<float, visualmesh::generator::Hexapizza> mesh(shape, h, k, max_distance);
    auto quality = check_quality(shape, mesh);
    print_quality(quality, k);
    std::cout << std::endl;
  }

  {
    std::cout << "Quadpizza Quality:" << std::endl;
    visualmesh::Mesh<float, visualmesh::generator::QuadPizza> mesh(shape, h, k, max_distance);
    auto quality = check_quality(shape, mesh);
    print_quality(quality, k);
    std::cout << std::endl;
  }
}
