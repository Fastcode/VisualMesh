#include <iostream>
#include "geometry/Circle.hpp"
#include "geometry/Cylinder.hpp"
#include "geometry/Sphere.hpp"
#include "mesh/mesh.hpp"

int main() {

  visualmesh::geometry::Circle<float> circle(0.1, 4, 100);
  visualmesh::geometry::Sphere<float> sphere(0.1, 4, 100);
  visualmesh::geometry::Cylinder<float> cylinder(1, 0.1, 4, 100);

  visualmesh::Mesh<float> mesh(cylinder, 0.5);

  std::cout << mesh.rows.size() << std::endl;
  for (auto& r : mesh.rows) {
    std::cout << r.phi << " " << (r.end - r.begin) << std::endl;
  }
}
