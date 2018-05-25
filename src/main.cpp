#include <iostream>
#include "geometry/Circle.hpp"
#include "geometry/Cylinder.hpp"
#include "geometry/Sphere.hpp"
#include "visualmesh.hpp"

int main() {

  visualmesh::geometry::Circle<float> circle(0.1, 4, 100);
  visualmesh::geometry::Sphere<float> sphere(0.1, 4, 100);
  visualmesh::geometry::Cylinder<float> cylinder(2, 0.1, 4, 100);

  visualmesh::VisualMesh<float> mesh(cylinder, 0.75, 1.25, 10);

  for (auto& r : mesh.height(1)->rows) {
    std::cout << r.phi << " " << (r.end - r.begin) << std::endl;
  }
}
