
#include <iostream>
#include "ArrayPrint.hpp"
#include "Timer.hpp"
//
#include "geometry/Circle.hpp"
#include "geometry/Sphere.hpp"
#include "mesh/test_mesh.hpp"
#include "visualmesh.hpp"

int main() {

  visualmesh::geometry::Circle<float> shape(0.0949996);
  visualmesh::Mesh<float> mesh(shape, 1.0, 1, 20);

  visualmesh::test_mesh(mesh, shape);
}
