
#include <iostream>
#include "Timer.hpp"
//
#include "geometry/Sphere.hpp"
#include "mesh/test_mesh.hpp"
#include "visualmesh.hpp"

int main() {

  visualmesh::geometry::Sphere<float> sphere(0.0949996);
  visualmesh::Mesh<float, visualmesh::generator::QuadPizza> mesh(sphere, 1.0, 1, 20);

  visualmesh::test_mesh(mesh, sphere);
}
