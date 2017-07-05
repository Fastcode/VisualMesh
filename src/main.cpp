#include <iostream>

#include "Sphere.hpp"
#include "VisualMesh.hpp"

int main() {

    mesh::Sphere<float> s(0, 0.075, 5);

    mesh::VisualMesh<> mesh(s, 0.4, 1.0, 10, M_PI / 1024.0);
}
