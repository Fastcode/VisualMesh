#include <array>
#include <iostream>

#include "Circle.hpp"
#include "Cylinder.hpp"
#include "Sphere.hpp"
#include "VisualMesh.hpp"

int main() {

    mesh::Cylinder<float> cylinder(0, 2.0, 0.075, 1, 20);
    mesh::Sphere<float> sphere(0, 0.075, 1, 10);
    mesh::Circle<float> circle(0, 0.075, 1, 10);

    mesh::VisualMesh<> mesh(cylinder, 1.0, 1.1, 1, M_PI / 1024.0);

    float a = 0;       // Rotation around z
    float b = M_PI_2;  // Rotation around y
    float c = 0;       // Rotation around x
    // Stored in row major order
    std::array<std::array<float, 4>, 4> Hco = {{
        {{std::cos(a) * std::cos(b),
          std::cos(a) * std::sin(b) * std::sin(c) - std::sin(a) * std::cos(c),
          std::cos(a) * std::sin(b) * std::cos(c) + std::sin(a) * std::sin(c),
          0}},  //
        {{std::sin(a) * std::cos(b),
          std::sin(a) * std::sin(b) * std::sin(c) + std::cos(a) * std::cos(c),
          std::sin(a) * std::sin(b) * std::cos(c) - std::cos(a) * std::sin(c),
          0}},                                                                      //
        {{-std::sin(a), std::cos(b) * std::sin(c), std::cos(b) * std::cos(c), 1}},  //
        {{0, 0, 0, 1}}                                                              //
    }};

    mesh::VisualMesh<float>::Lens lens;

    lens.type                = mesh::VisualMesh<float>::Lens::EQUIRECTANGULAR;
    lens.equirectangular.fov = {{1.0472, 0.785398}};

    // Perform our lookup
    const auto& lut    = mesh.height(1.0);
    const auto& ranges = mesh.lookup(Hco, lens);


    // Print out the mesh in a format for python
    std::cout << "[ ";
    for (auto& range : ranges) {
        for (size_t i = range.first; i < range.second; ++i) {

            const auto& node = lut.nodes[i];

            // for (const auto& n : lut[i].neighbours) {
            for (int j = 0; j < 6; ++j) {

                const auto& n = lut.nodes[i].neighbours[j];

                const auto& neighbour = lut.nodes[i + n];
                std::cout << "(" << node.ray[0] << ", " << node.ray[1] << ", " << node.ray[2] << ", "
                          << neighbour.ray[0] << ", " << neighbour.ray[1] << ", " << neighbour.ray[2] << "), ";
            }
        }
    }
    std::cout << "] " << std::endl;
}
