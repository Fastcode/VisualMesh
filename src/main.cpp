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

    float theta = 0;

    // Stored in row major order
    std::array<std::array<float, 4>, 4> Hco = {{
        {{std::cos(theta), -std::sin(theta), 0, 0}},  //
        {{std::sin(theta), std::cos(theta), 0, 0}},   //
        {{0, 0, 1, 1}},                               //
        {{0, 0, 0, 1}}                                //
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
