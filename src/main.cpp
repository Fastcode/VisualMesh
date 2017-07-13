#include <array>
#include <iostream>

#include "Circle.hpp"
#include "Cylinder.hpp"
#include "Sphere.hpp"
#include "VisualMesh.hpp"

int main() {

    mesh::Cylinder<float> cylinder(0, 2.0, 0.075, 1, 10);
    mesh::Sphere<float> sphere(0, 0.075, 1, 10);
    mesh::Circle<float> circle(0, 0.075, 1, 10);

    mesh::VisualMesh<> mesh(cylinder, 1.0, 1.1, 1, M_PI / 1024.0);

    //
    std::array<std::array<float, 4>, 4> Hoc = {{
        {{1, 0, 0, 0}},  //
        {{0, 1, 0, 0}},  //
        {{0, 0, 1, 0}},  //
        {{0, 0, 0, 1}}   //
    }};

    mesh::VisualMesh<float>::Lens lens;

    lens.equirectangular.fov = {1.0472, 0.785398};

    // Print the mesh
    const auto& lut    = mesh.height(0.5);
    const auto& ranges = mesh.lookup(Hoc, lens);

    // const auto& ranges = mesh.lookup(0.5, [](const float& phi) {

    //     std::vector<std::pair<float, float>> ret;

    //     // if (phi > M_PI_4 && phi < M_PI_2) {
    //     //     ret.emplace_back(3 * M_PI_2, M_PI_2);
    //     // }
    //     ret.emplace_back(0, M_PI_2);

    //     return ret;
    // });

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

    // Print all the unit vectors
    // std::cout << "[ ";
    // for (const auto& n : mesh.data(0.5)) {

    //     std::cout << "(" << n.ray[0] << ", " << n.ray[1] << ", " << n.ray[2] << "), ";
    // }
    // std::cout << "] " << std::endl;
}
