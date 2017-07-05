#include <iostream>

#include "Sphere.hpp"
#include "VisualMesh.hpp"

int main() {

    mesh::Sphere<float> s(0, 0.075, 1, 5);

    mesh::VisualMesh<> mesh(s, 0.4, 1.0, 10, M_PI / 1024.0);

    // Print the mesh
    const auto& lut = mesh.data(0.5);

    std::cout << "[ ";
    for (int i = 0; i < lut.size(); ++i) {

        const auto& node = lut[i];

        for (const auto& n : lut[i].neighbours) {
            const auto& neighbour = lut[i + n];
            std::cout << "(" << node.ray[0] << ", " << node.ray[1] << ", " << node.ray[2] << ", " << neighbour.ray[0]
                      << ", " << neighbour.ray[1] << ", " << neighbour.ray[2] << "), ";
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
