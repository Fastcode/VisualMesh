#include <array>
#include <iostream>

#include "Circle.hpp"
#include "Cylinder.hpp"
#include "Sphere.hpp"
#include "VisualMesh.hpp"

using Scalar = float;

int main() {

    mesh::Cylinder<Scalar> cylinder(0, 2.0, 0.075, 1, 20);
    mesh::Sphere<Scalar> sphere(0, 0.075, 1, 10);
    mesh::Circle<Scalar> circle(0, 0.075, 1, 10);
    mesh::VisualMesh<Scalar> mesh(cylinder, 1.0, 1.1, 1, M_PI / 1024.0);

    // theta is pitch, lambda is roll, and phi is yaw
    Scalar theta  = 0;
    Scalar phi    = 0;
    Scalar lambda = 0;

    Scalar ct = std::cos(theta);
    Scalar st = std::sin(theta);
    Scalar cp = std::cos(phi);
    Scalar sp = std::sin(phi);
    Scalar cl = std::cos(lambda);
    Scalar sl = std::sin(lambda);


    std::array<std::array<Scalar, 4>, 4> Hco;

    // Rotation matrix
    Hco[0][0] = ct * cp;
    Hco[0][1] = ct * sp;
    Hco[0][2] = -st;
    Hco[1][0] = sl * st * cp - cl * sp;
    Hco[1][1] = sl * st * sp + cl * cp;
    Hco[1][2] = ct * sl;
    Hco[2][0] = cl * st * cp + sl * sp;
    Hco[2][1] = cl * st * sp - sl * cp;
    Hco[2][2] = ct * cl;

    // Lower row
    Hco[3][0] = 0;
    Hco[3][1] = 0;
    Hco[3][2] = 0;
    Hco[3][3] = 1;

    // Translation
    Hco[0][3] = 0;
    Hco[1][3] = 0;
    Hco[2][3] = 1;


    mesh::VisualMesh<Scalar>::Lens lens;

    lens.type                = mesh::VisualMesh<Scalar>::Lens::EQUIRECTANGULAR;
    lens.equirectangular.fov = {{1.0472, 0.785398}};

    std::cout << "[ ";

    // Perform our lookup
    const auto& lut    = mesh.height(1.0);
    const auto& ranges = mesh.lookup(Hco, lens);

    // Print out the mesh in a format for python
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
