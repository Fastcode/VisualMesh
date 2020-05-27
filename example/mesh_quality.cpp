/*
 * Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <iomanip>
#include <iostream>

#include "ArrayPrint.hpp"
#include "Timer.hpp"
//
#include "geometry/Circle.hpp"
#include "geometry/Sphere.hpp"
#include "mesh/mesh.hpp"
#include "mesh/model/nmgrid4.hpp"
#include "mesh/model/nmgrid6.hpp"
#include "mesh/model/nmgrid8.hpp"
#include "mesh/model/radial4.hpp"
#include "mesh/model/radial6.hpp"
#include "mesh/model/radial8.hpp"
#include "mesh/model/ring4.hpp"
#include "mesh/model/ring6.hpp"
#include "mesh/model/ring8.hpp"
#include "mesh/model/xmgrid4.hpp"
#include "mesh/model/xmgrid6.hpp"
#include "mesh/model/xmgrid8.hpp"
#include "mesh/model/xygrid4.hpp"
#include "mesh/model/xygrid6.hpp"
#include "mesh/model/xygrid8.hpp"
#include "utility/math.hpp"
#include "utility/phi_difference.hpp"

template <typename Scalar>
using vec3 = visualmesh::vec3<Scalar>;

template <typename Scalar, int N_NEIGHBOURS>
struct NodeQuality {
    /// The distance this node is from the origin
    Scalar distance;
    /// The angle this node is around the z axis
    Scalar angle;
    /// The number of object jumps between this node and the nodes around it
    std::array<Scalar, N_NEIGHBOURS> radial;
    /// The number of object jumps between each neighbour and the subsequent neighbour
    std::array<Scalar, N_NEIGHBOURS> cyclical;
    /// The angle between each neighbour and the subsequent neighbour
    std::array<Scalar, N_NEIGHBOURS> angular;
};

template <typename Scalar, template <typename> class Shape, template <typename> class Model>
std::vector<NodeQuality<Scalar, Model<Scalar>::N_NEIGHBOURS>> check_quality(
  const Shape<Scalar>& shape, const visualmesh::Mesh<Scalar, Model>& mesh) {

    constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

    // Loop through all the nodes in the mesh
    std::vector<NodeQuality<Scalar, N_NEIGHBOURS>> nodes;
    for (const auto& node : mesh.nodes) {
        NodeQuality<Scalar, N_NEIGHBOURS> quality;

        // Our ray pointing in the centre of the cluster
        const auto& r0 = node.ray;

        // The rays location in the space
        quality.distance = (mesh.h - shape.c()) * std::sqrt(1 - r0[2] * r0[2]) / -r0[2];
        quality.angle    = std::atan2(r0[1], r0[0]);

        // By default set things to nan as a "this node did not exist"
        quality.radial.fill(std::numeric_limits<Scalar>::quiet_NaN());
        quality.cyclical.fill(std::numeric_limits<Scalar>::quiet_NaN());
        quality.angular.fill(std::numeric_limits<Scalar>::quiet_NaN());

        // We look through each of our neighbours to see how good we are
        for (unsigned int i = 0; i < node.neighbours.size(); ++i) {

            // We get our next two neighbours in a clockwise direction
            int n1 = node.neighbours[i];
            int n2 = node.neighbours[(i + 1) % node.neighbours.size()];

            // Ignore points that go off the screen
            if (n1 < int(mesh.nodes.size())) {
                // The neighbours ray
                const auto& r1 = mesh.nodes[n1].ray;

                // Radial difference to our neighbour
                auto r_d          = visualmesh::util::phi_difference(mesh.h, shape.c(), r0, r1);
                quality.radial[i] = std::abs(shape.n(r_d.phi_0, r_d.h_prime) - shape.n(r_d.phi_1, r_d.h_prime));

                // Ignore points that go off the screen
                if (n2 < int(mesh.nodes.size())) {
                    const auto& r2 = mesh.nodes[n2].ray;

                    // The distance difference between the two neighbour rays
                    auto c_d            = visualmesh::util::phi_difference<Scalar>(mesh.h, shape.c(), r1, r2);
                    quality.cyclical[i] = std::abs(shape.n(c_d.phi_0, c_d.h_prime) - shape.n(c_d.phi_1, c_d.h_prime));

                    // The angular difference between two neighbourhood rays
                    quality.angular[i] = std::acos(visualmesh::dot(visualmesh::normalise(visualmesh::cross(r0, r1)),
                                                                   visualmesh::normalise(visualmesh::cross(r0, r2))));
                }
            }
        }

        nodes.push_back(quality);
    }

    return nodes;
}

template <typename Scalar, int N_NEIGHBOURS>
struct Statistics {
    Statistics() {
        means.fill(0);
        sums.fill(0);
        counts.fill(0);
    }

    void update(const std::array<Scalar, N_NEIGHBOURS> input) {

        for (unsigned int i = 0; i < N_NEIGHBOURS; ++i) {
            if (!std::isnan(input[i]) && input[i] != 0) {
                sums[i] += input[i];
                counts[i]++;
                means[i] = sums[i] / counts[i];
            }
        }
    }

    std::array<Scalar, N_NEIGHBOURS> means;
    std::array<Scalar, N_NEIGHBOURS> sums;
    std::array<uint32_t, N_NEIGHBOURS> counts;
};

template <typename Scalar, int N_NEIGHBOURS>
void print_quality(const std::vector<NodeQuality<Scalar, N_NEIGHBOURS>>& nodes, const Scalar& k) {

    // Storage for the statistics
    Statistics<Scalar, N_NEIGHBOURS> radial;
    Statistics<Scalar, N_NEIGHBOURS> radial_var;
    Statistics<Scalar, N_NEIGHBOURS> cyclical;
    Statistics<Scalar, N_NEIGHBOURS> cyclical_var;
    Statistics<Scalar, N_NEIGHBOURS> angular;
    Statistics<Scalar, N_NEIGHBOURS> angular_var;

    // Go through all the nodes and accumulate for the mean value
    for (const auto& node : nodes) {
        radial.update(visualmesh::multiply(node.radial, k));
        cyclical.update(visualmesh::multiply(node.cyclical, k));
        angular.update(visualmesh::multiply(node.angular, Scalar(N_NEIGHBOURS * (M_PI * 2.0))));
    }

    // Sum up the variance
    for (const auto& node : nodes) {
        auto v = visualmesh::subtract(visualmesh::multiply(node.radial, k), radial.means);
        radial_var.update(visualmesh::multiply(v, v));
        auto c = visualmesh::subtract(visualmesh::multiply(node.cyclical, k), cyclical.means);
        cyclical_var.update(visualmesh::multiply(c, c));
        auto a =
          visualmesh::subtract(visualmesh::multiply(node.angular, Scalar(N_NEIGHBOURS * (M_PI * 2.0))), angular.means);
        angular_var.update(visualmesh::multiply(a, a));
    }

    std::cout << std::setprecision(4);
    std::cout << "Covered with " << nodes.size() << " nodes" << std::endl;
    for (unsigned int i = 0; i < N_NEIGHBOURS; ++i) {
        std::cout << "* " << (radial.means[i]) << "±" << (std::sqrt(radial_var.means[i]));
        std::cout << " o " << (cyclical.means[i]) << "±" << (std::sqrt(cyclical_var.means[i]));
        std::cout << " a " << (angular.means[i]) << "±" << (std::sqrt(angular_var.means[i]));
        std::cout << std::endl;
    }
}

int main(int argc, const char* argv[]) {

    const float h            = argc > 1 ? std::stof(argv[1]) : 1;
    const float r            = argc > 2 ? std::stof(argv[2]) : 0.0949996;
    const float k            = argc > 3 ? std::stof(argv[3]) : 1;
    const float max_distance = argc > 4 ? std::stof(argv[4]) : 20;

    visualmesh::geometry::Sphere<float> shape(r);

    {
        std::cout << "Ring 4 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::Ring4> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "Ring 6 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::Ring6> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "Ring 8 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::Ring8> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "Radial 4 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::Radial4> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "Radial 6 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::Radial6> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "Radial 8 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::Radial8> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "XM Grid 4 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::XMGrid4> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "XM Grid 6 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::XMGrid6> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "XM Grid 8 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::XMGrid8> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "XY Grid 4 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::XYGrid4> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "XY Grid 6 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::XYGrid6> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "XY Grid 8 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::XYGrid8> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "NM Grid 4 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::NMGrid4> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "NM Grid 6 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::NMGrid6> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }

    {
        std::cout << "NM Grid 8 Quality:" << std::endl;
        visualmesh::Mesh<float, visualmesh::model::NMGrid8> mesh(shape, h, k, max_distance);
        auto quality = check_quality(shape, mesh);
        print_quality(quality, k);
        std::cout << std::endl;
    }
}
