/*
 * Copyright (C) 2017-2018 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_MODEL_RADIAL8_HPP
#define VISUALMESH_MODEL_RADIAL8_HPP

#include <array>
#include <vector>

#include "mesh/node.hpp"
#include "polar_map.hpp"

namespace visualmesh {
namespace model {

    template <typename Scalar>
    struct Radial8 : public PolarMap<Scalar> {
    private:
        static inline vec3<Scalar> unit_vector(const Scalar& sin_phi, const Scalar& cos_phi, const Scalar& theta) {
            return vec3<Scalar>{{std::cos(theta) * sin_phi, std::sin(theta) * sin_phi, -cos_phi}};
        }

    public:
        static constexpr int N_NEIGHBOURS = 8;

        template <typename Shape>
        static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                                const Scalar& h,
                                                                const Scalar& k,
                                                                const Scalar& max_distance) {
            std::vector<Node<Scalar, N_NEIGHBOURS>> nodes;
            std::vector<Scalar> Theta_Offset;
            std::vector<int> number_points;

            // L, T, R, B
            int LEFT = 0;
            // TOP_LEFT = 1;
            int TOP = 2;
            // TOP_RIGHT = 3;
            int RIGHT = 4;
            // BELOW_RIGHT = 5;
            int BELOW = 6;
            // BELOW_LEFT = 7;

            std::array<int, 4> first_neighbours = {2, 3, 0, 1};

            Scalar phi_first = shape.phi(1 / k, h) / 2;
            Theta_Offset.push_back(0);

            for (int j = 0; j < 4; ++j) {
                Node<Scalar, N_NEIGHBOURS> first;
                first.ray = unit_vector(std::sin(phi_first / 2), std::cos(phi_first / 2), j * (2 * M_PI / 4));
                first.neighbours[BELOW] = first_neighbours[j];
                first.neighbours[LEFT]  = ((j + 1) % 4 + 4) % 4;
                first.neighbours[RIGHT] = ((j - 1) % 4 + 4) % 4;
                nodes.push_back(std::move(first));
            }
            number_points.emplace_back(4);

            int running_index = nodes.size();

            std::vector<int> origin_number_points;
            origin_number_points.emplace_back(0);
            origin_number_points.emplace_back(4);

            int stop;
            if (k < 9) { stop = 5; }
            else {
                stop = 7;
            }

            for (int i = 0; i < stop; ++i) {
                origin_number_points.emplace_back(8 + 8 * i);
            }

            for (int v = 1; h * std::tan(shape.phi(v / k, h)) < max_distance; ++v) {
                // for (int v = 1; v < 5; ++v) {
                Scalar phi_next = shape.phi(v / k, h);
                int begin       = running_index - number_points.back();
                int end         = running_index;
                bool every_one  = false;
                bool growing    = false;

                // hack // odd v generates clockwise, even v generates anti-clockwise.
                int one = std::round(std::pow(-1, v));

                Scalar theta_offset   = Theta_Offset.back();
                int number_points_now = number_points[v - 1];
                int number_points_next;

                // precalculate the number of points vector according to distribution and origin patch
                // fix this limit
                if (v < stop) { number_points_next = origin_number_points[v]; }
                else {
                    // floor vs ceil
                    number_points_next = std::ceil((2 * M_PI * k) / shape.theta(v / k, h));
                }

                int number_difference = number_points_next - number_points_now;
                Scalar theta_next;

                // *************** Calculate Split Distribution ***********************
                int distribution;
                if (number_difference == 0) {
                    // std::cout << "Difference is zero: " << std::endl;
                    distribution       = 0;
                    number_points_next = number_points_now;
                }
                else if (number_difference == 1) {
                    // std::cout << "Difference is 1: " << std::endl;
                    growing      = true;
                    distribution = 1;
                    // number_difference = 1;
                    number_points_next = 1 + number_points_now;
                }
                else if (number_difference > 1) {
                    growing = true;
                    if (number_difference < number_points_now) {
                        // std::cout << "Difference is reasonable: " << std::endl;
                        distribution = number_points_now / number_difference;
                        if (distribution == 1) { every_one = true; }
                    }
                    else {
                        // number_difference >= number_points_now
                        // std::cout << v << std::endl;
                        // std::cout << "Difference is greater than generating ring: " << std::endl;
                        distribution       = 1;
                        number_points_next = 2 * number_points_now;
                        number_difference  = number_points_now;
                    }
                }
                else {
                    // std::cout << "Difference is negative: " << std::endl;
                    distribution       = 0;
                    number_points_next = number_points_now;
                }
                // *********************************************************************************

                theta_next = 2 * M_PI / number_points_next;
                number_points.emplace_back(number_points_next);

                std::vector<int> indices;
                for (int i = begin; i < end; ++i) {
                    indices.push_back(i);
                }

                // condense this
                std::vector<int> vector_of_indices;
                if (v == 1) {
                    vector_of_indices.push_back(indices[0]);
                    for (int m = indices.size() - 1; m > 0; --m) {
                        vector_of_indices.push_back(indices[m]);
                    }
                }
                else {
                    vector_of_indices.push_back(indices[1]);
                    vector_of_indices.push_back(indices[0]);
                    for (int m = indices.size() - 1; m > 1; --m) {
                        vector_of_indices.push_back(indices[m]);
                    }
                }

                int relative_index_now  = 0;
                int relative_index_next = 0;
                int number_splits       = 0;

                for (auto it = vector_of_indices.begin(); it != vector_of_indices.end(); ++it) {
                    Node<Scalar, N_NEIGHBOURS> new_node;
                    new_node.ray = unit_vector(
                      std::sin(phi_next), std::cos(phi_next), theta_offset + one * relative_index_next * theta_next);

                    new_node.neighbours[LEFT] =
                      end
                      + (((relative_index_next + one) % number_points_next + number_points_next) % number_points_next);
                    new_node.neighbours[RIGHT] =
                      end
                      + (((relative_index_next - one) % number_points_next + number_points_next) % number_points_next);
                    new_node.neighbours[BELOW] = *it;

                    // nodes[*it].neighbours[TOP] = end + relative_index_next % number_points_next;
                    nodes[*it].neighbours[TOP] = end + relative_index_next;

                    nodes.push_back(std::move(new_node));
                    relative_index_next += 1;

                    if (relative_index_next == 1) {
                        Theta_Offset.push_back(theta_offset + one * relative_index_next * theta_next);
                    }

                    // *************** Generate Second Node ***********************
                    if (growing) {
                        if (every_one == true) {
                            if (number_splits <= number_points_now - number_difference) { distribution = 2; }
                            else {
                                distribution = 1;
                            }
                        }

                        // split every point of according to the distribution until difference is reached, or split
                        // every point.
                        if ((relative_index_now % distribution == 0 || distribution == 1)
                            && number_splits < number_difference) {
                            // std::cout << "Graph is growing!" << std::endl;
                            Node<Scalar, N_NEIGHBOURS> second_new_node;
                            second_new_node.ray = unit_vector(std::sin(phi_next),
                                                              std::cos(phi_next),
                                                              theta_offset + one * relative_index_next * theta_next);

                            second_new_node.neighbours[LEFT] =
                              end
                              + (((relative_index_next + one) % number_points_next + number_points_next)
                                 % number_points_next);
                            second_new_node.neighbours[RIGHT] =
                              end
                              + (((relative_index_next - one) % number_points_next + number_points_next)
                                 % number_points_next);
                            second_new_node.neighbours[BELOW] = *it;

                            nodes.push_back(std::move(second_new_node));

                            number_splits += 1;
                            relative_index_next += 1;
                        }
                    }
                    // *********************************************************************************
                    relative_index_now += 1;
                }
                running_index = nodes.size();
            }

            // specify the neighbours of the last ring of points
            for (unsigned int i = (nodes.size() - number_points.back()); i < nodes.size(); ++i) {
                nodes[i].neighbours[TOP] = i;
            }

            // Join the corners to make octa mesh from quadpizza.
            // Note because of the splits, it ends up most correct to take the top row using the top nodes neihgbours,
            // and to make the bottom row using the left and right below neighbours. This can be visualised as a a 2x2
            // lattice, with one of the lower quadrants merged into a triangle by the split.
            for (unsigned int p = 0; p < nodes.size(); ++p) {
                // TOP_LEFT
                nodes[p].neighbours[1] = nodes[nodes[p].neighbours[TOP]].neighbours[LEFT];
                // TOP_RIGHT
                nodes[p].neighbours[3] = nodes[nodes[p].neighbours[TOP]].neighbours[RIGHT];
                // BELOW_RIGHT
                nodes[p].neighbours[5] = nodes[nodes[p].neighbours[RIGHT]].neighbours[BELOW];
                // BELOW_LEFT
                nodes[p].neighbours[7] = nodes[nodes[p].neighbours[LEFT]].neighbours[BELOW];
            }

            // // print out mesh points
            // for (int i = 0; i < nodes.size(); ++i) {
            //   std::cout << "meshpoint: " << i << ": " << nodes[i].neighbours[TOP] << ", " <<
            //   nodes[i].neighbours[BELOW]
            //             << ", " << nodes[i].neighbours[RIGHT] << ", " << nodes[i].neighbours[LEFT] << ", "
            //             << nodes[i].neighbours[1] << ", " << nodes[i].neighbours[3] << ", " << nodes[i].neighbours[5]
            //             << ",
            //             "
            //             << nodes[i].neighbours[7] << std::endl;
            // }

            return nodes;
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_RADIAL8_HPP
