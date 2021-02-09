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

#ifndef VISUALMESH_ENGINE_CPU_ENGINE_HPP
#define VISUALMESH_ENGINE_CPU_ENGINE_HPP

#include <cstdint>
#include <numeric>

#include "apply_activation.hpp"
#include "pixel.hpp"
#include "visualmesh/classified_mesh.hpp"
#include "visualmesh/mesh.hpp"
#include "visualmesh/network_structure.hpp"
#include "visualmesh/projected_mesh.hpp"
#include "visualmesh/utility/fourcc.hpp"
#include "visualmesh/utility/math.hpp"
#include "visualmesh/visualmesh.hpp"

namespace visualmesh {
namespace engine {
    namespace cpu {

        /**
         * @brief The reference CPU implementation of the visual mesh inference engine
         *
         * @details
         *  The CPU implementation is designed to be a simple implementation of the visual mesh projection and
         *  classification code. It is only single threaded and is not designed to be used in high performance contexts.
         *  For those use another implementation that is able to take advantage of other system features such as GPUs or
         *  multithreading.
         *
         * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
         */
        template <typename Scalar>
        class Engine {
        public:
            /**
             * @brief Construct a new CPU Engine object
             *
             * @param structure the network structure to use classification
             */
            Engine(const NetworkStructure<Scalar>& structure = {}) : structure(structure) {
                // Transpose all the weights matrices to make it easier for us to multiply against
                for (auto& conv : this->structure) {
                    for (auto& layer : conv) {
                        auto& w = layer.weights;
                        Weights<Scalar> new_weights(w.front().size(), std::vector<Scalar>(w.size()));
                        for (unsigned int i = 0; i < w.size(); ++i) {
                            for (unsigned int j = 0; j < w[i].size(); ++j) {
                                new_weights[j][i] = w[i][j];
                            }
                        }
                        w = std::move(new_weights);
                    }
                }
            }

            /**
             * @brief Projects a provided mesh to pixel coordinates
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh the mesh table that we are projecting to pixel coordinates
             * @param Hoc  the homogenous transformation matrix from the camera to the observation plane
             * @param lens the lens parameters that describe the optics of the camera
             *
             * @return a projected mesh for the provided arguments
             */
            template <template <typename> class Model>
            ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Model>& mesh,
                                                                          const mat4<Scalar>& Hoc,
                                                                          const Lens<Scalar>& lens) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                // Lookup the on screen ranges
                auto ranges = mesh.lookup(Hoc, lens);

                // Convenience variables
                const auto& nodes = mesh.nodes;
                const mat3<Scalar> Rco(block<3, 3>(transpose(Hoc)));

                // Work out how many points total there are in the ranges
                unsigned int n_points = 0;
                for (auto& r : ranges) {
                    n_points += r.second - r.first;
                }

                // Output variables
                std::vector<int> global_indices;
                global_indices.reserve(n_points);
                std::vector<vec2<Scalar>> pixels;
                pixels.reserve(n_points);

                // Loop through adding global indices and pixel coordinates
                for (const auto& range : ranges) {
                    for (int i = range.first; i < range.second; ++i) {
                        // Even though we have already gone through a bsp to remove out of range points, sometimes it's
                        // not perfect and misses by a few pixels. So as we are projecting the points here we also need
                        // to check that they are on screen
                        auto px = project(multiply(Rco, nodes[i].ray), lens);
                        if (0 <= px[0] && px[0] + 1 < lens.dimensions[0] && 0 <= px[1]
                            && px[1] + 1 < lens.dimensions[1]) {
                            global_indices.emplace_back(i);
                            pixels.emplace_back(px);
                        }
                    }
                }

                // Update the number of points to account for how many pixels we removed
                n_points = pixels.size();

                // Build our reverse lookup, the default point goes to the null point
                std::vector<int> r_lookup(nodes.size() + 1, n_points);
                for (unsigned int i = 0; i < n_points; ++i) {
                    r_lookup[global_indices[i]] = i;
                }

                // Build our local neighbourhood map
                std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood(n_points + 1);  // +1 for the null point
                for (unsigned int i = 0; i < n_points; ++i) {
                    const Node<Scalar, N_NEIGHBOURS>& node = nodes[global_indices[i]];
                    for (unsigned int j = 0; j < node.neighbours.size(); ++j) {
                        const auto& n       = node.neighbours[j];
                        neighbourhood[i][j] = r_lookup[n];
                    }
                }
                // Last point is the null point
                neighbourhood[n_points].fill(n_points);

                return ProjectedMesh<Scalar, N_NEIGHBOURS>{
                  std::move(pixels), std::move(neighbourhood), std::move(global_indices)};
            }

            /**
             * @brief Projects a provided mesh to pixel coordinates from an aggregate VisualMesh object
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh the mesh table that we are projecting to pixel coordinates
             * @param Hoc  the homogenous transformation matrix from the camera to the observation plane
             * @param lens the lens parameters that describe the optics of the camera
             *
             * @return a projected mesh for the provided arguments
             */
            template <template <typename> class Model>
            inline ProjectedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const VisualMesh<Scalar, Model>& mesh,
                                                                                 const mat4<Scalar>& Hoc,
                                                                                 const Lens<Scalar>& lens) const {
                return operator()(mesh.height(Hoc[2][3]), Hoc, lens);
            }

            /**
             * @brief Project and classify a mesh using the neural network that is loaded into this engine
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh    the mesh table that we are projecting to pixel coordinates
             * @param Hoc     the homogenous transformation matrix from the camera to the observation plane
             * @param lens    the lens parameters that describe the optics of the camera
             * @param image   the data that represents the image the network will run from
             * @param format  the pixel format of this image as a fourcc code
             *
             * @return a classified mesh for the provided arguments
             */
            template <template <typename> class Model>
            ClassifiedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const Mesh<Scalar, Model>& mesh,
                                                                           const mat4<Scalar>& Hoc,
                                                                           const Lens<Scalar>& lens,
                                                                           const void* image,
                                                                           const uint32_t& format) const {
                static constexpr int N_NEIGHBOURS = Model<Scalar>::N_NEIGHBOURS;

                // Project the pixels to the display
                ProjectedMesh<Scalar, N_NEIGHBOURS> projected = operator()(mesh, Hoc, lens);
                auto& neighbourhood                           = projected.neighbourhood;
                unsigned int n_points                         = neighbourhood.size();

                if (projected.global_indices.empty()) { return ClassifiedMesh<Scalar, N_NEIGHBOURS>(); }

                // Based on the fourcc code, load the data from the image into input
                input.reserve(n_points * 4);

                for (const auto& px : projected.pixel_coordinates) {
                    const uint8_t* const im = reinterpret_cast<const uint8_t*>(image);

                    const vec4<Scalar> p = interpolate(px, im, lens.dimensions, format);
                    input.insert(input.end(), p.begin(), p.end());
                }

                // Four -1 values for the offscreen point
                input.insert(input.end(), {Scalar(-1.0), Scalar(-1.0), Scalar(-1.0), Scalar(-1.0)});

                // We start out with 4d input (RGBAesque)
                unsigned int input_dimensions  = 4;
                unsigned int output_dimensions = 0;

                // For each convolutional layer
                for (unsigned int conv_no = 0; conv_no < structure.size(); ++conv_no) {
                    const auto& conv = structure[conv_no];

                    // Ensure enough space for the convolutional gather
                    output.resize(0);
                    output.reserve(input.size() * (N_NEIGHBOURS + 1));
                    output_dimensions = input_dimensions * (N_NEIGHBOURS + 1);

                    // Gather over each of the neighbours
                    for (unsigned int i = 0; i < neighbourhood.size(); ++i) {
                        output.insert(output.end(),
                                      std::next(input.begin(), i * input_dimensions),
                                      std::next(input.begin(), (i + 1) * input_dimensions));
                        for (const auto& n : neighbourhood[i]) {
                            output.insert(output.end(),
                                          std::next(input.begin(), n * input_dimensions),
                                          std::next(input.begin(), (n + 1) * input_dimensions));
                        }
                    }

                    // Output becomes input
                    std::swap(input, output);
                    input_dimensions = output_dimensions;

                    // For each network layer
                    for (unsigned int layer_no = 0; layer_no < conv.size(); ++layer_no) {
                        const auto& weights    = conv[layer_no].weights;
                        const auto& biases     = conv[layer_no].biases;
                        const auto& activation = conv[layer_no].activation;

                        // Setup the shapes
                        output_dimensions = biases.size();
                        output.resize(0);
                        output.reserve(n_points * output_dimensions);

                        // Apply the weights and bias
                        auto in_point = input.begin();
                        for (unsigned int i = 0; i < n_points; ++i) {
                            for (unsigned int j = 0; j < output_dimensions; ++j) {
                                output.emplace_back(std::inner_product(
                                  in_point, in_point + input_dimensions, weights[j].begin(), biases[j]));
                            }
                            in_point += input_dimensions;
                        }

                        // Apply the activation function
                        apply_activation(activation, output, output_dimensions);

                        // Swap our values over
                        std::swap(input, output);
                        input_dimensions = output_dimensions;
                    }
                }

                return ClassifiedMesh<Scalar, N_NEIGHBOURS>{std::move(projected.pixel_coordinates),
                                                            std::move(projected.neighbourhood),
                                                            std::move(projected.global_indices),
                                                            std::move(input)};
            }

            /**
             * @brief Project and classify a mesh using the neural network that is loaded into this engine.
             * This version takes an aggregate VisualMesh object
             *
             * @tparam Model the mesh model that we are projecting
             *
             * @param mesh    the mesh table that we are projecting to pixel coordinates
             * @param Hoc     the homogenous transformation matrix from the camera to the observation plane
             * @param lens    the lens parameters that describe the optics of the camera
             * @param image   the data that represents the image the network will run from
             * @param format  the pixel format of this image as a fourcc code
             *
             * @return a classified mesh for the provided arguments
             */
            template <template <typename> class Model>
            ClassifiedMesh<Scalar, Model<Scalar>::N_NEIGHBOURS> operator()(const VisualMesh<Scalar, Model>& mesh,
                                                                           const mat4<Scalar>& Hoc,
                                                                           const Lens<Scalar>& lens,
                                                                           const void* image,
                                                                           const uint32_t& format) const {
                return operator()(mesh.height(Hoc[2][3]), Hoc, lens, image, format);
            }

        private:
            /// The network structure used to perform the operations
            NetworkStructure<Scalar> structure;

            /// An input buffer used to ping/pong when doing classification so we don't have to remake them
            mutable std::vector<Scalar> input;
            /// An output buffer used to ping/pong when doing classification so we don't have to remake them
            mutable std::vector<Scalar> output;
        };

    }  // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_ENGINE_HPP
