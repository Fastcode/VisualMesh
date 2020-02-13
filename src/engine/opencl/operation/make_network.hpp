/*
 * Copyright (C) 2017-2019 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_OPENCL_OPERATION_MAKE_NETWORK_HPP
#define VISUALMESH_OPENCL_OPERATION_MAKE_NETWORK_HPP

#include <iostream>
#include <utility>
#include <vector>

#include "mesh/network_structure.hpp"
#include "wrapper.hpp"

namespace visualmesh {
namespace engine {
  namespace opencl {
    namespace operation {

      template <typename Scalar>
      std::string make_network(const network_structure_t<Scalar>& structure) {
        // Generate the OpenCL kernels for the network
        std::stringstream code;

        // If our structure has no layers, return empty code
        if (structure.empty() || structure.front().empty()) { return ""; }

        // First layer has 4 inputs, so that tells us how many neighbours we have (minus ourself)
        const size_t n_neighbours = (structure.front().front().first.size() / 4) - 1;

        // Set our precision for how many digits our scalar has
        code << std::setprecision(std::numeric_limits<Scalar>::digits10 + 2);

        // Keep track of the input and output size of each layer for building the network
        // The first layer input is always 4 from the image
        unsigned int input_dimensions  = 4;
        unsigned int output_dimensions = 0;

        for (unsigned int conv_no = 0; conv_no < structure.size(); ++conv_no) {
          auto& conv = structure[conv_no];

          // Write our OpenCL kernel definition
          code << "kernel void conv" << conv_no
               << "(global const int* neighbourhood, global const Scalar* input, global Scalar* output) {" << std::endl
               << std::endl;

          code << "  // Get our kernel index" << std::endl;
          code << "  const int idx = get_global_id(0);" << std::endl << std::endl;

          /*************************************************
           *                    GATHER                     *
           *************************************************/

          code << "  // Gather from our neighbourhood " << std::endl;
          code << "  Scalar in0[" << (input_dimensions * (n_neighbours + 1)) << "] = {" << std::endl;

          // Read the ones for our own index
          for (unsigned int j = 0; j < input_dimensions; ++j) {
            code << "    input[idx * " << input_dimensions << " + " << j << "]," << std::endl;
          }

          // Read our neighbourhood
          for (unsigned int i = 0; i < n_neighbours; ++i) {
            for (unsigned int j = 0; j < input_dimensions; ++j) {
              code << "    input[neighbourhood[idx * " << n_neighbours << " + " << i << "] * " << input_dimensions
                   << " + " << j << "]";

              // Comma separated except for the end
              if (i < n_neighbours || j + 1 < input_dimensions) { code << ","; }
              code << std::endl;
            }
          }
          code << "  };";


          // We have gathered which increased the size of the input
          input_dimensions = input_dimensions * (n_neighbours + 1);

          code << std::endl << std::endl;

          /*************************************************
           *                WEIGHTS + BIAS                 *
           *************************************************/

          // Now we have to do our layer operations
          for (unsigned int layer_no = 0; layer_no < conv.size(); ++layer_no) {
            const auto& weights = conv[layer_no].first;
            const auto& biases  = conv[layer_no].second;

            // Update our output dimensions
            output_dimensions = biases.size();

            // Perform the matrix multiplication
            code << "  // Perform our matrix multiplication for weights and add bias for layer " << layer_no
                 << std::endl;
            code << "  Scalar in" << (layer_no + 1) << "[" << output_dimensions << "] = {" << std::endl;
            for (unsigned int i = 0; i < output_dimensions; ++i) {
              code << "    ";
              for (unsigned int j = 0; j < input_dimensions; ++j) {
                code << "in" << layer_no << "[" << j << "] * " << weights[j][i] << " + ";
              }
              code << biases[i];
              if (i + 1 < output_dimensions) { code << ","; }
              code << std::endl;
            }
            code << "  };" << std::endl << std::endl;


            /*************************************************
             *                  ACTIVATION.                  *
             *************************************************/

            // Apply our activation function
            code << "  // Apply the activation function" << std::endl;

            // selu constants
            constexpr const Scalar lambda = 1.0507009873554804934193349852946;
            constexpr const Scalar alpha  = 1.6732632423543772848170429916717;

            // Apply selu
            if (conv_no + 1 < structure.size() || layer_no + 1 < conv.size()) {
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  " << e << " = " << lambda << "f * (" << e << " > 0 ? " << e << " : " << alpha << "f * exp("
                     << e << ") - " << alpha << "f);" << std::endl;
              }
            }
            else {  // If this is our last layer, apply softmax
              code << "  // Apply softmax to our final output" << std::endl;

              // Apply exp to each of the elements
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  " << e << " = exp(" << e << ");" << std::endl;
              }

              // Sum up all the values
              code << "Scalar exp_sum = 0;" << std::endl;
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  exp_sum += " << e << ";" << std::endl;
              }

              // Divide all the values
              for (unsigned int i = 0; i < output_dimensions; ++i) {
                std::string e = "in" + std::to_string(layer_no + 1) + "[" + std::to_string(i) + "]";
                code << "  " << e << " /= exp_sum;" << std::endl;
              }
            }
            code << std::endl;

            // Update our input size for the next loop
            input_dimensions = output_dimensions;
          }

          /*************************************************
           *                    OUTPUT                     *
           *************************************************/
          code << "  // Save our value to the output" << std::endl;
          for (unsigned int i = 0; i < input_dimensions; ++i) {
            code << "  output[idx * " << input_dimensions << " + " << i << "] = in" << conv.size() << "[" << i << "];"
                 << std::endl;
          }

          code << "}" << std::endl << std::endl;

          // Update our input dimensions for the next round
          input_dimensions = output_dimensions;
        }

        return code.str();
      }

    }  // namespace operation
  }    // namespace opencl
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_OPENCL_OPERATION_MAKE_NETWORK_HPP
