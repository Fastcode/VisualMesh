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

#ifndef VISUALMESH_NETWORKSTRUCTURE_HPP
#define VISUALMESH_NETWORKSTRUCTURE_HPP

#include <vector>

namespace visualmesh {

/// Weights are a matrix (vector of vectors)
template <typename Scalar>
using Weights = std::vector<std::vector<Scalar>>;
/// Biases are a vector
template <typename Scalar>
using Biases = std::vector<Scalar>;

enum ActivationFunction {
    SELU,
    RELU,
    SOFTMAX,
    TANH,
};

/// A layer is made up of weights biases and activation function
template <typename Scalar>
struct Layer {
    Weights<Scalar> weights;
    Biases<Scalar> biases;
    ActivationFunction activation;
};

/// A convolutional layer is made up of a list of network layers
template <typename Scalar>
using ConvolutionalGroup = std::vector<Layer<Scalar>>;
/// A network is a list of convolutional layers
template <typename Scalar>
using NetworkStructure = std::vector<ConvolutionalGroup<Scalar>>;

}  // namespace visualmesh

#endif  // VISUALMESH_NETWORKSTRUCTURE_HPP
