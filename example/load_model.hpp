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

#ifndef LOAD_MODEL_HPP
#define LOAD_MODEL_HPP

#include <yaml-cpp/yaml.h>

#include "visualmesh/network_structure.hpp"

inline visualmesh::ActivationFunction activation_function(const std::string& name) {

    if (name == "selu") { return visualmesh::ActivationFunction::SELU; }
    if (name == "softmax") { return visualmesh::ActivationFunction::SOFTMAX; }
    if (name == "relu") { return visualmesh::ActivationFunction::RELU; }
    if (name == "tanh") { return visualmesh::ActivationFunction::TANH; }
    throw std::runtime_error("Unknown activation function " + name);
}

template <typename Scalar>
visualmesh::NetworkStructure<Scalar> load_model(const std::string& path) {

    visualmesh::NetworkStructure<Scalar> model;
    YAML::Node config = YAML::LoadFile(path);
    for (const auto& conv : config["network"]) {
        model.emplace_back();
        auto& net_conv = model.back();

        for (const auto& layer : conv) {
            net_conv.emplace_back(visualmesh::Layer<Scalar>{
              layer["weights"].as<std::vector<std::vector<Scalar>>>(),
              layer["biases"].as<std::vector<Scalar>>(),
              activation_function(layer["activation"].as<std::string>()),
            });
        }
    }
    return model;
}

#endif  // LOAD_MODEL_HPP
