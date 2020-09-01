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

#ifndef VISUALMESH_ENGINE_CPU_ACTIVATION_HPP
#define VISUALMESH_ENGINE_CPU_ACTIVATION_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "visualmesh/network_structure.hpp"

namespace visualmesh {
namespace engine {
    namespace cpu {

        namespace activation {

            template <typename Scalar>
            void selu(std::vector<Scalar>& data, const int& /*dimensions*/) {
                std::transform(data.begin(), data.end(), data.begin(), [](const Scalar& s) {
                    constexpr const Scalar lambda = 1.0507009873554804934193349852946;
                    constexpr const Scalar alpha  = 1.6732632423543772848170429916717;
                    return lambda * (s >= 0 ? s : alpha * std::exp(s) - alpha);
                });
            }

            template <typename Scalar>
            void relu(std::vector<Scalar>& data, const int& /*dimensions*/) {
                std::transform(
                  data.begin(), data.end(), data.begin(), [](const Scalar& s) { return std::max(s, Scalar(0.0)); });
            }

            template <typename Scalar>
            void tanh(std::vector<Scalar>& data, const int& /*dimensions*/) {
                std::transform(data.begin(), data.end(), data.begin(), [](const Scalar& s) {  //
                    return std::tanh(s);
                });
            }

            template <typename Scalar>
            void softmax(std::vector<Scalar>& data, const int& dimensions) {
                std::transform(data.begin(), data.end(), data.begin(), [](const Scalar& s) { return std::exp(s); });
                for (auto it = data.begin(); it < data.end(); std::advance(it, dimensions)) {
                    const auto end = std::next(it, dimensions);
                    Scalar total   = std::accumulate(it, end, Scalar(0.0));
                    std::transform(it, end, it, [total](const Scalar& s) { return s / total; });
                }
            }
        }  // namespace activation

        template <typename Scalar>
        void apply_activation(const ActivationFunction& fn, std::vector<Scalar>& data, const int& dimensions) {
            switch (fn) {
                case ActivationFunction::SELU: activation::selu(data, dimensions); break;
                case ActivationFunction::RELU: activation::relu(data, dimensions); break;
                case ActivationFunction::TANH: activation::tanh(data, dimensions); break;
                case ActivationFunction::SOFTMAX: activation::softmax(data, dimensions); break;
            }
        }

    }  // namespace cpu
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_CPU_ACTIVATION_HPP
