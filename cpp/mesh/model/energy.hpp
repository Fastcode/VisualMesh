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

#ifndef VISUALMESH_MODEL_MESHGRID_HPP
#define VISUALMESH_MODEL_MESHGRID_HPP

#include <array>
#include <list>
#include <vector>

#include "mesh/node.hpp"

namespace visualmesh {
namespace model {

    template <typename Submodel>
    struct Energy {
    public:
        static constexpr int N_NEIGHBOURS = Submodel::N_NEIGHBOURS;
        using Scalar                      = typename Submodel::Scalar;

        template <typename Shape>
        static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                                const Scalar& h,
                                                                const Scalar& k,
                                                                const Scalar& max_distance) {

            Scalar max_phi    = std::tan(max_distance, h);
            Scalar slices     = k * Scalar(2.0 * M_PI) / shape.theta(shape.n(max_phi, h), h);

            for(int i = 0; i < slices; ++i) {
                
            }
            // TODO Find top n for each node
            // TODO get the mean/median/something and split them and make new nodes there

            // TODO make a list of nodes that go around the edge
            // TODO set the links for the outward points to something that flags them as "edge" points

            // TODO loop until some condition

            // TODO find the n nearest neighbours and set the distance to them
        }
    };

}  // namespace model
}  // namespace visualmesh

#endif  // VISUALMESH_MODEL_MESHGRID_HPP
