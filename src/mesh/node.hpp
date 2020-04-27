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

#ifndef VISUALMESH_NODE_HPP
#define VISUALMESH_NODE_HPP

#include <array>

#include "utility/math.hpp"

namespace visualmesh {

/**
 * @brief This represents a single node in the visual mesh.
 *
 * @details
 *  A single node in the visual mesh is a single point, it is made up of a vector in observation plane space that points
 *  to the position, as well as a list of neighbours that are connected to this point. A collection of these with each
 *  having neighbours pointing to indices of other points in the list make up a Visual Mesh. The neighbours are ordered
 *  in a clockwise fashion.
 *
 * @tparam Scalar       the scalar type used for calculations and storage (normally one of float or double)
 * @tparam N_NEIGHBOURS the number of neighbours that each point has
 */
template <typename Scalar, int N_NEIGHBOURS>
struct Node {
    /// The unit vector in the direction for this node
    vec3<Scalar> ray;
    /// Absolute indices to the linked nodes ordered L, TL, TR, R, BR, BL (clockwise)
    std::array<int, N_NEIGHBOURS> neighbours;
};

}  // namespace visualmesh

#endif  // VISUALMESH_NODE_HPP
