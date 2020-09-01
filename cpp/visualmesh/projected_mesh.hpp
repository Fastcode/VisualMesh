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

#ifndef VISUALMESH_PROJECTED_MESH_HPP
#define VISUALMESH_PROJECTED_MESH_HPP

#include <array>
#include <vector>

namespace visualmesh {

/**
 * @brief A projected mesh is a Visual Mesh that has been projected onto a camera image
 *
 * @tparam Scalar       the scalar type used for calculations and storage (normally one of float or double)
 * @tparam N_NEIGHBOURS the number of neighbours that each point has
 */
template <typename Scalar, int N_NEIGHBOURS>
struct ProjectedMesh {

    /// The pixel coordinates (x,y) of the points projected from the visual mesh
    std::vector<std::array<Scalar, 2>> pixel_coordinates;
    /// The index graph giving the locations of the neighbours of each point
    std::vector<std::array<int, N_NEIGHBOURS>> neighbourhood;
    /// The original indicies of these points in the visual mesh
    std::vector<int> global_indices;
};

}  // namespace visualmesh

#endif  // VISUALMESH_PROJECTED_MESH_HPP
