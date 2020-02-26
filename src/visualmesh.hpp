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

#ifndef VISUALMESH_HPP
#define VISUALMESH_HPP

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "mesh/mesh.hpp"
#include "mesh/model/ring6.hpp"

namespace visualmesh {

/**
 * @brief An aggregate of many Visual Meshs at different heights that can be looked up for performance.
 *
 * @details
 *  Provides convenience functions for accessing projection and classification of the mesh using different engines.
 *  The available engines are currently limited to OpenCL and CPU, however CUDA and Vulkan can be added later.
 *
 * @tparam Scalar the type that will hold the vectors <float, double>
 * @tparam Model  the model used to generate the mesh in each of the individual heights
 */
template <typename Scalar = float, template <typename> class Model = model::Ring6>
class VisualMesh {
public:
    /**
     * @brief Makes an unallocated visual mesh with no LUTs
     */
    VisualMesh() {}

    /**
     * @brief Generate a new visual mesh for the given shape.
     *
     * @tparam Shape the shape type that this mesh will generate using
     *
     * @param shape        the shape we are generating a visual mesh for
     * @param min_height   the minimum height that our camera will be at
     * @param max_height   the maximum height our camera will be at
     * @param k            the number of intersections with the object
     * @param max_error    the maximum amount of error in terms of k that a mesh can have
     * @param max_distance the maximum distance that this mesh will project for
     */
    template <typename Shape>
    explicit VisualMesh(const Shape& shape,
                        const Scalar& min_height,
                        const Scalar& max_height,
                        const Scalar& k,
                        const Scalar& max_error,
                        const Scalar& max_distance) {

        // Add an element for the min and max height
        luts.insert(std::make_pair(min_height, Mesh<Scalar, Model>(shape, min_height, k, max_distance)));
        luts.insert(std::make_pair(max_height, Mesh<Scalar, Model>(shape, max_height, k, max_distance)));

        // Run through a stack splitting the range in two until the region is filled appropriately
        std::vector<vec2<Scalar>> stack;
        stack.emplace_back(vec2<Scalar>{min_height, max_height});

        while (!stack.empty()) {
            // Get the next element for consideration
            vec2<Scalar> range = stack.back();
            Scalar h           = (range[0] + range[1]) / 2;
            stack.pop_back();

            Scalar lower_err = std::abs(k - k * shape.k(range[0], h));
            Scalar upper_err = std::abs(k - k * shape.k(range[1], h));

            // If we aren't close enough to both elements
            if (lower_err > max_error || upper_err > max_error) {
                luts.insert(std::make_pair(h, Mesh<Scalar, Model>(shape, h, k, max_distance)));
                stack.emplace_back(vec2<Scalar>{range[0], h});
                stack.emplace_back(vec2<Scalar>{h, range[1]});
            }
        }
    }

    /**
     * Find a visual mesh that exists at a specific height above the observation plane.
     * This only looks up meshes that were created during instantiation.
     * If this lookup is out of range, it will return the highest or lowest mesh (whichever is closer)
     *
     * @param  height the height above the observation plane for the mesh we are trying to find
     *
     * @return the closest generated visual mesh to the provided height
     */
    const Mesh<Scalar, Model>& height(const Scalar& height) const {
        // Find the bounding height values
        auto range = luts.equal_range(height);

        // If we reached the end of the list return the lower bound
        if (range.second == luts.end()) {
            if (range.first == luts.end()) {  // We are off the larger end
                return luts.rbegin()->second;
            }
            else {  // We are off the smaller end
                return luts.begin()->second;
            }
        }
        // Otherwise see which has less error
        else if (std::abs(range.first->first - height) < std::abs(range.second->first - height)) {
            return range.first->second;
        }
        else {
            return range.second->second;
        }
    }

    /**
     * Performs a visual mesh lookup using the description of the lens provided to find visual mesh points on the image.
     *
     * @param Hoc   A 4x4 homogeneous transformation matrix that transforms from the observation plane to camera space.
     * @param lens  A description of the lens used to project the mesh.
     *
     * @return the mesh that was used for this lookup and a vector of start/end indices that are on the screen.
     */
    std::pair<const Mesh<Scalar, Model>&, std::vector<std::pair<uint, uint>>> lookup(const mat4<Scalar>& Hoc,
                                                                                     const Lens<Scalar>& lens) const {

        // z height from the transformation matrix
        const Scalar& h = Hoc[2][3];
        auto mesh       = height(h);
        return mesh->lookup(Hoc, lens);
    }

private:
    /// A map from heights to visual mesh tables
    std::map<Scalar, const Mesh<Scalar, Model>> luts;
};

}  // namespace visualmesh

#endif  // VISUALMESH_HPP
