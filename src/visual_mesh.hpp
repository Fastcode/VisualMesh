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

#ifndef VISUALMESH_HPP
#define VISUALMESH_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

namespace mesh {

/**
 * An aggregate of many Visual Meshs at different heights that can be looked up
 * Provides convienence functions for accessing projection and classification of the mesh using different engines.
 * The available engines are currently limited to OpenCL and CPU, however CUDA and Vulkan can be added later.
 *
 * @tparam Scalar the type that will hold the vectors <float, double>
 */
template <typename Scalar = float, typename Engine>
class VisualMesh {
public:
  /**
   * @brief Makes an unallocated visual mesh
   */
  VisualMesh() {}

  /**
   * @brief Generate a new visual mesh for the given shape.
   *
   * @param shape             the shape we are generating a visual mesh for
   * @param min_height        the minimum height that our camera will be at
   * @param max_height        the maximum height our camera will be at
   * @param height_resolution the number of look up tables to generated (height gradations)
   */
  template <typename Shape>
  explicit VisualMesh(const Shape& shape,
                      const Scalar& min_height,
                      const Scalar& max_height,
                      const uint& height_resolution) {  // TODO height resolution is a dumb name

    // Loop through to make a mesh for each of our height possibilities
    for (Scalar h = min_height; h < max_height; h += (max_height - min_height) / height_resolution) {

      // Insert our constructed mesh into the lookup
      luts.insert(std::make_pair(h, std::make_shared<Mesh>(shape, h)));
    }
  }

  /**
   * Find a visual mesh that exists at a specific height above the observation plane.
   * This only looks up meshes that were created during instantiation.
   * If this lookup is out of range, it will return the highest or lowest mesh (whichever is closer)
   *
   * @param  height the height above the observation plane for the mesh we are trying to find
   *
   * @return        the closest generated visual mesh to the provided height
   */
  const Mesh& height(const Scalar& height) const {
    // Find the bounding height values
    auto range = luts.equal_range(height);

    // If we reached the end of the list return the lower bound
    if (range.second == luts.end()) { return range.first->second; }
    // Otherwise see which is closer
    else if (std::abs(range.first->first - height) < std::abs(range.second->first - height)) {
      return range.first->second;
    }
    else {
      return range.second->second;
    }
  }

  /**
   * Performs a visual mesh lookup, finding the start and end indexes for visual mesh points that are on the screen.
   * It uses the provided `theta_limits` function to identify the start and end points on the screen for a specific phi.
   *
   * @param height       the height to use for looking up the mesh, follows the same rules as `VisualMesh::height`
   * @param theta_limits the function that is used to calculate the start/end indices for a specific phi
   *
   * @return             the mesh that was used for this lookup and a vector of start/end indices that are on the
   *                     screen.
   *
   * @tparam Func        the type of the function that identifies theta ranges given a phi value
   */
  template <typename Func>
  std::pair<std::shared_ptr<Mesh>, std::vector<std::pair<uint, uint>>> lookup(const Scalar& height,
                                                                              Func&& theta_limits) const {

    auto mesh = this->height(height);
    return std::make_pair(mesh, mesh->lookup(std::forward<Func>(theta_limits)));
  }

  /**
   * Performs a visual mesh lookupusing the description of the lens provided to find visual mesh points on the image.
   *
   * @param height       the height to use for looking up the mesh, follows the same rules as `VisualMesh::height`
   * @param theta_limits the function that is used to calculate the start/end indices for a specific phi
   *
   * @return             the mesh that was used for this lookup and a vector of start/end indices that are on the
   *                     screen.
   */
  std::pair<std::shared_ptr<Mesh>, std::vector<std::pair<uint, uint>>> lookup(const mat4& Hoc, const Lens& lens) {

    auto mesh = this->height(height);
    return return std::make_pair(mesh, mesh->lookup(Hoc, lens));
  }

  // TODO Project using engine
  // TODO classify using engine

private:
  /// A map from heights to visual mesh tables
  std::map<Scalar, std::shared_ptr<Mesh>> luts;
};

}  // namespace mesh

#endif  // VISUALMESH_HPP
