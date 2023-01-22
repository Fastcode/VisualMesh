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

#ifndef VISUALMESH_TENSORFLOW_MESH_CACHE_HPP
#define VISUALMESH_TENSORFLOW_MESH_CACHE_HPP

#include <memory>
#include <mutex>
#include <vector>

#include "visualmesh/mesh.hpp"

/**
 * @brief Given a shape, two heights and a k value, calculate the absolute number of intersections difference given
 * the new height.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 * @tparam Shape  the type of shape to use when calculating the error
 *
 * @param shape the instance of the shape that will be used to calculate the k error
 * @param h_0   the height of the camera in the mesh we are comparing to
 * @param h_1   the current height of the camera we want to get an error for
 * @param k     the k value that the original mesh was designed to use
 *
 * @return what the k value would be if we used this mesh at this height
 */
template <typename Scalar, template <typename> class Shape>
Scalar mesh_k_error(const Shape<Scalar>& shape, const Scalar& h_0, const Scalar& h_1, const Scalar& k) {
    return std::abs(k - k * shape.k(h_0, h_1));
}

/**
 * @brief Lookup an appropriate Visual Mesh to use for this lens and height given the provided tolerances
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 * @tparam Shape  the type of shape to use when calculating the error
 *
 * @param meshes  the list of meshes that we will be looking for the target in
 * @param shape   the shape that we will be using for the lookup
 * @param h       the current height of the camera above the ground
 * @param k       the number of cross sectional intersections that we want with the object
 * @param t       the tolerance for the number of cross sectional intersections before we need a new mesh
 * @param d       the maximum distance that the mesh should be generated for
 *
 * @return either returns a shared_ptr to the mesh that would best fit within our tolerance, or if none could be found a
 *         nullptr
 */
template <typename Scalar, template <typename> class Model, template <typename> class Shape>
std::shared_ptr<visualmesh::Mesh<Scalar, Model>> find_mesh(
  std::vector<std::shared_ptr<visualmesh::Mesh<Scalar, Model>>>& meshes,
  const Shape<Scalar>& shape,
  const Scalar& h,
  const Scalar& k,
  const Scalar& t,
  const Scalar& d) {

    // Nothing in the map!
    if (meshes.empty()) { return nullptr; }

    // Find the best mesh we have available
    auto best_it      = meshes.begin();
    Scalar best_error = std::numeric_limits<Scalar>::max();
    for (auto it = meshes.begin(); it != meshes.end(); ++it) {
        auto error = mesh_k_error(shape, (*it)->h, h, k);
        if (d == (*it)->max_distance && error < best_error) {
            best_error = error;
            best_it    = it;
        }
    }

    // If it was good enough return it, otherwise return null
    if (best_error <= t) {
        // Swap it to the top of the list so we can keep somewhat of which items are most used
        std::iter_swap(meshes.begin(), best_it);
        // iter_swap moves the lowest error mesh into the first position of meshes
        return *meshes.begin();
    }

    return nullptr;
}

/**
 * @brief Lookup or create an appropriate Visual Mesh to use for this lens and height given the provided tolerances
 *
 * @details
 *  This function gets the best fitting mesh that it can find that is within the number of intersections tolerance. If
 *  it cannot find a mesh that matches the tolerance it will create a new one for the provided details. The mesh will
 *  not match if the maximum distance has changed, only if the k difference is small enough. Additionally it will only
 *  cache `cached_meshes` number of meshes. Each time a mesh is used again it will move it to the top of the list, and
 *  if a new mesh must be added and this would exceed this limit the least recently used mesh will be dropped.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 * @tparam Shape  the type of shape to use when calculating the error
 *
 * @param shape                   the shape that we will be using for the lookup
 * @param height                  the current height of the camera above the ground
 * @param n_intersections         the number of cross sectional intersections that we want with the object
 * @param intersection_tolerance  tolerance for the number of cross sectional intersections before we need a new mesh
 * @param cached_meshes           the number of meshes to cache at any one time before we delete one
 * @param max_distance            the maximum distance that the mesh should be generated for
 *
 * @return std::shared_ptr<visualmesh::Mesh<Scalar>>
 */
template <typename Scalar, template <typename> class Model, template <typename> class Shape>
std::shared_ptr<visualmesh::Mesh<Scalar, Model>> get_mesh(const Shape<Scalar>& shape,
                                                          const Scalar& height,
                                                          const Scalar& n_intersections,
                                                          const Scalar& intersection_tolerance,
                                                          const int32_t& cached_meshes,
                                                          const Scalar& max_distance) {

    // Static map of heights to meshes
    static std::vector<std::shared_ptr<visualmesh::Mesh<Scalar, Model>>> meshes;
    static std::mutex mesh_mutex;

    // Find and return an element if one is appropriate
    /* mutex scope */ {
        std::lock_guard<std::mutex> lock(mesh_mutex);

        // If we found an acceptable mesh return it
        auto mesh = find_mesh(meshes, shape, height, n_intersections, intersection_tolerance, max_distance);
        if (mesh != nullptr) { return mesh; }
    }

    // We can't find an appropriate mesh, make a new one but don't hold the mutex while we do so others can still query
    // Generate the mesh using double precision and then cast it over to whatever we need
    auto generated_mesh = visualmesh::Mesh<double, Model>(shape, height, n_intersections, max_distance);

    /* mutex scope */ {
        std::lock_guard<std::mutex> lock(mesh_mutex);

        // Check again for an acceptable mesh in case someone else made one too
        auto mesh = find_mesh(meshes, shape, height, n_intersections, intersection_tolerance, max_distance);
        if (mesh != nullptr) { return mesh; }

        // Only cache a fixed number of meshes so remove the old ones
        while (static_cast<int32_t>(meshes.size()) > cached_meshes) {
            meshes.pop_back();
        }

        // Add our new mesh to the cache and return
        meshes.push_back(std::make_shared<visualmesh::Mesh<Scalar, Model>>(generated_mesh));
        return meshes.back();
    }
}

#endif  // VISUALMESH_TENSORFLOW_MESH_CACHE_HPP
