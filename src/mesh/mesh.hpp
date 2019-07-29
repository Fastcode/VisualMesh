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

#ifndef VISUALMESH_MESH_HPP
#define VISUALMESH_MESH_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>
#include "generator/hexapizza.hpp"
#include "lens.hpp"
#include "node.hpp"
#include "util/cone.hpp"
#include "util/math.hpp"
#include "util/projection.hpp"

namespace visualmesh {

template <typename Scalar>
struct Mesh {
private:
  struct BSP {
    // The bounds of the range that this BSP element represents (start to one past the end)
    std::pair<int, int> range;
    // The indicies of the two children in this BSP
    std::array<int, 2> children;
    // The paramters of the cone that describe this part of the BSP
    std::pair<vec3<Scalar>, vec2<Scalar>> cone;
  };

  /**
   * @brief Given a set of points, find the smallest cone that contains all points
   *
   * @details implements welzls algorithm for circles, but instead for cones
   *
   * @tparam Iterator the type of the iterator passed in
   *
   * @param
   */
  template <typename Iterator>
  std::pair<vec3<Scalar>, vec2<Scalar>> bounding_cone(Iterator start, Iterator end) {
    std::pair<vec3<Scalar>, vec2<Scalar>> cone(cone_from_points<Scalar>());
    for (auto i = start; i < end; ++i) {
      if (dot(cone.first, nodes[*i].ray) < cone.second[0]) {
        cone = cone_from_points(nodes[*i].ray);
        for (auto j = start; j < i; ++j) {
          if (dot(cone.first, nodes[*j].ray) < cone.second[0]) {
            cone = cone_from_points(nodes[*i].ray, nodes[*j].ray);
            for (auto k = start; k < j; ++k) {
              if (dot(cone.first, nodes[*k].ray) < cone.second[0]) {
                cone = cone_from_points(nodes[*i].ray, nodes[*j].ray, nodes[*k].ray);
              }
            }
          }
        }
      }
    }
    return cone;
  }

  template <typename Iterator>
  int build_bsp(Iterator start, Iterator end, int min_points = 8, int offset = 0) {
    // No points in this partition, this should never happen
    if (std::distance(start, end) == 0) { throw std::runtime_error("We tried to make a tree with no nodes"); }

    // If we have few enough points, terminate the search here and return what we have. It can be cheaper to project a
    // list of pixels than to do more BSP steps. This also makes it cheaper to build the BSP and less memory to store.
    if (std::distance(start, end) <= min_points) {
      int elem = bsp.size();

      // Add this element with children -1,-1 to signify it has no children
      bsp.push_back(BSP{std::make_pair(offset, static_cast<int>(offset + std::distance(start, end))),
                        {{-1, -1}},
                        bounding_cone(start, end)});

      // By default, sort by index that they were generated with to remove the remaining randomness
      std::sort(start, end);

      return elem;
    }
    // We have some points
    else {
      // Calculate our bounding cone for this cluster. We have to do a random sort of our segment here so that the
      // performance of the bounding cone algorithm is expected to be linear
      auto cone = bounding_cone(start, end);

      // Split the larger angle range so we have as close to cone shapes as we can
      auto minmax_phi   = std::minmax_element(start, end, [this](const int& a, const int& b) {
        return nodes[a].ray[2] < nodes[b].ray[2];  //
      });
      auto minmax_theta = std::minmax_element(start, end, [this](const int& a, const int& b) {
        return std::atan2(nodes[a].ray[1], nodes[a].ray[0]) < std::atan2(nodes[b].ray[1], nodes[b].ray[0]);
      });

      // Get our min and max phi and theta
      Scalar min_phi     = nodes[*minmax_phi.first].ray[2];
      Scalar max_phi     = nodes[*minmax_phi.second].ray[2];
      Scalar min_theta   = std::atan2(nodes[*minmax_theta.first].ray[1], nodes[*minmax_theta.first].ray[0]);
      Scalar max_theta   = std::atan2(nodes[*minmax_theta.second].ray[1], nodes[*minmax_theta.second].ray[0]);
      Scalar split_phi   = std::cos((std::acos(min_phi) + std::acos(max_phi)) * 0.5);
      Scalar split_theta = (min_theta + max_theta) * 0.5;

      // Partition based on either phi or theta
      Iterator mid =
        max_phi - min_phi > max_theta - min_theta
          ? std::partition(start, end, [this, split_phi](const int& a) { return nodes[a].ray[2] > split_phi; })
          : std::partition(start, end, [this, split_theta](const int& a) {
              return std::atan2(nodes[a].ray[1], nodes[a].ray[0]) < split_theta;
            });

      int elem = bsp.size();
      bsp.push_back(BSP{
        std::make_pair(offset, static_cast<int>(offset + std::distance(start, end))),
        {{-2, -2}},
        cone,
      });
      bsp[elem].children = {{
        build_bsp(start, mid, min_points, offset),
        build_bsp(mid, end, min_points, offset + std::distance(start, mid)),
      }};
      return elem;
    }
  }

  /**
   * Given the lens, get the cone objects that best fit each edge of the screen (or planes for the rectilinear case)
   * This is arranged as the axis (or normal) and the cos and sin of the angle for each of the edges.
   *
   * @return The cones that best describe the edges of the camera
   */
  static std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4> screen_edges(const mat4<Scalar> Hoc,
                                                                           const Lens<Scalar>& lens) {

    // Extract our rotation matrix
    const mat3<Scalar> Roc = {{
      {{Hoc[0][0], Hoc[0][1], Hoc[0][2]}},
      {{Hoc[1][0], Hoc[1][1], Hoc[1][2]}},
      {{Hoc[2][0], Hoc[2][1], Hoc[2][2]}},
    }};

    /* The labels for each of the corners of the screen in cam space are is shown below.
     * ^    T       U
     * |        C
     * z    W       V
     * <- y
     */

    // Unproject each of the four corners of the screen into camera space
    const vec2<Scalar> dimensions          = cast<Scalar>(lens.dimensions);
    const std::array<vec3<Scalar>, 4> rNCc = {{
      visualmesh::unproject(vec2<Scalar>{0, 0}, lens),                          // rTCc
      visualmesh::unproject(vec2<Scalar>{dimensions[0], 0}, lens),              // rUCc
      visualmesh::unproject(vec2<Scalar>{dimensions[0], dimensions[1]}, lens),  // rVCc
      visualmesh::unproject(vec2<Scalar>{0, dimensions[1]}, lens),              // rWCc
    }};

    // Rotate these vectors into world space
    const std::array<vec3<Scalar>, 4> rNCo = {{
      multiply(Roc, rNCc[0]),  // rTCo
      multiply(Roc, rNCc[1]),  // rUCo
      multiply(Roc, rNCc[2]),  // rVCo
      multiply(Roc, rNCc[3]),  // rWCo
    }};

    switch (lens.projection) {
      case LensProjection::RECTILINEAR: {
        // For the case of a plane, we have a cone with a 90 degrees which means cos(theta) = 0 and sin(theta) = 1
        return std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>{{
          std::make_pair(normalise(cross(rNCo[1], rNCo[0])), vec2<Scalar>{0, 1}),
          std::make_pair(normalise(cross(rNCo[2], rNCo[1])), vec2<Scalar>{0, 1}),
          std::make_pair(normalise(cross(rNCo[3], rNCo[2])), vec2<Scalar>{0, 1}),
          std::make_pair(normalise(cross(rNCo[0], rNCo[3])), vec2<Scalar>{0, 1}),
        }};
      }
      case LensProjection::EQUIDISTANT:
      case LensProjection::EQUISOLID: {
        /* The labels for each of the corners of the screen in cam space are is shown below.
         * ^        D
         * |    G   C   E
         * z        F
         * <- y
         */
        // Get the lens axis centre coordinates
        vec2<Scalar> centre = add(multiply(dimensions, static_cast<Scalar>(0.5)), lens.centre);

        // Unproject the centre of each of the edges into cam space using the lens axis as the centre
        const std::array<vec3<Scalar>, 4> rECc = {{
          visualmesh::unproject(vec2<Scalar>{centre[0], 0}, lens),              // rDCc
          visualmesh::unproject(vec2<Scalar>{dimensions[0], centre[1]}, lens),  // rECc
          visualmesh::unproject(vec2<Scalar>{centre[0], dimensions[1]}, lens),  // rFCc
          visualmesh::unproject(vec2<Scalar>{0, centre[1]}, lens),              // rGCc
        }};

        // Rotate these vectors into world space
        const std::array<vec3<Scalar>, 4> rECo = {{
          multiply(Roc, rECc[0]),  // rTCo
          multiply(Roc, rECc[1]),  // rUCo
          multiply(Roc, rECc[2]),  // rVCo
          multiply(Roc, rECc[3]),  // rWCo
        }};

        // Calculate cones from each of the four screen edges
        return std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>{{
          cone_from_points(rNCo[1], rECo[0], rNCo[0]),
          cone_from_points(rNCo[2], rECo[1], rNCo[1]),
          cone_from_points(rNCo[3], rECo[2], rNCo[2]),
          cone_from_points(rNCo[0], rECo[3], rNCo[3]),
        }};
      }
    }
  }

  /**
   * Check if a point is on the screen, given a description of the edges of the screen as cones, and the axis
   */
  static inline std::pair<bool, bool> check_on_screen(
    const mat3<Scalar>& Rco,
    const std::pair<vec3<Scalar>, vec2<Scalar>>& cone,
    const Lens<Scalar>& lens,
    const std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>& edges) {

    // Firstly check if the cone axis is on the screen
    vec2<Scalar> px = ::visualmesh::project(multiply(Rco, cone.first), lens);
    bool axis_on_screen =
      0 <= px[0] && px[0] + 1 <= lens.dimensions[0] && 0 <= px[1] && px[1] + 1 <= lens.dimensions[1];

    std::array<Scalar, 4> angles{{
      dot(cone.first, edges[0].first),
      dot(cone.first, edges[1].first),
      dot(cone.first, edges[2].first),
      dot(cone.first, edges[3].first),
    }};

    // Check if our cone is entirely contained within a screen edge
    // acos(dot(cone_axis, edge_axis)) < edge_angle - cone_angle
    std::array<bool, 4> contains{{
      angles[0] > edges[0].second[0] * cone.second[0] + edges[0].second[1] * cone.second[1],
      angles[1] > edges[1].second[0] * cone.second[0] + edges[1].second[1] * cone.second[1],
      angles[2] > edges[2].second[0] * cone.second[0] + edges[2].second[1] * cone.second[1],
      angles[3] > edges[3].second[0] * cone.second[0] + edges[3].second[1] * cone.second[1],
    }};

    // If we are off the screen, we are not on it?
    const bool outside = !axis_on_screen && (contains[0] || contains[1] || contains[2] || contains[3]);
    if (outside) { return std::make_pair(false, true); }

    // Check if we intersect with any of the edges of the screen
    // acos(dot(cone_axis, edge_axis) > edge_angle + cone_angle
    std::array<bool, 4> intersects{{
      angles[0] < edges[0].second[0] * cone.second[0] - edges[0].second[1] * cone.second[1],
      angles[1] < edges[1].second[0] * cone.second[0] - edges[1].second[1] * cone.second[1],
      angles[2] < edges[2].second[0] * cone.second[0] - edges[2].second[1] * cone.second[1],
      dot(cone.first, edges[3].first) < edges[3].second[0] * cone.second[0] - edges[3].second[1] * cone.second[1],
    }};

    // Inside if the axis is on the screen and we don't intersect with any of the edges
    const bool inside = axis_on_screen && intersects[0] && intersects[1] && intersects[2] && intersects[3];

    return std::make_pair(inside, outside);
  }


public:
  template <template <typename T> class Generator = generator::HexaPizza, typename Shape>
  Mesh(const Shape& shape, const Scalar& h, const Scalar& k, const Scalar& max_distance) : h(h) {
    Timer t;
    nodes = Generator<Scalar>::generate(shape, h, k, max_distance);
    t.measure("Generate Mesh");
    // To ensure that later we can fix the graph we need to perform our sorting on an index list
    std::vector<int> sorting(nodes.size());
    std::iota(sorting.begin(), sorting.end(), 0);
    t.measure("Init");

    // We need to shuffle our list to ensure that the bounding cone algorithm has roughly linear performance.
    // We could use std::random_shuffle here but since we only need the list to be "kinda shuffled" so that it's
    // unlikely that we hit the worst case of the bounding cone algorithm. We can actually just shuffle every nth
    // element and use a fairly bad random number generator algorithm
    for (int i = sorting.size() - 1; i > 0; i -= 5) {
      std::swap(sorting[i], sorting[rand() % i]);
    }
    t.measure("Shuffle");

    // Build our bsp tree
    // Reserve enough memory for the bsp as we know how many nodes it will need
    bsp.reserve(nodes.size() * 2);
    build_bsp(sorting.begin(), sorting.end());
    t.measure("Built BSP");

    // Make our reverse lookup so we can correct the neighbourhood indices
    std::vector<int> r_sorting(nodes.size() + 1);
    r_sorting[nodes.size()] = nodes.size();
    for (int i = 0; i < nodes.size(); ++i) {
      r_sorting[sorting[i]] = i;
    }
    t.measure("Built reverse map");

    // Sort the nodes and correct the neighbourhood graph based on our BSP sorting
    std::vector<Node<Scalar>> sorted_nodes;
    sorted_nodes.reserve(nodes.size());
    for (const auto& i : sorting) {
      sorted_nodes.push_back(nodes[i]);
      for (int& n : sorted_nodes.back().neighbours) {
        n = r_sorting[n];
      }
    }
    t.measure("Sorting");

    nodes = std::move(sorted_nodes);
    t.measure("Updating");
  }

  std::vector<std::pair<int, int>> lookup(const mat4<Scalar>& Hoc, const Lens<Scalar>& lens) const {

    // Our FOV is an easy check to exclude things outside our view
    const Scalar cos_fov = std::cos(lens.fov);
    const Scalar sin_fov = std::sin(lens.fov);

    // Get the x axis of the camera in world space and the cone equations that describe the edges of the screen
    const mat3<Scalar> Rco(block<3, 3>(transpose(Hoc)));
    const vec3<Scalar> cam_x{Hoc[0][0], Hoc[1][0], Hoc[2][0]};
    const auto edges = screen_edges(Hoc, lens);

    // Go through our BSP tree to work out which segments of the mesh are on screen
    // The first element of the tree is the root element of the bsp
    std::vector<int> stack(1, 0);
    std::vector<std::pair<int, int>> ranges;
    bool building   = false;
    int range_start = 0;
    int range_end   = 0;
    while (!stack.empty()) {
      // Grab our next bsp element
      int i = stack.back();
      stack.pop_back();

      // Get the data from our bsp element
      const auto& elem = bsp[i];
      const auto& cone = elem.cone;

      // Dot the camera x axis in world with the cone
      Scalar delta = dot(cam_x, cone.first);

      // Check if we are outside the field of view of the lens using an easy check
      // To check if we are inside or outside the cone we need to check how angle between the cones compares
      // outside == dot(cam, axis) < cos(fov + acos(gradient))
      // However given that the thetas don't change and we have gradient naturally from the dot product it's easier
      // to calculate it using the compound angle formula
      bool outside = delta < cos_fov * cone.second[0] - sin_fov * cone.second[1];
      bool inside  = false;

      // If we are not ruled as outside by the field of view we might be on the screen
      if (!outside) { std::tie(inside, outside) = check_on_screen(Rco, cone, lens, edges); }

      if (inside) {
        std::cout << "FOUND" << std::endl;
        // If we are building just update our end point
        if (building) {
          range_end = elem.range.second;  //
        }
        else {
          range_start = elem.range.first;
          range_end   = elem.range.second;
          building    = true;
        }
      }
      else if (outside) {
        std::cout << "SKIPPING" << std::endl;
        // If we found an outside point we have finished building our range
        if (building) {
          ranges.emplace_back(std::make_pair(range_start, range_end));
          building = false;
        }
      }
      // We have reached the end of a tree, from here we need to check each point on screen individually
      else if (elem.children[0] < 0) {
        for (int i = elem.range.first; i < elem.range.second; ++i) {
          auto px = visualmesh::project(multiply(Rco, nodes[i].ray), lens);
          std::cout << "GOTTA TEST " << i << " " << px << std::endl;
          if (0 <= px[0] && px[0] + 1 <= lens.dimensions[0] && 0 <= px[1] && px[1] + 1 <= lens.dimensions[1]) {
            // TODO the whole if building end thing
            std::cout << "\tON SCREEN!" << std::endl;
          }
        }
        // TODO go through the range and project each individual ray to the screen to see if it's on the screen
        std::cout << "LEAF_NODE!" << std::endl;
      }
      else {
        std::cout << "SPLITTING" << std::endl;
        // Add the children of this to the search in order 1,0 so we pop 0 first (contiguous indices)
        stack.push_back(elem.children[1]);
        stack.push_back(elem.children[0]);
        for (const auto& s : stack) {}
      }
    }
    // If we finished while building add the last point
    if (building) { ranges.emplace_back(std::make_pair(range_start, range_end)); }

    std::cout << ranges.size() << std::endl;
    exit(0);
    return ranges;
  }
  /// The height that this mesh is designed to run at
  Scalar h;
  /// The lookup table for this mesh
  std::vector<Node<Scalar>> nodes;

private:
  std::vector<BSP> bsp;
};

}  // namespace visualmesh

#endif  // VISUALMESH_MESH_HPP
