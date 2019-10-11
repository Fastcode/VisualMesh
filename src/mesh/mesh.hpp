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

/**
 * @brief Holds a description of a Visual Mesh
 *
 * @details
 *  This object holds a Visual Mesh for a single height. It utilises a generator class to create a mesh, and then
 *  transforms it into a structure that is suppored as a BSP tree. It then uses this tree to lookup the mesh given
 *  different lens paramters. Note that because of teh way it oes the lookup, this lookup isn't perfect esperically for
 *  fisheye lenses. In this case it will sometimes give points that are outside the bounds of the image. The CPU engine
 *  removes these extra points however the OpenCL engine does not. So if you are relying on these pixel coordinates
 *  being in bounds you should ensure that you use the CPU engine.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 */
template <typename Scalar>
struct Mesh {
private:
  /**
   * @brief An element of a binary search partitioning scheme to quickly work out which points are on the screen.
   *
   * @details
   *  This is a node in a binary search partition. It is represented by a bounding code that can be used to work out if
   *  any of the elements in that cone are on the screen. These cones will be split into sub cones to further limit the
   *  scope of the search until a list of a few elements are found that can be checked manually.
   */
  struct BSP {
    // The bounds of the range that this BSP element represents (start to one past the end)
    std::pair<int, int> range;
    // The indicies of the two children in this BSP
    std::array<int, 2> children;
    // The paramters of the cone that describe this part of the BSP
    // These include the unit axis in world space, and the cos and sin of the cone angle (θ)
    //
    //  \    a    /
    //   \   |   /
    //    \  |--/
    //     \ |θ/
    //      \|/
    //       V
    std::pair<vec3<Scalar>, vec2<Scalar>> cone;
  };

  /**
   * @brief Given a set of points, find the smallest cone that contains all points
   *
   * @details
   *  Implements welzls algorithm for circles, but instead for cones. You should randomize the iterator before running
   *  this algorithm, otherwise you can suffer from very poor performance
   *
   * @tparam Iterator the type of the iterator passed in
   *
   * @param start the start iterator of points to consider for the circle
   * @param end   the end iterator of points to consider for the circle
   */
  template <typename Iterator>
  std::pair<vec3<Scalar>, vec2<Scalar>> bounding_cone(Iterator start, Iterator end) {
    std::pair<vec3<Scalar>, Scalar> cone(cone_from_points<Scalar>());
    for (auto i = start; i < end; ++i) {
      if (dot(cone.first, nodes[*i].ray) < cone.second) {
        cone = cone_from_points(nodes[*i].ray);
        for (auto j = start; j < i; ++j) {
          if (dot(cone.first, nodes[*j].ray) < cone.second) {
            cone = cone_from_points(nodes[*i].ray, nodes[*j].ray);
            for (auto k = start; k < j; ++k) {
              if (dot(cone.first, nodes[*k].ray) < cone.second) {
                cone = cone_from_points(nodes[*i].ray, nodes[*j].ray, nodes[*k].ray);
              }
            }
          }
        }
      }
    }

    // Add in sin_theta when we return the final cone
    return std::make_pair(cone.first,
                          vec2<Scalar>{cone.second, std::sqrt(static_cast<Scalar>(1.0) - cone.second * cone.second)});
  }

  /**
   * @brief Given an iterator to a set of Visual Mesh nodes, calculate a binary search partition for it
   *
   * @details
   *  This algorithm takes in iterators to a set of Visual Mesh nodes and sorts them such that they conform to a binary
   *  search partitioning scheme. These partitions are described using bounding cones which can then be used to include
   *  or throw out points on mass
   *
   * @tparam Iterator the type of the iterator passed in, must evalute to an object of type Node
   *
   * @param start       the start iterator of points to sort into the bsp
   * @param end         the end iterator of points to sort into the bsp
   * @param min_points  the number of points that the algorithm terminates at
   * @param offset      the offset from the start of the nodes list to the region this BSP node represents
   */
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
    // We treat the first element specially
    else if (bsp.empty()) {
      // The first tree is always a split in the theta angle, and it is the split that will split +y from -y so that
      // future loops can sort purely based on x value making for a faster algorithm

      // Find the largest phi value for making the cone
      auto max_phi_element = std::max_element(start, end, [this](const int& a, const int& b) {
        return nodes[a].ray[2] < nodes[b].ray[2];  // comparing z is the same as comparing phi
      });

      // Negate as we would be dotting with the -z axis to get the angle
      const Scalar cone_cos = -nodes[*max_phi_element].ray[2];
      const Scalar cone_sin = std::sqrt(1 - cone_cos * cone_cos);

      // The cone will have a known axis (the -z axis) and our cos and sin theta come from the most positive z value
      std::pair<vec3<Scalar>, vec2<Scalar>> cone =
        std::make_pair(vec3<Scalar>{0, 0, -1}, vec2<Scalar>{cone_cos, cone_sin});

      // Partition based on the sign of the y component
      Iterator mid = std::partition(start, end, [this](const int& a) { return nodes[a].ray[1] > 0; });

      bsp.push_back(BSP{
        std::make_pair(offset, static_cast<int>(std::distance(start, end))),
        {{-2, -2}},
        cone,
      });
      bsp.front().children = {{
        build_bsp(start, mid, min_points, 0),
        build_bsp(mid, end, min_points, std::distance(start, mid)),
      }};
      return 0;
    }
    else {
      // Calculate our bounding cone for this cluster. We have to do a random sort of our segment here so that the
      // performance of the bounding cone algorithm is expected to be linear
      auto cone = bounding_cone(start, end);

      // Split the larger angle range so we have as close to cone shapes as we can
      auto minmax_phi   = std::minmax_element(start, end, [this](const int& a, const int& b) {
        return nodes[a].ray[2] < nodes[b].ray[2];  // comparing z is the same as comparing phi
      });
      auto minmax_theta = std::minmax_element(start, end, [this](const int& a, const int& b) {
        // Since we have already sorted such that our y value is either positive or negative, we can now just sort by
        // the x component once we normalise it to a 2d unit vector.
        return nodes[a].ray[0] / std::sqrt(1 - nodes[a].ray[2] * nodes[a].ray[2])
               < nodes[b].ray[0] / std::sqrt(1 - nodes[b].ray[2] * nodes[b].ray[2]);
      });

      // Get our min and max phi and theta
      Scalar min_phi   = nodes[*minmax_phi.first].ray[2];
      Scalar max_phi   = nodes[*minmax_phi.second].ray[2];
      Scalar min_theta = nodes[*minmax_theta.first].ray[0]
                         / std::sqrt(1 - nodes[*minmax_theta.first].ray[2] * nodes[*minmax_theta.first].ray[2]);
      Scalar max_theta = nodes[*minmax_theta.second].ray[0]
                         / std::sqrt(1 - nodes[*minmax_theta.second].ray[2] * nodes[*minmax_theta.second].ray[2]);

      // Work out the z and x values we need to split on
      Scalar split_phi   = std::cos((std::acos(min_phi) + std::acos(max_phi)) * 0.5);
      Scalar split_theta = std::cos((std::acos(std::max(min_theta, static_cast<Scalar>(-1)))
                                     + std::acos(std::min(max_theta, static_cast<Scalar>(1))))
                                    * 0.5);

      // Partition based on either phi or theta
      Iterator mid =
        max_phi - min_phi > max_theta - min_theta
          ? std::partition(start, end, [this, &split_phi](const int& a) { return nodes[a].ray[2] > split_phi; })
          : std::partition(start, end, [this, &split_theta](const int& a) {
              return nodes[a].ray[0] / std::sqrt(1 - nodes[a].ray[2] * nodes[a].ray[2]) < split_theta;
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
   * @details
   *  This function provides a heuristic for use with the BSP. It generates a set of cones that approximate the screen
   *  edges. These will be perfect approximations for rectilinear lenses as the cones will be planes (cones with 90
   *  degree angles). And for fisheye lenses they will give a decent approximation that will only have errors of a few
   *  pixels typically in the overselection rather than underselection.
   *
   * @param Hoc   the homogenous transformation matrix that transforms from camera space to observation plane space
   * @param lens  the lens object describing the type and geometry of the lens that is used
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

    // Unproject each of the four corners of the screen and rotate them into world space
    // Add a 1 pixel offset from the edge so that we don't go over the edge from rounding errors
    const vec2<Scalar> dimensions          = subtract(cast<Scalar>(lens.dimensions), static_cast<Scalar>(1.0));
    const std::array<vec3<Scalar>, 4> rNCo = {{
      multiply(Roc, visualmesh::unproject(vec2<Scalar>{1, 1}, lens)),                          // rTCo
      multiply(Roc, visualmesh::unproject(vec2<Scalar>{dimensions[0], 1}, lens)),              // rUCo
      multiply(Roc, visualmesh::unproject(vec2<Scalar>{dimensions[0], dimensions[1]}, lens)),  // rVCo
      multiply(Roc, visualmesh::unproject(vec2<Scalar>{1, dimensions[1]}, lens)),              // rWCo
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
        /* The labels for each of the edge centres of the screen in cam space are is shown below.
         * ^        D
         * |    G   C   E
         * z        F
         * <- y
         */
        // Get the lens axis centre coordinates
        vec2<Scalar> centre = add(multiply(dimensions, static_cast<Scalar>(0.5)), lens.centre);

        // Unproject the centre of each of the edges using the lens axis as the centre and rotate into world space
        const std::array<vec3<Scalar>, 4> rECo = {{
          multiply(Roc, visualmesh::unproject(vec2<Scalar>{centre[0], 1}, lens)),              // rTCo
          multiply(Roc, visualmesh::unproject(vec2<Scalar>{dimensions[0], centre[1]}, lens)),  // rUCo
          multiply(Roc, visualmesh::unproject(vec2<Scalar>{centre[0], dimensions[1]}, lens)),  // rVCo
          multiply(Roc, visualmesh::unproject(vec2<Scalar>{1, centre[1]}, lens)),              // rWCo
        }};

        // Calculate cones from each of the four screen edges
        const std::array<std::pair<vec3<Scalar>, Scalar>, 4> cones{{
          cone_from_points(rNCo[1], rECo[0], rNCo[0]),
          cone_from_points(rNCo[2], rECo[1], rNCo[1]),
          cone_from_points(rNCo[3], rECo[2], rNCo[2]),
          cone_from_points(rNCo[0], rECo[3], rNCo[3]),
        }};

        // Add in sin_theta
        return std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>{{
          std::make_pair(
            cones[0].first,
            vec2<Scalar>{cones[0].second, std::sqrt(static_cast<Scalar>(1.0) - cones[0].second * cones[0].second)}),
          std::make_pair(
            cones[1].first,
            vec2<Scalar>{cones[1].second, std::sqrt(static_cast<Scalar>(1.0) - cones[1].second * cones[1].second)}),
          std::make_pair(
            cones[2].first,
            vec2<Scalar>{cones[2].second, std::sqrt(static_cast<Scalar>(1.0) - cones[2].second * cones[2].second)}),
          std::make_pair(
            cones[3].first,
            vec2<Scalar>{cones[3].second, std::sqrt(static_cast<Scalar>(1.0) - cones[3].second * cones[3].second)}),
        }};
      }
      default: throw std::runtime_error("Unknown lens type");
    }
  }

  /**
   * @brief Check if a point is on the screen, given a description of the edges of the screen as cones, and the axis
   *
   * @param Rco     the 3x3 rotation matrix which rotates from observation plane space to camera space
   * @param cone    the cone object that we are checking if it is on the screen
   * @param lens    the lens object describing the type and geometry of the lens that is used
   * @param edges   the matrix of 4 cone objects that describe the edge of the screen
   *
   * @return two booleans that describe if this cone is inside (first) the screen and outside(second) the screen. If
   *         both are true then the cone is intersecting the screen edge.
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
      angles[3] < edges[3].second[0] * cone.second[0] - edges[3].second[1] * cone.second[1],
    }};

    // Inside if the axis is on the screen and we don't intersect with any of the edges
    const bool inside = axis_on_screen && intersects[0] && intersects[1] && intersects[2] && intersects[3];

    return std::make_pair(inside, outside);
  }


public:
  /**
   * @brief Construct a new Mesh object
   *
   * @details
   *  Constructs a new Mesh object using the provided generator type. This mesh object generates a BSP tree and holds
   *  the logic needed to quickly lookup points that are on screen and return valid index ranges.
   *
   * @tparam Generator the generator that is to be used to generate the Visual Mesh
   * @tparam Shape     the type of shape that will be used to generate the Visual Mesh
   *
   * @param shape         the shape instance that will be used to generate the Visual Mesh
   * @param h             the height of the camera above the observation plane
   * @param k             the number of cross section intersections that are needed for the object
   * @param max_distance  the maximum distance to generate the Visual Mesh for
   */
  template <template <typename T> class Generator = generator::Hexapizza, typename Shape>
  Mesh(const Shape& shape, const Scalar& h, const Scalar& k, const Scalar& max_distance)
    : h(h), max_distance(max_distance), nodes(Generator<Scalar>::generate(shape, h, k, max_distance)) {

    // To ensure that later we can fix the graph we need to perform our sorting on an index list
    std::vector<int> sorting(nodes.size());
    std::iota(sorting.begin(), sorting.end(), 0);

    // We need to shuffle our list to ensure that the bounding cone algorithm has roughly linear performance.
    // We could use std::random_shuffle here but since we only need the list to be "kinda shuffled" so that it's
    // unlikely that we hit the worst case of the bounding cone algorithm. We can actually just shuffle every nth
    // element and use a fairly bad random number generator algorithm
    for (int i = sorting.size() - 1; i > 0; i -= 5) {
      std::swap(sorting[i], sorting[rand() % i]);
    }

    // Build our bsp tree
    // Reserve enough memory for the bsp as we know how many nodes it will need
    bsp.reserve(nodes.size() * 2);
    build_bsp(sorting.begin(), sorting.end());

    // Make our reverse lookup so we can correct the neighbourhood indices
    std::vector<int> r_sorting(nodes.size() + 1);
    r_sorting[nodes.size()] = nodes.size();
    for (unsigned int i = 0; i < nodes.size(); ++i) {
      r_sorting[sorting[i]] = i;
    }

    // Sort the nodes and correct the neighbourhood graph based on our BSP sorting
    std::vector<Node<Scalar>> sorted_nodes;
    sorted_nodes.reserve(nodes.size());
    for (const auto& i : sorting) {
      sorted_nodes.push_back(nodes[i]);
      for (int& n : sorted_nodes.back().neighbours) {
        n = r_sorting[n];
      }
    }

    nodes = std::move(sorted_nodes);
  }

  /**
   * @brief Lookup which ranges in the Visual Mesh are on screen given the description of the camera lens/sensor and the
   *        orientation of the camera relative to the observation plane.
   *
   * @param Hoc   the homogenous transformation matrix that transforms from camera space to observation plane space
   * @param lens  the lens object describing the type and geometry of the lens that is used
   *
   * @return gives pairs of start/end ranges that are the points which are on the screen
   */
  std::vector<std::pair<int, int>> lookup(const mat4<Scalar>& Hoc, const Lens<Scalar>& lens) const {

    // Our FOV is an easy check to exclude things outside our view
    // Multiply by 0.5 to get the cone angle
    const Scalar cos_fov = std::cos(lens.fov * static_cast<Scalar>(0.5));
    const Scalar sin_fov = std::sin(lens.fov * static_cast<Scalar>(0.5));

    // Get the x axis of the camera in world space and the cone equations that describe the edges of the screen
    const mat3<Scalar> Rco(block<3, 3>(transpose(Hoc)));
    const vec3<Scalar>& rXCo = Rco[0];  // Camera x in world space
    const auto edges         = screen_edges(Hoc, lens);

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

      // Check if we are outside the field of view of the lens using an easy check
      // To check if we are inside or outside the cone we need to check how angle between the cones compares
      // outside == dot(cam, axis) < cos(fov + acos(gradient))
      // However given that the thetas don't change and we have gradient naturally from the dot product it's easier
      // to calculate it using the compound angle formula
      const Scalar delta = dot(rXCo, cone.first);
      bool outside       = delta < cos_fov * cone.second[0] - sin_fov * cone.second[1];
      bool inside        = delta > cos_fov * cone.second[0] + sin_fov * cone.second[1];

      // The FOV can either entirely exclude our points, or split based on intersection. If it can't do either of these
      // (entirely inside) we need to use the screen edges to do a proper check.
      if (!outside && inside) { std::tie(inside, outside) = check_on_screen(Rco, cone, lens, edges); }

      if (inside) {
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
        // If we found an outside point we have finished building our range
        if (building) {
          ranges.emplace_back(std::make_pair(range_start, range_end));
          building = false;
        }
      }
      // We have reached the end of a tree, from here we need to check each point on screen individually
      else if (elem.children[0] < 0) {
        for (int i = elem.range.first; i < elem.range.second; ++i) {
          // Check if the pixel is on the screen
          auto px              = visualmesh::project(multiply(Rco, nodes[i].ray), lens);
          const bool on_screen = dot(rXCo, nodes[i].ray) > cos_fov && 0 <= px[0] && px[0] + 1 <= lens.dimensions[0]
                                 && 0 <= px[1] && px[1] + 1 <= lens.dimensions[1];

          if (on_screen && building) {
            // Extend the end
            range_end = i + 1;
          }
          else if (!on_screen && building) {
            // Add the range we just closed
            ranges.emplace_back(std::make_pair(range_start, range_end));
            building = false;
          }
          else if (on_screen && !building) {
            // Start a new range
            range_start = i;
            range_end   = i + 1;
            building    = true;
          }
        }
      }

      else {
        // Add the children of this to the search in order 1,0 so we pop 0 first (contiguous indices)
        stack.push_back(elem.children[1]);
        stack.push_back(elem.children[0]);
      }
    }
    // If we finished while building add the last point
    if (building) { ranges.emplace_back(std::make_pair(range_start, range_end)); }

    return ranges;
  }
  /// The height that this mesh is designed to run at
  Scalar h;
  /// The maximum distance this mesh is setup for
  Scalar max_distance;
  /// The lookup table for this mesh
  std::vector<Node<Scalar>> nodes;

private:
  /// The binary search tree that is used for looking up which points are on screen in the mesh
  std::vector<BSP> bsp;
};

}  // namespace visualmesh

#endif  // VISUALMESH_MESH_HPP
