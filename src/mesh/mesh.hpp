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

#include <array>
#include <numeric>
#include <utility>
#include <vector>
#include "generator/hexapizza.hpp"
#include "lens.hpp"
#include "node.hpp"
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

  static std::pair<vec3<Scalar>, vec2<Scalar>> cone_from_points() {
    return std::make_pair(vec3<Scalar>{0, 0, 0}, vec2<Scalar>{1, 0});
  }

  static std::pair<vec3<Scalar>, vec2<Scalar>> cone_from_points(const vec3<Scalar>& p1) {
    return std::make_pair(p1, vec2<Scalar>{1, 0});
  }

  static std::pair<vec3<Scalar>, vec2<Scalar>> cone_from_points(const vec3<Scalar>& p1, const vec3<Scalar>& p2) {
    //  Get the axis and gradient by averaging the unit vectors and dotting with an edge point
    vec3<Scalar> axis = normalise(add(p1, p2));
    Scalar cos_theta  = dot(axis, p1);
    return std::make_pair(axis, vec2<Scalar>{cos_theta, std::sqrt(1 - cos_theta * cos_theta)});
  }

  static std::pair<vec3<Scalar>, vec2<Scalar>> cone_from_points(const vec3<Scalar>& p1,
                                                                const vec3<Scalar>& p2,
                                                                const vec3<Scalar>& p3) {
    // Put the rays into a matrix so we can solve it
    mat3<Scalar> mat{{p1, p2, p3}};
    mat3<Scalar> imat = invert(mat);

    // Transpose and multiply by 1 1 1 to get the axis
    vec3<Scalar> axis = normalise(vec3<Scalar>{
      dot(imat[0], vec3<Scalar>{1, 1, 1}),
      dot(imat[1], vec3<Scalar>{1, 1, 1}),
      dot(imat[2], vec3<Scalar>{1, 1, 1}),
    });

    Scalar cos_theta = dot(axis, p1);

    return std::make_pair(axis, vec2<Scalar>{cos_theta, std::sqrt(1 - cos_theta * cos_theta)});
  }

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
    std::pair<vec3<Scalar>, vec2<Scalar>> cone = cone_from_points();
    for (auto i = start; i < end; ++i) {
      if (dot(cone.first, head<3>(nodes[*i].ray)) < cone.second[0]) {
        cone = cone_from_points(head<3>(nodes[*i].ray));
        for (auto j = start; j < i; ++j) {
          if (dot(cone.first, head<3>(nodes[*j].ray)) < cone.second[0]) {
            cone = cone_from_points(head<3>(nodes[*i].ray), head<3>(nodes[*j].ray));
            for (auto k = start; k < j; ++k) {
              if (dot(cone.first, head<3>(nodes[*k].ray)) < cone.second[0]) {
                cone = cone_from_points(head<3>(nodes[*i].ray), head<3>(nodes[*j].ray), head<3>(nodes[*k].ray));
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
        {{
          build_bsp(start, mid, min_points, offset),
          build_bsp(mid, end, min_points, offset + std::distance(start, mid)),
        }},
        cone,
      });
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
      head<3>(visualmesh::unproject(vec2<Scalar>{0, 0}, lens)),                          // rTCc
      head<3>(visualmesh::unproject(vec2<Scalar>{dimensions[0], 0}, lens)),              // rUCc
      head<3>(visualmesh::unproject(vec2<Scalar>{dimensions[0], dimensions[1]}, lens)),  // rVCc
      head<3>(visualmesh::unproject(vec2<Scalar>{0, dimensions[1]}, lens)),              // rWCc
    }};

    // Rotate these vectors into world space
    const std::array<vec3<Scalar>, 4> rNCo = {{
      {{dot(rNCc[0], Roc[0]), dot(rNCc[0], Roc[1]), dot(rNCc[0], Roc[2])}},  // rTCo
      {{dot(rNCc[1], Roc[0]), dot(rNCc[1], Roc[1]), dot(rNCc[1], Roc[2])}},  // rUCo
      {{dot(rNCc[2], Roc[0]), dot(rNCc[2], Roc[1]), dot(rNCc[2], Roc[2])}},  // rVCo
      {{dot(rNCc[3], Roc[0]), dot(rNCc[3], Roc[1]), dot(rNCc[3], Roc[2])}},  // rWCo
    }};

    switch (lens.projection) {
      case LensProjection::RECTILINEAR: {
        // For the case of a plane, we have a cone with a 90 degrees which means cos(theta) = 0 and sin(theta) = 1
        return std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>{{
          std::make_pair(normalise(cross(rNCo[0], rNCo[1])), vec2<Scalar>{0, 1}),
          std::make_pair(normalise(cross(rNCo[1], rNCo[2])), vec2<Scalar>{0, 1}),
          std::make_pair(normalise(cross(rNCo[2], rNCo[3])), vec2<Scalar>{0, 1}),
          std::make_pair(normalise(cross(rNCo[3], rNCo[0])), vec2<Scalar>{0, 1}),
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
          head<3>(visualmesh::unproject(vec2<Scalar>{centre[0], 0}, lens)),              // rDCc
          head<3>(visualmesh::unproject(vec2<Scalar>{dimensions[0], centre[1]}, lens)),  // rECc
          head<3>(visualmesh::unproject(vec2<Scalar>{centre[0], dimensions[1]}, lens)),  // rFCc
          head<3>(visualmesh::unproject(vec2<Scalar>{0, centre[1]}, lens)),              // rGCc
        }};

        // Rotate these vectors into world space
        const std::array<vec3<Scalar>, 4> rECo = {{
          {{dot(rECc[0], Roc[0]), dot(rECc[0], Roc[1]), dot(rECc[0], Roc[2])}},  // rTCo
          {{dot(rECc[1], Roc[0]), dot(rECc[1], Roc[1]), dot(rECc[1], Roc[2])}},  // rUCo
          {{dot(rECc[2], Roc[0]), dot(rECc[2], Roc[1]), dot(rECc[2], Roc[2])}},  // rVCo
          {{dot(rECc[3], Roc[0]), dot(rECc[3], Roc[1]), dot(rECc[3], Roc[2])}},  // rWCo
        }};

        // Calculate cones from each of the four screen edges
        return std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>{{
          cone_from_points(rNCo[0], rECo[0], rNCo[1]),
          cone_from_points(rNCo[1], rECo[1], rNCo[2]),
          cone_from_points(rNCo[2], rECo[2], rNCo[3]),
          cone_from_points(rNCo[3], rECo[3], rNCo[0]),
        }};
      }
    }
  }

  /**
   * Check if a point is on the screen, given a description of the edges of the screen as cones, and the axis
   */
  static inline std::pair<bool, bool> check_on_screen(
    const mat4<Scalar>& Hoc,
    const vec3<Scalar>& axis,
    const Lens<Scalar>& lens,
    const std::array<std::pair<vec3<Scalar>, vec2<Scalar>>, 4>& edges) {

    // TODO check if the axis projects to the screen

    // TODO check against FOV

    // TODO check aginst screen edges

    // Firstly check if the cone axis is on the screen
    vec2<Scalar> px = ::visualmesh::project(vec4<Scalar>{axis[0], axis[1], axis[2], 0}, lens);
    bool axis_on_screen =
      0 <= px[0] && px[0] + 1 <= lens.dimensions[0] && 0 <= px[1] && px[1] + 1 <= lens.dimensions[1];


    // Find the angular equation

    // Project through the lens equation

    // Polar to cartesian

    // Check for intersection with edges of the screen


    return std::make_pair(false, false);
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

    // A random shuffle here before we build the BSP means that partitions will be in general randomish
    // This is required for the linear performance of the bounding cone algorithm to work
    std::random_shuffle(sorting.begin(), sorting.end());
    t.measure("Shuffle");

    // Build our bsp tree
    // Reserve enough memory for the bsp as we know how many nodes it will need
    bsp.reserve(nodes.size() * 2);
    build_bsp(sorting.begin(), sorting.end());
    t.measure("Built BSP");

    // Sort the nodes and correct the neighbourhood graph based on our BSP sorting
    std::vector<Node<Scalar>> sorted_nodes;
    sorted_nodes.reserve(nodes.size());
    for (const auto& i : sorting) {
      sorted_nodes.push_back(nodes[i]);
      for (int& n : sorted_nodes.back().neighbours) {
        n = sorting[n];
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
    const vec3<Scalar> cam_x{Hoc[0][0], Hoc[1][0], Hoc[2][0]};
    const auto edges = screen_edges(Hoc, lens);

    // Go through our BSP tree to work out which segments of the mesh are on screen
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
      // Inside == cam * axis >  cos(theta_1)cos(theta_2) - sin(theta_1)sin(theta_2)
      bool outside = delta < cos_fov * cone.second[0] - sin_fov * cone.second[1];
      bool inside  = false;

      // If we are not ruled as outside by the field of view we might be on the screen
      if (!outside) {
        // TODO construct
      }

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
      else {
        // Add the children of this to the search in order 1,0 so we pop 0 first (contiguous indices)
        stack.push_back(elem.children[1]);
        stack.push_back(elem.children[0]);
      }
    }

    std::cout << "BYE BYE" << std::endl;

    exit(1);
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
