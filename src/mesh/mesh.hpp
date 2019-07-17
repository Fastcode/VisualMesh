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
    // The bounds of the range that this BSP element represents
    int start;
    int end;

    std::array<int, 2> children;
    vec3<Scalar> axis;
    Scalar cos_theta;
    Scalar sin_theta;
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
  int build_bsp(Iterator start, Iterator end, int offset = 0) {
    // No points in this partition, this should never happen
    if (std::distance(start, end) == 0) { throw std::runtime_error("We tried to make a tree with no nodes"); }

    // Single point
    if (std::distance(start, end) == 1) {
      int elem = bsp.size();
      bsp.push_back(
        BSP{offset, static_cast<int>(offset + std::distance(start, end)), {{-1, -1}}, head<3>(nodes[*start].ray), 0});
      return elem;
    }
    // We have some points
    else {
      // Calculate our bounding cone for this cluster. We have to do a random sort of our segment here so that the
      // performance of the bounding cone algorithm is expected to be linear
      auto cone = bounding_cone(start, end);

      std::cout << cone.second[0] << std::endl;

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

      // std::cout << "Split: " << (max_phi - min_phi > max_theta - min_theta ? "p" : "t")
      //           << " len: " << std::distance(start, end) << " p1: " << std::distance(start, mid)
      //           << " p2: " << std::distance(mid, end) << std::endl;


      int elem = bsp.size();
      bsp.push_back(BSP{
        offset,
        static_cast<int>(offset + std::distance(start, end)),
        {{
          build_bsp(start, mid, offset),
          build_bsp(mid, end, offset + std::distance(start, mid)),
        }},
        cone.first,
        cone.second[0],
        cone.second[1],
      });
      return elem;
    }
  }

public:
  template <template <typename T> class Generator = generator::HexaPizza, typename Shape>
  Mesh(const Shape& shape, const Scalar& h, const Scalar& k, const Scalar& max_distance)
    : h(h), nodes(Generator<Scalar>::generate(shape, h, k, max_distance)) {

    // To ensure that later we can fix the graph we need to perform our sorting on an index list
    std::vector<int> sorting(nodes.size());
    std::iota(sorting.begin(), sorting.end(), 0);

    // A random shuffle here before we build the BSP means that partitions will be in general randomish
    // This is required for the linear performance of the bounding cone algorithm to work
    std::random_shuffle(sorting.begin(), sorting.end());

    // Build our bsp tree
    // Reserve enough memory for the bsp as we know how many nodes it will need
    Timer t;
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

    nodes = std::move(sorted_nodes);
  }

  std::vector<std::pair<int, int>> lookup(const mat4<Scalar>& Hoc, const Lens<Scalar>& lens) const {

    // Centre of the lens projection so we can look at edges relative to this rather than the centre of the sensor
    vec2<Scalar> proj_centre(add(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), lens.centre));

    // Get the four inner edges measured from the lens axis and then dot with the x axis (take the 0th element)
    // The smallest angle from these is the minimum cone we can 100% see
    vec4<Scalar> inner_options = {
      unproject({proj_centre[0], 0}, lens)[0],
      unproject({proj_centre[0], static_cast<Scalar>(lens.dimensions[1])}, lens)[0],
      unproject({0, proj_centre[1]}, lens)[0],
      unproject({static_cast<Scalar>(lens.dimensions[0]), proj_centre[1]}, lens)[0],
    };

    // The four outer corners that define our "Absolutely out" region
    // If a value is outside of this region, there is no way that any point on it intersects with our view
    vec4<Scalar> outer_options = {
      unproject({0, 0}, lens)[0],
      unproject({static_cast<Scalar>(lens.dimensions[0]), static_cast<Scalar>(lens.dimensions[1])}, lens)[0],
      unproject({0, static_cast<Scalar>(lens.dimensions[1])}, lens)[0],
      unproject({static_cast<Scalar>(lens.dimensions[0]), 0}, lens)[0],
    };

    // Get the inner and outer cone gradients
    Scalar cos_inner = *std::max_element(inner_options.begin(), inner_options.end());
    Scalar cos_outer = *std::min_element(outer_options.begin(), outer_options.end());

    // Our FOV may be a more harsh inner cutoff, or a better outer cutoff
    Scalar cos_fov   = std::cos(lens.fov);
    cos_inner        = cos_inner > cos_fov ? cos_inner : cos_fov;
    cos_outer        = cos_outer > cos_fov ? cos_outer : cos_fov;
    Scalar sin_inner = std::sqrt(1 - cos_inner * cos_inner);
    Scalar sin_outer = std::sqrt(1 - cos_outer * cos_outer);

    // Get the x axis of the camera in world space
    vec3<Scalar> cam_x = head<3>(Hoc[0]);

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
      const auto& elem      = bsp[i];
      const auto& axis      = elem.axis;
      const auto& cos_theta = elem.cos_theta;
      const auto& sin_theta = elem.sin_theta;

      // Dot the camera x axis with the cone
      Scalar delta = dot(cam_x, elem.axis);

      // Do our checks for inside and outside
      // To check if we are inside or outside the cone we need to check how angle between the cones compares
      // Inside == cam * axis > cos(acos(inner) + acos(gradient))
      // However given that the thetas don't change and  we have cos(theta) naturally from the dot productit's easier
      // to calculate it using the compound angle formula
      // Inside == cam * axis >  cos(theta_1)cos(theta_2) - sin(theta_1)sin(theta_2)
      bool inside  = delta > cos_inner * cos_theta - sin_inner * sin_theta;
      bool outside = delta < cos_outer * cos_theta - sin_outer * sin_theta;

      // If we have a single point and we were not ruled out as outside, we need to check if we are on screen directly
      if (!inside && !outside && elem.children[0] == -1) {
        auto px = ::visualmesh::project(nodes[elem.start].ray, lens);
        inside  = 0 <= px[0] && px[0] + 1 <= lens.dimensions[0] && 0 <= px[1] && px[1] + 1 <= lens.dimensions[1];
        outside = !inside;
      }

      std::cout << inside << " " << outside << std::endl;

      if (inside) {
        // If we are building just update our end point
        if (building) {
          range_end = elem.end;  //
        }
        else {
          range_start = elem.start;
          range_end   = elem.end;
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
