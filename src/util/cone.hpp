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

#ifndef VISUALMESH_UTILITY_CONE_HPP
#define VISUALMESH_UTILITY_CONE_HPP

#include "math.hpp"

namespace visualmesh {

template <typename Scalar>
inline std::pair<vec3<Scalar>, Scalar> cone_from_points() {
  return std::make_pair(vec3<Scalar>{0, 0, 0}, 1);
}

template <typename Scalar>
inline std::pair<vec3<Scalar>, Scalar> cone_from_points(const vec3<Scalar>& p1) {
  return std::make_pair(p1, 1);
}

template <typename Scalar>
inline std::pair<vec3<Scalar>, Scalar> cone_from_points(const vec3<Scalar>& p1, const vec3<Scalar>& p2) {
  //  Get the axis and gradient by averaging the unit vectors and dotting with an edge point
  vec3<Scalar> axis = normalise(add(p1, p2));
  Scalar cos_theta  = dot(axis, p1);
  return std::make_pair(axis, cos_theta);
}

template <typename Scalar>
inline std::pair<vec3<Scalar>, Scalar> cone_from_points(const vec3<Scalar>& p1,
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

  return std::make_pair(axis, cos_theta);
}

}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_CONE_HPP
