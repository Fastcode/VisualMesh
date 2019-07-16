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

#ifndef VISUALMESH_UTILITY_MATH_HPP
#define VISUALMESH_UTILITY_MATH_HPP

#include <array>
#include <cmath>

namespace visualmesh {

// Typedef some value types we commonly use
template <typename Scalar>
using vec2 = std::array<Scalar, 2>;
template <typename Scalar>
using vec3 = std::array<Scalar, 3>;
template <typename Scalar>
using vec4 = std::array<Scalar, 4>;
template <typename Scalar>
using mat3 = std::array<vec3<Scalar>, 3>;
template <typename Scalar>
using mat4 = std::array<vec4<Scalar>, 4>;

/**
 * Vector casting
 */
template <typename OutScalar, typename InScalar, std::size_t L, std::size_t... I>
inline std::array<OutScalar, L> cast(const std::array<InScalar, L>& a, const std::index_sequence<I...>&) {
  return {{static_cast<OutScalar>(a[I])...}};
}

template <typename OutScalar, typename InScalar, std::size_t L>
inline std::array<OutScalar, L> cast(const std::array<InScalar, L>& a) {
  return cast<OutScalar>(a, std::make_index_sequence<L>());
}

/**
 * Vector subtraction
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> subtract(const std::array<Scalar, L>& a,
                                      const std::array<Scalar, L>& b,
                                      const std::index_sequence<I...>&) {
  return {{(a[I] - b[I])...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> subtract(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
  return subtract(a, b, std::make_index_sequence<L>());
}

/**
 * Vector addition
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> add(const std::array<Scalar, L>& a,
                                 const std::array<Scalar, L>& b,
                                 const std::index_sequence<I...>&) {
  return {{(a[I] + b[I])...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> add(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
  return add(a, b, std::make_index_sequence<L>());
}

/**
 * Vector multiply by scalar
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> multiply(const std::array<Scalar, L>& a,
                                      const Scalar& s,
                                      const std::index_sequence<I...>&) {
  return {{(a[I] * s)...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> multiply(const std::array<Scalar, L>& a, const Scalar& s) {
  return multiply(a, s, std::make_index_sequence<L>());
}

/**
 * Dot product
 */
template <typename T, std::size_t I>
struct Dot;

template <typename Scalar, std::size_t L>
struct Dot<std::array<Scalar, L>, 0> {
  static inline Scalar dot(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
    return a[0] * b[0];
  }
};

template <typename Scalar, std::size_t L, std::size_t I>
struct Dot<std::array<Scalar, L>, I> {
  static inline Scalar dot(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
    return a[I] * b[I] + Dot<std::array<Scalar, L>, I - 1>::dot(a, b);
  }
};

template <typename Scalar, std::size_t L>
inline Scalar dot(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
  return Dot<std::array<Scalar, L>, L - 1>::dot(a, b);
}

/**
 * Vector norm
 */
template <typename Scalar, std::size_t L>
inline Scalar norm(const std::array<Scalar, L>& a) {
  return std::sqrt(dot(a, a));
}

/**
 * Vector normalise
 */
template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> normalise(const std::array<Scalar, L>& a) {
  const Scalar len = static_cast<Scalar>(1.0) / norm(a);
  return multiply(a, len);
}

/**
 * Vector cross product
 */
template <typename Scalar>
inline vec3<Scalar> cross(const vec3<Scalar>& a, const vec3<Scalar>& b) {
  return {{
    a[1] * b[2] - a[2] * b[1],  // x
    a[2] * b[0] - a[0] * b[2],  // y
    a[0] * b[1] - a[1] * b[0]   // z
  }};
}

/**
 * Matrix transpose
 */
template <std::size_t X, typename Scalar, std::size_t L, std::size_t M, std::size_t... Y>
inline std::array<Scalar, M> transpose_vector(const std::array<std::array<Scalar, L>, M>& mat,
                                              const std::index_sequence<Y...>&) {
  return {{mat[Y][X]...}};
}

template <typename Scalar, std::size_t L, std::size_t M, std::size_t... X>
inline std::array<std::array<Scalar, L>, L> transpose(const std::array<std::array<Scalar, L>, M>& mat,
                                                      const std::index_sequence<X...>&) {
  return {{transpose_vector<X>(mat, std::make_index_sequence<M>())...}};
}

template <typename Scalar, std::size_t L, std::size_t M>
inline std::array<std::array<Scalar, M>, L> transpose(const std::array<std::array<Scalar, L>, M>& mat) {
  return transpose(mat, std::make_index_sequence<L>());
}

}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_MATH_HPP
