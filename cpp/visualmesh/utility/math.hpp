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
inline std::array<OutScalar, L> cast(const std::array<InScalar, L>& a, const std::index_sequence<I...>& /*unused*/) {
    return {{static_cast<OutScalar>(a[I])...}};
}

template <typename OutScalar, typename InScalar, std::size_t L>
inline std::array<OutScalar, L> cast(const std::array<InScalar, L>& a) {
    return cast<OutScalar>(a, std::make_index_sequence<L>());
}

/**
 * Get the head of a vector
 */
template <std::size_t S, typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, S> head(const std::array<Scalar, L>& a, const std::index_sequence<I...>& /*unused*/) {
    return {{a[I]...}};
}

template <std::size_t S, typename Scalar, std::size_t L>
inline std::enable_if_t<(S < L), std::array<Scalar, S>> head(const std::array<Scalar, L>& a) {
    return head<S>(a, std::make_index_sequence<S>());
}

/**
 * Vector Scalar subtraction
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> subtract(const std::array<Scalar, L>& a,
                                      const Scalar& b,
                                      const std::index_sequence<I...>& /*unused*/) {
    return {{(a[I] - b)...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> subtract(const std::array<Scalar, L>& a, const Scalar& b) {
    return subtract(a, b, std::make_index_sequence<L>());
}

/**
 * Vector Vector subtraction
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> subtract(const std::array<Scalar, L>& a,
                                      const std::array<Scalar, L>& b,
                                      const std::index_sequence<I...>& /*unused*/) {
    return {{(a[I] - b[I])...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> subtract(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
    return subtract(a, b, std::make_index_sequence<L>());
}

/**
 * Vector Scalar addition
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> add(const std::array<Scalar, L>& a,
                                 const Scalar& b,
                                 const std::index_sequence<I...>& /*unused*/) {
    return {{(a[I] + b)...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> add(const std::array<Scalar, L>& a, const Scalar& b) {
    return add(a, b, std::make_index_sequence<L>());
}

/**
 * Vector Vector addition
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> add(const std::array<Scalar, L>& a,
                                 const std::array<Scalar, L>& b,
                                 const std::index_sequence<I...>& /*unused*/) {
    return {{(a[I] + b[I])...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> add(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
    return add(a, b, std::make_index_sequence<L>());
}

/**
 * Vector Scalar multiplication
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> multiply(const std::array<Scalar, L>& a,
                                      const Scalar& s,
                                      const std::index_sequence<I...>& /*unused*/) {
    return {{(a[I] * s)...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> multiply(const std::array<Scalar, L>& a, const Scalar& s) {
    return multiply(a, s, std::make_index_sequence<L>());
}

/**
 * Vector Vector multiplication
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<Scalar, L> multiply(const std::array<Scalar, L>& a,
                                      const std::array<Scalar, L>& b,
                                      const std::index_sequence<I...>& /*unused*/) {
    return {{(a[I] * b[I])...}};
}

template <typename Scalar, std::size_t L>
inline std::array<Scalar, L> multiply(const std::array<Scalar, L>& a, const std::array<Scalar, L>& b) {
    return multiply(a, b, std::make_index_sequence<L>());
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
    const Scalar len = Scalar(1.0) / norm(a);
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
 * Matrix block (head of matrix)
 */
template <std::size_t S, std::size_t T, typename Scalar, std::size_t L, std::size_t M, std::size_t... I>
inline std::array<std::array<Scalar, T>, S> block(const std::array<std::array<Scalar, M>, L>& a,
                                                  const std::index_sequence<I...>& /*unused*/) {
    return {{head<T>(a[I])...}};
}

template <std::size_t S, std::size_t T, typename Scalar, std::size_t L, std::size_t... I>
inline std::array<std::array<Scalar, T>, S> block(const std::array<std::array<Scalar, T>, L>& a,
                                                  const std::index_sequence<I...>& /*unused*/) {
    return {{a[I]...}};
}

template <std::size_t S, std::size_t T, typename Scalar, std::size_t L, std::size_t M>
inline std::enable_if_t<(S <= L && T < M) || (S < L && T <= M), std::array<std::array<Scalar, T>, S>> block(
  const std::array<std::array<Scalar, M>, L>& a) {
    return block<S, T>(a, std::make_index_sequence<S>());
}

/**
 * Matrix transpose
 */
template <std::size_t X, typename Scalar, std::size_t L, std::size_t M, std::size_t... Y>
inline std::array<Scalar, M> transpose_vector(const std::array<std::array<Scalar, L>, M>& mat,
                                              const std::index_sequence<Y...>& /*unused*/) {
    return {{mat[Y][X]...}};
}

template <typename Scalar, std::size_t L, std::size_t M, std::size_t... X>
inline std::array<std::array<Scalar, L>, L> transpose(const std::array<std::array<Scalar, L>, M>& mat,
                                                      const std::index_sequence<X...>& /*unused*/) {
    return {{transpose_vector<X>(mat, std::make_index_sequence<M>())...}};
}

template <typename Scalar, std::size_t L, std::size_t M>
inline std::array<std::array<Scalar, M>, L> transpose(const std::array<std::array<Scalar, L>, M>& mat) {
    return transpose(mat, std::make_index_sequence<L>());
}


/**
 * Matrix Vector multiplication
 * (the vector is treated as a column vector for this function)
 */
template <typename Scalar>
inline vec3<Scalar> multiply(const mat3<Scalar>& a, const vec3<Scalar>& b) {
    return vec3<Scalar>{{dot(a[0], b), dot(a[1], b), dot(a[2], b)}};
}


/**
 * Matrix Scalar multiplication
 */
template <typename Scalar, std::size_t L, std::size_t... I>
inline std::array<std::array<Scalar, L>, L> multiply(const std::array<std::array<Scalar, L>, L>& a,
                                                     const Scalar& s,
                                                     const std::index_sequence<I...>& /*unused*/) {
    return {{multiply(a[I], s)...}};
}

template <typename Scalar, std::size_t L>
inline std::array<std::array<Scalar, L>, L> multiply(const std::array<std::array<Scalar, L>, L>& a, const Scalar& s) {
    return multiply(a, s, std::make_index_sequence<L>());
}


/**
 * Matrix inverse
 */
template <typename Scalar>
inline mat3<Scalar> invert(const mat3<Scalar>& m) {

    // computes the inverse of a matrix m
    const Scalar det = (m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -  //
                        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +  //
                        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));  //

    // Matrix is not invertible
    if (det == 0) {
        const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
        return {vec3<Scalar>{nan, nan, nan}, vec3<Scalar>{nan, nan, nan}, vec3<Scalar>{nan, nan, nan}};
    }

    const Scalar idet = Scalar(1) / det;

    return {vec3<Scalar>{(m[1][1] * m[2][2] - m[2][1] * m[1][2]) * idet,
                         (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * idet,
                         (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * idet},
            vec3<Scalar>{(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * idet,
                         (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * idet,
                         (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * idet},
            vec3<Scalar>{(m[1][0] * m[2][1] - m[2][0] * m[1][1]) * idet,
                         (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * idet,
                         (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * idet}};
}

template <typename Scalar>
inline mat4<Scalar> invert_affine(const mat4<Scalar>& Hab) {
    // Invert the rotation component.
    // R^{-1} = R^{T}
    const mat3<Scalar> Rba = transpose(block<3, 3>(Hab));

    // Invert the translation component
    // T^{-1} = R^{T} * -T
    const vec3<Scalar> T = multiply(Rba, vec3<Scalar>{-Hab[0][3], -Hab[1][3], -Hab[2][3]});

    // Construct the inverted matrix
    return {{vec4<Scalar>{Rba[0][0], Rba[0][1], Rba[0][2], T[0]},
             vec4<Scalar>{Rba[1][0], Rba[1][1], Rba[1][2], T[1]},
             vec4<Scalar>{Rba[2][0], Rba[2][1], Rba[2][2], T[2]},
             vec4<Scalar>{Scalar(0), Scalar(0), Scalar(0), Scalar(1)}}};
}
}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_MATH_HPP
