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

#ifndef VISUALMESH_UTILITY_PROJECTION_HPP
#define VISUALMESH_UTILITY_PROJECTION_HPP

#include "math.hpp"
#include "mesh/lens.hpp"

namespace visualmesh {

/**
 * @brief Projects a unit vector into a pixel coordinate using an equidistant fisheye lens model
 *
 * @details
 *  This function expects a unit vector in camera space. For this camera space is defined as a coordinate system with
 *  the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up in the
 *  resulting image. The pixel coordinate that results will have (0,0) at the top left of the image, with x to the right
 *  and y down.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the unit vector in camera space to project
 * @param lens  the paramters that describe the lens that we are using to project
 *
 * @return a pixel coordinate that this vector projects into
 */
template <typename Scalar>
inline vec2<Scalar> project_equidistant(const vec3<Scalar>& p, const Lens<Scalar>& lens) {
  // Calculate some intermediates
  Scalar theta      = std::acos(p[0]);
  Scalar r          = lens.focal_length * theta;
  Scalar rsin_theta = static_cast<Scalar>(1) / std::sqrt(static_cast<Scalar>(1) - p[0] * p[0]);

  // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
  // Sometimes x is greater than one due to floating point error, this almost certainly means that we are facing
  // directly forward
  vec2<Scalar> screen = p[0] >= 1 ? vec2<Scalar>{{static_cast<Scalar>(0.0), static_cast<Scalar>(0.0)}}
                                  : vec2<Scalar>{{r * p[1] * rsin_theta, r * p[2] * rsin_theta}};

  // Apply our offset to move into image space (0 at top left, x to the right, y down)
  // Then apply the offset to the centre of our lens
  return subtract(subtract(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), screen), lens.centre);
}

/**
 * @brief Unprojects a pixel coordinate into a unit vector using an equidistant fisheye lens model
 *
 * @details
 *  This function expects a pixel coordinate having (0,0) at the top left of the image, with x to the right and y down.
 *  It will then convert this into a unit vector in camera space. For this camera space is defined as a coordinate
 *  system with the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the pixel coordinate to unproject
 * @param lens  the paramters that describe the lens that we are using to unproject
 *
 * @return the unit vector that this pixel represents in camera space
 */
template <typename Scalar>
inline vec3<Scalar> unproject_equidistant(const vec2<Scalar>& point, const Lens<Scalar>& lens) {
  vec2<Scalar> screen =
    subtract(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), add(point, lens.centre));
  Scalar r     = norm(screen);
  Scalar theta = r / lens.focal_length;
  return normalise(vec3<Scalar>{std::cos(theta), std::sin(theta) * screen[0] / r, std::sin(theta) * screen[1] / r});
}

/**
 * @brief Projects a unit vector into a pixel coordinate using an equisolid fisheye lens model
 *
 * @details
 *  This function expects a unit vector in camera space. For this camera space is defined as a coordinate system with
 *  the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up in the
 *  resulting image. The pixel coordinate that results will have (0,0) at the top left of the image, with x to the right
 *  and y down.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the unit vector to project
 * @param lens  the paramters that describe the lens that we are using to project
 *
 * @return a pixel coordinate that this vector projects into
 */
template <typename Scalar>
inline vec2<Scalar> project_equisolid(const vec3<Scalar>& p, const Lens<Scalar>& lens) {
  // Calculate some intermediates
  Scalar theta      = std::acos(p[0]);
  Scalar r          = static_cast<Scalar>(2.0) * lens.focal_length * std::sin(theta * static_cast<Scalar>(0.5));
  Scalar rsin_theta = static_cast<Scalar>(1) / std::sqrt(static_cast<Scalar>(1) - p[0] * p[0]);

  // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
  // Sometimes x is greater than one due to floating point error, this almost certainly means that we are facing
  // directly forward
  vec2<Scalar> screen = p[0] >= 1 ? vec2<Scalar>{{static_cast<Scalar>(0.0), static_cast<Scalar>(0.0)}}
                                  : vec2<Scalar>{{r * p[1] * rsin_theta, r * p[2] * rsin_theta}};

  // Apply our offset to move into image space (0 at top left, x to the right, y down)
  // Then apply the offset to the centre of our lens
  return subtract(subtract(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), screen), lens.centre);
}

/**
 * @brief Unprojects a pixel coordinate into a unit vector using an equisolid fisheye lens model
 *
 * @details
 *  This function expects a pixel coordinate having (0,0) at the top left of the image, with x to the right and y down.
 *  It will then convert this into a unit vector in camera space. For this camera space is defined as a coordinate
 *  system with the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the pixel coordinate to unproject
 * @param lens  the paramters that describe the lens that we are using to unproject
 *
 * @return the unit vector that this pixel represents in camera space
 */
template <typename Scalar>
inline vec3<Scalar> unproject_equisolid(const vec2<Scalar>& point, const Lens<Scalar>& lens) {
  vec2<Scalar> screen =
    subtract(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), add(point, lens.centre));
  Scalar r     = norm(screen);
  Scalar theta = 2.0 * std::asin(r / (2.0 * lens.focal_length));
  return normalise(vec3<Scalar>{std::cos(theta), std::sin(theta) * screen[0] / r, std::sin(theta) * screen[1] / r});
}

/**
 * @brief Projects a unit vector into a pixel coordinate using an rectilinear (standard) lens model
 *
 * @details
 *  This function expects a unit vector in camera space. For this camera space is defined as a coordinate system with
 *  the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up in the
 *  resulting image. The pixel coordinate that results will have (0,0) at the top left of the image, with x to the right
 *  and y down.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the unit vector to project
 * @param lens  the paramters that describe the lens that we are using to project
 *
 * @return a pixel coordinate that this vector projects into
 */
template <typename Scalar>
inline vec2<Scalar> project_rectilinear(const vec3<Scalar>& p, const Lens<Scalar>& lens) {
  // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
  vec2<Scalar> screen = {{lens.focal_length * p[1] / p[0], lens.focal_length * p[2] / p[0]}};

  // Apply our offset to move into image space (0 at top left, x to the right, y down)
  // Then apply the offset to the centre of our lens
  return subtract(subtract(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), screen), lens.centre);
}

/**
 * @brief Unprojects a pixel coordinate into a unit vector using an rectilinear (standard) lens model
 *
 * @details
 *  This function expects a pixel coordinate having (0,0) at the top left of the image, with x to the right and y down.
 *  It will then convert this into a unit vector in camera space. For this camera space is defined as a coordinate
 *  system with the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the pixel coordinate to unproject
 * @param lens  the paramters that describe the lens that we are using to unproject
 *
 * @return the unit vector that this pixel represents in camera space
 */
template <typename Scalar>
inline vec3<Scalar> unproject_rectilinear(const vec2<Scalar>& point, const Lens<Scalar>& lens) {
  vec2<Scalar> screen =
    subtract(multiply(cast<Scalar>(lens.dimensions), static_cast<Scalar>(0.5)), add(point, lens.centre));
  return normalise(vec3<Scalar>{lens.focal_length, screen[0], screen[1]});
}

/**
 * @brief Projects a unit vector into a pixel coordinate while working out which lens model to use via the lens
 *        parameters.
 *
 * @details
 *  This function expects a unit vector in camera space. For this camera space is defined as a coordinate system with
 *  the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up in the
 *  resulting image. The pixel coordinate that results will have (0,0) at the top left of the image, with x to the right
 *  and y down.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the unit vector to project
 * @param lens  the paramters that describe the lens that we are using to project
 *
 * @return a pixel coordinate that this vector projects into
 */
template <typename Scalar>
inline vec2<Scalar> project(const vec3<Scalar>& p, const Lens<Scalar>& lens) {
  switch (lens.projection) {
    case EQUISOLID: return project_equisolid(p, lens);
    case EQUIDISTANT: return project_equidistant(p, lens);
    case RECTILINEAR: return project_rectilinear(p, lens);
    default: throw std::runtime_error("Unknown projection type");
  }
}

/**
 * @brief Unprojects a pixel coordinate into a unit vector working out which lens model to use via the lens parameters.
 *
 * @details
 *  This function expects a pixel coordinate having (0,0) at the top left of the image, with x to the right and y down.
 *  It will then convert this into a unit vector in camera space. For this camera space is defined as a coordinate
 *  system with the x axis going down the viewing direction of the camera, y is to the left of the image, and z is up.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param p     the pixel coordinate to unproject
 * @param lens  the paramters that describe the lens that we are using to unproject
 *
 * @return the unit vector that this pixel represents in camera space
 */
template <typename Scalar>
inline vec3<Scalar> unproject(const vec2<Scalar>& p, const Lens<Scalar>& lens) {
  switch (lens.projection) {
    case EQUISOLID: return unproject_equisolid(p, lens);
    case EQUIDISTANT: return unproject_equidistant(p, lens);
    case RECTILINEAR: return unproject_rectilinear(p, lens);
    default: throw std::runtime_error("Unknown projection type");
  }
}

}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_PROJECTION_HPP
