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

#ifndef VISUALMESH_UTILITY_PROJECTION_HPP
#define VISUALMESH_UTILITY_PROJECTION_HPP

#include "math.hpp"
#include "mesh/lens.hpp"

namespace visualmesh {

namespace equidistant {
    template <typename Scalar>
    inline Scalar r(const Scalar& theta, const Scalar& f) {
        return f * theta;
    }

    template <typename Scalar>
    inline Scalar theta(const Scalar& r, const Scalar& f) {
        return r / f;
    }
}  // namespace equidistant

namespace equisolid {
    template <typename Scalar>
    inline Scalar r(const Scalar& theta, const Scalar& f) {
        return Scalar(2.0) * f * std::sin(theta * Scalar(0.5));
    }

    template <typename Scalar>
    inline Scalar theta(const Scalar& r, const Scalar& f) {
        return Scalar(2.0) * std::asin(r / (Scalar(2.0) * f));
    }
}  // namespace equisolid

namespace rectilinear {
    template <typename Scalar>
    inline Scalar r(const Scalar& theta, const Scalar& f) {
        return f * std::tan(theta);
    }

    template <typename Scalar>
    inline Scalar theta(const Scalar& r, const Scalar& f) {
        return std::atan(r / f);
    }
}  // namespace rectilinear

/**
 * @brief Calculates polynomial coefficients that approximate the inverse distortion
 *
 * @details
 *  These coefficients are based on math from the paper
 *  An Exact Formula for Calculating Inverse Radial Lens Distortions
 *  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934233/pdf/sensors-16-00807.pdf
 *  These terms have been stripped back to only include k1 and k2 and only uses the first 4 terms
 *  In general for most cases this provides an accuracy of around 0.2 pixels which is sufficient.
 *  If more accuracy is required in the future or more input parameters are used they can be adjusted here.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param k the forward coefficients that go from a distorted image to an undistorted one
 *
 * @return the inverse coefficients that go from an undistorted image to a distorted one
 */
template <typename Scalar>
inline vec4<Scalar> inverse_coefficients(const vec2<Scalar>& k) {
    return vec4<Scalar>{{
      -k[0],
      Scalar(3.0) * (k[0] * k[0]) - k[1],
      Scalar(-12.0) * (k[0] * k[0]) * k[0] + Scalar(8.0) * k[0] * k[1],
      Scalar(55.0) * (k[0] * k[0]) * (k[0] * k[0]) - Scalar(55.0) * (k[0] * k[0]) * k[1] + Scalar(5.0) * (k[1] * k[1]),
    }};
}

/**
 * @brief Undistorts radial distortion using the provided distortion coefficients
 *
 * @details
 *  Given a radial distance from the optical centre, this applies a polynomial distortion model in order to approximate
 *  an ideal lens. After the radial distance has gone through this function it will approximate the equivilant radius
 *  in an ideal lens projection (depending on which base lens projection you are using).
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param r the radial distance from the optical centre
 * @param k the distortion coefficients to use for undistortion
 *
 * @return the undistorted radial distance from the optical centre
 */
template <typename Scalar>
inline Scalar distort(const Scalar& r, const vec2<Scalar>& k) {
    // Uses the math from the paper
    // An Exact Formula for Calculating Inverse Radial Lens Distortions
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934233/pdf/sensors-16-00807.pdf
    // These terms have been stripped back to only include k1 and k2 and only uses the first 4 terms
    // if more are needed in the future go and get them from the original paper
    // TODO if performance ever becomes an issue, this can be precomputed for the same k values
    const vec4<Scalar> ik = inverse_coefficients(k);
    return r
           * (1.0                                                  //
              + ik[0] * (r * r)                                    //
              + ik[1] * ((r * r) * (r * r))                        //
              + ik[2] * ((r * r) * (r * r)) * (r * r)              //
              + ik[3] * ((r * r) * (r * r)) * ((r * r) * (r * r))  //
           );
}

/**
 * @brief Undistorts radial distortion using the provided distortion coefficients
 *
 * @details
 *  Given a radial distance from the optical centre, this applies a polynomial distortion model in order to approximate
 *  an ideal lens. After the radial distance has gone through this function it will approximate the equivilant radius
 *  in an ideal lens projection (depending on which base lens projection you are using).
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 *
 * @param r the radial distance from the optical centre
 * @param k the distortion coefficients to use for undistortion
 *
 * @return the undistorted radial distance from the optical centre
 */
template <typename Scalar>
inline Scalar undistort(const Scalar& r, const vec2<Scalar>& k) {
    // These parenthesis are important as they allow the compiler to optimise further
    // Since floating point multiplication is not commutative r * r * r * r != (r * r) * (r * r)
    // This means that the first needs 3 multiplication operations while the second needs only 2
    return r * (1.0 + k[0] * (r * r) + k[1] * ((r * r) * (r * r)));
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
vec2<Scalar> project(const vec3<Scalar>& ray, const Lens<Scalar>& lens) {

    // Perform the projection math
    const Scalar& f         = lens.focal_length;
    const Scalar theta      = std::acos(ray[0]);
    const Scalar rsin_theta = Scalar(1) / std::sqrt(Scalar(1) - ray[0] * ray[0]);
    Scalar r_u;
    switch (lens.projection) {
        case RECTILINEAR: r_u = rectilinear::r(theta, f); break;
        case EQUISOLID: r_u = equisolid::r(theta, f); break;
        case EQUIDISTANT: r_u = equidistant::r(theta, f); break;
        default: throw std::runtime_error("Cannot project: Unknown lens type"); break;
    }
    const Scalar r_d = distort(r_u, lens.k);

    // Work out our pixel coordinates as a 0 centred image with x to the left and y up (screen space)
    // Sometimes x is greater than one due to floating point error, this almost certainly means that we are facing
    // directly forward
    vec2<Scalar> screen = ray[0] >= 1 ? vec2<Scalar>{{Scalar(0.0), Scalar(0.0)}}
                                      : vec2<Scalar>{{r_d * ray[1] * rsin_theta, r_d * ray[2] * rsin_theta}};

    // Apply our offset to move into image space (0 at top left, x to the right, y down)
    // Then apply the offset to the centre of our lens
    return subtract(subtract(multiply(cast<Scalar>(lens.dimensions), Scalar(0.5)), screen), lens.centre);
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
 * @param px    the pixel coordinate to unproject
 * @param lens  the paramters that describe the lens that we are using to unproject
 *
 * @return the unit vector that this pixel represents in camera space
 */
template <typename Scalar>
vec3<Scalar> unproject(const vec2<Scalar>& px, const Lens<Scalar>& lens) {

    // Transform to centre of the screen:
    vec2<Scalar> screen = subtract(multiply(cast<Scalar>(lens.dimensions), Scalar(0.5)), add(px, lens.centre));

    // Perform the unprojection math
    const Scalar& f  = lens.focal_length;
    const Scalar r_d = norm(screen);
    if (r_d == 0) return {{1.0, 0.0, 0.0}};
    const Scalar r_u = undistort(r_d, lens.k);
    Scalar theta;
    switch (lens.projection) {
        case RECTILINEAR: theta = rectilinear::theta(r_u, f); break;
        case EQUISOLID: theta = equisolid::theta(r_u, f); break;
        case EQUIDISTANT: theta = equidistant::theta(r_u, f); break;
        default: throw std::runtime_error("Cannot project: Unknown lens type"); break;
    }
    const Scalar sin_theta = std::sin(theta);

    return vec3<Scalar>{{std::cos(theta), sin_theta * screen[0] / r_d, sin_theta * screen[1] / r_d}};
}

}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_PROJECTION_HPP
