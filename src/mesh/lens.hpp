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

#ifndef VISUALMESH_LENS_HPP
#define VISUALMESH_LENS_HPP

#include <array>

namespace visualmesh {

/**
 * @brief An enum that describes the lens projection type
 */
enum LensProjection { RECTILINEAR, EQUISOLID, EQUIDISTANT };

/**
 * @brief A description of a lens that will be used for projection of unit vectors into camera space.
 *
 * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
 */
template <typename Scalar>
struct Lens {
    /// The dimensions of the image
    std::array<int, 2> dimensions;
    /// The projection that this image is using
    LensProjection projection;
    /// The focal length of the camera, normalised to the image width
    Scalar focal_length;
    /// The offset required to move the centre of the lens to the centre of the image
    std::array<Scalar, 2> centre;
    /// The distortion parameters for the camera model
    std::array<Scalar, 2> k;
    /// The field of view of the camera measured in radians
    /// This field of view is used to cut off sections of the image that the lens does not project for (think the black
    /// sections on a fisheye camera). If there are no black sections on the image this should be set to the diagonal
    /// field of view of the lens.
    Scalar fov;
};

}  // namespace visualmesh

#endif  // VISUALMESH_LENS_HPP
