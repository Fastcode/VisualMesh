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

const sampler_t bayer_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
const sampler_t interp_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

enum FOURCC {
    GREY    = 0x59455247,
    Y12     = 0x20323159,
    Y16     = 0x20363159,
    GRBG    = 0x47425247,
    RGGB    = 0x42474752,
    GBRG    = 0x47524247,
    BGGR    = 0x52474742,
    GR12    = 0x32315247,
    RG12    = 0x32314752,
    GB12    = 0x32314247,
    BG12    = 0x32314742,
    GR16    = 0x36315247,
    RG16    = 0x36314752,
    GB16    = 0x36314247,
    BG16    = 0x36314742,
    Y411    = 0x31313459,
    UYVY    = 0x59565955,
    YUYV    = 0x56595559,
    YM24    = 0x34324d59,
    RGB3    = 0x33424752,
    RGBA    = 0x41424752,
    BGR3    = 0x33524742,
    BGRA    = 0x41524742,
    JPEG    = 0x4745504a,
    UNKNOWN = 0
};

/**
 * @brief Given an image, fetches the r component for use with debayering
 *
 * @param raw_image the raw image that the sampler will be accessing from
 * @param sampler   the sampler that will be used for accessing memory
 * @param coord     the floating point coordinates to access
 *
 * @return returns the pixel at the given coordinates
 */
float fetch(read_only image2d_t raw_image, sampler_t sampler, float2 coord) {
    return read_imagef(raw_image, sampler, coord).x;
}

/**
 * @brief Converts a single pixel from a bayer pattern to RGB and returns it
 *
 * @details Code adapted from http://graphics.cs.williams.edu/papers/BayerJGT09/
 *
 * @param raw_image   the raw bayer pattern image that we are reading from
 * @param sampler     the sampler to use for reading the raw bayer image
 * @param coord       the coordinate to read from
 * @param first_red   the coordinate for the first red pixel in the bayer pattern
 *
 * @return the RGB pixel at the given location in the bayer image
 */
float4 bayerToRGB(read_only image2d_t raw_image, sampler_t sampler, float2 coord, float2 first_red) {
    float4 centre = (float4){0.0f, 0.0f, 0.0f, 0.0f};
    centre.xy     = coord;
    centre.zw     = coord + first_red;

    float4 x_coord = centre.x + (float4){-2.0f, -1.0f, 1.0f, 2.0f};
    float4 y_coord = centre.y + (float4){-2.0f, -1.0f, 1.0f, 2.0f};

    float C         = fetch(raw_image, sampler, centre.xy);  // ( 0, 0)
    const float4 kC = {0.5f, 0.75f, 0.625f, 0.625f};

    // Determine which of four types of pixels we are on.
    float2 alternate = fmod(floor(centre.zw), 2.0f);

    float4 Dvec = (float4){fetch(raw_image, sampler, (float2){x_coord.y, y_coord.y}),   // (-1,-1)
                           fetch(raw_image, sampler, (float2){x_coord.y, y_coord.z}),   // (-1, 1)
                           fetch(raw_image, sampler, (float2){x_coord.z, y_coord.y}),   // ( 1,-1)
                           fetch(raw_image, sampler, (float2){x_coord.z, y_coord.z})};  // ( 1, 1)

    float4 PATTERN = (kC.xyz * C).xyzz;

    // Can also be a dot product with (1,1,1,1) on hardware where that is
    // specially optimized.
    // Equivalent to: D = Dvec.x + Dvec.y + Dvec.z + Dvec.w;
    Dvec.xy += Dvec.zw;
    Dvec.x += Dvec.y;

    float4 value = (float4){fetch(raw_image, sampler, (float2){centre.x, y_coord.x}),   // ( 0,-2)
                            fetch(raw_image, sampler, (float2){centre.x, y_coord.y}),   // ( 0,-1)
                            fetch(raw_image, sampler, (float2){x_coord.x, centre.y}),   // (-1, 0)
                            fetch(raw_image, sampler, (float2){x_coord.y, centre.y})};  // (-2, 0)

    float4 temp = (float4){fetch(raw_image, sampler, (float2){centre.x, y_coord.w}),   // ( 0, 2)
                           fetch(raw_image, sampler, (float2){centre.x, y_coord.z}),   // ( 0, 1)
                           fetch(raw_image, sampler, (float2){x_coord.w, centre.y}),   // ( 2, 0)
                           fetch(raw_image, sampler, (float2){x_coord.z, centre.y})};  // ( 1, 0)

    // Even the simplest compilers should be able to constant-fold these to avoid the division.
    // Note that on scalar processors these constants force computation of some identical products twice.
    const float4 kA = {-0.125f, -0.1875f, 0.0625f, -0.125f};
    const float4 kB = {0.25f, 0.0f, 0.0f, 0.5f};
    const float4 kD = {0.0f, 0.25f, -0.125f, -0.125f};

    // Conserve constant registers and take advantage of free swizzle on load
    const float4 kE = kA.xywz;
    const float4 kF = kB.xywz;

    value += temp;

    // There are five filter patterns (identity, cross, checker,
    // theta, phi).  Precompute the terms from all of them and then
    // use swizzles to assign to color channels.
    //
    // Channel   Matches
    //   x       cross   (e.g., EE G)
    //   y       checker (e.g., EE B)
    //   z       theta   (e.g., EO R)
    //   w       phi     (e.g., EO R)
    const float A = value.x;
    const float B = value.y;
    const float D = Dvec.x;
    const float E = value.z;
    const float F = value.w;

    // Avoid zero elements. On a scalar processor this saves two MADDs and it has no
    // effect on a vector processor.
    PATTERN.yzw += (kD.yz * D).xyy;

    PATTERN += (kA.xyz * A).xyzx + (kE.xyw * E).xyxz;
    PATTERN.xw += kB.xw * B;
    PATTERN.xz += kF.xz * F;

    float4 result;

    if (alternate.y == 0.0f) {
        if (alternate.x == 0.0f) { result = (float4){C, PATTERN.xy, 1.0f}; }

        else {
            result = (float4){PATTERN.z, C, PATTERN.w, 1.0f};
        }
    }

    else {
        if (alternate.x == 0.0f) { result = (float4){PATTERN.w, C, PATTERN.z, 1.0f}; }

        else {
            result = (float4){PATTERN.yx, C, 1.0f};
        }
    }

    return result;
}

/**
 * @brief Reads a pixel at a position and returns a vec4 value
 *
 * @param image   the image to read from
 * @param format  the fourcc code for the image to decode from
 * @param pos
 */
float4 read_image(read_only image2d_t image, const enum FOURCC format, const float2 coord) {
    switch (format) {
        // Bayer formats use the bayer sampler (nearest neighbour)
        case GRBG: return bayerToRGB(image, bayer_sampler, coord, (float2)(1.0, 0.0));
        case RGGB: return bayerToRGB(image, bayer_sampler, coord, (float2)(0.0, 0.0));
        case GBRG: return bayerToRGB(image, bayer_sampler, coord, (float2)(0.0, 1.0));
        case BGGR: return bayerToRGB(image, bayer_sampler, coord, (float2)(1.0, 1.0));
        // RGB formats read with interpolation
        case RGB3:
        case BGRA:
        case RGBA: {
            return read_imagef(image, interp_sampler, coord);
        }
        default: {
            return (float4)(0);
        }
    }
}

/**
 * @brief Reads data from an image into a network layer using the projected visual mesh points
 *
 * @param image   the image to read from
 * @param format  the format that the image is encoded in
 * @param coords  the pixel coordinates for each element in the visual mesh graph
 * @param network the memory storage for the first layer of the network
 */
kernel void load_image(read_only image2d_t image,
                       const enum FOURCC format,
                       global float2* coords,
                       global float4* network) {

    const int idx = get_global_id(0);

    // Read our pixel coordinate into the image
    network[idx] = read_image(image, format, coords[idx]);
}
