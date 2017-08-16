/*
 * Copyright (C) 2017 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_HPP
#define VISUALMESH_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

namespace mesh {

/**
 * @brief Constructs and holds a visual mesh
 * @details [long description]
 *
 * @tparam Scalar the type that will hold the vectors <float, double>
 */
template <typename Scalar = float>
class VisualMesh {
public:
    // Typedef some value types we commonly use
    using vec2 = std::array<Scalar, 2>;
    using vec3 = std::array<Scalar, 3>;
    using vec4 = std::array<Scalar, 4>;
    using mat3 = std::array<vec3, 3>;
    using mat4 = std::array<vec4, 4>;

    struct Lens {
        enum Type { EQUIRECTANGULAR, RADIAL };
        struct Radial {
            Scalar fov;
        };
        struct Equirectangular {
            vec2 fov;
        };

        Type type;
        union {
            Radial radial;
            Equirectangular equirectangular;
        };
    };

    struct Node {
        /// The unit vector in the direction for this node
        vec4 ray;
        /// Relative indices to the linked hexagon nodes in the LUT ordered TL, TR, L, R, BL, BR,
        int neighbours[6];
    };

    struct Row {
        Row(const Scalar& phi, const size_t& begin, const size_t& end) : phi(phi), begin(begin), end(end) {}

        /// The phi value this row represents
        Scalar phi;
        /// The index of the beginning of this row in the node table
        size_t begin;
        /// The index of one past the end of this row in the node table
        size_t end;

        /**
         * @brief Compare based on phi
         *
         * @param other other item to compare to
         *
         * @return if our phi is less than other phi
         */
        bool operator<(const Row& other) const {
            return phi < other.phi;
        }
    };

    struct Mesh {
        Mesh(std::vector<Node>&& nodes, std::vector<Row>&& rows) : nodes(nodes), rows(rows) {}

        /// The lookup table for this mesh
        std::vector<Node> nodes;
        /// A set of individual rows for phi values. `begin` and `end` refer to the table with end being 1 past the end
        std::vector<Row> rows;
    };

    /**
     * @brief Makes an unallocated visual mesh
     */
    VisualMesh() {}

    /**
     * @brief Generate a new visual mesh for the given shape.
     *
     * @param shape             the shape we are generating a visual mesh for
     * @param min_height        the minimum height that our camera will be at
     * @param max_height        the maximum height our camera will be at
     * @param height_resolution the number of look up tables to generated (height gradations)
     * @param min_angular_res   the smallest angular size to generate for
     */
    template <typename Shape>
    explicit VisualMesh(const Shape& shape,
                        const Scalar& min_height,
                        const Scalar& max_height,
                        const size_t& height_resolution,
                        const Scalar& min_angular_res)
        : min_angular_res(min_angular_res)
        , min_height(min_height)
        , max_height(max_height)
        , height_resolution(height_resolution) {

        // Loop through to make a mesh for each of our height possibilities
        for (Scalar h = min_height; h < max_height; h += (max_height - min_height) / height_resolution) {

            // This is a list of phi values along with the delta theta values associated with them
            std::vector<std::pair<Scalar, size_t>> phis;

            // Loop from directly down up to the horizon (if phi is nan it will stop)
            // So we don't have a single point at the base, we move half a jump forward
            for (Scalar phi = shape.phi(0, h) * Scalar(0.5); phi < M_PI_2;) {

                // Calculate our theta
                Scalar theta = std::max(shape.theta(phi, h), min_angular_res);

                if (!std::isnan(theta)) {
                    // Push back the phi, and the number of whole shapes we can fit
                    phis.emplace_back(phi, size_t(std::ceil(Scalar(2.0) * M_PI / theta)));
                }

                // Move to our next phi
                phi = std::max(phi + min_angular_res, shape.phi(phi, h));
            }

            // Loop from directly up down to the horizon (if phi is nan it will stop)
            for (Scalar phi = (M_PI + shape.phi(M_PI, h)) * Scalar(0.5); phi > M_PI_2;) {

                // Calculate our theta
                Scalar theta = std::max(shape.theta(phi, h), min_angular_res);

                if (!std::isnan(theta)) {
                    // Push back the phi, and the number of whole shapes we can fit
                    phis.emplace_back(phi, size_t(std::ceil(Scalar(2.0) * M_PI / theta)));
                }

                // Move to our next phi
                phi = std::min(phi - min_angular_res, shape.phi(phi, h));
            }


            // Sort the list by phi to create a contiguous area
            std::sort(phis.begin(), phis.end());

            // From this generate unit vectors for the full lut
            std::vector<Node> lut;

            // Work out how big our LUT will be
            size_t lut_size = 0;
            for (const auto& v : phis) {
                lut_size += v.second;
            }
            lut.reserve(lut_size);

            // The start and end of each row in the final lut
            std::vector<Row> rows;
            rows.reserve(phis.size());

            // Loop through our LUT and calculate our left and right neighbours
            for (const auto& v : phis) {

                // Get our phi and delta theta values for a clean circle
                const auto& phi   = v.first;
                const auto& steps = v.second;
                Scalar dtheta     = (Scalar(2.0) * M_PI) / steps;

                // We will use the start position of each row later for linking the graph
                rows.emplace_back(phi, lut.size(), lut.size() + steps);

                // Generate for each of the theta values from 0 to 2 pi
                Scalar theta = 0;
                for (size_t i = 0; i < steps; ++i) {
                    Node n;

                    // Calculate our unit vector with origin facing forward
                    n.ray[0] = std::sin(M_PI - phi) * std::cos(theta);
                    n.ray[1] = std::sin(M_PI - phi) * std::sin(theta);
                    n.ray[2] = std::cos(M_PI - phi);
                    n.ray[3] = 0;

                    // Get the indices for our left/right neighbours relative to this row
                    const int l = i == 0 ? steps - 1 : i - 1;
                    const int r = i == steps - 1 ? 0 : i + 1;

                    // Set these two neighbours
                    n.neighbours[2] = l - i;  // L
                    n.neighbours[3] = r - i;  // R

                    // Move on to the next theta value
                    theta += dtheta;

                    lut.push_back(std::move(n));
                }
            }


            /**
             * This function links to the previous and next rows
             *
             * @param i       the absolute index to the node we are linking
             * @param pos     the position of this node in its row as a value between 0 and 1
             * @param start   the start of the row to link to
             * @param size    the size of the row we are linking to
             * @param offset  the offset for our neighbour (0 for TL,TR 4 for BL BR)
             */
            auto link = [](std::vector<Node>& lut,
                           const size_t& i,
                           const Scalar& pos,
                           const int& start,
                           const int& size,
                           const size_t offset) {

                // Grab our current node
                auto& node = lut[i];

                // Work out if we are closer to the left or right and make an offset var for it
                // Note this bool is used like a bool and int. It is 0 when we should access TR first
                // and 1 when we should access TL first. This is to avoid accessing values which wrap around
                // and instead access a non wrap element and use its neighbours to work out ours
                bool left = pos > Scalar(0.5);

                // Get our closest neighbour on the previous row and use it to work out where the other one
                // is This will be the Right element when < 0.5 and Left when > 0.5
                size_t o1 = start + std::floor(pos * size + !left);  // Use `left` to add one to one
                size_t o2 = o1 + lut[o1].neighbours[2 + left];       // But not the other

                // Now use these to set our TL and TR neighbours
                node.neighbours[offset]     = (left ? o1 : o2) - i;
                node.neighbours[offset + 1] = (left ? o2 : o1) - i;
            };

            // Now we upwards and downwards to fill in the missing links
            for (size_t r = 1; r < rows.size() - 1; ++r) {

                // Alias for convenience
                const auto& prev    = rows[r - 1];
                const auto& current = rows[r];
                const auto& next    = rows[r + 1];

                // Work out how big our rows are if they are within valid indices
                int prev_size    = prev.end - prev.begin;
                int current_size = current.end - current.begin;
                int next_size    = next.end - next.begin;

                // Go through all the nodes on our current row
                for (size_t i = current.begin; i < current.end; ++i) {

                    // Find where we are in our row as a value between 0 and 1
                    Scalar pos = Scalar(i - current.begin) / Scalar(current_size);

                    // Perform both links
                    link(lut, i, pos, prev.begin, prev_size, 0);
                    link(lut, i, pos, next.begin, next_size, 4);
                }
            }

            // Now we have to deal with the very first, and very last rows as they can't be linked in the normal way
            if (!rows.empty()) {

                auto& front    = rows.front();
                int front_size = front.end - front.begin;

                auto& back    = rows.back();
                int back_size = back.end - back.begin;

                // Link the front to itself
                for (size_t i = front.begin; i < front.end; ++i) {
                    // Alias our node
                    auto& node = lut[i];

                    // Work out which two points are on the opposite side to us
                    size_t index = i - front.begin + (front_size / 2);

                    // Find where we are in our row as a value between 0 and 1
                    Scalar pos = Scalar(i - front.begin) / Scalar(front_size);

                    // Link to ourself
                    node.neighbours[0] = front.begin + (index % front_size) - i;
                    node.neighbours[1] = front.begin + ((index + 1) % front_size) - i;

                    // Link to our next row normally
                    auto& r2 = rows[1];
                    link(lut, i, pos, r2.begin, r2.end - r2.begin, 4);
                }

                // Link the back to itself
                for (size_t i = back.begin; i < back.end; ++i) {
                    // Alias our node
                    auto& node = lut[i];

                    // Work out which two points are on the opposite side to us
                    size_t index = i - back.begin + (back_size / 2);

                    // Find where we are in our row as a value between 0 and 1
                    Scalar pos = Scalar(i - back.begin) / Scalar(back_size);

                    // Link to ourself on the other side
                    node.neighbours[4] = back.begin + (index % back_size) - i;
                    node.neighbours[5] = back.begin + ((index + 1) % back_size) - i;

                    // Link to our previous row normally
                    auto& r2 = rows[rows.size() - 2];
                    link(lut, i, pos, r2.begin, r2.end - r2.begin, 0);
                }
            }

            // Insert our constructed mesh into the lookup
            luts.insert(std::make_pair(h, Mesh(std::move(lut), std::move(rows))));
        }
    }

    const Mesh& height(const Scalar& height) const {
        return luts.lower_bound(height)->second;
    }

    template <typename Func>
    std::vector<std::pair<size_t, size_t>> lookup(const Scalar& height, Func&& theta_limits) const {

        const auto& mesh = luts.lower_bound(height)->second;
        std::vector<std::pair<size_t, size_t>> indices;

        // Loop through each phi row
        for (auto& row : mesh.rows) {

            auto row_size = row.end - row.begin;

            // Get the theta values that are valid for this phi
            auto theta_ranges = theta_limits(row.phi);

            // Work out what this range means in terms of theta
            for (auto& range : theta_ranges) {

                // Convert our theta values into local indices
                size_t begin = std::ceil(row_size * range.first * (Scalar(1.0) / (Scalar(2.0) * M_PI)));
                size_t end   = std::ceil(row_size * range.second * (Scalar(1.0) / (Scalar(2.0) * M_PI)));

                // Floating point numbers are annoying... did you know pi * 1/pi is slightly larger than 1?
                // It's also possible that our theta ranges cross the wrap around but the indices mean they don't
                // This will cause segfaults unless we fix the wrap
                begin = begin > row_size ? 0 : begin;
                end   = end > row_size ? row_size : end;

                // If we define a nice enclosed range range add it
                if (end >= begin) {
                    indices.emplace_back(row.begin + begin, row.begin + end);
                }
                // Our phi values wrap around so we need two ranges
                else {
                    indices.emplace_back(row.begin, row.begin + end);
                    indices.emplace_back(row.begin + begin, row.end);
                }
            }
        }

        return indices;
    }

    std::vector<std::pair<size_t, size_t>> lookup(const mat4& Hco, const Lens& lens) {

        // We multiply a lot of things by 2
        constexpr const Scalar x2 = Scalar(2.0);

        switch (lens.type) {
            case Lens::EQUIRECTANGULAR: {

                // Extract our rotation matrix
                const mat3 Rco = {{
                    {{Hco[0][0], Hco[0][1], Hco[0][2]}},  //
                    {{Hco[1][0], Hco[1][1], Hco[1][2]}},  //
                    {{Hco[2][0], Hco[2][1], Hco[2][2]}}   //
                }};

                // Extract our z height
                // TODO this is wrong, fix it when it matters
                const vec3 rOCc = {{Hco[0][3], Hco[1][3], Hco[2][3]}};

                // Print our camera vector
                const std::array<Scalar, 3> cam = {{Hco[0][0], Hco[1][0], Hco[2][0]}};
                std::cerr << "Cam: " << cam << std::endl << std::endl;
                std::cout << "[0, 0, 0, " << cam[0] << ", " << cam[1] << ", " << cam[2] << ", 0, 0, 0],";

                // Solution to finding the edges is an intersection between a line and a cone
                // Based on a simplified version of the math found at
                // https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf

                // Work out how much additional y and z we get from our field of view if we have a focal length of 1
                const Scalar y_extent = std::tan(lens.equirectangular.fov[0] * Scalar(0.5));
                const Scalar z_extent = std::tan(lens.equirectangular.fov[1] * Scalar(0.5));

                /* The labels for each of the corners of the frustum is shown below.
                    ^    T       U
                    |        C
                    z    W       V
                    <- y
                 */

                // Make vectors to the corners in cam space
                const std::array<vec3, 4> rNCc = {{
                    {{Scalar(1.0), +y_extent, +z_extent}},  // rTCc
                    {{Scalar(1.0), -y_extent, +z_extent}},  // rUCc
                    {{Scalar(1.0), -y_extent, -z_extent}},  // rVCc
                    {{Scalar(1.0), +y_extent, -z_extent}}   // rWCc
                }};

                // Rotate these into world space by multiplying by the rotation matrix
                // Because of the way we are performing our dot product here (row->row), we are transposing Rco
                const std::array<vec3, 4> rNCo = {{
                    {{dot(rNCc[0], Rco[0]), dot(rNCc[0], Rco[1]), dot(rNCc[0], Rco[2])}},  // rTCo
                    {{dot(rNCc[1], Rco[0]), dot(rNCc[1], Rco[1]), dot(rNCc[1], Rco[2])}},  // rUCo
                    {{dot(rNCc[2], Rco[0]), dot(rNCc[2], Rco[1]), dot(rNCc[2], Rco[2])}},  // rVCo
                    {{dot(rNCc[3], Rco[0]), dot(rNCc[3], Rco[1]), dot(rNCc[3], Rco[2])}},  // rWCo
                }};

                // Make our corner to next corner vectors
                // In cam space these are 0,1,0 style vectors so we just get a col of the other matrix
                // But since we are multiplying by the transpose we get a row of the matrix
                // When we are storing this matrix we represent each corner as N and the following clockwise corner as M
                // Then it is multiplied by the extent to make a vector of the length of the edge of the frustum
                const std::array<vec3, 4> rMNo = {{
                    {{-Rco[0][1] * x2 * y_extent, -Rco[1][1] * x2 * y_extent, -Rco[2][1] * x2 * y_extent}},  // rUTo
                    {{-Rco[0][2] * x2 * z_extent, -Rco[1][2] * x2 * z_extent, -Rco[2][2] * x2 * z_extent}},  // rVUo
                    {{+Rco[0][1] * x2 * y_extent, +Rco[1][1] * x2 * y_extent, +Rco[2][1] * x2 * y_extent}},  // rWVo
                    {{+Rco[0][2] * x2 * z_extent, +Rco[1][2] * x2 * z_extent, +Rco[2][2] * x2 * z_extent}}   // rTWo
                }};

                // Make our normals to the frustum edges
                const std::array<vec3, 4> edges = {{
                    cross(rNCo[0], rNCo[1]),  // Top edge
                    cross(rNCo[1], rNCo[2]),  // Left edge
                    cross(rNCo[2], rNCo[3]),  // Base edge
                    cross(rNCo[3], rNCo[0]),  // Right edge
                }};

                // These calculations are intermediates for the solution to the cone/line equation. Since these parts
                // are the same for all phi values, we can pre-calculate them here to save effort later
                std::array<std::array<Scalar, 6>, 4> eq_parts;
                for (int i = 0; i < 4; ++i) {
                    const auto& o = rNCo[i];  // Line origin
                    const auto& d = rMNo[i];  // Line direction

                    // Later we will use these constants like so
                    // (p[0] + c2 * p[1] ± sqrt(c2 * p[2] + p[3]))/(p[4] + c2 * p[5]);

                    // c2 dependant part of numerator
                    eq_parts[i][0] = d[2] * o[2];  // -dz oz

                    // Non c2 dependant part of numerator
                    eq_parts[i][1] = -d[1] * o[1] - d[0] * o[0];  // -dy oy - dx ox

                    // c2 dependant part of discriminant
                    eq_parts[i][2] = d[0] * d[0] * o[2] * o[2]         // dx^2 oz^2
                                     - x2 * d[0] * d[2] * o[0] * o[2]  // 2 dx dz ox oz
                                     + d[1] * d[1] * o[2] * o[2]       // dy^2 oz^2
                                     - x2 * d[1] * d[2] * o[1] * o[2]  // 2 dy dz oy oz
                                     + d[2] * d[2] * o[0] * o[0]       // d_z^2 o_x^2
                                     + d[2] * d[2] * o[1] * o[1];      // d_z^2 o_y^2

                    // non c2 dependant part of discriminant
                    eq_parts[i][3] = -d[0] * d[0] * o[1] * o[1]        // dx^2 oy^2
                                     + x2 * d[0] * d[1] * o[0] * o[1]  // 2 dx dy ox oy
                                     - d[1] * d[1] * o[0] * o[0];      // dy^2 ox^2

                    // c2 dependant part of denominator
                    eq_parts[i][4] = -d[2] * d[2];  // -(dz^2)

                    // non c2 dependant part of denominator
                    eq_parts[i][5] = d[0] * d[0] + d[1] * d[1];  // dx^2 + dy^2
                }


                // TODO remove this when you're finished it
                for (int i = 0; i < 4; ++i) {
                    std::cout << "[0, 0, 0," << rNCo[i][0] << ", " << rNCo[i][1] << ", " << rNCo[i][2]
                              << ", 255, 0, 0], ";

                    std::cout << "[0, 0, 0, " << edges[i][0] << ", " << edges[i][1] << ", " << edges[i][2]
                              << ", 255, 0, 0], ";

                    std::cout << "[" << rNCo[i][0] << ", " << rNCo[i][1] << ", " << rNCo[i][2] << ", "
                              << (rMNo[i][0] + rNCo[i][0]) << ", " << (rMNo[i][1] + rNCo[i][1]) << ", "
                              << (rMNo[i][2] + rNCo[i][2]) << ", 255, 0, 0], ";
                }

                // Calculate our theta limits
                auto theta_limits = [&](const Scalar& phi) {

                    // Cone gradient squared
                    const Scalar cos_phi = std::cos(phi);
                    const Scalar tan_phi = std::tan(phi);
                    const Scalar c2      = tan_phi * tan_phi;

                    // Store any limits we find
                    std::vector<Scalar> limits;

                    // Count how many complex solutions we get
                    int complex_sols = 0;

                    for (int i = 0; i < 4; ++i) {
                        // We make a line origin + ray to define a parametric line
                        // Note that both of these vectors are always unit length
                        const auto& o = rNCo[i];  // Line origin
                        const auto& d = rMNo[i];  // Line direction

                        // Calculate the first half of our numerator
                        const Scalar num = c2 * eq_parts[i][0] + eq_parts[i][1];

                        // Calculate our discriminant.
                        const Scalar disc = c2 * eq_parts[i][2] + eq_parts[i][3];

                        // Calculate our denominator
                        const Scalar denom = c2 * eq_parts[i][4] + eq_parts[i][5];

                        // We need to count how many complex solutions we get, if all 4 are we totally enclose phi
                        if (disc < Scalar(0.0)) {
                            ++complex_sols;
                        }
                        else if (denom != Scalar(0.0)) {

                            // We have two intersections with either the upper or lower cone
                            Scalar root = std::sqrt(disc);

                            // Get our two solutions for t
                            for (const Scalar t : {(num + root) / denom, (num - root) / denom}) {

                                // Check we are within the valid range for our segment.
                                // Since we set the length of the direction vector to the length of the side we can
                                // check it's less than one
                                if (t >= Scalar(0.0) && t <= Scalar(1.0)) {

                                    // We check z first to make sure it's on the correct side
                                    Scalar z = o[2] + d[2] * t;

                                    // If we are both above, or both below the horizon
                                    if ((z > 0) == (phi > M_PI_2)) {

                                        Scalar x = o[0] + d[0] * t;
                                        Scalar y = o[1] + d[1] * t;
                                        std::cout << "[0, 0, 0, " << x << ", " << y << ", " << z << ", 255, 0, 0],";
                                        Scalar theta = std::atan2(y, x);
                                        // atan2 gives a result from -pi -> pi, we need 0 -> 2 pi
                                        limits.emplace_back(theta > 0 ? theta : theta + M_PI * Scalar(2.0));
                                    }
                                }
                            }
                        }
                    }

                    // If all solutions are complex we totally enclose the phi however we still need to check the cone
                    // is on the correct side
                    if (complex_sols == 4 && ((cos_phi > 0) == (-cam[2] > 0))) {
                        return std::vector<std::pair<Scalar, Scalar>>(1, std::make_pair(0, Scalar(2.0) * M_PI));
                    }
                    // If we have intersections
                    else if (!limits.empty()) {
                        // If we have an even number of intersections
                        if (limits.size() % 2 == 0) {
                            // Sort the limits
                            std::sort(limits.begin(), limits.end());

                            // Get a test point half way between the first two points
                            const Scalar test_theta = (limits[0] + limits[1]) * Scalar(0.5);
                            const Scalar sin_phi    = std::sin(phi);
                            const Scalar sin_theta  = std::sin(test_theta);
                            const Scalar cos_theta  = std::cos(test_theta);

                            // Make a unit vector from the phi and theta
                            vec3 test_vec = {{cos_theta * sin_phi, sin_theta * sin_phi, -cos_phi}};

                            bool first_is_end = false;
                            for (int i = 0; i < 4; ++i) {
                                // If we get a negative dot product our first point is an end segment
                                first_is_end |= Scalar(0.0) > dot(test_vec, edges[i]);
                            }

                            // If this is entering, point 0 is a start, and point 1 is an end
                            std::vector<std::pair<Scalar, Scalar>> output;
                            for (size_t i = first_is_end ? 1 : 0; i < limits.size() - 1; i += 2) {
                                output.emplace_back(limits[i], limits[i + 1]);
                            }
                            if (first_is_end) {
                                output.emplace_back(limits.back(), limits.front());
                            }
                            return output;
                        }
                        // If we have an odd number of intersections something is wrong
                        else {
                            throw std::runtime_error("Odd number of intersections found with cone");
                        }
                    }

                    // Default to returning an empty list
                    return std::vector<std::pair<Scalar, Scalar>>();
                };

                return lookup(rOCc[2], theta_limits);
            }

            case Lens::RADIAL: {
                // Solution for intersections on the edge is the intersection between a unit sphere, a plane, and a cone
                // The cone is the cone made by the phi angle, and the plane intersects with the unit sphere to form
                // The circle that defines the edge of the field of view of the camera.
                //
                // Unit sphere
                // x^2 + y^2 + z^2 = 1
                //
                // Cone (don't need to check side for phi since it's squared)
                // z^2 = (x^2+y^2)/c^2
                // c = tan(phi)
                //
                // Plane
                // N = the unit vector in the direction of the camera
                // r_0 = N * cos(fov/2)
                // N . (r - r_0) = 0
                //
                // To simplify things however, we remove the y component and assume the camera vector is only ever
                // on the x/z plane. We calculate the offset to make this happen and re apply it at the end

                // The gradient of our field of view cone
                const Scalar cos_half_fov = std::cos(lens.radial.fov * Scalar(0.5));
                const vec3 cam            = {{Hco[0][0], Hco[1][0], Hco[2][0]}};
                std::cerr << "Cam: " << cam << std::endl << std::endl;
                std::cout << "[0, 0, 0, " << cam[0] << ", " << cam[1] << ", " << cam[2] << ", 255, 0, 0],";

                auto theta_limits = [&](const Scalar& phi) -> std::array<std::pair<Scalar, Scalar>, 1> {

                    // Check if we are intersecting with an upper or lower cone
                    const bool upper = phi > M_PI_2;

                    // The cameras inclination from straight down (same reference frame as phi)
                    const Scalar cam_inc  = std::acos(-cam[2]);
                    const Scalar half_fov = lens.radial.fov * 0.5;

                    // First we should check if this phi is totally contained in our fov
                    // Work out what our largest fully contained phi value is
                    // We can work this out by subtracting our offset angle from our fov and checking if phi is smaller
                    if ((upper && half_fov - (M_PI - cam_inc) > M_PI - phi) || (!upper && half_fov - cam_inc > phi)) {
                        return {{std::make_pair(0, M_PI * 2.0)}};
                    }
                    // Also if we can tell that the phi is totally outside we can bail out early
                    // To check this we check phi is greater than our inclination plus our fov
                    if ((upper && half_fov + (M_PI - cam_inc) < M_PI - phi) || (!upper && half_fov + cam_inc < phi)) {
                        return {{std::make_pair(0, 0)}};
                    }

                    // The solution only works for camera vectors that lie in the x/z plane
                    // So we have to rotate our vector into that space, solve it and then rotate them back
                    // Normally this would be somewhat unsafe as cam[1] and cam[0] could be both 0
                    // However, that case is resolved by the checks above that confirm we intersect
                    const Scalar offset     = std::atan2(cam[1], cam[0]);
                    const Scalar sin_offset = std::sin(offset);
                    const Scalar cos_offset = std::cos(offset);

                    // Now we must rotate our cam vector before doing the solution
                    // Since y will be 0, and z doesn't change we only need this one
                    const Scalar r_x = cam[0] * cos_offset + cam[1] * sin_offset;

                    // The z component of our solution
                    const Scalar z = -std::cos(phi);

                    // Calculate intermediate products
                    const Scalar a = Scalar(1.0) - z * z;  // aka sin^2(phi)
                    const Scalar x = (cos_half_fov - cam[2] * z) / r_x;

                    // The y component is ± this square root
                    const Scalar y_disc = a - x * x;

                    if (y_disc < 0) {
                        return {{std::make_pair(0, 0)}};
                    }

                    const Scalar y  = std::sqrt(y_disc);
                    const Scalar t1 = offset + std::atan2(-y, x);
                    const Scalar t2 = offset + std::atan2(y, x);

                    //  Print our two solutions
                    std::cout << "[0, 0, 0, " << (x * cos_offset - y * sin_offset) << ", "
                              << (x * sin_offset + y * cos_offset) << ", " << z << ", 255, 0, 0],";
                    std::cout << "[0, 0, 0, " << (x * cos_offset + y * sin_offset) << ", "
                              << (x * sin_offset - y * cos_offset) << ", " << z << ", 255, 0, 0],";

                    return {
                        {std::make_pair(t1 > 0 ? t1 : t1 + M_PI * Scalar(2.0), t2 > 0 ? t2 : t2 + M_PI * Scalar(2.0))}};
                };

                // TODO work out the height of the camera properly
                return lookup(0.5, theta_limits);
            }
            default: { throw std::runtime_error("Unknown lens type"); }
        }
    }

private:
    inline Scalar dot(const vec3& a, const vec3& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    inline vec3 cross(const vec3& a, const vec3& b) {
        return {{
            a[1] * b[2] - a[2] * b[1],  // x
            a[2] * b[0] - a[0] * b[2],  // y
            a[0] * b[1] - a[1] * b[0]   // z
        }};
    }

    inline vec3 normalise(const vec3& a) {
        Scalar length = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] + a[2]);
        return {{a[0] * length, a[1] * length, a[2] * length}};
    }

    /// A map from heights to visual mesh tables
    std::map<Scalar, Mesh> luts;

    /// The smallest angular width the LUT should be generated for
    Scalar min_angular_res;
    /// The minimum height the the luts are generated for
    Scalar min_height;
    // The maximum height the luts are generated for
    Scalar max_height;
    // The number gradations in height
    size_t height_resolution;
};

}  // namespace mesh

#endif  // VISUALMESH_HPP
