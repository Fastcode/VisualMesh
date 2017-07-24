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
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

namespace mesh {

template <typename T>
struct Printer;

// Print a matrix
template <typename Scalar, std::size_t n, std::size_t m>
struct Printer<std::array<std::array<Scalar, n>, m>> {
    static inline void print(std::ostream& out, const std::array<std::array<Scalar, n>, m>& s) {
        for (std::size_t j = 0; j < m; ++j) {
            out << "[";
            for (std::size_t i = 0; i < n - 1; ++i) {
                out << s[j][i] << ", ";
            }
            if (n > 0) {
                out << s[j][n - 1];
            }
            out << "]";

            if (j < m - 1) {
                out << std::endl;
            }
        }
    }
};

// Print a vector
template <typename Scalar, std::size_t n>
struct Printer<std::array<Scalar, n>> {
    static inline void print(std::ostream& out, const std::array<Scalar, n>& s) {
        out << "[";
        for (std::size_t i = 0; i < n - 1; ++i) {
            out << s[i] << ", ";
        }
        if (n > 0) {
            out << s[n - 1];
        }
        out << "]";
    }
};

template <typename T, std::size_t n>
std::ostream& operator<<(std::ostream& out, const std::array<T, n>& s) {
    Printer<std::array<T, n>>::print(out, s);
    return out;
}

/**
 * @brief Constructs and holds a visual mesh
 * @details [long description]
 *
 * @tparam Scalar the type that will hold the vectors <float, double>
 */
template <typename Scalar = float>
class VisualMesh {
public:
    struct Lens {
        enum Type { EQUIRECTANGULAR, RADIAL };
        struct Radial {};
        struct Equirectangular {
            std::array<Scalar, 2> fov;
        };

        Type type;
        union {
            Radial radial;
            Equirectangular equirectangular;
        };
    };

    struct Node {
        /// The unit vector in the direction for this node
        Scalar ray[4];
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
            for (Scalar phi = shape.phi(0, h) / 2.0; phi < M_PI_2;) {

                // Calculate our theta
                Scalar theta = std::max(shape.theta(phi, h), min_angular_res);

                if (!std::isnan(theta)) {
                    // Push back the phi, and the number of whole shapes we can fit
                    phis.emplace_back(phi, size_t(std::ceil(2.0 * M_PI / theta)));
                }

                // Move to our next phi
                phi = std::max(phi + min_angular_res, shape.phi(phi, h));
            }

            // Loop from directly up down to the horizon (if phi is nan it will stop)
            for (Scalar phi = (M_PI + shape.phi(M_PI, h)) / 2.0; phi > M_PI_2;) {

                // Calculate our theta
                Scalar theta = std::max(shape.theta(phi, h), min_angular_res);

                if (!std::isnan(theta)) {
                    // Push back the phi, and the number of whole shapes we can fit
                    phis.emplace_back(phi, size_t(std::ceil(2.0 * M_PI / theta)));
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
                Scalar dtheta     = (2.0 * M_PI) / steps;

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

            // Now we upwards and downwards to fill in the missing links
            for (size_t r = 0; r < rows.size(); ++r) {

                // Alias for convenience
                const auto& prev    = rows[r - 1];
                const auto& current = rows[r];
                const auto& next    = rows[r + 1];

                // Work out how big our rows are if they are within valid indices
                int prev_size    = r > 0 ? prev.end - prev.begin : 0;
                int current_size = current.end - current.begin;
                int next_size    = r < rows.size() - 1 ? next.end - next.begin : 0;

                // Go through all the nodes on our current row
                for (size_t i = current.begin; i < current.end; ++i) {

                    // Grab our current node
                    auto& node = lut[i];

                    // Find where we are in our row as a value between 0 and 1
                    Scalar pos = Scalar(i - current.begin) / Scalar(current_size);

                    /**
                     * This function links to the previous and next rows
                     *
                     * @param start         the start of the row to link to
                     * @param size          the size of the row we are linking to
                     * @param invalid_row   the row that is invalid (first for prev, last for next)
                     * @param offset        the offset for our neighbour (0 for TL,TR 4 for BL BR)
                     */
                    auto link = [&](const int& start, const int& size, const size_t& invalid_row, const size_t offset) {

                        if (r != invalid_row) {
                            // Work out if we are closer to the left or right and make an offset var for it
                            // Note this bool is used like a bool and int. It is 0 when we should access TR first
                            // and 1 when we should access TL first. This is to avoid accessing values which wrap around
                            // and instead access a non wrap element and use its neighbours to work out ours
                            bool left = pos > 0.5;

                            // Get our closest neighbour on the previous row and use it to work out where the other one
                            // is This will be the Right element when < 0.5 and Left when > 0.5
                            size_t o1 = start + std::floor(pos * size + !left);  // Use `left` to add one to one
                            size_t o2 = o1 + lut[o1].neighbours[2 + left];       // But not the other

                            // Now use these to set our TL and TR neighbours
                            node.neighbours[offset]     = (left ? o1 : o2) - i;
                            node.neighbours[offset + 1] = (left ? o2 : o1) - i;
                        }
                        // If we don't have a row to link to, we are at the end, so instead link to our own row
                        // However if we only do this if we only do this if we are ending on our side of the horizon
                        else if ((node.ray[2] < 0 && invalid_row == 0) || (node.ray[2] > 0 && invalid_row != 0)) {
                            // Work out which two points are on the opposite side to us
                            size_t index = i - current.begin + (current_size / 2);

                            // Link to them
                            node.neighbours[offset]     = current.begin + (index % current_size) - i;
                            node.neighbours[offset + 1] = current.begin + ((index + 1) % current_size) - i;
                        }
                        // If we can't link, just link to our left and right
                        else {
                            node.neighbours[offset]     = node.neighbours[2];
                            node.neighbours[offset + 1] = node.neighbours[3];
                        }
                    };

                    // Perform both links
                    if (r > 0) link(prev.begin, prev_size, 0, 0);
                    if (r < rows.size() - 1) link(next.begin, next_size, rows.size() - 1, 4);
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
        std::vector<std::pair<size_t, size_t>> indicies;

        // Loop through each phi row
        for (auto& row : mesh.rows) {

            auto row_size = row.end - row.begin;

            // Get the theta values that are valid for this phi
            auto theta_ranges = theta_limits(row.phi);

            // Work out what this range means in terms of theta
            for (auto& range : theta_ranges) {

                // Convert our theta values into local indices
                size_t begin = std::ceil(row_size * range.first / (2.0 * M_PI));
                size_t end   = std::ceil(row_size * range.second / (2.0 * M_PI));

                // If we define a nice enclosed range range add it
                if (end >= begin) {
                    indicies.emplace_back(row.begin + begin, row.begin + end);
                }
                // Our phi values wrap around so we need two ranges
                else {
                    indicies.emplace_back(row.begin, row.begin + end);
                    indicies.emplace_back(row.begin + begin, row.end);
                }
            }
        }

        return indicies;
    }

    std::vector<std::pair<size_t, size_t>> lookup(const std::array<std::array<Scalar, 4>, 4>& Hco, const Lens& lens) {

        switch (lens.type) {
            case Lens::EQUIRECTANGULAR: {

                // Extract our rotation matrix
                const std::array<std::array<Scalar, 3>, 3> Rco = {{
                    {{Hco[0][0], Hco[0][1], Hco[0][2]}},  //
                    {{Hco[1][0], Hco[1][1], Hco[1][2]}},  //
                    {{Hco[2][0], Hco[2][1], Hco[2][2]}}   //
                }};

                // Solution to finding the edges is an intersection between a line and a cone
                // https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf

                // Extract our z height
                // TODO this is wrong, fix it when it matters
                const std::array<Scalar, 3> rOCc = {{Hco[0][3], Hco[1][3], Hco[2][3]}};

                // Work out how much additional y and z we get from our field of view if we have a focal length of 1
                Scalar y_extent = std::tan(lens.equirectangular.fov[0] * 0.5);
                Scalar z_extent = std::tan(lens.equirectangular.fov[1] * 0.5);

                std::cout << "Y extent: " << y_extent << std::endl;
                std::cout << "Z extent: " << z_extent << std::endl;

                // Prenormalise these values as they will all be the same length
                Scalar length = 1.0 / std::sqrt(y_extent * y_extent + z_extent * z_extent + 1);
                y_extent      = y_extent * length;
                z_extent      = z_extent * length;

                /* The names of the corners are as follows.
                   When stored as a group we use the identifier N
                    ^    T       U
                    |        C
                    z    W       V
                    <- y
                 */

                // Make corners in cam space as unit vectors
                const std::array<std::array<Scalar, 3>, 4> rNCc = {{
                    {{length, +y_extent, +z_extent}},  // rTCc
                    {{length, -y_extent, +z_extent}},  // rUCc
                    {{length, -y_extent, -z_extent}},  // rVCc
                    {{length, +y_extent, -z_extent}}   // rWCc
                }};

                // Rotate these into world space by multiplying by the rotation matrix
                // Because of the way we are performing our dot product here (row->row), we are transposing Rco
                const std::array<std::array<Scalar, 3>, 4> rNCo = {{
                    {{dot(rNCc[0], Rco[0]), dot(rNCc[0], Rco[1]), dot(rNCc[0], Rco[2])}},  // rTCo
                    {{dot(rNCc[1], Rco[0]), dot(rNCc[1], Rco[1]), dot(rNCc[1], Rco[2])}},  // rUCo
                    {{dot(rNCc[2], Rco[0]), dot(rNCc[2], Rco[1]), dot(rNCc[2], Rco[2])}},  // rVCo
                    {{dot(rNCc[3], Rco[0]), dot(rNCc[3], Rco[1]), dot(rNCc[3], Rco[2])}},  // rWCo
                }};

                // Make our corner to next corner vectors
                // In cam space these are 0,1,0 style vectors so we just get a col of the other matrix
                // But since we are multiplying by the transpose we get a row of the matrix
                const std::array<std::array<Scalar, 3>, 4> rMNo = {{
                    {{-Rco[1][0], -Rco[1][1], -Rco[1][2]}},  // normalise(rUTc(0, -1,  0)) -> normalise(rUTo)
                    {{-Rco[2][0], -Rco[2][1], -Rco[2][2]}},  // normalise(rVUc(0,  0, -1)) -> normalise(rVUo)
                    {{+Rco[1][0], +Rco[1][1], +Rco[1][2]}},  // normalise(rWVc(0,  1,  0)) -> normalise(rWVo)
                    {{+Rco[2][0], +Rco[2][1], +Rco[2][2]}}   // normalise(rTWc(0,  0,  1)) -> normalise(rTWo)
                }};

                // Later we need dot(rNCo, rMNo) so calculate now.
                // Also the values are identical for each side so only calculate half
                // TODO this line can be greatly optimised using the extends/length/knowledge that it's 0,1,0 and 0,0,1
                const std::array<Scalar, 2> rNCo_dot_rMNo = {dot(rNCo[0], rMNo[0]), dot(rNCo[1], rMNo[1])};


                // Calculate our theta limits
                auto theta_limits = [&](const Scalar& phi) {

                    // Store any limits we find
                    std::vector<Scalar> limits;

                    // Should we intersect with an upper or lower cone
                    // If upper, the axis for the cone is +z, lower is -z
                    const bool upper = phi > M_PI_2;

                    // No need for an upper/lower check here as cos^2(pi-x) == cos^2(x)
                    const Scalar cos_phi  = std::cos(phi);
                    const Scalar cos_phi2 = cos_phi * cos_phi;

                    for (int i = 0; i < 4; ++i) {
                        // We make a line origin + ray to define a parametric line
                        const auto& line_origin    = rNCo[i];
                        const auto& line_direction = rMNo[i];

                        // If we are using the upper cone then the cone axis is z = 1 and if the lower cone z = -1
                        // Therefore these dot products are the same as selecting either +- the z component of the
                        // vectors
                        Scalar DdU   = upper ? line_direction[2] : -line_direction[2];
                        Scalar DdPmV = upper ? line_origin[2] : -line_origin[2];

                        // rNCo_dot_rMNo[i % 2];  // Each side is the same for this
                        Scalar UdPmV = dot(line_origin, line_direction);

                        Scalar c2 = DdU * DdU - cos_phi2;
                        Scalar c1 = DdU * DdPmV - cos_phi2 * UdPmV;
                        Scalar c0 = DdPmV * DdPmV - cos_phi2;

                        if (c2 != 0) {
                            Scalar discriminant = c1 * c1 - c0 * c2;

                            if (discriminant > 0) {
                                // We have two intersections with either the upper or lower code
                                Scalar root   = std::sqrt(discriminant);
                                Scalar inv_c2 = 1.0 / c2;

                                // Get our two solutions for t
                                for (const Scalar t : {(-c1 - root) * inv_c2, (-c1 + root) * inv_c2}) {

                                    bool validp = t >= 0;
                                    bool validm = t <= (i % 2 == 0 ? y_extent : z_extent);
                                    bool valids = DdU * t + DdPmV >= 0;

                                    if (validp && validm && valids) {
                                        // This T is totally a valid value for our range! work out it's theta!
                                        std::cout << i << " " << t << ", ";
                                    }
                                    else if (validp && validm && !valids) {
                                        std::cout << i << " 🍿, ";
                                    }
                                    else if (validp && !validm && valids) {
                                        std::cout << i << " 💣, ";
                                    }
                                    else if (validp && !validm && !valids) {
                                        std::cout << i << " ⛄, ";
                                    }
                                    else if (!validp && validm && valids) {
                                        std::cout << i << " 🚰, ";
                                    }
                                    else if (!validp && validm && !valids) {
                                        std::cout << i << " 💔, ";
                                    }
                                    else if (!validp && !validm && valids) {
                                        std::cout << i << " 🔎, ";
                                    }
                                    else if (!validp && !validm && !valids) {
                                        std::cout << i << " 🍰, ";
                                    }
                                    else {
                                        std::cout << i << " 🥚, ";
                                    }
                                }
                            }
                            else {
                                std::cout << i << " 🍪, ";
                            }
                        }
                        else {
                            std::cout << i << " 😈, ";
                        }
                    }

                    std::cout << std::endl;
                    std::cout << std::endl;

                    // Sort them
                    std::sort(limits.begin(), limits.end());

                    // Now make pairs
                    std::vector<std::pair<Scalar, Scalar>> output;
                    for (size_t i = 0; (i + 1) < limits.size(); i += 2) {
                        output.emplace_back(limits[i], limits[i + 1]);
                    }
                    return output;
                };

                return lookup(rOCc[2], theta_limits);
            }

            case Lens::RADIAL: {


                // Solution for intersections on the edge is the intersection between a unit sphere, a plane, and a cone
                // The cone is the cone made by the phi angle, and the plane intersects with the unit sphere to form
                // The circle that defines the edge of the field of view of the camera.
                //
                // Unit sphere
                // x^2 + y^2 + x^2 = 1
                //
                // Cone
                // z^2 = (x^2+y^2)/c^2
                // c = phi > pi/2 ? tan(phi) : tan(pi - phi)
                //
                // Plane
                // N = the unit vector in the direction of the camera
                // r_0 = N * cos(fov/2)
                // N . (r - r_0) = 0

                // TODO work out the phi value of the camera vector

                // TODO add/subtract the phi value from the FOV

                // TODO normalise

                throw std::runtime_error("Not implemented");
            }
            default: { throw std::runtime_error("Unknown lens type"); }
        }
    }

private:
    Scalar dot(const std::array<Scalar, 3>& a, const std::array<Scalar, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    std::array<Scalar, 3> cross(const std::array<Scalar, 3>& a, const std::array<Scalar, 3>& b) {
        return {{
            a[1] * b[2] - a[2] * b[1],  // x
            a[2] * b[0] - a[0] * b[2],  // y
            a[0] * b[1] - a[1] * b[0]   // z
        }};
    }

    std::array<Scalar, 3> normalise(const std::array<Scalar, 3>& a) {
        Scalar length = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] + a[2]);
        return {{a[0] * length, a[1] * length, a[2] * length}};
    }

    std::pair<Scalar, Scalar> a_cos_theta_plus_b_sin_theta_equals_c(float a, float b, float c) {

        float r = std::sqrt(a * a + b * b);

        if (std::abs(c) <= r) {
            float alpha = std::atan2(b, a);
            float beta  = std::acos(c / r);
            if (beta == 0) {
                return {alpha + beta, alpha + beta};
            }
            else {
                return {alpha + beta, alpha - beta};
            }
        }
        else {
            return {std::numeric_limits<Scalar>::quiet_NaN(), std::numeric_limits<Scalar>::quiet_NaN()};
        }
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
