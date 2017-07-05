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
    struct alignas(16) Node {
        /// The unit vector in the direction for this node
        Scalar ray[4];
        /// Relative indicies to the linked hexagon nodes in the LUT ordered TL, TR, L, R, BL, BR,
        int neighbours[6];
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
        : min_height(min_height)
        , max_height(max_height)
        , height_resolution(height_resolution)
        , min_angular_res(min_angular_res) {

        for (Scalar h = min_height; h < max_height; h += (max_height - min_height) / height_resolution) {


            // This is a list of phi values along with the delta theta values associated with them
            std::vector<std::pair<Scalar, size_t>> phis;

            // Loop from directly down up to the horizon (if phi is nan it will stop)
            // So we don't have a single point at the base, we move half a jump forward
            for (Scalar phi = shape.phi(0, h) / 2.0; phi < M_PI_2;) {

                // Calculate our theta
                Scalar theta = std::max(shape.theta(phi, h), min_angular_res);

                if (!isnan(theta)) {
                    // Push back the phi, and the number of whole shapes we can fit
                    phis.emplace_back(phi, size_t(std::ceil(2.0 * M_PI / theta)));
                }

                // Move to our next phi
                phi = std::max(phi + min_angular_res, shape.phi(phi, h));
            }


            // Loop from directly up down to the horizon (if phi is nan it will stop)
            for (Scalar phi = shape.phi(M_PI, h) / 2.0; phi > M_PI_2;) {

                // Calculate our theta
                Scalar theta = std::max(shape.theta(phi, h), min_angular_res);

                if (!isnan(theta)) {
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
            std::vector<std::pair<size_t, size_t>> rows;
            rows.reserve(phis.size());

            // Loop through our LUT and calculate our left and right neighbours
            for (const auto& v : phis) {

                // Get our phi and delta theta values for a clean circle
                const auto& phi   = v.first;
                const auto& steps = v.second;
                Scalar dtheta     = (2.0 * M_PI) / steps;

                // We will use the start position of each row later for linking the graph
                rows.emplace_back(lut.size(), lut.size() + steps);

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
                int prev_size    = r > 0 ? prev.second - prev.first : 0;
                int current_size = current.second - current.first;
                int next_size    = r < rows.size() - 1 ? next.second - next.first : 0;


                // Go through all the nodes on our current row
                for (size_t i = current.first; i < current.second; ++i) {

                    // Grab our current node
                    auto& node = lut[i];


                    // Find where we are in our row
                    Scalar pos = Scalar(i - current.first) / Scalar(current_size);

                    // Link to our previous row if something is available
                    if (r > 0) {

                        // Work out if we are closer to the left or right and make an offset var for it
                        // Note this bool is used like a bool and int. It is 0 when we should access TR first
                        // and 1 when we should access TL first
                        bool left = pos > 0.5;

                        // Get our closest neighbour on the previous row and use it to work out where the other one is
                        // This will be the Right element when < 0.5 and Left when > 0.5
                        size_t o1 = prev.first + std::floor(pos * prev_size + !left);  // Use `left` to add one to one
                        size_t o2 = o1 + lut[o1].neighbours[2 + left];                 // But not the other

                        // Now use these to set our TL and TR neighbours
                        node.neighbours[0] = (left ? o1 : o2) - i;
                        node.neighbours[1] = (left ? o2 : o1) - i;
                    }
                    // If we don't have a previous row, we are at the end, so instead link to our own row
                    else {
                        // Work out which two points are on the opposite side to us
                        size_t index = i - current.first + (current_size / 2);

                        // Link to them
                        node.neighbours[0] = current.first + (index % current_size);
                        node.neighbours[1] = current.first + ((index + 1) % current_size);
                    }

                    // Link to our next row if something is available
                    if (r < rows.size() - 1) {

                        // Work out if we are closer to the left or right and make an offset var for it
                        // Note this bool is used like a bool and int. It is 0 when we should access TR first
                        // and 1 when we should access TL first
                        bool left = pos > 0.5;

                        // Get our closest neighbour on the previous row and use it to work out where the other one is
                        // This will be the Right element when < 0.5 and Left when > 0.5
                        size_t o1 = next.first + std::floor(pos * next_size + !left);  // Use `left` to add one to one
                        size_t o2 = o1 + lut[o1].neighbours[2 + left];                 // But not the other

                        // Now use these to set our TL and TR neighbours
                        node.neighbours[4] = (left ? o1 : o2) - i;
                        node.neighbours[5] = (left ? o2 : o1) - i;
                    }
                    // If we don't have a previous row, we are at the end, so instead link to our own row
                    else {
                        // Work out which two points are on the opposite side to us
                        size_t index = i - current.first + (current_size / 2);

                        // Link to them
                        node.neighbours[4] = current.first + (index % current_size);
                        node.neighbours[5] = current.first + ((index + 1) % current_size);
                    }
                }
            }

            // Insert our constructed lut into the lookup
            luts.insert(std::make_pair(h, std::move(lut)));
        }
    }

    // std::vector<std::pair<iterator, iterator>> lookup() const {}

    const std::vector<Node>& data(const Scalar& height) const {

        return luts.lower_bound(height)->second;
    }

private:
    /// A map from heights to visual mesh tables
    std::map<Scalar, std::vector<Node>> luts;

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
