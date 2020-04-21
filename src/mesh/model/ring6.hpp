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

#ifndef VISUALMESH_MODEL_RING6_HPP
#define VISUALMESH_MODEL_RING6_HPP

#include <array>

#include "mesh/node.hpp"
#include "ring_base.hpp"

namespace visualmesh {
namespace model {

    /**
     * @brief Model utilising the Ring6 method
     *
     * @tparam Scalar the scalar type used for calculations and storage (normally one of float or double)
     */
    template <typename Scalar>
    struct Ring6 : public RingBase<Scalar, Ring6, 6> {

        /// The number of neighbours that each node in the graph has
        static constexpr size_t N_NEIGHBOURS = 6;

        /**
         * @brief Calculates the neighbour indicies for this ring form
         *
         * @param i the index of the current point in the current ring
         * @param p the number of points in the previous ring
         * @param c the number of points in the current ring
         * @param n the number of points in the next ring
         *
         * @return the list of neighbours of this point in a clockwise order
         */
        static std::array<int, N_NEIGHBOURS> neighbours(const int& i, const int& p, const int& c, const int& n) {

            // Get how far we are through this ring as a value between 0 and 1
            const Scalar f = Scalar(i) / Scalar(c);

            // Left and right is just our index += 1 with wraparound
            const int l = static_cast<int>(i > 0 ? i - 1 : c - 1);
            const int r = static_cast<int>(i + 1 < c ? i + 1 : 0);

            // Top left and top right are the next ring around nearest left and right with wraparound
            const int tl = (static_cast<int>(f * n) % n) + c;
            const int tr = ((static_cast<int>(f * n) + 1) % n) + c;

            // Bottom left and bottom right are the next ring around nearest left and right with wraparound
            const int bl = (static_cast<int>(f * p) % p) - p;
            const int br = ((static_cast<int>(f * p) + 1) % p) - p;

            // The indices of our neighbours presented in a clockwise arrangement relative to the start of our row
            return std::array<int, N_NEIGHBOURS>{{l, tl, tr, r, br, bl}};
        }
    };

}  // namespace model
}  // namespace visualmesh


#endif  // VISUALMESH_MODEL_RING6_HPP
