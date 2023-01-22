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

#ifndef VISUALMESH_UTILITY_FOURCC_HPP
#define VISUALMESH_UTILITY_FOURCC_HPP

#include <string>

namespace visualmesh {

/**
 * @brief Given a fourcc (four character code), in string form convert it into it's uint32_t representation
 *
 * @param code the four characters to convert
 *
 * @return the uint32_t representing this four character code
 */
// We allow c arrays here as it lets us pass in string literals with no copying
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
inline constexpr uint32_t fourcc(const char (&code)[5]) {
    return uint32_t(code[0] | (code[1] << 8) | (code[2] << 16) | (code[3] << 24));
}

/**
 * @brief Given a fourcc (four character code), in uint32_t form convert it into it's four character representation
 *
 * @param code
 * @return std::string
 */
inline std::string fourcc_text(const uint32_t& code) {
    return std::string({char(code & 0xFF), char(code >> 8 & 0xFF), char(code >> 16 & 0xFF), char(code >> 24 & 0xFF)});
}

}  // namespace visualmesh

#endif  // VISUALMESH_UTILITY_FOURCC_HPP
