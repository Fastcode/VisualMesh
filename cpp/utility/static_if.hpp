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

#ifndef VISUALMESH_UTILITY_STATIC_IF_HPP
#define VISUALMESH_UTILITY_STATIC_IF_HPP

// https://baptiste-wicht.com/posts/2015/07/simulate-static_if-with-c11c14.html
namespace static_if_detail {

struct identity {
    template <typename T>
    T operator()(T&& x) const {
        return std::forward<T>(x);
    }
};

template <bool Cond>
struct statement {
    template <typename F>
    void then(const F& f) {
        f(identity());
    }

    template <typename F>
    void else_(const F&) {}
};

template <>
struct statement<false> {
    template <typename F>
    void then(const F&) {}

    template <typename F>
    void else_(const F& f) {
        f(identity());
    }
};

}  // namespace static_if_detail

template <bool Cond, typename F>
static_if_detail::statement<Cond> static_if(F const& f) {
    static_if_detail::statement<Cond> if_;
    if_.then(f);
    return if_;
}

#endif  // VISUALMESH_UTILITY_STATIC_IF_HPP
