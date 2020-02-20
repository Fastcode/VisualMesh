/*
 * Copyright (C) 2017-2018 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_ENGINE_VULKAN_VULKAN_ERROR_CATEGORY_HPP
#define VISUALMESH_ENGINE_VULKAN_VULKAN_ERROR_CATEGORY_HPP

#include <system_error>
#include <vulkan/vulkan.hpp>

namespace std {
template <>
struct is_error_condition_enum<vk::Result> : public true_type {};
}  // namespace std

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            class vulkan_error_category_t : public std::error_category {
            public:
                virtual const char* name() const noexcept;

                virtual std::error_condition default_error_condition(int code) const noexcept;

                virtual bool equivalent(const std::error_code& code, int condition) const noexcept;
                bool equivalent(const std::error_code& code, const vk::Result& condition) const noexcept;

                virtual std::string message(int code) const noexcept;
                std::string message(const vk::Result& code) const noexcept;
            };

            inline const std::error_category& vulkan_error_category() {
                static vulkan_error_category_t instance;
                return instance;
            }

            inline std::error_condition make_error_condition(const vk::Result& e) {
                return std::error_condition(static_cast<int>(e), vulkan_error_category());
            }

            const char* vulkan_error_category_t::name() const noexcept {
                return "vulkan_error_category";
            }

            std::error_condition vulkan_error_category_t::default_error_condition(int code) const noexcept {
                return std::error_condition(static_cast<vk::Result>(code));
            }

            bool vulkan_error_category_t::equivalent(const std::error_code& code, int condition) const noexcept {
                return *this == code.category()
                       && static_cast<int>(default_error_condition(code.value()).value()) == condition;
            }

            bool vulkan_error_category_t::equivalent(const std::error_code& code, const vk::Result& condition) const
              noexcept {
                return *this == code.category()
                       && static_cast<vk::Result>(default_error_condition(code.value()).value()) == condition;
            }

            std::string vulkan_error_category_t::message(const vk::Result& code) const noexcept {
                return vk::to_string(code);
            }

            std::string vulkan_error_category_t::message(int code) const noexcept {
                return message(static_cast<vk::Result>(code));
            }

        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_VULKAN_ERROR_CATEGORY_HPP
