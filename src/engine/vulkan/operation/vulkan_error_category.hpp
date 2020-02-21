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

extern "C" {
#include <vulkan/vulkan.h>
}

#include <system_error>

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {
            enum class vulkan_error_code {
                SUCCESS                              = VK_SUCCESS,
                NOT_READY                            = VK_NOT_READY,
                TIMEOUT                              = VK_TIMEOUT,
                EVENT_SET                            = VK_EVENT_SET,
                EVENT_RESET                          = VK_EVENT_RESET,
                INCOMPLETE                           = VK_INCOMPLETE,
                ERROR_OUT_OF_HOST_MEMORY             = VK_ERROR_OUT_OF_HOST_MEMORY,
                ERROR_OUT_OF_DEVICE_MEMORY           = VK_ERROR_OUT_OF_DEVICE_MEMORY,
                ERROR_INITIALIZATION_FAILED          = VK_ERROR_INITIALIZATION_FAILED,
                ERROR_DEVICE_LOST                    = VK_ERROR_DEVICE_LOST,
                ERROR_MEMORY_MAP_FAILED              = VK_ERROR_MEMORY_MAP_FAILED,
                ERROR_LAYER_NOT_PRESENT              = VK_ERROR_LAYER_NOT_PRESENT,
                ERROR_EXTENSION_NOT_PRESENT          = VK_ERROR_EXTENSION_NOT_PRESENT,
                ERROR_FEATURE_NOT_PRESENT            = VK_ERROR_FEATURE_NOT_PRESENT,
                ERROR_INCOMPATIBLE_DRIVER            = VK_ERROR_INCOMPATIBLE_DRIVER,
                ERROR_TOO_MANY_OBJECTS               = VK_ERROR_TOO_MANY_OBJECTS,
                ERROR_FORMAT_NOT_SUPPORTED           = VK_ERROR_FORMAT_NOT_SUPPORTED,
                ERROR_FRAGMENTED_POOL                = VK_ERROR_FRAGMENTED_POOL,
                ERROR_UNKNOWN                        = VK_ERROR_UNKNOWN,
                ERROR_OUT_OF_POOL_MEMORY             = VK_ERROR_OUT_OF_POOL_MEMORY,
                ERROR_INVALID_EXTERNAL_HANDLE        = VK_ERROR_INVALID_EXTERNAL_HANDLE,
                ERROR_FRAGMENTATION                  = VK_ERROR_FRAGMENTATION,
                ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
                ERROR_SURFACE_LOST_KHR               = VK_ERROR_SURFACE_LOST_KHR,
                ERROR_NATIVE_WINDOW_IN_USE_KHR       = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,
                SUBOPTIMAL_KHR                       = VK_SUBOPTIMAL_KHR,
                ERROR_OUT_OF_DATE_KHR                = VK_ERROR_OUT_OF_DATE_KHR,
                ERROR_INCOMPATIBLE_DISPLAY_KHR       = VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,
                ERROR_VALIDATION_FAILED_EXT          = VK_ERROR_VALIDATION_FAILED_EXT,
                ERROR_INVALID_SHADER_NV              = VK_ERROR_INVALID_SHADER_NV,
                ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT =
                  VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
                ERROR_NOT_PERMITTED_EXT                   = VK_ERROR_NOT_PERMITTED_EXT,
                ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT = VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                UNKNOWN
            };
        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

namespace std {
template <>
struct is_error_condition_enum<visualmesh::engine::vulkan::operation::vulkan_error_code> : public true_type {};
}  // namespace std

namespace visualmesh {
namespace engine {
    namespace vulkan {
        namespace operation {

            class vulkan_error_category_t : public std::error_category {
            public:
                inline virtual const char* name() const noexcept {
                    return "vulkan_error_category";
                }

                inline virtual std::error_condition default_error_condition(int code) const noexcept {
                    using vke = vulkan_error_code;
                    switch (code) {
                        case VK_SUCCESS: return std::error_condition(vke::SUCCESS);
                        case VK_NOT_READY: return std::error_condition(vke::NOT_READY);
                        case VK_TIMEOUT: return std::error_condition(vke::TIMEOUT);
                        case VK_EVENT_SET: return std::error_condition(vke::EVENT_SET);
                        case VK_EVENT_RESET: return std::error_condition(vke::EVENT_RESET);
                        case VK_INCOMPLETE: return std::error_condition(vke::INCOMPLETE);
                        case VK_ERROR_OUT_OF_HOST_MEMORY: return std::error_condition(vke::ERROR_OUT_OF_HOST_MEMORY);
                        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                            return std::error_condition(vke::ERROR_OUT_OF_DEVICE_MEMORY);
                        case VK_ERROR_INITIALIZATION_FAILED:
                            return std::error_condition(vke::ERROR_INITIALIZATION_FAILED);
                        case VK_ERROR_DEVICE_LOST: return std::error_condition(vke::ERROR_DEVICE_LOST);
                        case VK_ERROR_MEMORY_MAP_FAILED: return std::error_condition(vke::ERROR_MEMORY_MAP_FAILED);
                        case VK_ERROR_LAYER_NOT_PRESENT: return std::error_condition(vke::ERROR_LAYER_NOT_PRESENT);
                        case VK_ERROR_EXTENSION_NOT_PRESENT:
                            return std::error_condition(vke::ERROR_EXTENSION_NOT_PRESENT);
                        case VK_ERROR_FEATURE_NOT_PRESENT: return std::error_condition(vke::ERROR_FEATURE_NOT_PRESENT);
                        case VK_ERROR_INCOMPATIBLE_DRIVER: return std::error_condition(vke::ERROR_INCOMPATIBLE_DRIVER);
                        case VK_ERROR_TOO_MANY_OBJECTS: return std::error_condition(vke::ERROR_TOO_MANY_OBJECTS);
                        case VK_ERROR_FORMAT_NOT_SUPPORTED:
                            return std::error_condition(vke::ERROR_FORMAT_NOT_SUPPORTED);
                        case VK_ERROR_FRAGMENTED_POOL: return std::error_condition(vke::ERROR_FRAGMENTED_POOL);
                        case VK_ERROR_UNKNOWN: return std::error_condition(vke::ERROR_UNKNOWN);
                        case VK_ERROR_OUT_OF_POOL_MEMORY: return std::error_condition(vke::ERROR_OUT_OF_POOL_MEMORY);
                        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
                            return std::error_condition(vke::ERROR_INVALID_EXTERNAL_HANDLE);
                        case VK_ERROR_FRAGMENTATION: return std::error_condition(vke::ERROR_FRAGMENTATION);
                        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
                            return std::error_condition(vke::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
                        case VK_ERROR_SURFACE_LOST_KHR: return std::error_condition(vke::ERROR_SURFACE_LOST_KHR);
                        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
                            return std::error_condition(vke::ERROR_NATIVE_WINDOW_IN_USE_KHR);
                        case VK_SUBOPTIMAL_KHR: return std::error_condition(vke::SUBOPTIMAL_KHR);
                        case VK_ERROR_OUT_OF_DATE_KHR: return std::error_condition(vke::ERROR_OUT_OF_DATE_KHR);
                        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
                            return std::error_condition(vke::ERROR_INCOMPATIBLE_DISPLAY_KHR);
                        case VK_ERROR_VALIDATION_FAILED_EXT:
                            return std::error_condition(vke::ERROR_VALIDATION_FAILED_EXT);
                        case VK_ERROR_INVALID_SHADER_NV: return std::error_condition(vke::ERROR_INVALID_SHADER_NV);
                        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
                            return std::error_condition(vke::ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
                        case VK_ERROR_NOT_PERMITTED_EXT: return std::error_condition(vke::ERROR_NOT_PERMITTED_EXT);
                        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
                            return std::error_condition(vke::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
                        default: return std::error_condition(vke::UNKNOWN);
                    }
                }

                inline virtual bool equivalent(const std::error_code& code, int condition) const noexcept {
                    return *this == code.category()
                           && static_cast<int>(default_error_condition(code.value()).value()) == condition;
                }

                inline virtual std::string message(int code) const noexcept {
                    switch (code) {
                        case VK_SUCCESS: return "Success";
                        case VK_NOT_READY: return "Not ready";
                        case VK_TIMEOUT: return "Timeout";
                        case VK_EVENT_SET: return "Event set";
                        case VK_EVENT_RESET: return "Event reset";
                        case VK_INCOMPLETE: return "Incomplete";
                        case VK_ERROR_OUT_OF_HOST_MEMORY: return "Error out of host memory";
                        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "Error out of device memory";
                        case VK_ERROR_INITIALIZATION_FAILED: return "Error initialization failed";
                        case VK_ERROR_DEVICE_LOST: return "Error device lost";
                        case VK_ERROR_MEMORY_MAP_FAILED: return "Error memory map failed";
                        case VK_ERROR_LAYER_NOT_PRESENT: return "Error layer not present";
                        case VK_ERROR_EXTENSION_NOT_PRESENT: return "Error extension not present";
                        case VK_ERROR_FEATURE_NOT_PRESENT: return "Error feature not present";
                        case VK_ERROR_INCOMPATIBLE_DRIVER: return "Error incompatible driver";
                        case VK_ERROR_TOO_MANY_OBJECTS: return "Error too many objects";
                        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "Error format not supported";
                        case VK_ERROR_FRAGMENTED_POOL: return "Error fragmented pool";
                        case VK_ERROR_OUT_OF_POOL_MEMORY: return "Error out of pool memory";
                        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "Error invalid external handle";
                        case VK_ERROR_FRAGMENTATION: return "Error fragmentation";
                        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "Error invalid opaque capture address";
                        case VK_ERROR_SURFACE_LOST_KHR: return "Error surface lost khr";
                        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "Error native window in use khr";
                        case VK_SUBOPTIMAL_KHR: return "Suboptimal khr";
                        case VK_ERROR_OUT_OF_DATE_KHR: return "Error out of date khr";
                        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "Error incompatible display khr";
                        case VK_ERROR_VALIDATION_FAILED_EXT: return "Error validation failed ext";
                        case VK_ERROR_INVALID_SHADER_NV: return "Error invalid shader nv";
                        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
                            return "Error invalid drm format modifier plane layout ext";
                        case VK_ERROR_NOT_PERMITTED_EXT: return "Error not permitted ext";
                        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
                            return "Error full screen exclusive mode lost ext";
                        case VK_ERROR_UNKNOWN:
                        default: return "Unknown error";
                    }
                }
            };

            inline const std::error_category& vulkan_error_category() {
                static vulkan_error_category_t instance;
                return instance;
            }

            inline std::error_condition make_error_condition(vulkan_error_code e) {
                return std::error_condition(static_cast<int>(e), vulkan_error_category());
            }

        }  // namespace operation
    }      // namespace vulkan
}  // namespace engine
}  // namespace visualmesh

#endif  // VISUALMESH_ENGINE_VULKAN_VULKAN_ERROR_CATEGORY_HPP
