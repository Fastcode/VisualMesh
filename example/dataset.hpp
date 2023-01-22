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

#ifndef EXAMPLE_DATASET_HPP
#define EXAMPLE_DATASET_HPP

#include <dirent.h>
#include <sys/types.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

#include "visualmesh/lens.hpp"
#include "visualmesh/utility/math.hpp"

template <typename Scalar>
struct dataset_element {
    std::string number;
    cv::Mat image;
    visualmesh::Lens<Scalar> lens{};
    visualmesh::mat4<Scalar> Hoc{};
};

template <typename Scalar>
std::vector<dataset_element<Scalar>> load_dataset(const std::string& path) {

    auto* dir = ::opendir(path.c_str());

    if (dir != nullptr) {
        std::vector<dataset_element<Scalar>> dataset;
        for (dirent* ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {
            auto file = std::string(ent->d_name);

            if (file.substr(0, 4) == "lens" && !(ent->d_type & DT_DIR)) {

                dataset_element<Scalar> element;

                // Extract the number so we can find the other files
                element.number = file.substr(4, 7);

                // NOLINTNEXTLINE(performance-inefficient-string-concatenation) Performance doesn't matter here
                element.image = cv::imread(path + "/image" + element.number + ".jpg");
                // NOLINTNEXTLINE(performance-inefficient-string-concatenation) Performance doesn't matter here
                YAML::Node lens        = YAML::LoadFile(path + "/" + file);
                std::string projection = lens["projection"].as<std::string>();

                // Convert the image from BGR to BGRA
                cv::cvtColor(element.image, element.image, cv::COLOR_BGR2BGRA);

                // Load the lens parameters
                element.lens.projection = projection == "RECTILINEAR"   ? visualmesh::RECTILINEAR
                                          : projection == "EQUIDISTANT" ? visualmesh::EQUIDISTANT
                                          : projection == "EQUISOLID"   ? visualmesh::EQUISOLID
                                                                        : static_cast<visualmesh::LensProjection>(-1);

                element.lens.dimensions   = {element.image.cols, element.image.rows};
                element.lens.focal_length = lens["focal_length"].as<Scalar>();
                element.lens.centre       = lens["centre"].as<visualmesh::vec2<Scalar>>();
                element.lens.k            = lens["k"].as<visualmesh::vec2<Scalar>>();
                element.lens.fov          = lens["fov"].as<Scalar>();

                // Load the Hoc matrix
                element.Hoc = lens["Hoc"].as<visualmesh::mat4<Scalar>>();

                dataset.push_back(element);
            }
        }
        ::closedir(dir);
        return dataset;
    }

    throw std::system_error(errno, std::system_category(), "Failed to open directory " + path);
}

#endif  // EXAMPLE_DATASET_HPP
