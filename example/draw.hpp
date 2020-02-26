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

#ifndef EXAMPLE_DRAW_HPP
#define EXAMPLE_DRAW_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mesh/classified_mesh.hpp"
#include "mesh/projected_mesh.hpp"

template <typename Scalar, size_t N_NEIGHBOURS>
void draw(const std::string& window,
          const cv::Mat& image,
          const visualmesh::ClassifiedMesh<Scalar, N_NEIGHBOURS>& mesh,
          const std::vector<cv::Scalar>& colours) {

    const auto& classifications   = mesh.classifications;
    const auto& pixel_coordinates = mesh.pixel_coordinates;
    const auto& neighbourhood     = mesh.neighbourhood;
    const size_t num_classes      = colours.size();

    cv::Mat scratch = image.clone();

    for (unsigned int i = 0; i < pixel_coordinates.size(); ++i) {
        cv::Point p1(pixel_coordinates[i][0], pixel_coordinates[i][1]);

        // Work out what colour based on mixing
        const Scalar* cl = classifications.data() + (i * num_classes);
        cv::Scalar colour(0, 0, 0);

        for (unsigned int i = 0; i < colours.size(); ++i) {
            colour += colours[i] * cl[i];
        }

        for (const auto& n : neighbourhood[i]) {
            if (n < static_cast<int>(pixel_coordinates.size())) {
                cv::Point p2(pixel_coordinates[n][0], pixel_coordinates[n][1]);
                cv::Point p2x = p1 + ((p2 - p1) * 0.5);
                cv::line(scratch, p1, p2x, colour, 1);
            }
        }
    }

    cv::imshow(window, scratch);
}


template <typename Scalar, size_t N_NEIGHBOURS>
void draw(const std::string& window,
          const cv::Mat& image,
          const visualmesh::ProjectedMesh<Scalar, N_NEIGHBOURS>& mesh,
          const cv::Scalar& colour) {

    const auto& pixel_coordinates = mesh.pixel_coordinates;
    const auto& neighbourhood     = mesh.neighbourhood;

    cv::Mat scratch = image.clone();

    for (unsigned int i = 0; i < pixel_coordinates.size(); ++i) {
        cv::Point p1(pixel_coordinates[i][0], pixel_coordinates[i][1]);

        for (const auto& n : neighbourhood[i]) {
            if (n < static_cast<int>(pixel_coordinates.size())) {
                cv::Point p2(pixel_coordinates[n][0], pixel_coordinates[n][1]);
                cv::Point p2x = p1 + ((p2 - p1) * 0.5);
                cv::line(scratch, p1, p2x, colour, 1, cv::LINE_AA);
            }
        }
    }

    cv::imshow(window, scratch);
}

#endif  // EXAMPLE_DRAW_HPP
