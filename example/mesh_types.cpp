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

#include "dataset.hpp"
#include "draw.hpp"
#include "engine/cpu/engine.hpp"
#include "engine/opencl/engine.hpp"
#include "geometry/Sphere.hpp"
#include "mesh/model/radial4.hpp"
#include "mesh/model/radial6.hpp"
#include "mesh/model/radial8.hpp"
#include "mesh/model/ring4.hpp"
#include "mesh/model/ring6.hpp"
#include "mesh/model/ring8.hpp"
#include "mesh/model/xgrid4.hpp"
#include "visualmesh.hpp"

using Scalar = float;

int main() {
    std::string image_path = "../example/images";

    // Create windows
    cv::namedWindow("Ring 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ring 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ring 8", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Radial 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Radial 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Radial 8", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("X Grid 4", cv::WINDOW_AUTOSIZE);

    visualmesh::geometry::Sphere<Scalar> sphere(0.0949996);

    // Build meshes
    visualmesh::VisualMesh<Scalar, visualmesh::model::Radial4> radial4(sphere, 0.5, 1.5, 6, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Radial6> radial6(sphere, 0.5, 1.5, 6, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Radial8> radial8(sphere, 0.5, 1.5, 6, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring4> ring4(sphere, 0.5, 1.5, 6, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring6> ring6(sphere, 0.5, 1.5, 6, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring8> ring8(sphere, 0.5, 1.5, 6, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XGrid4> xgrid4(sphere, 0.5, 1.5, 6, 0.5, 20);

    // Build engines
    visualmesh::engine::opencl::Engine<Scalar> engine;

    // Load dataset
    auto dataset = load_dataset<Scalar>(image_path);

    for (const auto& element : dataset) {
        draw("Radial 4", element.image, engine.project(radial4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Radial 6", element.image, engine.project(radial6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Radial 8", element.image, engine.project(radial8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Ring 4", element.image, engine.project(ring4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Ring 6", element.image, engine.project(ring6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Ring 8", element.image, engine.project(ring8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("X Grid 4", element.image, engine.project(xgrid4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        if (char(cv::waitKey(0)) == 27) break;
    }

    // Run through the images doing projections
}
