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

#include <array>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "draw.hpp"
#include "visualmesh/geometry/Sphere.hpp"
#include "visualmesh/model/nmgrid4.hpp"
#include "visualmesh/model/nmgrid6.hpp"
#include "visualmesh/model/nmgrid8.hpp"
#include "visualmesh/model/radial4.hpp"
#include "visualmesh/model/radial6.hpp"
#include "visualmesh/model/radial8.hpp"
#include "visualmesh/model/ring4.hpp"
#include "visualmesh/model/ring6.hpp"
#include "visualmesh/model/ring8.hpp"
#include "visualmesh/model/xmgrid4.hpp"
#include "visualmesh/model/xmgrid6.hpp"
#include "visualmesh/model/xmgrid8.hpp"
#include "visualmesh/model/xygrid4.hpp"
#include "visualmesh/model/xygrid6.hpp"
#include "visualmesh/model/xygrid8.hpp"
#include "visualmesh/visualmesh.hpp"

using Scalar = float;

// Use OpenCL engine if available, then Vulkan then CPU
#if defined(VISUALMESH_ENABLE_OPENCL)

#include "visualmesh/engine/opencl/engine.hpp"
template <typename Scalar>
using Engine = visualmesh::engine::opencl::Engine<Scalar>;

#elif defined(VISUALMESH_ENABLE_VULKAN)

#include "visualmesh/engine/opencl/engine.hpp"
template <typename Scalar>
using Engine = visualmesh::engine::vulkan::Engine<Scalar, false>;

#else

#include "visualmesh/engine/cpu/engine.hpp"
template <typename Scalar>
using Engine = visualmesh::engine::cpu::Engine<Scalar>;

#endif

int main() {
    std::string image_path = "../example/images";

    // Create windows
    cv::namedWindow("Radial 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Radial 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Radial 8", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ring 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ring 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ring 8", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XM Grid 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XM Grid 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XM Grid 8", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XY Grid 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XY Grid 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("XY Grid 8", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("NM Grid 4", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("NM Grid 6", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("NM Grid 8", cv::WINDOW_AUTOSIZE);

    visualmesh::geometry::Sphere<Scalar> sphere(0.0949996);

    // Build meshes
    visualmesh::VisualMesh<Scalar, visualmesh::model::Radial4> radial4(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Radial6> radial6(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Radial8> radial8(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring4> ring4(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring6> ring6(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring8> ring8(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XMGrid4> xmgrid4(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XMGrid6> xmgrid6(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XMGrid8> xmgrid8(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XYGrid4> xygrid4(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XYGrid6> xygrid6(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::XYGrid8> xygrid8(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::NMGrid4> nmgrid4(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::NMGrid6> nmgrid6(sphere, 0.5, 1.5, 4, 0.5, 20);
    visualmesh::VisualMesh<Scalar, visualmesh::model::NMGrid8> nmgrid8(sphere, 0.5, 1.5, 4, 0.5, 20);

    // Build engines
    Engine<Scalar> engine;

    // Load dataset
    auto dataset = load_dataset<Scalar>(image_path);

    for (const auto& element : dataset) {
        draw("Radial 4", element.image, engine(radial4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Radial 6", element.image, engine(radial6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Radial 8", element.image, engine(radial8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Ring 4", element.image, engine(ring4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Ring 6", element.image, engine(ring6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("Ring 8", element.image, engine(ring8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("XM Grid 4", element.image, engine(xmgrid4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("XM Grid 6", element.image, engine(xmgrid6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("XM Grid 8", element.image, engine(xmgrid8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("XY Grid 4", element.image, engine(xygrid4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("XY Grid 6", element.image, engine(xygrid6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("XY Grid 8", element.image, engine(xygrid8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("NM Grid 4", element.image, engine(nmgrid4, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("NM Grid 6", element.image, engine(nmgrid6, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        draw("NM Grid 8", element.image, engine(nmgrid8, element.Hoc, element.lens), cv::Scalar(255, 255, 255));
        if (char(cv::waitKey(0)) == 27) break;
    }

    // Run through the images doing projections
}
