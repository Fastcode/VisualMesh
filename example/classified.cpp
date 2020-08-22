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

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <system_error>

#include "ArrayPrint.hpp"
#include "Timer.hpp"
#include "dataset.hpp"
#include "draw.hpp"
#include "engine/cpu/engine.hpp"
#include "engine/opencl/engine.hpp"
#include "engine/vulkan/engine.hpp"
#include "geometry/Sphere.hpp"
#include "load_model.hpp"
#include "mesh/network_structure.hpp"
#include "mesh/visualmesh.hpp"
#include "utility/fourcc.hpp"

template <typename Scalar>
using Model  = visualmesh::model::Ring6<Scalar>;
using Scalar = float;

int main() {

    // Input image path
    std::string image_path = "../example/images";
    std::string model_path = "../example/model.yaml";

    std::vector<cv::Scalar> colours = {
      // Ball
      cv::Scalar(0, 0, 255),
      // Goal
      cv::Scalar(0, 255, 255),
      // Field Line
      cv::Scalar(255, 255, 255),
      // Field
      cv::Scalar(0, 255, 0),
      // Unclassified
      cv::Scalar(0, 0, 0),
    };

    // Create the window to show the images
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);

    // Time how long each stage takes
    Timer t;

    // Build our classification network
    visualmesh::NetworkStructure<Scalar> network = load_model<Scalar>(model_path);
    t.measure("Loaded network from YAML file");

    visualmesh::geometry::Sphere<Scalar> sphere(0.0949996);
    visualmesh::VisualMesh<Scalar, Model> mesh(sphere, 0.5, 1.5, 6, 0.5, 20);
    t.measure("Built Visual Mesh");

    visualmesh::engine::cpu::Engine<Scalar> cpu_engine(network);
    t.measure("Loaded CPU Engine");

#if !defined(VISUALMESH_DISABLE_OPENCL)
    visualmesh::engine::opencl::Engine<Scalar> cl_engine(network);
    t.measure("Loaded OpenCL Engine");
#endif  // !defined(VISUALMESH_DISABLE_OPENCL)

#if !defined(VISUALMESH_DISABLE_VULKAN)
    visualmesh::engine::vulkan::Engine<Scalar, false> vk_engine(network);
    t.measure("Loaded Vulkan Engine");
#endif  // !defined(VISUALMESH_DISABLE_VULKAN)

    auto dataset = load_dataset<Scalar>(image_path);
    t.measure("Loaded dataset");

    // Go through all our training data
    std::cerr << "Looping through training data" << std::endl;
    for (const auto& element : dataset) {

        std::cerr << "Processing file " << element.number << std::endl;

        Timer t;

        // Run the classifiers
#if !defined(VISUALMESH_DISABLE_OPENCL)
        {
            t.reset();
            auto classified =
              cl_engine(mesh, element.Hoc, element.lens, element.image.data, visualmesh::fourcc("BGRA"));
            t.measure("\tOpenCL Classified Mesh");
            draw("Image", element.image, classified, colours);
            if (char(cv::waitKey(0)) == 27) break;
        }
#endif  // !defined(VISUALMESH_DISABLE_OPENCL)

#if !defined(VISUALMESH_DISABLE_VULKAN)
        {
            t.reset();
            auto classified =
              vk_engine(mesh, element.Hoc, element.lens, element.image.data, visualmesh::fourcc("BGRA"));
            t.measure("\tVulkan Classified Mesh");
            draw("Image", element.image, classified, colours);
            if (char(cv::waitKey(0)) == 27) break;
        }
#endif  // !defined(VISUALMESH_DISABLE_VULKAN)

        {
            t.reset();
            auto classified =
              cpu_engine(mesh, element.Hoc, element.lens, element.image.data, visualmesh::fourcc("BGRA"));
            t.measure("\tCPU Classified Mesh");
            draw("Image", element.image, classified, colours);
            if (char(cv::waitKey(0)) == 27) break;
        }
    }
}
