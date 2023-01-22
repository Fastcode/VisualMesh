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
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "Timer.hpp"
#include "dataset.hpp"
#include "load_model.hpp"
#include "visualmesh/engine/cpu/engine.hpp"
#include "visualmesh/engine/opencl/engine.hpp"
#include "visualmesh/engine/vulkan/engine.hpp"
#include "visualmesh/geometry/Sphere.hpp"
#include "visualmesh/model/ring6.hpp"
#include "visualmesh/network_structure.hpp"
#include "visualmesh/utility/fourcc.hpp"
#include "visualmesh/visualmesh.hpp"

using Scalar = float;

template <typename Engine, typename Mesh>
class Benchmarker {
public:
    Benchmarker(const visualmesh::NetworkStructure<Scalar>& network,
                const Mesh& mesh,
                const std::vector<dataset_element<Scalar>>& dataset,
                const int& loops)
      : total(0), engine(network), mesh(mesh), dataset(dataset), loops(loops) {}

    void start() {
        thread = std::thread([this] {
            for (int i = 0; i < loops; ++i) {
                for (const auto& element : dataset) {
                    // steady_clock::time_point start = steady_clock::now();
                    engine(mesh, element.Hoc, element.lens, element.image.data, visualmesh::fourcc("BGRA"));
                    // total += steady_clock::now() - start;
                }
            }
        });
    }

    void join() {
        thread.join();
    }

    std::chrono::steady_clock::duration total;

private:
    Engine engine;
    const Mesh& mesh;
    const std::vector<dataset_element<Scalar>>& dataset;
    int loops;
    std::thread thread;
};

template <typename Engine, typename Mesh>
void benchmark(const visualmesh::NetworkStructure<Scalar>& network,
               const std::vector<dataset_element<Scalar>>& dataset,
               const Mesh& mesh,
               const int& loops,
               const unsigned int& parallelity = 1) {

    using namespace std::chrono;  // NOLINT(google-build-using-namespace) fine in function scope
    Timer t;
    // Build engines
    std::vector<Benchmarker<Engine, Mesh>> benchmarkers;
    benchmarkers.reserve(parallelity);
    for (unsigned int t = 0; t < parallelity; ++t) {
        benchmarkers.emplace_back(network, mesh, dataset, loops);
    }
    t.measure("Built benchmarkers");

    steady_clock::time_point start = steady_clock::now();
    for (auto& b : benchmarkers) {
        b.start();
    }

    // Wait until all the threads are done
    for (auto& b : benchmarkers) {
        b.join();
    }
    steady_clock::time_point end = steady_clock::now();

    steady_clock::duration total = end - start;
    double fps = double(parallelity * loops * dataset.size()) / duration_cast<duration<double>>(total).count();
    std::cout << "FPS: " << fps << std::endl;
}

// NOLINTNEXTLINE(bugprone-exception-escape) This is debugging code, I would prefer exceptions crash the program
int main() {
    std::string image_path = "../example/images";
    std::string model_path = "../example/model.yaml";

    Timer t;

    // Load the classification network
    visualmesh::NetworkStructure<Scalar> network = load_model<Scalar>(model_path);
    t.measure("Loaded network from YAML file");

    // Build Mesh
    visualmesh::geometry::Sphere<Scalar> sphere(0.0949996);
    visualmesh::VisualMesh<Scalar, visualmesh::model::Ring6> mesh(sphere, 0.5, 1.5, 6, 0.5, 20);
    t.measure("Built mesh");


    // Load dataset
    auto dataset = load_dataset<Scalar>(image_path);
    t.measure("Loaded dataset");

// Do benchmarks
#if !defined(VISUALMESH_DISABLE_OPENCL)
    std::cout << "Benchmarking OpenCL Engine" << std::endl;
    benchmark<visualmesh::engine::opencl::Engine<Scalar>>(network, dataset, mesh, 100, 4);
#endif  // !defined(VISUALMESH_DISABLE_OPENCL)

#if !defined(VISUALMESH_DISABLE_VULKAN)
    std::cout << "Benchmarking Vulkan Engine" << std::endl;
    benchmark<visualmesh::engine::vulkan::Engine<Scalar>>(network, dataset, mesh, 100, 4);
#endif  // !defined(VISUALMESH_DISABLE_VULKAN)

    std::cout << "Benchmarking CPU Engine" << std::endl;
    benchmark<visualmesh::engine::cpu::Engine<Scalar>>(network, dataset, mesh, 2, std::thread::hardware_concurrency());
}
