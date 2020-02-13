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
#include "geometry/Sphere.hpp"
#include "utility/fourcc.hpp"
#include "visualmesh.hpp"

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
  std::vector<std::vector<std::pair<std::vector<std::vector<Scalar>>, std::vector<Scalar>>>> network;
  YAML::Node config = YAML::LoadFile(model_path);
  for (const auto& conv : config) {
    network.emplace_back();
    auto& net_conv = network.back();

    for (const auto& layer : conv) {
      net_conv.emplace_back(layer["weights"].as<std::vector<std::vector<Scalar>>>(),
                            layer["biases"].as<std::vector<Scalar>>());
    }
  }
  t.measure("Loaded network from YAML file");

  visualmesh::geometry::Sphere<Scalar> sphere(0.0949996);
  visualmesh::VisualMesh<Scalar, Model> mesh(sphere, 0.5, 1.5, 6, 0.5, 20);
  t.measure("Built Visual Mesh");

  visualmesh::engine::cpu::Engine<Scalar> cpu_engine(network);
  visualmesh::engine::opencl::Engine<Scalar> cl_engine(network);
  t.measure("Loaded engines");

  auto dataset = load_dataset<Scalar>(image_path);
  t.measure("Loaded dataset");

  // Go through all our training data
  std::cerr << "Looping through training data" << std::endl;
  for (const auto& element : dataset) {

    std::cerr << "Processing file " << element.number << std::endl;

    Timer t;

    // Run the classifiers
    {
      t.reset();
      auto classified = cl_engine(mesh, element.Hoc, element.lens, element.image.data, visualmesh::fourcc("BGRA"));
      t.measure("\tOpenCL Classified Mesh");
      draw("Image", element.image, classified, colours);
      if (char(cv::waitKey(0)) == 27) break;
    }

    {
      t.reset();
      auto classified = cpu_engine(mesh, element.Hoc, element.lens, element.image.data, visualmesh::fourcc("BGRA"));
      t.measure("\tCPU Classified Mesh");
      draw("Image", element.image, classified, colours);
      if (char(cv::waitKey(0)) == 27) break;
    }
  }
}
