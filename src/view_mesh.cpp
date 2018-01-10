#include <dirent.h>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ArrayPrint.hpp"
#include "Sphere.hpp"
#include "Timer.hpp"
#include "VisualMesh.hpp"
#include "json.hpp"

// List the contents of a directory
std::vector<std::string> listdir(const std::string& path) {

    auto dir = opendir(path.c_str());
    std::vector<std::string> result;

    if (dir != nullptr) {
        for (dirent* ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {

            auto file = std::string(ent->d_name);

            if (file == "." || file == "..") {
                continue;
            }

            if (ent->d_type & DT_DIR) {
                result.push_back(file + "/");
            }
            else {
                result.push_back(file);
            }
        }

        closedir(dir);
    }
    else {
        // TODO Throw an error or something
    }

    return result;
}

int main() {

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);

    // Input image path
    std::string image_path = "/Users/trent/Code/VisualMesh/training/raw";
    std::string model_path = "/Users/trent/Code/VisualMesh/4_4_4_4_2.yaml";

    // Construct our VisualMesh
    std::cerr << "Building VisualMesh" << std::endl;
    mesh::Sphere<float> sphere(0, 0.075, 4, 10);
    mesh::VisualMesh<float> mesh(sphere, 0.5, 1.5, 100, M_PI / 1024.0);
    std::cerr << "Finished building VisualMesh" << std::endl;

    // Build our classification network
    std::vector<std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>>> network;

    YAML::Node config = YAML::LoadFile(model_path);

    for (const auto& conv : config) {

        // New conv layer
        network.emplace_back();
        auto& net_conv = network.back();

        for (const auto& layer : conv) {

            // New network layer
            net_conv.emplace_back();
            auto& net_layer = net_conv.back();

            // Copy across our weights
            for (const auto& l : layer["weights"]) {
                net_layer.first.emplace_back();
                auto& weight = net_layer.first.back();

                for (const auto& v : l) {
                    weight.push_back(v.as<float>());
                }
            }

            // Copy across our biases
            for (const auto& v : layer["biases"]) {
                net_layer.second.push_back(v.as<float>());
            }
        }
    }

    auto classifier = mesh.make_classifier(network);

    // Go through all our training data
    std::cerr << "Looping through training data" << std::endl;
    auto files = listdir(image_path);
    std::sort(files.begin(), files.end());

    for (const auto& p : files) {
        if (p.substr(0, 4) == "meta") {

            // Extract the number so we can find the other files
            auto number = p.substr(4, 7);

            std::cerr << "Processing file " << number << std::endl;

            Timer t;

            // Load our metadata and two images
            nlohmann::json meta;
            std::ifstream(image_path + "/" + p) >> meta;
            cv::Mat img     = cv::imread(image_path + "/image" + number + ".png", CV_LOAD_IMAGE_UNCHANGED);
            cv::Mat stencil = cv::imread(image_path + "/stencil" + number + ".png");

            t.measure("\tLoaded files");

            // Construct our rotation matrix from the camera angles
            auto r       = meta["camera"]["rotation"];
            float height = meta["camera"]["height"];

            // Oh no! the coordinate systems are wrong!
            // We are expecting
            //      x forward
            //      y to the left
            //      z up
            // However blender's camera objects have
            //      z facing away from the object,
            //      y up
            //      z to the right
            // So to fix this we have to make x = -z, y = -x, z = y (swap cols)
            std::array<std::array<float, 4>, 4> Hoc = {{
                {{-float(r[0][2]), -float(r[0][0]), r[0][1], 0}},
                {{-float(r[1][2]), -float(r[1][0]), r[1][1], 0}},
                {{-float(r[2][2]), -float(r[2][0]), r[2][1], height}},
                {{0, 0, 0, 1}},
            }};

            // Make our lens object
            mesh::VisualMesh<float>::Lens lens;
            lens.dimensions = {{img.size().width, img.size().height}};
            if (meta["camera"]["lens"]["type"] == "PERSPECTIVE") {

                // Horizontal field of view
                float h_fov = meta["camera"]["lens"]["fov"];

                // Construct rectilinear projection
                lens.projection   = mesh::VisualMesh<float>::Lens::RECTILINEAR;
                lens.fov          = h_fov;
                lens.focal_length = (lens.dimensions[0] * 0.5) / std::tan(h_fov * 0.5);
            }
            else if (meta["camera"]["lens"]["type"] == "FISHEYE") {
                float fov             = meta["camera"]["lens"]["fov"];
                float height_mm       = meta["camera"]["lens"]["sensor_height"];
                float width_mm        = meta["camera"]["lens"]["sensor_width"];
                float focal_length_mm = meta["camera"]["lens"]["focal_length"];

                // Get conversion from mm to pixels
                float sensor_density = lens.dimensions[0] / width_mm;

                // Blender was rendered with an equisolid lens type
                lens.projection   = mesh::VisualMesh<float>::Lens::EQUISOLID;
                lens.fov          = fov;
                lens.focal_length = focal_length_mm * sensor_density;
            }

            // Run our classifier
            auto classified = classifier(img.data, mesh::VisualMesh<float>::BGRA, Hoc, lens);

            auto& neighbourhood                               = classified.first.neighbourhood;
            std::vector<std::array<int, 2>> pixel_coordinates = classified.first.pixel_coordinates;

            t.measure("\tClassified Mesh");

            cv::imshow("Image", img);

            cv::Mat scratch = img.clone();

            // Wait for esc key
            if (char(cv::waitKey(0)) == 27) break;

            for (int i = 0; i < pixel_coordinates.size(); ++i) {
                cv::Point p1(pixel_coordinates[i][0], pixel_coordinates[i][1]);

                // cv::Scalar colour(uint8_t(classified[i][0] * 255), 0, uint8_t(classified[i][1] * 255));
                cv::Scalar colour(classified.second[i][0] > 0.5 ? 255 : 0, 0, classified.second[i][1] >= 0.5 ? 255 : 0);

                for (const auto& n : neighbourhood[i]) {
                    if (n < pixel_coordinates.size()) {
                        cv::Point p2(pixel_coordinates[n][0], pixel_coordinates[n][1]);
                        cv::Point p2x = p1 + ((p2 - p1) * 0.5);
                        cv::line(scratch, p1, p2x, colour, 1);
                    }
                }
            }

            cv::imshow("Image", scratch);
            // Wait for esc key
            if (char(cv::waitKey(0)) == 27) break;


            cv::imshow("Image", img);
            // Wait for esc key
            if (char(cv::waitKey(0)) == 27) break;
        }
    }
}
