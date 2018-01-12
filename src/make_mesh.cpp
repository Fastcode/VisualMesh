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

    // Input image path
    std::string image_path = "/Users/trent/Code/VisualMesh/training/raw";

    // Construct our VisualMesh
    std::cerr << "Building VisualMesh" << std::endl;
    const int n_intersections = 4;
    mesh::Sphere<float> sphere(0, 0.075, n_intersections, 10);
    mesh::VisualMesh<float> mesh(sphere, 0.5, 1.5, 100, M_PI / 1024.0);
    std::cerr << "Finished building VisualMesh" << std::endl;

    // Go through all our training data
    std::cerr << "Looping through data" << std::endl;
    auto files = listdir(image_path);
    std::sort(files.begin(), files.end());

    for (const auto& p : files) {
        if (p.substr(0, 4) == "meta") {
            try {

                // Extract the number so we can find the other files
                auto number = p.substr(4, 7);

                std::cerr << "Processing file " << number << std::endl;

                // Load our metadata and two images
                nlohmann::json meta;
                std::ifstream(image_path + "/" + p) >> meta;

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
                lens.dimensions = {{1280, 1024}};
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
                auto projection = mesh.project(Hoc, lens);

                // Get our relevant data
                std::vector<std::array<int, 6>>& mesh_neighbours = projection.neighbourhood;
                std::vector<std::array<int, 2>> mesh_px          = projection.pixel_coordinates;

                // Strip our pixel coordinates down to the ones on the screen
                std::vector<std::array<int, 2>> pixel_coordinates;
                std::vector<int> pixel_coordinates_index;
                px_idx.reserve(mesh_px.size());
                px_val.reserve(mesh_px.size());

                for (const int i = 0; i < mesh_px.size(); ++mesh_px) {
                    const auto& px = mesh_px[i];

                    // Only copy across if our pixel is on the screen
                    if (px[0] > 0 && px[0] < lens.dimensions[0] && px[1] > 0 && px[1] > lens.dimensions[1]) {
                        px_idx.push_back(i);
                        px_val.push_back(px)
                    }
                }

                // Make a reverse lookup list
                std::vector<int> rev_idx(mesh_neighbours.size(), px_val.size());
                for (int i = 0; i < px_idx; ++i) {
                    rev_idx[px_idx[i]] = i;
                }

                // Work out how big the ball is
                const float angular = (2 * std::asin(0.075 / 10.0)) / float(n_intersections);
                // Work out how many pixels the ball takes up
                float ball_px_size = lens.focal_length * std::tan(angular);

                // Make a file
                std::ofstream out(fmt::format("{}/{}mesh{}.bin", image_path, n_intersections, number),
                                  std::ios::trunc | std::ios::binary);

                // Output the ball pixel size for hexidense training
                out.write(reinterpret_cast<const char*>(&ball_px_size), sizeof(ball_px_size));

                // Output our pixel coordinates
                out.write(reinterpret_cast<const char*>(pixel_coordinates.data()),
                          pixel_coordinates.size() * sizeof(pixel_coordinates[0]));

                for (int32_t i = 0; i < neighbourhood.size(); ++i) {

                    // Output our own index and then our neighbours indices
                    out.write(reinterpret_cast<const char*>(&i), sizeof(int32_t));
                    out.write(reinterpret_cast<const char*>(neighbourhood[i].data()), sizeof(neighbourhood[i]));
                }

                out.close();
            }
            // If there is an exception just skip the file
            catch (const std::exception& ex) {
            }
        }
    }
}
