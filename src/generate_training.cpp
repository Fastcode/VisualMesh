#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Sphere.hpp"
#include "VisualMesh.hpp"
#include "json.hpp"

extern "C" {
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
}

#include "ArrayPrint.hpp"

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

    // OpenCV window for displaying stuff
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);

    std::string path = "/Users/trent/Code/VisualMesh/training/raw";

    // Construct our VisualMesh
    std::cerr << "Building VisualMesh" << std::endl;
    mesh::Sphere<float> sphere(0, 0.075, 3, 20);
    mesh::VisualMesh<float> mesh(sphere, 0.5, 1.5, 100, M_PI / 1024.0);
    std::cerr << "Finished building VisualMesh" << std::endl;

    // Go through all our training data
    std::cerr << "Looping through training data" << std::endl;
    auto files = listdir(path);
    std::sort(files.begin(), files.end());
    for (const auto& p : files) {

        if (p.substr(0, 4) == "meta") {

            auto number = p.substr(4, 7);

            std::cerr << "Processing file " << number << std::endl;

            // Load our metadata and two images
            nlohmann::json meta;
            std::ifstream(path + "/" + p) >> meta;
            cv::Mat img     = cv::imread(path + "/image" + number + ".png");
            cv::Mat stencil = cv::imread(path + "/stencil" + number + ".png");

            std::cerr << "\tLoaded files" << std::endl;

            cv::imshow("Image", img);
            cv::waitKey(0);

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
            lens.dimensions = {img.size().width, img.size().height};
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

            std::cerr << "\tBuilt lens geometry" << std::endl;

            // Project our visual mesh coordinates
            auto projection = mesh.project_mesh(Hoc, lens);

            // DEBUGGING, DRAW THE PROJECTION
            for (int i = 0; i < projection.pixel_coordinates.size(); ++i) {

                cv::Point p1(projection.pixel_coordinates[i][0], projection.pixel_coordinates[i][1]);

                for (int j = 0; j < 6; ++j) {

                    projection.pixel_coordinates[i];

                    const auto& neighbour = projection.neighbourhood[i][j];

                    if (neighbour >= 0) {
                        cv::Point p2(projection.pixel_coordinates[neighbour][0],
                                     projection.pixel_coordinates[neighbour][1]);
                        cv::line(img, p1, p2, cv::Scalar(255, 255, 255), 1);
                    }
                }
            }

            cv::imshow("Image", img);
            cv::imwrite("mesh.jpg", img);
            cv::waitKey(0);
        }
    }
}
