#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Sphere.hpp"
#include "VisualMesh.hpp"
#include "json.hpp"

extern "C" {
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
}

// Generate our transformation matrix
template <typename Scalar>
std::array<std::array<Scalar, 4>, 4> generateHoc(const Scalar& theta,
                                                 const Scalar& phi,
                                                 const Scalar& lambda,
                                                 const Scalar& height) {


    Scalar ct = std::cos(theta);
    Scalar st = std::sin(theta);
    Scalar cp = std::cos(phi);
    Scalar sp = std::sin(phi);
    Scalar cl = std::cos(lambda);
    Scalar sl = std::sin(lambda);


    std::array<std::array<Scalar, 4>, 4> Hoc;

    // Rotation matrix
    Hoc[0][0] = ct * cp;
    Hoc[0][1] = ct * sp;
    Hoc[0][2] = -st;
    Hoc[1][0] = sl * st * cp - cl * sp;
    Hoc[1][1] = sl * st * sp + cl * cp;
    Hoc[1][2] = ct * sl;
    Hoc[2][0] = cl * st * cp + sl * sp;
    Hoc[2][1] = cl * st * sp - sl * cp;
    Hoc[2][2] = ct * cl;

    // Lower row
    Hoc[3][0] = 0;
    Hoc[3][1] = 0;
    Hoc[3][2] = 0;
    Hoc[3][3] = 1;

    // Translation
    Hoc[0][3] = 0;
    Hoc[1][3] = 0;
    Hoc[2][3] = height;

    return Hoc;
}

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

    std::string path = "/Users/trent/Code/VisualMesh/training/raw";

    // Construct our visual mesh
    mesh::Sphere<float> sphere(0, 0.075, 1, 20);
    mesh::VisualMesh<float> mesh(sphere, 0.5, 1.5, 100, M_PI / 1024.0);

    // Go through all our training data
    auto files = listdir(path);
    std::sort(files.begin(), files.end());
    for (auto& p : files) {

        if (p.substr(0, 4) == "meta") {

            auto number = p.substr(4, 7);

            std::cout << number << std::endl;

            // Load our metadata and two images
            nlohmann::json meta;
            std::ifstream(path + "/" + p) >> meta;
            cv::Mat img     = cv::imread(path + "/image" + number + ".png");
            cv::Mat stencil = cv::imread(path + "/stencil" + number + ".png");

            // Construct our rotation matrix from the camera angles
            auto Hoc = generateHoc<float>(meta["camera"]["rotation"][0],
                                          meta["camera"]["rotation"][1],
                                          meta["camera"]["rotation"][2],
                                          meta["camera"]["height"]);

            mesh::VisualMesh<float>::Lens lens;
            if (meta["camera"]["lens"]["type"] == "PERSPECTIVE") {

                // Horizontal field of view
                float h_fov = meta["camera"]["lens"]["fov"];

                lens.type            = mesh::VisualMesh<float>::Lens::RECTILINEAR;
                lens.rectilinear.fov = ;
                lens.rectilinear.focal_length_pixels;
            }
            else {

                lens.type = mesh::VisualMesh<float>::Lens::RADIAL;
            }

            mesh.classify(nullptr, 0, mesh::VisualMesh<Scalar>::FOURCC::RGBA, Hoc, lens);

            cv::namedWindow("Boo", cv::WINDOW_AUTOSIZE);
            cv::imshow("Boo", img);
            cv::waitKey(0);
            cv::imshow("Boo", stencil);
            cv::waitKey(0);

            std::cout << meta << std::endl;

            std::cout << p << std::endl;
        }
    }
}
