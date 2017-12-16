#include <dirent.h>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/stat.h>
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

void save_data(std::ofstream& out,
               const mesh::VisualMesh<float>::ProjectedMesh& mesh,
               const cv::Mat& img,
               const cv::Mat& stencil) {

    // Make a rectangle for testing if points are on the image
    cv::Rect rect(cv::Point(), img.size());

    // Write the number of points in the mesh
    int32_t size = int32_t(mesh.pixel_coordinates.size());

    out.write(reinterpret_cast<const char*>(&size), sizeof(int32_t));

    // Write the neighbourhood information
    for (int32_t i = 0; i < mesh.neighbourhood.size(); ++i) {

        // First write our indexes
        out.write(reinterpret_cast<const char*>(&i), sizeof(int32_t));

        // Write out the indexes of our neighbours
        for (int32_t n : mesh.neighbourhood[i]) {
            out.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
        }
    }

    // Write out each of our image values
    for (const auto& p : mesh.pixel_coordinates) {
        // Get the point
        cv::Point q(cv::Point(p[0], p[1]));

        // RGB floats
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;

        // Test if the point is in the image
        if (rect.contains(q)) {
            cv::Vec3b colour = img.at<cv::Vec3b>(q);

            r = double(colour[2]) / 255.0;
            g = double(colour[1]) / 255.0;
            b = double(colour[0]) / 255.0;
        }

        // Write out to file
        out.write(reinterpret_cast<const char*>(&r), sizeof(float));
        out.write(reinterpret_cast<const char*>(&g), sizeof(float));
        out.write(reinterpret_cast<const char*>(&b), sizeof(float));
    }

    for (const auto& p : mesh.pixel_coordinates) {
        // Get the point
        cv::Point q(cv::Point(p[0], p[1]));

        // Classification
        float v = 0.0f;

        // Test if the point is in the image
        if (rect.contains(q)) {
            v = stencil.at<cv::Vec3b>(q)[0] > 0 ? 1.0f : 0.0f;
        }

        // Write out to file
        out.write(reinterpret_cast<const char*>(&v), sizeof(float));
    }
}

int main() {

    // Seed our random number generator
    srand(time(nullptr));

    // Input path
    std::string path = "/Users/trent/Code/VisualMesh/training/raw";

    // Construct our VisualMesh
    std::cerr << "Building VisualMesh" << std::endl;
    mesh::Sphere<float> sphere(0, 0.075, 4, 10);
    mesh::VisualMesh<float> mesh(sphere, 0.5, 1.5, 100, M_PI / 1024.0);
    std::cerr << "Finished building VisualMesh" << std::endl;

    // Go through all our training data
    std::cerr << "Looping through training data" << std::endl;
    auto files = listdir(path);
    std::sort(files.begin(), files.end());

    int num_files = 0;
    int file_num  = 0;

    // Build our blur kernels
    std::vector<cv::Mat> blur_kernels;

    std::vector<std::array<float, 2>> blur_directions = {
        {{1, 0}}, {{1, -0.5}}, {{1, -1}}, {{0.5, -1}}, {{0, 1}}, {{0.5, 1}}, {{1, 1}}, {{1, 0.5}}};

    for (int size = 11; size < 33; size += 1) {

        // Gaussian kernel
        float sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8;
        blur_kernels.push_back(cv::getGaussianKernel(size, sigma, CV_32F));

        // Motion blur kernel
        for (const auto& d : blur_directions) {
            cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);

            int x                  = int((size - 1) * 0.5);
            int y                  = int((size - 1) * 0.5);
            kernel.at<float>(x, y) = 1.0 / size;

            for (int i = 0; i < x + 1; ++i) {

                int p1x = x - int(i * d[0]);
                int p1y = y - int(i * d[1]);
                int p2x = x + int(i * d[0]);
                int p2y = y + int(i * d[1]);

                kernel.at<float>(p1y, p1x) = 1.0 / size;
                kernel.at<float>(p2y, p2x) = 1.0 / size;
            }

            blur_kernels.push_back(kernel);
        }
    }


    Timer ftimer;

    auto file_name = fmt::format("data{:04d}.bin", ++file_num);
    auto file      = std::ofstream(file_name, std::ios::binary);

    for (const auto& p : files) {
        if (p.substr(0, 4) == "meta") {

            // After 250 files, make a new file (around 1GB chunks)
            ++num_files;
            if (num_files >= 250) {

                // Compress our file
                file.close();
                num_files = 0;
                ftimer.measure("Finished batch");
                std::cerr << "Compressing file " << file_name << std::endl;
                ::system(fmt::format("pigz -9 {}", file_name).c_str());
                ftimer.measure("Compressed file");
                file_name = fmt::format("data{:04d}.bin", ++file_num);
                file      = std::ofstream(file_name, std::ios::binary);
            }

            // Extract the number so we can find the other files
            auto number = p.substr(4, 7);

            std::cerr << "Processing file " << number << std::endl;

            Timer t;

            // Load our metadata and two images
            nlohmann::json meta;
            std::ifstream(path + "/" + p) >> meta;
            cv::Mat img     = cv::imread(path + "/image" + number + ".png");
            cv::Mat scratch = img.clone();
            cv::Mat stencil = cv::imread(path + "/stencil" + number + ".png");

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

            // Project our visual mesh coordinates
            decltype(mesh.project_mesh(Hoc, lens)) projection;
            try {
                projection = mesh.project_mesh(Hoc, lens);
            }
            catch (...) {
                continue;
            }

            t.measure("\tProjected Visual Mesh");

            // Save our original image file
            save_data(file, projection, img, stencil);

            t.measure("\tSaved first file");

            // For checking if a pixel is in the image
            cv::Rect rect(cv::Point(), img.size());

            // Make 10 random variations
            for (int var = 0; var < 10; ++var) {
                // Clone our original for manipulation
                scratch = img.clone();

                std::cerr << "\tVariant" << std::endl;

                // Choose which blur we should do (if any)
                int blur_roll = rand() % (blur_kernels.size() * 2);

                if (blur_roll < blur_kernels.size()) {

                    // Get our blur kernel
                    const auto& kernel = blur_kernels[blur_roll];

                    // Apply our blur kernel
                    cv::filter2D(scratch, scratch, -1, kernel);

                    t.measure("\t\tApplied Blur");
                }

                // 50/50 on brightness changes
                if (rand() % 2 == 0) {
                    int v = (rand() % 128) - 64;

                    // Just manipulate the values the mesh will select
                    for (const auto& p : projection.pixel_coordinates) {
                        // Test if the point is in the image
                        cv::Point q(cv::Point(p[0], p[1]));
                        if (rect.contains(q)) {
                            auto& px = scratch.at<cv::Vec3b>(q);
                            for (int i = 0; i < 3; ++i) {
                                px[i] = std::min(std::max(int(px[i]) + v, 0), 255);
                            }
                        }
                    }

                    t.measure("\t\tApplied Brightness");
                }

                // 50/50 on contrast changes
                if (rand() % 2 == 0) {
                    int v       = (rand() % 256) - 128;
                    auto factor = (259.0 * (float(v) + 255.0)) / (255.0 * (259.0 - float(v)));

                    // Just manipulate the values the mesh will select
                    for (const auto& p : projection.pixel_coordinates) {
                        // Test if the point is in the image
                        cv::Point q(cv::Point(p[0], p[1]));
                        if (rect.contains(q)) {
                            auto& px = scratch.at<cv::Vec3b>(q);
                            for (int i = 0; i < 3; ++i) {
                                px[i] =
                                    std::min(std::max(int(std::round(factor * (float(px[i]) - 128.0)) + 128), 0), 255);
                            }
                        }
                    }


                    t.measure("\t\tApplied Contrast");
                }

                // Save our variant
                save_data(file, projection, scratch, stencil);
            }
        }
    }

    file.close();
}
