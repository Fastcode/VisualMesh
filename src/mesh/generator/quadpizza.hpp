/*
 * Copyright (C) 2017-2018 Trent Houliston <trent@houliston.me>
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

#ifndef VISUALMESH_GENERATOR_QUADPIZZA_HPP
#define VISUALMESH_GENERATOR_QUADPIZZA_HPP

#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include "mesh/node.hpp"

namespace visualmesh {
namespace generator {

  template <typename Scalar>
  struct QuadPizza {
  private:
    static inline vec3<Scalar> unit_vector(const Scalar& sin_phi, const Scalar& cos_phi, const Scalar& theta) {
      return vec3<Scalar>{{std::cos(theta) * sin_phi, std::sin(theta) * sin_phi, -cos_phi}};
    }

  public:
    static constexpr size_t N_NEIGHBOURS = 4;

    template <typename Shape>
    static std::vector<Node<Scalar, N_NEIGHBOURS>> generate(const Shape& shape,
                                                            const Scalar& h,
                                                            const Scalar& k,
                                                            const Scalar& max_distance) {
      std::vector<Node<Scalar, N_NEIGHBOURS>> nodes;
      std::vector<Scalar> Theta_Offset;
      std::vector<int> number_points;

      // L, T, R, B
      int LEFT  = 0;
      int TOP   = 1;
      int RIGHT = 2;
      int BELOW = 3;


      // **********************************DATA*******************************

      std::vector<int> DISTRIBUTION;
      std::vector<Scalar> THETA_NEXT;
      // **********************************************************************

      std::array<int, 4> first_neighbours = {2, 3, 0, 1};

      Scalar phi_first = shape.phi(1 / k, h) / 2;
      Theta_Offset.push_back(0);

      for (int j = 0; j < 4; ++j) {
        Node<Scalar, N_NEIGHBOURS> first;
        first.ray               = unit_vector(std::sin(phi_first / 2), std::cos(phi_first / 2), j * (2 * M_PI / 4));
        first.neighbours[BELOW] = first_neighbours[j];
        first.neighbours[LEFT]  = ((j + 1) % 4 + 4) % 4;
        first.neighbours[RIGHT] = ((j - 1) % 4 + 4) % 4;
        nodes.push_back(std::move(first));
      }
      number_points.emplace_back(4);
      THETA_NEXT.push_back(2 * M_PI / 4);
      DISTRIBUTION.push_back(1);

      int running_index = nodes.size();

      std::vector<int> origin_number_points;
      origin_number_points.emplace_back(0);
      origin_number_points.emplace_back(4);

      int stop;
      if (k < 9) { stop = 5; }
      else {
        stop = 8;
      }

      for (int i = 0; i < stop; ++i) {
        origin_number_points.emplace_back(8 + 8 * i);
      }

      bool half_offset = false;
      for (int v = 1; h * std::tan(shape.phi(v / k, h)) < max_distance; ++v) {
        // check this int to Scalar conversion
        Scalar phi_next = shape.phi(v / k, h);
        int begin       = running_index - number_points.back();
        int end         = running_index;
        bool every_one  = false;
        bool growing    = false;

        // hack // odd v generates clockwise, even v generates anti-clockwise.
        int one               = std::round(std::pow(-1, v));
        int number_points_now = number_points[v - 1];
        int number_points_next;

        // precalculate the number of points vector according to distribution and origin patch
        // fix this limit
        if (v < stop) { number_points_next = origin_number_points[v]; }
        else {
          number_points_next = std::ceil((2 * M_PI * k) / shape.theta(phi_next, h));
        }

        int number_difference = number_points_next - number_points_now;
        Scalar theta_next;

        // *************** Calculate Split Distribution ***********************
        int distribution;
        if (number_difference == 0) {
          // std::cout << "Difference is zero: " << std::endl;
          distribution       = 0;
          number_points_next = number_points_now;
        }
        else if (number_difference == 1) {
          // std::cout << "Difference is 1: " << std::endl;
          growing      = true;
          distribution = 1;
          // number_difference = 1;
          number_points_next = 1 + number_points_now;
        }
        else if (number_difference > 1) {
          growing = true;
          if (number_difference < number_points_now) {
            // std::cout << "Difference is reasonable: " << std::endl;
            distribution = number_points_now / number_difference;
            if (distribution == 1) { every_one = true; }
          }
          else {
            // number_difference >= number_points_now
            // std::cout << v << std::endl;
            // std::cout << "Difference is greater than generating ring: " << std::endl;
            distribution       = 1;
            number_points_next = 2 * number_points_now;
            number_difference  = number_points_now;
          }
        }
        else {
          // std::cout << "Difference is negative: " << std::endl;
          distribution       = 0;
          number_points_next = number_points_now;
        }
        // *********************************************************************************

        theta_next = 2 * M_PI / number_points_next;
        number_points.emplace_back(number_points_next);

        THETA_NEXT.push_back(theta_next);
        DISTRIBUTION.push_back(distribution);

        if (distribution >= 2 * k + 2) { half_offset = true; };

        std::vector<int> indices;
        for (int i = begin; i < end; ++i) {
          indices.push_back(i);
        }

        // Chooses starting node
        int new_offset = 1;
        if (v == 1) { new_offset = 0; }
        else if (half_offset) {
          new_offset = std::floor(distribution / 2);
        }
        else {
          new_offset = 1;
        }

        std::vector<int> vector_of_indices;

        for (int m = new_offset; m >= 0; --m) {
          vector_of_indices.push_back(indices[m]);
        }
        for (int p = indices.size() - 1; p > new_offset; --p) {
          vector_of_indices.push_back(indices[p]);
        }

        int relative_index_now  = 0;
        int relative_index_next = 0;
        int number_splits       = 0;
        Scalar theta_offset =
          std::atan2(nodes[*vector_of_indices.begin()].ray[1], nodes[*vector_of_indices.begin()].ray[0]);

        for (auto it = vector_of_indices.begin(); it != vector_of_indices.end(); ++it) {
          Node<Scalar, N_NEIGHBOURS> new_node;
          new_node.ray =
            unit_vector(std::sin(phi_next), std::cos(phi_next), theta_offset + one * relative_index_next * theta_next);

          new_node.neighbours[LEFT] =
            end + (((relative_index_next + one) % number_points_next + number_points_next) % number_points_next);
          new_node.neighbours[RIGHT] =
            end + (((relative_index_next - one) % number_points_next + number_points_next) % number_points_next);
          new_node.neighbours[BELOW] = *it;

          // nodes[*it].neighbours[TOP] = end + relative_index_next % number_points_next;
          nodes[*it].neighbours[TOP] = end + relative_index_next;

          nodes.push_back(std::move(new_node));
          relative_index_next += 1;

          // *************** Generate Second Node ***********************
          if (growing) {
            if (every_one == true) {
              if (number_splits <= number_points_now - number_difference) { distribution = 2; }
              else {
                distribution = 1;
              }
            }

            // split every point of according to the distribution until difference is reached, or split every point.
            if ((relative_index_now % distribution == 0 || distribution == 1) && number_splits < number_difference) {
              // std::cout << "Graph is growing!" << std::endl;
              Node<Scalar, N_NEIGHBOURS> second_new_node;
              second_new_node.ray = unit_vector(
                std::sin(phi_next), std::cos(phi_next), theta_offset + one * relative_index_next * theta_next);

              second_new_node.neighbours[LEFT] =
                end + (((relative_index_next + one) % number_points_next + number_points_next) % number_points_next);
              second_new_node.neighbours[RIGHT] =
                end + (((relative_index_next - one) % number_points_next + number_points_next) % number_points_next);
              second_new_node.neighbours[BELOW] = *it;

              nodes.push_back(std::move(second_new_node));

              number_splits += 1;
              relative_index_next += 1;
            }
          }
          // *********************************************************************************
          relative_index_now += 1;
        }
        running_index = nodes.size();
      }

      // specify the neighbours of the last ring of points
      for (int i = (nodes.size() - number_points.back()); i < nodes.size(); ++i) {
        nodes[i].neighbours[TOP] = i;
      }

      //***********************************Data Collector***************************************
      std::vector<Scalar> PHI;
      std::vector<Scalar> THETA_RAW;
      std::vector<int> NUMBER;


      for (int j = 0; h * std::tan(shape.phi(j / k, h)) < max_distance; ++j) {
        PHI.push_back(shape.phi(j / k, h));
        THETA_RAW.push_back(shape.theta(shape.phi(j / k, h), h));
        NUMBER.push_back(std::ceil((2 * M_PI * k) / shape.theta(shape.phi(j / k, h), h)));
      }

      int last;
      last = std::max(
        {PHI.size(), THETA_RAW.size(), NUMBER.size(), number_points.size(), THETA_NEXT.size(), DISTRIBUTION.size()});

      // if (h >= 0.94 - 0.001 && h <= 0.94 + 0.001) {
      //   std::ofstream outfile;
      //   outfile.open("/home/asugo/LatexPlotData/Error/Robot6.csv");
      //   if (outfile.fail()) { std::cout << "Couldn't open the file!" << std::endl; }
      //   // outfile << "#,"
      //   // << "kNumber," << k << ",Height," << h << ",Stop," << stop << ",LastRing," << last << ",Distance,"
      //   // << max_distance << "\n";
      //   // outfile << "PhiNumber,Phi,THETA_RAW,NumberOfPoints,PointsByGraph,Difference,\n";
      //   outfile
      //     <<
      //     "PhiNumber,Phi,Distance,Theta_Raw,NumberOfPoints,PointsByGraph,Difference,Distribution,Theta_next,Ratio\n";
      //   int diff;
      //   for (size_t j = 5; j < last; ++j) {
      //     if (std::isfinite(THETA_RAW[j]) && std::isfinite(THETA_RAW[j - 1])) { diff = NUMBER[j] - NUMBER[j - 1]; }
      //     else {
      //       diff = 0;
      //     }
      //     Scalar distance = h * std::tan(PHI[j]);
      //     Scalar T        = std::isfinite(THETA_RAW[j]) ? THETA_RAW[j] : 10000;
      //     Scalar N        = std::isfinite(THETA_RAW[j]) ? NUMBER[j] : 0;
      //     int Ratio       = std::floor(THETA_NEXT[j] / (THETA_NEXT[j - 1] - THETA_NEXT[j]));
      //     outfile << j << "," << PHI[j] << "," << distance << "," << T << "," << N << "," << number_points[j] << ","
      //             << diff << "," << DISTRIBUTION[j] << "," << THETA_NEXT[j] << "," << Ratio << "\n";
      //   }
      //   outfile.close();
      // }


      if (h >= 0.94 - 0.001 && h <= 0.94 + 0.001) {

        std::vector<std::vector<Scalar>> Variation;

        for (int i = 1; i < 4; ++i) {
          // Scalar counter = nodes.size() - 1 - (1000 * i);
          Scalar counter = nodes.size() - 1 - 4000 + (900 * i);
          std::cout << counter << std::endl;
          std::vector<Scalar> variation;
          for (int t = last - 1; t > 2; --t) {
            variation.push_back(std::atan2(nodes[counter].ray[1], nodes[counter].ray[0]));
            counter = nodes[counter].neighbours[BELOW];
          }
          Variation.push_back(variation);
        }


        std::ofstream outfile2;
        outfile2.open("/home/asugo/LatexPlotData/ThetaVariationQuadAlt8.csv");
        outfile2 << "Phi_Number,Distance,Theta0,Theta1,Theta2\n";

        for (int t = 0; t < Variation[0].size(); ++t) {
          outfile2 << last - 1 - t << "," << h * std::tan(PHI[last - 1 - t]) << "," << Variation[0][t] << ","
                   << Variation[1][t] << "," << Variation[2][t] << "\n";
        }
      }

      //****************************************************************************************

      // // print out mesh points
      // for (int i = 0; i < nodes.size(); ++i) {
      //   std::cout << "meshpoint: " << i << ": " << nodes[i].neighbours[LEFT] << ", " << nodes[i].neighbours[RIGHT]
      //             << ", " << nodes[i].neighbours[TOP] << ", " << nodes[i].neighbours[BELOW] << std::endl;
      // }

      return nodes;
    }  // namespace generator
  };   // namespace generator

}  // namespace generator
}  // namespace visualmesh

#endif  // VISUALMESH_GENERATOR_QUADPIZZA_HPP
