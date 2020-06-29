/*
 * Copyright (C) 2017-2020 Alex Biddulph <Alexander.Biddulph@uon.edu.au>
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

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <Eigen/Core>
#include <limits>
#include <utility>

#include "shape_op_base.hpp"
#include "utility/phi_difference.hpp"

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Tensor = Eigen::Tensor<T, 2, Eigen::RowMajor>;

enum Args {
    V_A       = 0,
    V_B       = 1,
    G_B       = 2,
    HOC_A     = 3,
    HOC_B     = 4,
    GEOMETRY  = 5,
    RADIUS    = 6,
    THRESHOLD = 7,
};

enum Outputs {
    MATCHES = 0,
};

REGISTER_OP("LinkNearest")
  .Attr("T: {float, double}")
  .Attr("U: {int32, int64}")
  .Input("v_a: T")
  .Input("v_b: T")
  .Input("g_b: T")
  .Input("hoc_a: T")
  .Input("hoc_b: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Input("distance_threshold: T")
  .Output("matches: U")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(Outputs::MATCHES, c->input(Args::V_B));
      return tensorflow::Status::OK();
  });

/**
 * @brief The Align Graphs tensorflow op
 *
 * @details
 *   This op with search through both graphs finding the nodes that are closest to the nodes in the opposite graph.
 *
 *   Given a node from the right graph, a search is made for the node in the left graph that is closest to it. This
 * search is repeated for every node in the right graph, generating a list of indices linking each node in the right
 * graph to its match in the left graph.
 *
 *   The same search is performed for nodes in the left graph, the resulting index list linking nodes in the left graph
 * to their match in the right graph.
 *
 * @tparam T The scalar type used for floating point numbers
 * @tparam U The scalar type used for integer numbers
 */
template <typename T, typename U>
class LinkNearestOp : public ShapeOpBase<T, LinkNearestOp<T, U>, Args::GEOMETRY, Args::RADIUS> {
public:
    explicit LinkNearestOp(tensorflow::OpKernelConstruction* context)
      : ShapeOpBase<T, LinkNearestOp<T, U>, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {
        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::V_A).shape())
                      && context->input(Args::V_A).shape().dim_size(1) == 3,
                    tensorflow::errors::InvalidArgument("A vectors must be an nx3 matrix of unit vectors"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::V_B).shape())
                      && context->input(Args::V_B).shape().dim_size(1) == 3,
                    tensorflow::errors::InvalidArgument("B vectors must be an nx3 matrix of unit vectors"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::G_B).shape()),
                    tensorflow::errors::InvalidArgument("B graph must be a matrix of ints"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsSquareMatrix(context->input(Args::HOC_A).shape())
                      && context->input(Args::HOC_A).shape().dim_size(0) == 4,
                    tensorflow::errors::InvalidArgument("A Hoc must be a 4x4 matrix"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsSquareMatrix(context->input(Args::HOC_B).shape())
                      && context->input(Args::HOC_B).shape().dim_size(0) == 4,
                    tensorflow::errors::InvalidArgument("B Hoc must be a 4x4 matrix"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::THRESHOLD).shape()),
                    tensorflow::errors::InvalidArgument("Distance threshold must be a scalar"));

        // Extract information from our input tensors
        T distance_threshold      = context->input(Args::THRESHOLD).scalar<T>()(0);
        tensorflow::Tensor tV_a   = context->input(Args::V_A);
        tensorflow::Tensor tV_b   = context->input(Args::V_B);
        tensorflow::Tensor tG_b   = context->input(Args::G_B);
        tensorflow::Tensor tHoc_a = context->input(Args::HOC_A);
        tensorflow::Tensor tHoc_b = context->input(Args::HOC_B);
        Matrix<T> v_a =
          Eigen::Map<Matrix<T>>(tV_a.tensor<T, 2>().data(), tV_a.shape().dim_size(0), tV_a.shape().dim_size(1));
        Matrix<T> v_b =
          Eigen::Map<Matrix<T>>(tV_b.tensor<T, 2>().data(), tV_b.shape().dim_size(0), tV_b.shape().dim_size(1));
        Matrix<T> g_b =
          Eigen::Map<Matrix<T>>(tG_b.tensor<T, 2>().data(), tG_b.shape().dim_size(0), tG_b.shape().dim_size(1));
        Matrix<T> Hoc_a =
          Eigen::Map<Matrix<T>>(tHoc_a.tensor<T, 2>().data(), tHoc_a.shape().dim_size(0), tHoc_a.shape().dim_size(1));
        Matrix<T> Hoc_b =
          Eigen::Map<Matrix<T>>(tHoc_b.tensor<T, 2>().data(), tHoc_b.shape().dim_size(0), tHoc_b.shape().dim_size(1));


        // Project left and right vectors on to the observation plane
        v_a = (v_a * (Hoc_a(2, 3) - shape.c())).cwiseQuotient(v_a.template rightCols<1>());

        // Translate left vectors to right view and right vectors to left view and renormalise
        // Points:
        //    A - left camera origin
        //    B - right camera origin
        //    O - observation plane origin
        //    N - projected node position
        // Spaces:
        //    o - observation plane origin
        const Eigen::Matrix<T, 3, 1> rAOo(Hoc_a(0, 3), Hoc_a(1, 3), Hoc_a(2, 3));
        const Eigen::Matrix<T, 3, 1> rBOo(Hoc_b(0, 3), Hoc_b(1, 3), Hoc_b(2, 3));
        const Eigen::Matrix<T, 3, 1> rABo = rAOo - rBOo;

        // These are the A vectors projected into the B camera
        // rNBo = rNAo + rABo
        Matrix<T> rNBo = (v_a + rABo).normalized();

        // Calculate radial distance between the two nodes
        auto radial_distance = [&](const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 1>& b) -> T {
            auto r_d =
              visualmesh::util::phi_difference(Hoc_b(2, 3), shape.c(), {a.x(), a.y(), a.z()}, {b.x(), b.y(), b.z()});
            return std::abs(shape.n(r_d.phi_0, r_d.h_prime) - shape.n(r_d.phi_1, r_d.h_prime));
        };

        // Given a node index check the distance of each of its neighbours to a given reference vector
        auto check_neighbours = [&](const U& index,
                                    const Matrix<T>& vectors,
                                    const Matrix<T>& reference,
                                    const Matrix<T>& neighbours) -> std::pair<U, T> {
            // Get distance to the first neighbour
            U min_index    = neighbours(index, 0);
            T min_distance = radial_distance(vectors.row(min_index), reference);

            // Now find the minumum distance over all neighbours
            for (U neighbour_index = 1; neighbour_index < neighbours.cols(); ++neighbour_index) {
                U new_index         = neighbours(index, neighbour_index);
                Matrix<T> neighbour = vectors.row(new_index);

                T new_distance = radial_distance(neighbour, reference);

                if (new_distance < min_distance) {
                    min_index    = new_index;
                    min_distance = new_distance;
                }
            }

            // Return the node index and distance for the neighbour with the smallest distance
            return std::make_pair(min_index, min_distance);
        };

        // For each reference vector find the index to the vector in the pool that is closest to it
        auto match_vectors =
          [&](const Matrix<T>& pool, const Matrix<T>& references, const Matrix<T>& neighbours) -> Matrix<U> {
            Matrix<U> matches = Matrix<U>::Zero(references.rows(), 1);

            for (U reference_index = 0; reference_index < references.rows(); ++reference_index) {
                const Matrix<T>& reference = references.row(reference_index);

                // Calculate pool vector starting index (hopefully this index is close to the final index)
                U min_index = reference_index * pool.rows() / references.rows();

                // Set initial distance as the distance between the reference vector and selected pool vector
                T min_distance = radial_distance(pool.row(min_index), reference);

                // Check distance to the neighbours of the selected index in the pool to find the neighbour to
                // start the graph search from.
                U new_min_index;
                T new_min_distance;
                std::tie(new_min_index, new_min_distance) = check_neighbours(min_index, pool, reference, neighbours);

                // Start checking the rest of the graph
                // Find the neighbour of the current index that is closest to the reference vector
                // If none of the neighbours are closer to the reference vector than the current pool vector
                // then our search is over.
                while (new_min_distance < min_distance) {
                    min_distance = new_min_distance;
                    min_index    = new_min_index;
                    std::tie(new_min_index, new_min_distance) =
                      check_neighbours(min_index, pool, reference, neighbours);
                }

                // Check to see if the closest point is offscreen.
                if (new_min_distance > distance_threshold) { min_index = -std::numeric_limits<U>::max(); }
                else {
                    min_index = new_min_index;
                }

                // min_index is now the index of the pool vector that is closest to the current reference
                // vector
                matches(reference_index) = min_index;
            }
            return matches;
        };

        // Create the output matrix to hold the matching indices
        tensorflow::Tensor* matches = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::MATCHES, tV_b.shape(), &matches));

        // Find the matching indices for each vector in the right view
        matches->matrix<U>() = Eigen::TensorMap<Tensor<U>, Eigen::Aligned>(
          match_vectors(v_a, v_b, g_b).data(), {tV_b.shape().dim_size(0), tV_b.shape().dim_size(1)});
    }
};

// Register a version for all the combinations of float/double and int32/int64
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int32>("U"),
  LinkNearestOp<double, tensorflow::int32>)
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int64>("U"),
  LinkNearestOp<double, tensorflow::int64>)
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int32>("U"),
  LinkNearestOp<double, tensorflow::int32>)
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int64>("U"),
  LinkNearestOp<double, tensorflow::int64>)
