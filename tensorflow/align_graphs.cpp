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
#include <memory>

#include "shape_op_base.hpp"

template <typename T>
using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

enum Args {
    V_LEFT    = 0,
    V_RIGHT   = 1,
    G_LEFT    = 2,
    G_RIGHT   = 3,
    HOC_LEFT  = 4,
    HOC_RIGHT = 5,
    GEOMETRY  = 6,
    RADIUS    = 7,
};

enum Outputs {
    MATCHED_LEFT  = 0,
    MATCHED_RIGHT = 1,
};

REGISTER_OP("AlignGraphs")
  .Attr("T: {float, double}")
  .Attr("U: {int32, int64}")
  .Input("left_vectors: T")
  .Input("right_vectors: T")
  .Input("left_graph: T")
  .Input("right_graph: T")
  .Input("left_hoc: T")
  .Input("right_hoc: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Output("left_matches: U")
  .Output("right_matches: U")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(Outputs::MATCHED_LEFT, c->input(Args::V_RIGHT));
      c->set_output(Outputs::MATCHED_RIGHT, c->input(Args::V_LEFT));
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
class AlignGraphsOp : public ShapeOpBase<T, AlignGraphsOp<T, U>, Args::GEOMETRY, Args::RADIUS> {
public:
    explicit AlignGraphsOp(tensorflow::OpKernelConstruction* context)
      : ShapeOpBase<T, AlignGraphsOp<T, U>, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {
        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::V_LEFT).shape())
                      && context->input(Args::V_LEFT).shape().dim_size(1) == 3,
                    tensorflow::errors::InvalidArgument("Left vectors must be an nx3 matrix of unit vectors"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::V_RIGHT).shape())
                      && context->input(Args::V_RIGHT).shape().dim_size(1) == 3,
                    tensorflow::errors::InvalidArgument("Right vectors must be an nx3 matrix of unit vectors"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::G_LEFT).shape()),
                    tensorflow::errors::InvalidArgument("Left graph must be a matrix of ints"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::G_RIGHT).shape()),
                    tensorflow::errors::InvalidArgument("Right graph must be a matrix of ints"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsSquareMatrix(context->input(Args::HOC_LEFT).shape())
                      && context->input(Args::HOC_LEFT).shape().dim_size(0) == 4,
                    tensorflow::errors::InvalidArgument("Left Hoc must be a 4x4 matrix"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsSquareMatrix(context->input(Args::HOC_RIGHT).shape())
                      && context->input(Args::HOC_RIGHT).shape().dim_size(0) == 4,
                    tensorflow::errors::InvalidArgument("Right Hoc must be a 4x4 matrix"));

        // Extract information from our input tensors
        tensorflow::Tensor v_l   = context->input(Args::V_LEFT);
        tensorflow::Tensor v_r   = context->input(Args::V_RIGHT);
        tensorflow::Tensor g_l   = context->input(Args::G_LEFT);
        tensorflow::Tensor g_r   = context->input(Args::G_RIGHT);
        tensorflow::Tensor Hoc_l = context->input(Args::HOC_LEFT);
        tensorflow::Tensor Hoc_r = context->input(Args::HOC_RIGHT);
        MatrixType<T> v_left =
          Eigen::Map<MatrixType<T>>(v_l.tensor<T, 2>().data(), v_l.shape().dim_size(0), v_l.shape().dim_size(1));
        MatrixType<T> v_right =
          Eigen::Map<MatrixType<T>>(v_r.tensor<T, 2>().data(), v_r.shape().dim_size(0), v_r.shape().dim_size(1));
        MatrixType<T> g_left =
          Eigen::Map<MatrixType<T>>(g_l.tensor<T, 2>().data(), g_l.shape().dim_size(0), g_l.shape().dim_size(1));
        MatrixType<T> g_right =
          Eigen::Map<MatrixType<T>>(g_r.tensor<T, 2>().data(), g_r.shape().dim_size(0), g_r.shape().dim_size(1));
        MatrixType<T> Hoc_left =
          Eigen::Map<MatrixType<T>>(Hoc_l.tensor<T, 2>().data(), Hoc_l.shape().dim_size(0), Hoc_l.shape().dim_size(1));
        MatrixType<T> Hoc_right =
          Eigen::Map<MatrixType<T>>(Hoc_r.tensor<T, 2>().data(), Hoc_r.shape().dim_size(0), Hoc_r.shape().dim_size(1));


        // Determine the height of the observation plane
        T shape_c = shape.c();

        // Project left and right vectors on to the observation plane
        v_left  = (v_left * (Hoc_left(2, 3) - shape_c)).cwiseQuotient(Hoc_left.template rightCols<1>());
        v_right = (v_right * (Hoc_right(2, 3) - shape_c)).cwiseQuotient(Hoc_right.template rightCols<1>());

        // Create the output matrix to hold the matching indices
        tensorflow::Tensor* matched_left  = nullptr;
        tensorflow::Tensor* matched_right = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::MATCHED_LEFT, v_r.shape(), &matched_left));
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::MATCHED_RIGHT, v_l.shape(), &matched_right));

        // Given a node index check the distance of each of its neighbours to a given reference vector
        auto check_neighbours = [&](const U& index,
                                    const T& distance,
                                    const MatrixType<T>& reference,
                                    const MatrixType<T>& neighbours) -> std::pair<U, T> {
            T min_distance = distance;
            U min_index    = index;
            for (U neighbour_index = 0; neighbour_index < neighbours.cols(); ++neighbour_index) {
                U new_index             = neighbours(index, neighbour_index);
                MatrixType<T> neighbour = v_left.row(new_index);

                T new_distance = (neighbour - reference).squaredNorm();

                if (new_distance < min_distance) {
                    min_index    = new_index;
                    min_distance = new_distance;
                }
            }
            return std::make_pair(min_index, min_distance);
        };

        // For each reference vector find the index to the vector in the pool that is closest to it
        auto match_vectors =
          [&](const MatrixType<T>& pool, const MatrixType<T>& references, const MatrixType<T>& neighbours) {
              MatrixType<U> matches = MatrixType<U>::Zero(references.rows(), 1);

              for (U reference_index = 0; reference_index < references.rows(); ++reference_index) {
                  const MatrixType<T>& reference = references.row(reference_index);

                  // Calculate pool vector starting index (hopefully this index is close to the final index)
                  U min_index     = reference_index * pool.rows() / references.rows();
                  U new_min_index = -1;

                  // Set initial distance as the distance between the reference vector and selected pool vector
                  T min_distance = (pool.row(min_index) - reference).squaredNorm();

                  // Check distance to the neighbours of the selected index in the pool to find the neighbour to
                  // start the graph search from.
                  std::tie(min_index, min_distance) = check_neighbours(min_index, min_distance, reference, neighbours);

                  // Start checking the rest of the graph
                  // Find the neighbour of the current index that is closest to the reference vector
                  // If none of the neighbours are closer to the reference vector than the current pool vector then our
                  // search is over.
                  while (new_min_index != min_index) {
                      min_index = new_min_index;
                      std::tie(min_index, min_distance) =
                        check_neighbours(min_index, min_distance, reference, neighbours);
                  }

                  // min_index is now the index of the pool vector that is closest to the current reference vector
                  matches(reference_index) = min_index;
              }
              return matches;
          };

        // // Find the matching indices for each vector in the right view
        MatrixType<U> left_matches = match_vectors(v_left, v_right, g_left);
        matched_left->matrix<U>()  = Eigen::TensorMap<Eigen::Tensor<U, 2, Eigen::RowMajor>, Eigen::Aligned>(
          left_matches.data(), {v_r.shape().dim_size(0), v_r.shape().dim_size(1)});

        // // Find the matching indices for each vector in the left view
        MatrixType<U> right_matches = match_vectors(v_right, v_left, g_right);
        matched_right->matrix<U>()  = Eigen::TensorMap<Eigen::Tensor<U, 2, Eigen::RowMajor>, Eigen::Aligned>(
          right_matches.data(), {v_l.shape().dim_size(0), v_l.shape().dim_size(1)});
    }
};

// Register a version for all the combinations of float/double and int32/int64
REGISTER_KERNEL_BUILDER(
  Name("AlignGraphs").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int32>("U"),
  AlignGraphsOp<double, tensorflow::int32>)
REGISTER_KERNEL_BUILDER(
  Name("AlignGraphs").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int64>("U"),
  AlignGraphsOp<double, tensorflow::int64>)
REGISTER_KERNEL_BUILDER(
  Name("AlignGraphs").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int32>("U"),
  AlignGraphsOp<double, tensorflow::int32>)
REGISTER_KERNEL_BUILDER(
  Name("AlignGraphs").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int64>("U"),
  AlignGraphsOp<double, tensorflow::int64>)
