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
  .Input("g_b: U")
  .Input("hoc_a: T")
  .Input("hoc_b: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Input("distance_threshold: T")
  .Output("matches: U")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(Outputs::MATCHES, c->MakeShape({c->Dim(c->input(Args::V_A), 0), 1}));
      return tensorflow::Status::OK();
  });

/**
 * @brief The Link Nearest tensorflow op
 *
 * @details
 *   For each vector in v_a find the vector in v_b that is radially closest to it.
 *
 *   Vectors in v_a are first transformed so that they are viewed from the same perspective as the vectors in v_b.
 *
 *   A graph search through g_b is then performed searching for the vector in v_b that has the smallest radial distance
 * to any given vector in v_a. If the shortest distance to all vectors in v_b exceeds a specified distance_threshold
 * then the vector in v_a is deemed to be offscreen.
 *
 * @tparam T The scalar type used for floating point numbers
 * @tparam U The scalar type used for integer numbers
 */
template <typename T, typename U>
class LinkNearestOp : public ShapeOpBase<T, LinkNearestOp<T, U>, Args::GEOMETRY, Args::RADIUS> {
private:
    template <typename Shape>
    // Calculate radial distance between the two nodes
    T radial_distance(const Eigen::Matrix<T, 3, 1>& a,
                      const Eigen::Matrix<T, 3, 1>& b,
                      const Eigen::Matrix<T, 4, 4>& Hoc,
                      const Shape& shape) {
        auto r_d = visualmesh::util::phi_difference(Hoc(2, 3), shape.c(), {a.x(), a.y(), a.z()}, {b.x(), b.y(), b.z()});
        return std::abs(shape.n(r_d.phi_0, r_d.h_prime) - shape.n(r_d.phi_1, r_d.h_prime));
    }

    // Given a node index check the distance of each of its neighbours to a given reference vector
    template <typename Shape>
    std::pair<U, T> check_neighbours(const U& index,
                                     const Eigen::Matrix<T, 3, 1>& reference,
                                     const Eigen::Matrix<T, Eigen::Dynamic, 3>& pool,
                                     const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>& neighbours,
                                     const Eigen::Matrix<T, 4, 4>& Hoc,
                                     const Shape& shape) {
        // Get distance to the first neighbour that is not offscreen
        U min_index;
        U neighbour_index = 0;
        do {
            min_index = neighbours(index, neighbour_index);
            neighbour_index++;
        } while (min_index < 0 && neighbour_index < neighbours.cols());

        // Make sure we found an onscreen neighbour
        if (min_index >= 0) {
            T min_distance = radial_distance(pool.row(min_index).transpose(), reference, Hoc, shape);

            // Now find the minumum distance over all neighbours
            for (; neighbour_index < neighbours.cols(); ++neighbour_index) {
                // Get next neighbour
                U new_index = neighbours(index, neighbour_index);
                if (new_index >= 0) {
                    Eigen::Matrix<T, 3, 1> neighbour = pool.row(new_index).transpose();

                    // Calculate distance to neighbour
                    T new_distance = radial_distance(neighbour, reference, Hoc, shape);

                    // Update minimum distance
                    if (new_distance < min_distance) {
                        min_index    = new_index;
                        min_distance = new_distance;
                    }
                }
            }
            // Return the node index and distance for the neighbour with the smallest distance
            return std::make_pair(min_index, min_distance);
        }

        // Infinite distance for offscreen points
        return std::make_pair(-std::numeric_limits<U>::max(), std::numeric_limits<T>::infinity());
    }

    // For each reference vector find the index to the vector in the pool that is closest to it
    template <typename Shape>
    Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> match_vectors(
      const Eigen::Matrix<T, Eigen::Dynamic, 3>& references,
      const Eigen::Matrix<T, Eigen::Dynamic, 3>& pool,
      const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>& neighbours,
      const Eigen::Matrix<T, 4, 4>& Hoc,
      const Shape& shape,
      const T& distance_threshold) {
        // Create output matrix
        // Populate it with -intmax so that if we happen to miss one it defaults to offscreen
        Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> matches =
          Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>::Constant(
            references.rows(), 1, -std::numeric_limits<U>::max());

        // Match each reference vector to a vector in the pool
        for (U reference_index = 0; reference_index < references.rows(); ++reference_index) {
            // Extract the reference vector
            const Eigen::Matrix<T, 3, 1> reference = references.row(reference_index).transpose();

            // Calculate pool vector starting index (hopefully this index is close to the final index)
            U min_index = reference_index * pool.rows() / references.rows();

            // Set initial distance as the distance between the reference vector and selected pool vector
            T min_distance = radial_distance(pool.row(min_index), reference, Hoc, shape);

            // Check distance to the neighbours of the selected index in the pool to find the neighbour to
            // start the graph search from
            U new_min_index;
            T new_min_distance;
            std::tie(new_min_index, new_min_distance) =
              check_neighbours(min_index, reference, pool, neighbours, Hoc, shape);

            // Start checking the rest of the graph
            // Find the neighbour of the current index that is closest to the reference vector
            // If none of the neighbours are closer to the reference vector than the current pool vector
            // then our search is over
            while (new_min_distance < min_distance) {
                min_distance = new_min_distance;
                min_index    = new_min_index;
                std::tie(new_min_index, new_min_distance) =
                  check_neighbours(min_index, reference, pool, neighbours, Hoc, shape);
            }

            // Check to see if the closest point is offscreen
            if (new_min_distance > distance_threshold) { min_index = -std::numeric_limits<U>::max(); }
            else {
                min_index = new_min_index;
            }

            // min_index is now the index of the pool vector that is closest to the current reference vector
            matches(reference_index) = min_index;
        }
        return matches;
    }

public:
    explicit LinkNearestOp(tensorflow::OpKernelConstruction* context)
      : ShapeOpBase<T, LinkNearestOp<T, U>, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {

        /***********************************************************
         *** Check that the shape of each of the inputs is valid ***
         ***********************************************************/

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


        /**************************************************
         *** Extract information from our input tensors ***
         **************************************************/

        T distance_threshold      = context->input(Args::THRESHOLD).scalar<T>()(0);
        tensorflow::Tensor tV_a   = context->input(Args::V_A);
        tensorflow::Tensor tV_b   = context->input(Args::V_B);
        tensorflow::Tensor tG_b   = context->input(Args::G_B);
        tensorflow::Tensor tHoc_a = context->input(Args::HOC_A);
        tensorflow::Tensor tHoc_b = context->input(Args::HOC_B);


        /****************************************************
         *** Convert TensorFlow tensors to Eigen Matrices ***
         ****************************************************/

        Eigen::Matrix<T, Eigen::Dynamic, 3> v_a =
          Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 3>>(tV_a.tensor<T, 2>().data(), tV_a.shape().dim_size(0), 3);
        Eigen::Matrix<T, Eigen::Dynamic, 3> v_b =
          Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 3>>(tV_b.tensor<T, 2>().data(), tV_b.shape().dim_size(0), 3);
        Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> g_b =
          Eigen::Map<Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>>(
            tG_b.tensor<U, 2>().data(), tG_b.shape().dim_size(0), tG_b.shape().dim_size(1));
        Eigen::Matrix<T, 4, 4> Hoc_a = Eigen::Map<Eigen::Matrix<T, 4, 4>>(
          tHoc_a.tensor<T, 2>().data(), tHoc_a.shape().dim_size(0), tHoc_a.shape().dim_size(1));
        Eigen::Matrix<T, 4, 4> Hoc_b = Eigen::Map<Eigen::Matrix<T, 4, 4>>(
          tHoc_b.tensor<T, 2>().data(), tHoc_b.shape().dim_size(0), tHoc_b.shape().dim_size(1));


        /****************************************
         *** Project v_a into the view of v_b ***
         ****************************************/


        // Project v_a on to the observation plane
        v_a = v_a * (Hoc_a(2, 3) - shape.c());
        v_a = (v_a.array().colwise() / v_a.template rightCols<1>().array()).matrix();

        // Translate A vectors to the B view and renormalise
        // Points:
        //    A - A camera origin
        //    B - B camera origin
        //    O - observation plane origin
        //    N - projected node position
        // Spaces:
        //    o - observation plane
        const Eigen::Matrix<T, 1, 3> rAOo(Hoc_a(0, 3), Hoc_a(1, 3), Hoc_a(2, 3));
        const Eigen::Matrix<T, 1, 3> rBOo(Hoc_b(0, 3), Hoc_b(1, 3), Hoc_b(2, 3));
        const Eigen::Matrix<T, 1, 3> rABo = rAOo - rBOo;

        // These are the A vectors from the perspective of the B camera
        // rNBo = rNAo + rABo
        Eigen::Matrix<T, Eigen::Dynamic, 3> rNBo = (v_a.rowwise() + rABo).rowwise().normalized();


        /**********************
         *** Compute output ***
         **********************/


        // Create the output matrix to hold the matching indices
        tensorflow::Tensor* matches = nullptr;
        tensorflow::TensorShape matches_shape;
        matches_shape.AddDim(tV_a.shape().dim_size(0));
        matches_shape.AddDim(1);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::MATCHES, matches_shape, &matches));

        // Find the matching indices for each vector in the right view
        matches->matrix<U>() = Eigen::TensorMap<Tensor<U>, Eigen::Aligned>(
          match_vectors(rNBo, v_b, g_b, Hoc_b, shape, distance_threshold).data(), {tV_a.shape().dim_size(0), 1});
    }
};

// Register a version for all the combinations of float/double and int32/int64
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int32>("U"),
  LinkNearestOp<float, tensorflow::int32>)
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int64>("U"),
  LinkNearestOp<float, tensorflow::int64>)
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int32>("U"),
  LinkNearestOp<double, tensorflow::int32>)
REGISTER_KERNEL_BUILDER(
  Name("LinkNearest").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int64>("U"),
  LinkNearestOp<double, tensorflow::int64>)
