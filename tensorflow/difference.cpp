/*
 * Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
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

#include <memory>

#include "model_op_base.hpp"

enum Args {
    COORDINATES_A = 0,
    COORDINATES_B = 1,
    MESH_MODEL    = 2,
    HEIGHT        = 3,
    GEOMETRY      = 4,
    RADIUS        = 5,
};

enum Outputs {
    DIFFERENCES = 0,
};

REGISTER_OP("DifferenceVisualMesh")
  .Attr("T: {float, double}")
  .Input("coordinates_a: T")
  .Input("coordinates_b: T")
  .Input("model: string")
  .Input("height: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Output("vectors: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(Outputs::DIFFERENCES, c->input(Args::COORDINATES_A));
      return tensorflow::Status::OK();
  });

/**
 * @brief The Visual Mesh tensorflow op
 *
 * @details
 *  This op will perform a projection using the visual mesh and will return the neighbourhood graph and the pixel
 * coordinates for the points that would be on screen for the lens paramters provided.
 *
 * @tparam T The scalar type used for floating point numbers
 */
template <typename T>
class DifferenceVisualMeshOp
  : public ModelOpBase<T, DifferenceVisualMeshOp<T>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS> {
public:
    explicit DifferenceVisualMeshOp(tensorflow::OpKernelConstruction* context)
      : ModelOpBase<T, DifferenceVisualMeshOp<T>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <template <typename> class Model, typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {

        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::COORDINATES_A).shape())
                      && context->input(Args::COORDINATES_A).shape().dim_size(1) == 2,
                    tensorflow::errors::InvalidArgument("First coordinates must be an nx2 matrix of nm coordinates"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::COORDINATES_B).shape())
                      && context->input(Args::COORDINATES_B).shape().dim_size(1) == 2,
                    tensorflow::errors::InvalidArgument("Second coordinates must be an nx2 matrix of nm coordinates"));
        OP_REQUIRES(context,
                    context->input(Args::COORDINATES_A).shape() == context->input(Args::COORDINATES_B).shape(),
                    tensorflow::errors::InvalidArgument("Second coordinates must be an nx2 matrix of nm coordinates"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::HEIGHT).shape()),
                    tensorflow::errors::InvalidArgument("The height must be a scalar"));

        // Extract information from our input tensors
        auto c_a     = context->input(Args::COORDINATES_A).matrix<T>();
        auto c_b     = context->input(Args::COORDINATES_B).matrix<T>();
        T height     = context->input(Args::HEIGHT).scalar<T>()(0);
        auto n_elems = context->input(Args::COORDINATES_A).shape().dim_size(0);

        // Create the output matrix to hold the vectors
        tensorflow::Tensor* vectors = nullptr;
        tensorflow::TensorShape vectors_shape;
        vectors_shape.AddDim(n_elems);
        vectors_shape.AddDim(2);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::DIFFERENCES, vectors_shape, &vectors));

        // Perform the map operation for this shape
        auto ds = vectors->matrix<T>();
        for (int i = 0; i < n_elems; ++i) {
            visualmesh::vec2<T> d = Model<T>::difference(
              shape, height, visualmesh::vec2<T>({c_a(i, 0), c_a(i, 1)}), visualmesh::vec2<T>({c_b(i, 0), c_b(i, 1)}));
            ds(i, 0) = d[0];
            ds(i, 1) = d[1];
        }
    }
};

// Register a version for all the combinations of float/double
REGISTER_KERNEL_BUILDER(Name("DifferenceVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
                        DifferenceVisualMeshOp<float>)
REGISTER_KERNEL_BUILDER(Name("DifferenceVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T"),
                        DifferenceVisualMeshOp<double>)
