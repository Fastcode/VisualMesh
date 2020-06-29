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
    COORDINATES = 0,
    MESH_MODEL  = 1,
    HEIGHT      = 2,
    GEOMETRY    = 3,
    RADIUS      = 4,
};

enum Outputs {
    VECTORS = 0,
};

REGISTER_OP("MapVisualMesh")
  .Attr("T: {float, double}")
  .Input("coordinates: T")
  .Input("model: string")
  .Input("height: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Output("vectors: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(Outputs::VECTORS, c->Matrix(c->Dim(c->input(Args::COORDINATES), 0), 3));
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
class MapVisualMeshOp : public ModelOpBase<T, MapVisualMeshOp<T>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS> {
public:
    explicit MapVisualMeshOp(tensorflow::OpKernelConstruction* context)
      : ModelOpBase<T, MapVisualMeshOp<T>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <template <typename> class Model, typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {

        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::COORDINATES).shape())
                      && context->input(Args::COORDINATES).shape().dim_size(1) == 2,
                    tensorflow::errors::InvalidArgument("Coordinates must be an nx2 matrix of nm coordinates"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::HEIGHT).shape()),
                    tensorflow::errors::InvalidArgument("The height must be a scalar"));

        // Extract information from our input tensors
        auto coordinates = context->input(Args::COORDINATES).matrix<T>();
        T height         = context->input(Args::HEIGHT).scalar<T>()(0);
        auto n_elems     = context->input(Args::COORDINATES).shape().dim_size(0);

        // Create the output matrix to hold the vectors
        tensorflow::Tensor* vectors = nullptr;
        tensorflow::TensorShape vectors_shape;
        vectors_shape.AddDim(n_elems);
        vectors_shape.AddDim(3);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::VECTORS, vectors_shape, &vectors));

        // Perform the map operation for this shape
        auto vs = vectors->matrix<T>();
        for (int i = 0; i < n_elems; ++i) {
            visualmesh::vec3<T> v = visualmesh::normalise(
              Model<T>::map(shape, height, visualmesh::vec2<T>({coordinates(i, 0), coordinates(i, 1)})));
            vs(i, 0) = v[0];
            vs(i, 1) = v[1];
            vs(i, 2) = v[2];
        }
    }
};

// Register a version for all the combinations of float/double
REGISTER_KERNEL_BUILDER(Name("MapVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
                        MapVisualMeshOp<float>)
REGISTER_KERNEL_BUILDER(Name("MapVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T"),
                        MapVisualMeshOp<double>)
