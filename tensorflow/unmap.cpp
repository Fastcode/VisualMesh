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

#include <cmath>
#include <functional>

#include "model_op_base.hpp"
#include "visualmesh/utility/math.hpp"

enum Args {
    VECTORS    = 0,
    MESH_MODEL = 1,
    HEIGHT     = 2,
    GEOMETRY   = 3,
    RADIUS     = 4,
};

enum Outputs {
    COORDINATES = 0,
};

// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_OP("UnmapVisualMesh")
  .Attr("T: {float, double}")
  .Input("vectors: T")
  .Input("model: string")
  .Input("height: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Output("coordinates: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(Outputs::COORDINATES, c->Matrix(c->Dim(c->input(Args::VECTORS), 0), 2));
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
class UnmapVisualMeshOp : public ModelOpBase<T, UnmapVisualMeshOp<T>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS> {
public:
    explicit UnmapVisualMeshOp(tensorflow::OpKernelConstruction* context)
      : ModelOpBase<T, UnmapVisualMeshOp<T>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <template <typename> class Model, typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {

        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::VECTORS).shape())
                      && context->input(Args::VECTORS).shape().dim_size(1) == 3,
                    tensorflow::errors::InvalidArgument("Vectors must be an nx3 matrix of unit vectors"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::HEIGHT).shape()),
                    tensorflow::errors::InvalidArgument("The height must be a scalar"));

        // Extract information from our input tensors
        auto vectors = context->input(Args::VECTORS).matrix<T>();
        T height     = context->input(Args::HEIGHT).scalar<T>()(0);
        auto n_elems = context->input(Args::VECTORS).shape().dim_size(0);

        // Create the output matrix to hold the coordinates
        tensorflow::Tensor* coordinates = nullptr;
        tensorflow::TensorShape coordinates_shape;
        coordinates_shape.AddDim(n_elems);
        coordinates_shape.AddDim(2);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::COORDINATES, coordinates_shape, &coordinates));

        // Perform the map operation for this shape
        auto cs = coordinates->matrix<T>();
        for (int i = 0; i < n_elems; ++i) {
            visualmesh::vec2<T> c =
              Model<T>::unmap(shape, height, visualmesh::vec3<T>({vectors(i, 0), vectors(i, 1), vectors(i, 2)}));
            cs(i, 0) = c[0];
            cs(i, 1) = c[1];
        }
    }
};

// Register a version for all the combinations of float/double
// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_KERNEL_BUILDER(Name("UnmapVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
                        UnmapVisualMeshOp<float>)
// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_KERNEL_BUILDER(Name("UnmapVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T"),
                        UnmapVisualMeshOp<double>)
