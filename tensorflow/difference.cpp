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

#include "geometry/Circle.hpp"
#include "geometry/Sphere.hpp"
#include "mesh/model/nmgrid4.hpp"
#include "mesh/model/nmgrid6.hpp"
#include "mesh/model/nmgrid8.hpp"
#include "mesh/model/radial4.hpp"
#include "mesh/model/radial6.hpp"
#include "mesh/model/radial8.hpp"
#include "mesh/model/ring4.hpp"
#include "mesh/model/ring6.hpp"
#include "mesh/model/ring8.hpp"
#include "mesh/model/xmgrid4.hpp"
#include "mesh/model/xmgrid6.hpp"
#include "mesh/model/xmgrid8.hpp"
#include "mesh/model/xygrid4.hpp"
#include "mesh/model/xygrid6.hpp"
#include "mesh/model/xygrid8.hpp"

enum Args {
    COORDINATES_A = 0,
    COORDINATES_B = 1,
    MODEL         = 2,
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
class DifferenceVisualMeshOp : public tensorflow::OpKernel {
private:
    template <template <typename> class Model>
    void ComputeModel(tensorflow::OpKernelContext* context) {

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
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::GEOMETRY).shape()),
                    tensorflow::errors::InvalidArgument("Geometry must be a single string value"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::RADIUS).shape()),
                    tensorflow::errors::InvalidArgument("The radius must be a scalar"));

        // Extract information from our input tensors
        auto coordinates_a   = context->input(Args::COORDINATES_A).matrix<T>();
        auto coordinates_b   = context->input(Args::COORDINATES_B).matrix<T>();
        T height             = context->input(Args::HEIGHT).scalar<T>()(0);
        std::string geometry = *context->input(Args::GEOMETRY).flat<tensorflow::string>().data();
        T radius             = context->input(Args::RADIUS).scalar<T>()(0);
        auto n_elems         = context->input(Args::COORDINATES_A).shape().dim_size(1);

        // Perform some runtime checks on the actual values to make sure they make sense
        OP_REQUIRES(context,
                    geometry == "SPHERE" || geometry == "CIRCLE",
                    tensorflow::errors::InvalidArgument("Geometry must be one of SPHERE or CIRCLE"));


        // Create the output matrix to hold the vectors
        tensorflow::Tensor* vectors = nullptr;
        tensorflow::TensorShape vectors_shape;
        vectors_shape.AddDim(n_elems);
        vectors_shape.AddDim(2);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::DIFFERENCES, vectors_shape, &vectors));

        // Perform the map operation for this shape
        auto do_difference =
          [](const auto& n, const auto& c_a, const auto& c_b, auto& ds, const auto& shape, const auto& height) {
              for (int i = 0; i < n; ++i) {
                  visualmesh::vec2<T> d = Model<T>::difference(shape,
                                                               height,
                                                               visualmesh::vec2<T>({c_a(i, 0), c_a(i, 1)}),
                                                               visualmesh::vec2<T>({c_b(i, 0), c_b(i, 1)}));
                  ds(i, 0)              = d[0];
                  ds(i, 1)              = d[1];
              }
          };

        auto ds = vectors->matrix<T>();
        if (geometry == "SPHERE") {  //
            do_difference(n_elems, coordinates_a, coordinates_b, ds, visualmesh::geometry::Sphere<T>(radius), height);
        }
        else if (geometry == "CIRCLE") {
            do_difference(n_elems, coordinates_a, coordinates_b, ds, visualmesh::geometry::Circle<T>(radius), height);
        }
    }

public:
    explicit DifferenceVisualMeshOp(tensorflow::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext* context) override {

        // Check that the model is a string
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::MODEL).shape()),
                    tensorflow::errors::InvalidArgument("Model must be a single string value"));

        // Grab the Visual Mesh model we are using
        std::string model = *context->input(Args::MODEL).flat<tensorflow::string>().data();

        // clang-format off
        if (model == "RADIAL4") { ComputeModel<visualmesh::model::Radial4>(context); }
        else if (model == "RADIAL6") { ComputeModel<visualmesh::model::Radial6>(context); }
        else if (model == "RADIAL8") { ComputeModel<visualmesh::model::Radial8>(context); }
        else if (model == "RING4") { ComputeModel<visualmesh::model::Ring4>(context); }
        else if (model == "RING6") { ComputeModel<visualmesh::model::Ring6>(context); }
        else if (model == "RING8") { ComputeModel<visualmesh::model::Ring8>(context); }
        else if (model == "NMGRID4") { ComputeModel<visualmesh::model::NMGrid4>(context); }
        else if (model == "NMGRID6") { ComputeModel<visualmesh::model::NMGrid6>(context); }
        else if (model == "NMGRID8") { ComputeModel<visualmesh::model::NMGrid8>(context); }
        else if (model == "XMGRID4") { ComputeModel<visualmesh::model::XMGrid4>(context); }
        else if (model == "XMGRID6") { ComputeModel<visualmesh::model::XMGrid6>(context); }
        else if (model == "XMGRID8") { ComputeModel<visualmesh::model::XMGrid8>(context); }
        else if (model == "XYGRID4") { ComputeModel<visualmesh::model::XYGrid4>(context); }
        else if (model == "XYGRID6") { ComputeModel<visualmesh::model::XYGrid6>(context); }
        else if (model == "XYGRID8") { ComputeModel<visualmesh::model::XYGrid8>(context); }
        // clang-format on

        else {
            OP_REQUIRES(
              context,
              false,
              tensorflow::errors::InvalidArgument("The provided Visual Mesh model was not one of the known models"));
        }
    }
};

// Register a version for all the combinations of float/double
REGISTER_KERNEL_BUILDER(Name("DifferenceVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
                        DifferenceVisualMeshOp<float>)
REGISTER_KERNEL_BUILDER(Name("DifferenceVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T"),
                        DifferenceVisualMeshOp<double>)
