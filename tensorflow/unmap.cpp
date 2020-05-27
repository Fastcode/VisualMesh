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
    VECTORS  = 0,
    MODEL    = 1,
    HEIGHT   = 2,
    GEOMETRY = 3,
    RADIUS   = 4,
};

enum Outputs {
    COORDINATES = 0,
};

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
class UnmapVisualMeshOp : public tensorflow::OpKernel {
private:
    template <template <typename> class Model>
    void ComputeModel(tensorflow::OpKernelContext* context) {

        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsMatrix(context->input(Args::VECTORS).shape())
                      && context->input(Args::VECTORS).shape().dim_size(1) == 3,
                    tensorflow::errors::InvalidArgument("Vectors must be an nx3 matrix of unit vectors"));
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
        auto vectors         = context->input(Args::VECTORS).matrix<T>();
        T height             = context->input(Args::HEIGHT).scalar<T>()(0);
        std::string geometry = *context->input(Args::GEOMETRY).flat<tensorflow::tstring>().data();
        T radius             = context->input(Args::RADIUS).scalar<T>()(0);
        auto n_elems         = context->input(Args::VECTORS).shape().dim_size(1);

        // Perform some runtime checks on the actual values to make sure they make sense
        OP_REQUIRES(context,
                    geometry == "SPHERE" || geometry == "CIRCLE",
                    tensorflow::errors::InvalidArgument("Geometry must be one of SPHERE or CIRCLE"));


        // Create the output matrix to hold the coordinates
        tensorflow::Tensor* coordinates = nullptr;
        tensorflow::TensorShape coordinates_shape;
        coordinates_shape.AddDim(n_elems);
        coordinates_shape.AddDim(3);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::COORDINATES, coordinates_shape, &coordinates));

        // Perform the map operation for this shape
        auto do_unmap = [](const auto& n, const auto& vectors, auto& cs, const auto& shape, const auto& height) {
            for (int i = 0; i < n; ++i) {
                visualmesh::vec2<T> c =
                  Model<T>::unmap(shape, height, visualmesh::vec3<T>({vectors(i, 0), vectors(i, 1), vectors(i, 2)}));
                cs(i, 0) = c[0];
                cs(i, 1) = c[1];
            }
        };

        auto cs = coordinates->matrix<T>();
        if (geometry == "SPHERE") {  //
            do_unmap(n_elems, vectors, cs, visualmesh::geometry::Sphere<T>(radius), height);
        }
        else if (geometry == "CIRCLE") {
            do_unmap(n_elems, vectors, cs, visualmesh::geometry::Circle<T>(radius), height);
        }
    }

public:
    explicit UnmapVisualMeshOp(tensorflow::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext* context) override {

        // Check that the model is a string
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::MODEL).shape()),
                    tensorflow::errors::InvalidArgument("Model must be a single string value"));

        // Grab the Visual Mesh model we are using
        std::string model = *context->input(Args::MODEL).flat<tensorflow::tstring>().data();

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
REGISTER_KERNEL_BUILDER(Name("UnmapVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
                        UnmapVisualMeshOp<float>)
REGISTER_KERNEL_BUILDER(Name("UnmapVisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T"),
                        UnmapVisualMeshOp<double>)
