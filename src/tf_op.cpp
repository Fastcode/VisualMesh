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

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "engine/cpu/cpu_engine.hpp"
#include "geometry/Circle.hpp"
#include "geometry/Cylinder.hpp"
#include "geometry/Sphere.hpp"
#include "mesh/mesh.hpp"

REGISTER_OP("VisualMesh")
  .Attr("T: {float, double}")
  .Attr("U: {int32, int64}")
  .Input("image_dimensions: U")
  .Input("lens_type: string")
  .Input("lens_focal_length: T")
  .Input("lens_fov: T")
  .Input("lens_centre: T")
  .Input("cam_to_observation_plane: T")
  .Input("height: T")
  .Input("geometry: string")
  .Input("geometry_params: T")
  .Output("pixels: T")
  .Output("neighbours: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // Lots of these for checking the various things


    // nx2 points on image, and n+1x7 neighbours (including off screen point)
    c->set_output(0, c->MakeShape({c->kUnknownDim, 2}));
    c->set_output(1, c->MakeShape({c->kUnknownDim, 7}));
    return tensorflow::Status::OK();
  });

enum Args {
  DIMENSIONS      = 0,
  PROJECTION      = 1,
  FOCAL_LENGTH    = 2,
  FIELD_OF_VIEW   = 3,
  LENS_CENTRE     = 4,
  ROC             = 5,
  HEIGHT          = 6,
  GEOMETRY        = 7,
  GEOMETRY_PARAMS = 8
};

template <typename T, typename U>
class VisualMeshOp : public tensorflow::OpKernel {
public:
  explicit VisualMeshOp(tensorflow::OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {

    // Check that the shape of each of the inputs is valid
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsVector(context->input(Args::DIMENSIONS).shape())
                  && context->input(Args::DIMENSIONS).shape().dim_size(0) == 2,
                tensorflow::errors::InvalidArgument("The image dimensions must be a 2d vector of [y_size, x_size]"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsScalar(context->input(Args::FOCAL_LENGTH).shape()),
                tensorflow::errors::InvalidArgument("The focal length must be a scalar"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsScalar(context->input(Args::FIELD_OF_VIEW).shape()),
                tensorflow::errors::InvalidArgument("The field of view must be a scalar"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsVector(context->input(Args::LENS_CENTRE).shape())
                  && context->input(Args::LENS_CENTRE).shape().dim_size(0) == 2,
                tensorflow::errors::InvalidArgument("The lens centre must be a 2d vector of [y_size, x_size]"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsSquareMatrix(context->input(Args::ROC).shape())
                  && context->input(Args::LENS_CENTRE).shape().dim_size(0) == 3,
                tensorflow::errors::InvalidArgument("Roc must be a 3x3 matrix"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsScalar(context->input(Args::HEIGHT).shape()),
                tensorflow::errors::InvalidArgument("The height value must be a scalar"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsScalar(context->input(Args::GEOMETRY).shape()),
                tensorflow::errors::InvalidArgument("Geometry must be a single string value"));
    OP_REQUIRES(context,
                tensorflow::TensorShapeUtils::IsVector(context->input(Args::GEOMETRY_PARAMS).shape())
                  tensorflow::errors::InvalidArgument("The geometry params must be a vector of the required parts"));

    // Extract information from our input tensors, flip x and y as tensorflow has them reversed compared to us
    auto image_dimensions                = context->input(Args::DIMENSIONS).vec<U>();
    visualmesh::vec2<int32_t> dimensions = {{int32_t(image_dimensions(1)), int32_t(image_dimensions(0))}};
    std::string projection               = *context->input(Args::PROJECTION).flat<tensorflow::string>().data();
    T focal_length                       = context->input(Args::FOCAL_LENGTH).scalar<T>()(0);
    T fov                                = context->input(Args::FIELD_OF_VIEW).scalar<T>()(0);
    auto lens_centre                     = context->input(Args::LENS_CENTRE).flat<T>();
    auto tRoc                            = context->input(Args::ROC).matrix<T>();
    T height                             = context->input(Args::HEIGHT).scalar<T>()(0);
    std::string geometry                 = *context->input(Args::GEOMETRY).flat<tensorflow::string>().data();
    auto g_params                        = context->input(Args::GEOMETRY_PARAMS).vec<T>();

    // Perform some runtime checks on the actual values to make sure they make sense
    OP_REQUIRES(context,
                !(projection == "EQUISOLID" || projection == "EQUIDISTANT" || projection == "RECTILINEAR"),
                tensorflow::errors::InvalidArgument("Projection must be one of EQUISOLID, EQUIDISTANT or RECTILINEAR"));
    OP_REQUIRES(context,
                !(geometry == "SPHERE" || geometry == "CIRCLE" || geometry == "CYLINDER"),
                tensorflow::errors::InvalidArgument("Geometry must be one of SPHERE, CIRCLE or CYLINDER"));

    // Create our transformation matrix
    visualmesh::mat4<T> Hoc = {{
      visualmesh::vec4<T>{tRoc(0, 0), tRoc(0, 1), tRoc(0, 2), 0},
      visualmesh::vec4<T>{tRoc(1, 0), tRoc(1, 1), tRoc(1, 2), 0},
      visualmesh::vec4<T>{tRoc(2, 0), tRoc(2, 1), tRoc(2, 2), height},
      visualmesh::vec4<T>{0, 0, 0, 1},
    }};

    // Create our lens
    visualmesh::Lens<T> lens;
    lens.dimensions   = dimensions;
    lens.focal_length = focal_length;
    lens.fov          = fov;
    lens.centre       = {{lens_centre(1), lens_centre(0)}};  // Swap from tf coordinates to our coordinates
    if (projection == "EQUISOLID") {
      lens.projection = visualmesh::EQUISOLID;  //
    }
    else if (projection == "EQUIDISTANT") {
      lens.projection = visualmesh::EQUIDISTANT;
    }
    else if (projection == "RECTILINEAR") {
      lens.projection = visualmesh::RECTILINEAR;
    }
    else {
      // TODO work out how to throw an error
    }

    // Project the mesh using our engine and shape
    visualmesh::engine::cpu::Engine<T> engine;
    visualmesh::ProjectedMesh<T> projected;
    if (geometry == "SPHERE") {
      visualmesh::geometry::Sphere<T> shape(g_params(0));
      // TODO cache this, building a BSP is no joke
      visualmesh::Mesh<T> mesh(shape, height, g_params(1), g_params(2));
      projected = engine.project(mesh, mesh.lookup(Hoc, lens), Hoc, lens);
    }
    else if (geometry == "CIRCLE") {
      visualmesh::geometry::Circle<T> shape(g_params(0));
      // TODO cache this, building a BSP is no joke
      visualmesh::Mesh<T> mesh(shape, height, g_params(1), g_params(2));
      projected = engine.project(mesh, mesh.lookup(Hoc, lens), Hoc, lens);
    }
    else if (geometry == "CYLINDER") {
      visualmesh::geometry::Cylinder<T> shape(g_params(0), g_params(1));
      // TODO cache this, building a BSP is no joke
      visualmesh::Mesh<T> mesh(shape, height, g_params(2), g_params(3));
      projected = engine.project(mesh, mesh.lookup(Hoc, lens), Hoc, lens);
    }
    else {
      // TODO work out how to throw an error
    }

    // Get the interesting things out of the projected mesh
    const auto& px            = projected.pixel_coordinates;
    const auto& neighbourhood = projected.neighbourhood;

    // Fill in our tensorflow output matrix
    tensorflow::Tensor* coordinates = nullptr;
    tensorflow::TensorShape coords_shape;
    coords_shape.AddDim(px.size());
    coords_shape.AddDim(2);
    OP_REQUIRES_OK(context, context->allocate_output(0, coords_shape, &coordinates));

    // Copy across our pixel coordinates remembering to reverse the order from x,y to y,x
    auto c = coordinates->matrix<T>();
    for (size_t i = 0; i < px.size(); ++i) {
      // Swap x and y here since tensorflow expects them reversed
      const auto& p = px[i];
      c(i, 0)       = p[1];
      c(i, 1)       = p[0];
    }

    // Build our tensorflow neighbourhood graph
    tensorflow::Tensor* neighbours = nullptr;
    tensorflow::TensorShape neighbours_shape;
    neighbours_shape.AddDim(neighbourhood.size());
    neighbours_shape.AddDim(7);
    OP_REQUIRES_OK(context, context->allocate_output(1, neighbours_shape, &neighbours));

    // Copy across our neighbourhood graph, adding in a point for itself
    auto n = neighbours->matrix<U>();
    for (int i = 0; i < neighbourhood.size(); ++i) {
      // Get our old neighbours from original output
      const auto& m = neighbourhood[i];
      n(i, 0)       = i;
      n(i, 1)       = m[0];
      n(i, 2)       = m[1];
      n(i, 3)       = m[2];
      n(i, 4)       = m[3];
      n(i, 5)       = m[4];
      n(i, 6)       = m[5];
    }
  }
};

REGISTER_KERNEL_BUILDER(
  Name("VisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int32>("U"),
  VisualMeshOp<float, tensorflow::int32>);
REGISTER_KERNEL_BUILDER(
  Name("VisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T").TypeConstraint<tensorflow::int64>("U"),
  VisualMeshOp<float, tensorflow::int64>);
REGISTER_KERNEL_BUILDER(
  Name("VisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int32>("U"),
  VisualMeshOp<double, tensorflow::int32>);
REGISTER_KERNEL_BUILDER(
  Name("VisualMesh").Device(tensorflow::DEVICE_CPU).TypeConstraint<double>("T").TypeConstraint<tensorflow::int64>("U"),
  VisualMeshOp<double, tensorflow::int64>);
