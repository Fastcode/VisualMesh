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

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mesh_cache.hpp"
#include "model_op_base.hpp"
#include "visualmesh/lens.hpp"
#include "visualmesh/mesh.hpp"
#include "visualmesh/utility/math.hpp"

enum Args {
    DIMENSIONS             = 0,
    PROJECTION             = 1,
    FOCAL_LENGTH           = 2,
    LENS_CENTRE            = 3,
    LENS_DISTORTION        = 4,
    FIELD_OF_VIEW          = 5,
    HOC                    = 6,
    MESH_MODEL             = 7,
    CACHED_MESHES          = 8,
    MAX_DISTANCE           = 9,
    GEOMETRY               = 10,
    RADIUS                 = 11,
    N_INTERSECTIONS        = 12,
    INTERSECTION_TOLERANCE = 13,
};

enum Outputs {
    VECTORS    = 0,
    NEIGHBOURS = 1,
};

// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_OP("LookupVisualMesh")
  .Attr("T: {float, double}")
  .Attr("U: {int32, int64}")
  .Input("image_dimensions: U")
  .Input("lens_projection: string")
  .Input("lens_focal_length: T")
  .Input("lens_centre: T")
  .Input("lens_distortion: T")
  .Input("lens_fov: T")
  .Input("cam_to_observation_plane: T")
  .Input("mesh_model: string")
  .Input("cached_meshes: int32")
  .Input("max_distance: T")
  .Input("geometry: string")
  .Input("radius: T")
  .Input("n_intersections: T")
  .Input("intersection_tolerance: T")
  .Output("vectors: T")
  .Output("neighbours: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      constexpr auto kUnknownDim = ::tensorflow::shape_inference::InferenceContext::kUnknownDim;
      // nx2 vectors on image, n+1xG neighbours (including off screen point), and n global indices
      c->set_output(Outputs::VECTORS, c->MakeShape({kUnknownDim, 3}));
      c->set_output(Outputs::NEIGHBOURS, c->MakeShape({kUnknownDim, kUnknownDim}));
      return tensorflow::OkStatus();
  });

/**
 * @brief The Visual Mesh projection op
 *
 * @details
 *  This op will perform a projection using the visual mesh and will return the neighbourhood graph and the pixel
 * coordinates for the points that would be on screen for the lens paramters provided.
 *
 * @tparam T The scalar type used for floating point numbers
 * @tparam U The scalar type used for integer numbers
 */
template <typename T, typename U>
class LookupVisualMeshOp
  : public ModelOpBase<T, LookupVisualMeshOp<T, U>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS> {
public:
    explicit LookupVisualMeshOp(tensorflow::OpKernelConstruction* context)
      : ModelOpBase<T, LookupVisualMeshOp<T, U>, Args::MESH_MODEL, Args::GEOMETRY, Args::RADIUS>(context) {}

    template <template <typename> class Model, typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {

        // Check that the shape of each of the inputs is valid
        OP_REQUIRES(
          context,
          tensorflow::TensorShapeUtils::IsVector(context->input(Args::DIMENSIONS).shape())
            && context->input(Args::DIMENSIONS).shape().dim_size(0) == 2,
          tensorflow::errors::InvalidArgument("The image dimensions must be a 2d vector of [y_size, x_size]"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::FOCAL_LENGTH).shape()),
                    tensorflow::errors::InvalidArgument("The focal length must be a scalar"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsVector(context->input(Args::LENS_CENTRE).shape())
                      && context->input(Args::LENS_CENTRE).shape().dim_size(0) == 2,
                    tensorflow::errors::InvalidArgument("The lens centre must be a 2d vector of [y_size, x_size]"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsVector(context->input(Args::LENS_DISTORTION).shape())
                      && context->input(Args::LENS_DISTORTION).shape().dim_size(0) == 2,
                    tensorflow::errors::InvalidArgument("The lens distortion must be a 2d vector of [y_size, x_size]"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::FIELD_OF_VIEW).shape()),
                    tensorflow::errors::InvalidArgument("The field of view must be a scalar"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsSquareMatrix(context->input(Args::HOC).shape())
                      && context->input(Args::HOC).shape().dim_size(0) == 4,
                    tensorflow::errors::InvalidArgument("Hoc must be a 4x4 matrix"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::N_INTERSECTIONS).shape()),
                    tensorflow::errors::InvalidArgument("The number of intersections must be a scalar"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::CACHED_MESHES).shape()),
                    tensorflow::errors::InvalidArgument("The number cached meshes must be a scalar"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::INTERSECTION_TOLERANCE).shape()),
                    tensorflow::errors::InvalidArgument("The intersection tolerance must be a scalar"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(Args::MAX_DISTANCE).shape()),
                    tensorflow::errors::InvalidArgument("The maximum distance must be a scalar"));

        // Extract information from our input tensors, flip x and y as tensorflow has them reversed compared to us
        auto image_dimensions                = context->input(Args::DIMENSIONS).vec<U>();
        visualmesh::vec2<int32_t> dimensions = {{int32_t(image_dimensions(1)), int32_t(image_dimensions(0))}};
        std::string projection               = *context->input(Args::PROJECTION).flat<tensorflow::tstring>().data();
        T focal_length                       = context->input(Args::FOCAL_LENGTH).scalar<T>()(0);
        auto lens_centre                     = context->input(Args::LENS_CENTRE).flat<T>();
        auto lens_distortion                 = context->input(Args::LENS_DISTORTION).flat<T>();
        T fov                                = context->input(Args::FIELD_OF_VIEW).scalar<T>()(0);
        auto tHoc                            = context->input(Args::HOC).matrix<T>();
        T max_distance                       = context->input(Args::MAX_DISTANCE).scalar<T>()(0);
        T n_intersections                    = context->input(Args::N_INTERSECTIONS).scalar<T>()(0);
        tensorflow::int32 cached_meshes      = context->input(Args::CACHED_MESHES).scalar<tensorflow::int32>()(0);
        T intersection_tolerance             = context->input(Args::INTERSECTION_TOLERANCE).scalar<T>()(0);

        // Perform some runtime checks on the actual values to make sure they make sense
        OP_REQUIRES(
          context,
          projection == "EQUISOLID" || projection == "EQUIDISTANT" || projection == "RECTILINEAR",
          tensorflow::errors::InvalidArgument("Projection must be one of EQUISOLID, EQUIDISTANT or RECTILINEAR"));

        // Create our transformation matrix
        visualmesh::mat4<T> Hoc = {{
          visualmesh::vec4<T>{tHoc(0, 0), tHoc(0, 1), tHoc(0, 2), tHoc(0, 3)},
          visualmesh::vec4<T>{tHoc(1, 0), tHoc(1, 1), tHoc(1, 2), tHoc(1, 3)},
          visualmesh::vec4<T>{tHoc(2, 0), tHoc(2, 1), tHoc(2, 2), tHoc(2, 3)},
          visualmesh::vec4<T>{tHoc(3, 0), tHoc(3, 1), tHoc(3, 2), tHoc(3, 3)},
        }};

        // Create our lens
        visualmesh::Lens<T> lens{};
        lens.dimensions   = dimensions;
        lens.focal_length = focal_length;
        lens.centre       = {{lens_centre(1), lens_centre(0)}};  // Swap from tf coordinates to our coordinates
        lens.k            = {{lens_distortion(0), lens_distortion(1)}};
        lens.fov          = fov;

        // clang-format off
        if (projection == "EQUISOLID") { lens.projection = visualmesh::EQUISOLID; }
        else if (projection == "EQUIDISTANT") { lens.projection = visualmesh::EQUIDISTANT; }
        else if (projection == "RECTILINEAR") { lens.projection = visualmesh::RECTILINEAR; }
        // clang-format on

        // Get a mesh that matches from the mesh cache
        std::shared_ptr<visualmesh::Mesh<T, Model>> mesh =
          get_mesh<T, Model>(shape, Hoc[2][3], n_intersections, intersection_tolerance, cached_meshes, max_distance);

        // Grab the ranges
        auto ranges       = mesh->lookup(Hoc, lens);
        const auto& nodes = mesh->nodes;

        // Work out how many points total there are in the ranges
        unsigned int n_points = 0;
        for (auto& r : ranges) {
            n_points += r.second - r.first;
        }

        // Allocate our outputs
        tensorflow::Tensor* vectors = nullptr;
        tensorflow::TensorShape vectors_shape;
        vectors_shape.AddDim(n_points);
        vectors_shape.AddDim(3);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::VECTORS, vectors_shape, &vectors));

        tensorflow::Tensor* neighbours = nullptr;
        tensorflow::TensorShape neighbours_shape;
        neighbours_shape.AddDim(n_points);
        neighbours_shape.AddDim(Model<T>::N_NEIGHBOURS + 1);
        OP_REQUIRES_OK(context, context->allocate_output(Outputs::NEIGHBOURS, neighbours_shape, &neighbours));

        // Build the lookup for the graph so we can find the new location of points
        std::vector<int> r_lookup(nodes.size() + 1, std::numeric_limits<tensorflow::int32>::lowest());
        {
            int idx = 0;
            for (const auto& r : ranges) {
                for (int i = r.first; i < r.second; ++i) {
                    r_lookup[i] = idx++;
                }
            }
        }

        // Copy across the unit vectors we looked up
        {
            auto v  = vectors->matrix<T>();
            auto n  = neighbours->matrix<tensorflow::int32>();
            int idx = 0;
            for (const auto& r : ranges) {
                for (int i = r.first; i < r.second; ++i) {

                    // Copy across the ray
                    const auto& r = nodes[i].ray;
                    v(idx, 0)     = r[0];
                    v(idx, 1)     = r[1];
                    v(idx, 2)     = r[2];

                    // Copy across the graph points in their new position
                    const auto& node = nodes[i];
                    n(idx, 0)        = idx;
                    for (int j = 0; j < Model<T>::N_NEIGHBOURS; ++j) {
                        n(idx, j + 1) = r_lookup[node.neighbours[j]];
                    }

                    // Next value to fill
                    ++idx;
                }
            }
        }
    }
};

// Register a version for all the combinations of float/double and int32/int64
// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_KERNEL_BUILDER(Name("LookupVisualMesh")
                          .Device(tensorflow::DEVICE_CPU)
                          .TypeConstraint<float>("T")
                          .TypeConstraint<tensorflow::int32>("U"),
                        LookupVisualMeshOp<float, tensorflow::int32>)
// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_KERNEL_BUILDER(Name("LookupVisualMesh")
                          .Device(tensorflow::DEVICE_CPU)
                          .TypeConstraint<float>("T")
                          .TypeConstraint<tensorflow::int64>("U"),
                        LookupVisualMeshOp<float, tensorflow::int64>)
// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_KERNEL_BUILDER(Name("LookupVisualMesh")
                          .Device(tensorflow::DEVICE_CPU)
                          .TypeConstraint<double>("T")
                          .TypeConstraint<tensorflow::int32>("U"),
                        LookupVisualMeshOp<double, tensorflow::int32>)
// NOLINTNEXTLINE(cert-err58-cpp) this macro makes a static variable
REGISTER_KERNEL_BUILDER(Name("LookupVisualMesh")
                          .Device(tensorflow::DEVICE_CPU)
                          .TypeConstraint<double>("T")
                          .TypeConstraint<tensorflow::int64>("U"),
                        LookupVisualMeshOp<double, tensorflow::int64>)
