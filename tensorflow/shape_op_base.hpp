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

#ifndef VISUALMESH_TENSORFLOW_SHAPE_OP_BASE_HPP
#define VISUALMESH_TENSORFLOW_SHAPE_OP_BASE_HPP

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "geometry/Circle.hpp"
#include "geometry/Sphere.hpp"

template <typename T, typename Subclass, int GEOMETRY, int RADIUS>
class ShapeOpBase : public tensorflow::OpKernel {
public:
    explicit ShapeOpBase(tensorflow::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext* context) override {

        // Check the geometry type and radius
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(GEOMETRY).shape()),
                    tensorflow::errors::InvalidArgument("Geometry must be a single string value"));
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(RADIUS).shape()),
                    tensorflow::errors::InvalidArgument("The radius must be a scalar"));

        std::string geometry = *context->input(GEOMETRY).flat<tensorflow::tstring>().data();
        T radius             = context->input(RADIUS).scalar<T>()(0);

        OP_REQUIRES(context,
                    geometry == "SPHERE" || geometry == "CIRCLE",
                    tensorflow::errors::InvalidArgument("Geometry must be one of SPHERE or CIRCLE"));

        // clang-format off
        using namespace visualmesh::geometry;
        if (geometry == "SPHERE") { static_cast<Subclass*>(this)->DoCompute(context, Sphere<T>(radius)); }
        else if (geometry == "CIRCLE") { static_cast<Subclass*>(this)->DoCompute(context, Circle<T>(radius)); }
        // clang-format on
    }
};

#endif  // VISUALMESH_TENSORFLOW_SHAPE_OP_BASE_HPP
