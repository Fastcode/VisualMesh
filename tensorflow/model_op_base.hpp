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

#ifndef VISUALMESH_TENSORFLOW_MODEL_OP_BASE_HPP
#define VISUALMESH_TENSORFLOW_MODEL_OP_BASE_HPP

#include "shape_op_base.hpp"
#include "visualmesh/model/nmgrid4.hpp"
#include "visualmesh/model/nmgrid6.hpp"
#include "visualmesh/model/nmgrid8.hpp"
#include "visualmesh/model/ring4.hpp"
#include "visualmesh/model/ring6.hpp"
#include "visualmesh/model/ring8.hpp"
#include "visualmesh/model/xmgrid4.hpp"
#include "visualmesh/model/xmgrid6.hpp"
#include "visualmesh/model/xmgrid8.hpp"
#include "visualmesh/model/xygrid4.hpp"
#include "visualmesh/model/xygrid6.hpp"
#include "visualmesh/model/xygrid8.hpp"

template <typename T, typename Subclass, int MESH_MODEL, int GEOMETRY, int RADIUS>
class ModelOpBase : public ShapeOpBase<T, ModelOpBase<T, Subclass, MESH_MODEL, GEOMETRY, RADIUS>, GEOMETRY, RADIUS> {
public:
    explicit ModelOpBase(tensorflow::OpKernelConstruction* context)
      : ShapeOpBase<T, ModelOpBase<T, Subclass, MESH_MODEL, GEOMETRY, RADIUS>, GEOMETRY, RADIUS>(context) {}

    template <typename Shape>
    void DoCompute(tensorflow::OpKernelContext* context, const Shape& shape) {

        // Check that the model is a string
        OP_REQUIRES(context,
                    tensorflow::TensorShapeUtils::IsScalar(context->input(MESH_MODEL).shape()),
                    tensorflow::errors::InvalidArgument("Model must be a single string value"));

        // Grab the Visual Mesh model we are using
        std::string m = *context->input(MESH_MODEL).flat<tensorflow::tstring>().data();

        // clang-format off
        using namespace visualmesh::model; // NOLINT(google-build-using-namespace) function scope is fine
        if (m == "RING4") { static_cast<Subclass*>(this)->template DoCompute<Ring4>(context, shape); }
        else if (m == "RING6") { static_cast<Subclass*>(this)->template DoCompute<Ring6>(context, shape); }
        else if (m == "RING8") { static_cast<Subclass*>(this)->template DoCompute<Ring8>(context, shape); }
        else if (m == "NMGRID4") { static_cast<Subclass*>(this)->template DoCompute<NMGrid4>(context, shape); }
        else if (m == "NMGRID6") { static_cast<Subclass*>(this)->template DoCompute<NMGrid6>(context, shape); }
        else if (m == "NMGRID8") { static_cast<Subclass*>(this)->template DoCompute<NMGrid8>(context, shape); }
        else if (m == "XMGRID4") { static_cast<Subclass*>(this)->template DoCompute<XMGrid4>(context, shape); }
        else if (m == "XMGRID6") { static_cast<Subclass*>(this)->template DoCompute<XMGrid6>(context, shape); }
        else if (m == "XMGRID8") { static_cast<Subclass*>(this)->template DoCompute<XMGrid8>(context, shape); }
        else if (m == "XYGRID4") { static_cast<Subclass*>(this)->template DoCompute<XYGrid4>(context, shape); }
        else if (m == "XYGRID6") { static_cast<Subclass*>(this)->template DoCompute<XYGrid6>(context, shape); }
        else if (m == "XYGRID8") { static_cast<Subclass*>(this)->template DoCompute<XYGrid8>(context, shape); }
        // clang-format on

        else {
            OP_REQUIRES(
              context,
              false,
              tensorflow::errors::InvalidArgument("The provided Visual Mesh model was not one of the known models"));
        }
    }
};

#endif  // VISUALMESH_TENSORFLOW_MODEL_OP_BASE_HPP
