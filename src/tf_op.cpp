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

REGISTER_OP("VisualMesh")
  .Input("shape: ")
  .Input("lens: lens")
  .Input("n_sample_points: int32")
  .Input("cam_to_observation_plane: float32")
  .Output("pixels: float32")
  .Output("neighbors: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // c->set_output(0, SHAPE_OF_PIXELS);
    // c->set_output(1, SHAPE_OF_NEIGHBOURS);
    return tensorflow::Status::OK();
  });

class VisualMeshOp : public tensorflow::OpKernel {
public:
  explicit VisualMeshOp(tensorflow::OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {

    // TODO grab the shape, lens, n_sample_points and cam_to_observation_plane

    // TODO Perform a projection operation using the visual mesh

    // Return the pixels and neighbourhood graph
  }
};

REGISTER_KERNEL_BUILDER(Name("VisualMesh").Device(tensorflow::DEVICE_CPU), VisualMeshOp);
