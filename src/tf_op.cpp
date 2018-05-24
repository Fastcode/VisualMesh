#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

REGISTER_OP("VisualMesh")
  .Input("shape: ")
  .Input("sample_points: int32")
  .Input("cam_to_observation_plane: float32")
  .Output("name: type")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) { return tensorflow::Status::OK(); });

class VisualMeshOp : public tensorflow::OpKernel {
public:
  explicit VisualMeshOp(tensorflow::OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Grab the input tensor
    const tensorflow::Tensor& input_tensor = context->input(0);
    auto input                             = input_tensor.flat<tensorflow::int32>();

    // Create an output tensor
    tensorflow::Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<tensorflow::int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("VisualMesh").Device(tensorflow::DEVICE_CPU), VisualMeshOp);
