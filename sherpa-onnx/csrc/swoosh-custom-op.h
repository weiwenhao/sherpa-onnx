// sherpa-onnx/csrc/swoosh-custom-op.h
//
// 把 ZipVoice decoder 里的 SwooshR/SwooshL 激活注册成一个自定义算子。
//
// ONNX 导出后每处 Swoosh 是 9 个逐元素算子串联，每个算子都完整遍历一遍激活
// 张量。decoder 里有 80 处，实测占 decoder 25% 的耗时、31% 的能耗。
// 模型侧用 scripts/build_zipvoice_swoosh_fused.py 把这些链替换成本算子。
#ifndef SHERPA_ONNX_CSRC_SWOOSH_CUSTOM_OP_H_
#define SHERPA_ONNX_CSRC_SWOOSH_CUSTOM_OP_H_

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/swoosh-kernel.h"

namespace sherpa_onnx {

// 与模型改写脚本里的 domain 必须一致
constexpr char kSwooshOpDomain[] = "k2fsa.zipvoice";

class SwooshKernelImpl {
 public:
  SwooshKernelImpl(const OrtApi & /*api*/, const OrtKernelInfo *info) {
    Ort::ConstKernelInfo kernel_info{info};
    offset_ = kernel_info.GetAttribute<float>("offset");
    bias_ = kernel_info.GetAttribute<float>("bias");
  }

  void Compute(OrtKernelContext *context) {
    Ort::KernelContext ctx(context);
    auto input = ctx.GetInput(0);
    auto type_shape = input.GetTensorTypeAndShapeInfo();
    auto shape = type_shape.GetShape();
    auto output = ctx.GetOutput(0, shape);
    SwooshForward(input.GetTensorData<float>(),
                  output.GetTensorMutableData<float>(),
                  type_shape.GetElementCount(), offset_, bias_);
  }

 private:
  float offset_ = 1.0f;
  float bias_ = 0.313261687f;
};

struct SwooshCustomOp : Ort::CustomOpBase<SwooshCustomOp, SwooshKernelImpl> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return new SwooshKernelImpl(api, info);
  }

  const char *GetName() const { return "Swoosh"; }

  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  // 逐元素算子，输出形状同输入。
  // ORT 的自定义算子形状推导跑在常规推导之前，此时上游形状可能还是未知的；
  // 这种情况下不要写出一个空形状去覆盖模型里已有的 value_info。
  static Ort::Status InferOutputShape(Ort::ShapeInferContext &ctx) {
    const auto &shape = ctx.GetInputShape(0);
    if (!shape.empty()) {
      ctx.SetOutputShape(0, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    return Ort::Status{nullptr};
  }
};

// ORT 不接管 op 与 domain 的所有权，二者必须活得比所有 session 长，
// 所以用函数内静态变量。
inline Ort::CustomOpDomain &GetSwooshCustomOpDomain() {
  static SwooshCustomOp op;
  static Ort::CustomOpDomain domain = [] {
    Ort::CustomOpDomain d{kSwooshOpDomain};
    d.Add(&op);
    return d;
  }();
  return domain;
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SWOOSH_CUSTOM_OP_H_
