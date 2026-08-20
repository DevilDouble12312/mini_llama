#pragma once 


#include "mini_llama/model.h"
#include "mini_llama/tensor.h"


namespace mini_llama {
  Tensor Matmul(const Tensor& a, const Tensor& b);

  Tensor Linear(const Tensor& x, const Tensor& weight);
  Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

  Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

  Tensor Softmax(const Tensor& x);

  Tensor Silu(const Tensor& x);

  Tensor SwiGlu(const Tensor& gate, const Tensor& up);

  Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

  int ArgMax(const Tensor& x);

  void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type = RopeType::kNormal);
}
