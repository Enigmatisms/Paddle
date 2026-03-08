// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/reduce_nansum_grad_kernel.h"

#include <set>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void NanSumGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& dims_arr,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  reduce_all = recompute_reduce_all(x, dims_arr, reduce_all);
  auto dims = dims_arr.GetData();
  dev_ctx.Alloc(x_grad, x.dtype());
  auto* out_data = reinterpret_cast<const XPUType*>(out_grad.data());
  auto* x_grad_data = reinterpret_cast<XPUType*>(x_grad->data());
  const auto& input_dim_size = x.dims().size();
  std::vector<int64_t> true_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      true_dims.push_back(dims[i] + input_dim_size);
    } else {
      true_dims.push_back(dims[i]);
    }
  }

  std::vector<int64_t> ydims(input_dim_size);
  std::vector<int64_t> xdims(input_dim_size);
  std::set<int64_t> dims_set(true_dims.begin(), true_dims.end());
  for (auto i = 0; i < input_dim_size; i++) {
    xdims[i] = x.dims()[i];
    if (dims_set.find(i) != dims_set.end() || reduce_all) {
      ydims[i] = 1;
    } else {
      ydims[i] = x.dims()[i];
    }
  }

  // use [1] to replace [], because xpu not support []
  if (xdims.size() == 0) {
    xdims = std::vector<int64_t>({1});
  }
  if (ydims.size() == 0) {
    ydims = std::vector<int64_t>({1});
  }

  // Step 1: broadcast out_grad to x_grad shape (same as sum_grad)
  int r = xpu::broadcast<XPUType>(
      dev_ctx.x_context(), out_data, x_grad_data, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  // Step 2: zero out gradient where x is NaN
  int64_t numel = x.numel();
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  bool* nan_mask = RAII_GUARD.alloc<bool>(numel);
  PADDLE_ENFORCE_NOT_NULL(
      nan_mask, common::errors::ResourceExhausted("XPU has no enough memory"));
  r = xpu::isnan<XPUType>(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(x.data<T>()),
                          nan_mask,
                          numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "isnan");

  XPUType* zeros = RAII_GUARD.alloc<XPUType>(numel);
  PADDLE_ENFORCE_NOT_NULL(
      zeros, common::errors::ResourceExhausted("XPU has no enough memory"));
  r = xpu::constant<XPUType>(dev_ctx.x_context(), zeros, numel, XPUType(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  // where(condition=nan_mask, x_true=zeros, y_false=x_grad) → pick 0 where NaN
  std::vector<int64_t> shape = {numel};
  r = xpu::where<XPUType>(dev_ctx.x_context(),
                          nan_mask,
                          zeros,
                          x_grad_data,
                          x_grad_data,
                          shape,
                          shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "where");
}

}  // namespace phi

PD_REGISTER_KERNEL(nansum_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::NanSumGradKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
