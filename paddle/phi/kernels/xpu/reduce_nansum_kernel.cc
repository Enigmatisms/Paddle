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

#include "paddle/phi/kernels/reduce_nansum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void NanSumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  DataType out_dtype,
                  bool keep_dim,
                  DenseTensor* out) {
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    out_dtype = out->dtype();
  }

  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    if (out_dtype == DataType::INT64) {
      Full<int64_t, Context>(dev_ctx, out->dims(), 0, out);
    } else {
      Full<T, Context>(dev_ctx, out->dims(), 0, out);
    }
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t numel = x.numel();

  // Step 1: compute isnan mask (bool)
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  bool* nan_mask = RAII_GUARD.alloc<bool>(numel);
  PADDLE_ENFORCE_NOT_NULL(
      nan_mask, common::errors::ResourceExhausted("XPU has no enough memory"));
  int r = xpu::isnan<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(x.data<T>()),
                              nan_mask,
                              numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "isnan");

  // Step 2: create cleaned_x = where(isnan, 0, x)
  DenseTensor cleaned_x;
  cleaned_x.Resize(x.dims());
  dev_ctx.template Alloc<T>(&cleaned_x);

  XPUType* zeros = RAII_GUARD.alloc<XPUType>(numel);
  PADDLE_ENFORCE_NOT_NULL(
      zeros, common::errors::ResourceExhausted("XPU has no enough memory"));
  r = xpu::constant<XPUType>(dev_ctx.x_context(), zeros, numel, XPUType(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  // where(condition=nan_mask, x_true=zeros, y_false=x) → pick 0 where NaN, x
  // otherwise
  std::vector<int64_t> shape = {numel};
  r = xpu::where<XPUType>(dev_ctx.x_context(),
                          nan_mask,
                          zeros,
                          reinterpret_cast<const XPUType*>(x.data<T>()),
                          reinterpret_cast<XPUType*>(cleaned_x.data<T>()),
                          shape,
                          shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "where");

  // Step 3: delegate to sum
  bool reduce_all = recompute_reduce_all(x, dims);
  XPUReduce<Context, T, phi::SumFunctor>(
      dev_ctx, cleaned_x, dims.GetData(), keep_dim, reduce_all, out_dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(nansum,
                   XPU,
                   ALL_LAYOUT,
                   phi::NanSumKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
