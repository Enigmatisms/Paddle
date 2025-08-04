// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/op/contrib/topk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/ir_simplify.h"

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::CINNValue;
using cinn::common::CINNValuePack;

inline common::Type GetArgIdxType(const Type &elem_type) {
  std::string rtype_name =
      "argidx" + pe::Type2StrForArgReduce(elem_type) + "_i64";
  Type customized_type(ir::Type::type_t::Customized,
                       /* bits = */ 128,
                       /* width = */ 1);
  customized_type.set_customized_type(rtype_name);
  customized_type.set_cpp_const(false);
  return customized_type;
}

Expr CallExternWithWriteArgs(const std::string &func_name,
                             std::vector<Expr> &&read_args,
                             std::vector<Expr> &&write_args,
                             const std::map<std::string, attr_t> &attrs) {
  auto *proto =
      backends::ExternFunctionProtoRegistry::Global().Lookup(func_name);
  PADDLE_ENFORCE_NOT_NULL(
      proto,
      ::common::errors::InvalidArgument(
          "No extern function prototype %s found\nExisting records are:\n%s",
          func_name,
          backends::ExternFunctionProtoRegistry::Global().debug_string()));
  PADDLE_ENFORCE_EQ(
      proto->ret_type.is_void(),
      true,
      ::common::errors::InvalidArgument(
          "Function '%s' does not return void, please check", func_name));
  PADDLE_ENFORCE_NE(
      write_args.empty(),
      true,
      ::common::errors::InvalidArgument(
          "Function '%s' write args should not be empty, please check",
          func_name));
  PADDLE_ENFORCE_EQ(
      write_args.size(),
      proto->mutable_arg_types.size(),
      ::common::errors::InvalidArgument(
          "Input write args size (%lu) is not expected (%lu) for function '%s'",
          write_args.size(),
          proto->mutable_arg_types.size(),
          func_name));

  auto call = ir::Call::Make(proto->ret_type,
                             func_name,
                             std::move(read_args),
                             {},
                             ir::CallType::Extern,
                             ir::FunctionRef(),
                             0,
                             attrs);
  for (size_t i = 0; i < write_args.size(); i++) {
    auto arg_expr = write_args[i];
    if (arg_expr.As<ir::Tensor>() == nullptr) continue;
    auto tensor = arg_expr.as_tensor_ref();
    auto expected_type = proto->mutable_arg_types[i];
    auto actual_type = tensor->type();
    PADDLE_ENFORCE_EQ(
        expected_type,
        actual_type,
        ::common::errors::InvalidArgument("Function '%s' write arg (%lu) type "
                                          "mismatch: expected '%s', got '%s'",
                                          func_name,
                                          expected_type.to_string().c_str(),
                                          actual_type.to_string().c_str()));
    auto op = ir::CallOp::Make(func_name, call);
    op->as<ir::CallOp>()->value_slot = i;
    op->as<ir::CallOp>()->is_tuple_get = true;
    tensor->operation = std::move(op);
  }
  call.As<ir::Call>()->write_args = std::move(write_args);

  return call;
}

// This function returns a GPU local memory stored array, used for insertion
// sort of parallel topk. This is a self-defined version of Compute
ir::Tensor ComputeWithLocalBuffer(const ir::Tensor &input_tensor,
                                  const common::Type &output_type,
                                  const std::string &sort_func_name,
                                  Expr topk_axis,
                                  Expr num_elements,
                                  Expr argidx_buffer_size) {
  auto unique_name = cinn::UniqName("topk_sorted");
  std::vector<Expr> local_shape = {argidx_buffer_size};
  VLOG(3) << "topk local tensor " << unique_name
          << "'s domain is : " << argidx_buffer_size;
  // local tensor is actually small (avoid using excessive amount of local
  // memory)

  auto op = ir::ComputeOp::Make(
      unique_name,
      [=](const std::vector<Expr> &indices) -> Expr {
        Expr offset(0);
        Expr stride(1);
        for (int i = 0; i < indices.size(); i++) {
          offset = offset * input_tensor->shape[i] + indices[i];
          if (i > topk_axis.as_int32()) {
            stride = stride * input_tensor->shape[i];
          }
        }
        offset = optim::ArithSimplify(offset);
        stride = optim::ArithSimplify(stride);

        std::vector<Expr> func_args = {
            input_tensor, num_elements, argidx_buffer_size, offset, stride};
        // TODO(heqianyue): is this good to have looped dependency:
        // local_array->operation (ComputeOp) needs local array itself
        return CallExternWithWriteArgs(
            sort_func_name, std::move(func_args), {local_array})
      },
      local_shape,
      local_shape,
      {});

  local_array->operation = op;
  for (int i = 0; i < local_array->axis_.size(); ++i) {
    local_array->axis_[i]->is_keepdim = true;
  }
  return local_array;
}

std::vector<ir::Tensor> TopK(const ir::Tensor &A,
                             const cinn::common::Target &target,
                             const int &axis,
                             const int topk,
                             const bool largest,
                             const std::string &name) {
  std::string insert_sort_func;
  std::string block_reduce_func;
#define THROW_WITH_UNSUPPORTED_TARGET(arch)             \
  [&](common::arch) {                                   \
    PADDLE_THROW(::common::errors::Fatal(               \
        "TopK only supports NVGPU ! Please Check.\n")); \
  }

  // TODO(heqianyue): cinn_nvgpu_merge_sorted might need a optimization pass to
  // decide whether we need block reduce merge or grid reduce merge

  target.arch.Match(THROW_WITH_UNSUPPORTED_TARGET(UnknownArch),
                    THROW_WITH_UNSUPPORTED_TARGET(X86Arch),
                    THROW_WITH_UNSUPPORTED_TARGET(ARMArch),
                    THROW_WITH_UNSUPPORTED_TARGET(HygonDCUArchHIP),
                    THROW_WITH_UNSUPPORTED_TARGET(HygonDCUArchSYCL),
                    THROW_WITH_UNSUPPORTED_TARGET(HygonDCUArchSYCL),
                    [&](common::NVGPUArch) {
                      insert_sort_func.assign("cinn_nvgpu_insertion_sort");
                      block_reduce_func.assign("cinn_nvgpu_merge_sorted");
                    });
#undef THROW_WITH_UNSUPPORTED_TARGET
  // phase 1: insert sort and maintain a thread local sorted argidx array

  // eliminate negative axis
  int topk_axis = axis;
  if (topk_axis < 0) topk_axis += A->shape.size();

  PADDLE_ENFORCE_GE(
      topk,
      0,
      ::common::errors::Fatal(
          "TopK topk value should be greater than or equal to 0."));
  PADDLE_ENFORCE_LE(
      topk,
      A->shape[topk_axis].as_int(),
      ::common::errors::Fatal("TopK topk value should be less than or equal to "
                              "the length of the axis dimension."));

  const int argidx_buffer_size = std::min(4, topk);
  // insertion sort the topk elements in each thread, stored in local buffer
  auto pair_type = GetArgIdxType(A->type());

  insert_sort_func = insert_sort_func + pe::Type2StrForArgReduce(A->type());
  insert_sort_func += largest ? "_max" : "_min";
  block_reduce_func = block_reduce_func + pe::Type2StrForArgReduce(A->type());
  block_reduce_func += largest ? "_max" : "_min";
  // stored on GPU thread local memory, the returned local array has only
  // `argidx_buffer_size` elements

  ir::Tensor local_sorted = ComputeWithLocalBuffer(A,
                                                   pair_type,
                                                   insert_sort_func,
                                                   topk_axis,
                                                   A->shape[topk_axis],
                                                   Expr(argidx_buffer_size));

  std::vector<Expr> output_shape(A->shape.begin(), A->shape.end());
  std::vector<Expr> shape_wo_topk_axis(A->shape.begin(), A->shape.end());
  output_shape[topk_axis] = Expr(topk);
  shape_wo_topk_axis[topk_axis] = Expr(1);
  // here we should have shared memory buffers for array merging

  // phase 2 reduce merge: write to output buffer, but we need a global memory
  // ptr we also need a shared memory buffer for reduction we also need a grid
  // merge sort reduction

  auto res_idx = Compute(
      output_shape,
      [=](const std::vector<Expr> &indices) {
        // TODO(heqianyue): it is well possible that int should be replaced by
        // int64_t, including cuda template, registry, etc.
        Expr offset(0), ptr_offset(0);
        Expr stride(1);
        for (int i = 0; i < indices.size(); i++) {
          offset = offset * output_shape[i] + indices[i];
          ptr_offset = ptr_offset * shape_wo_topk_axis[i] + indices[i];
          if (i > topk_axis) {
            stride = stride * output_shape[i];
          }
        }
        offset = optim::ArithSimplify(offset);
        ptr_offset = optim::ArithSimplify(ptr_offset);
        stride = optim::ArithSimplify(stride);
        // Allocate shared memory buffer for reduction

        // TODO(heqianyue): currently I don't want to deal with the performance
        // issue. A fixed 32 * 4 * 16B size of shared buffer is used, which will
        // definitely introduce bank conflict and the wasting of shared memory
        // if k is small.
        auto merge_sort_buffer_name = cinn::UniqName("shm_merge_sort");
        auto merge_buffer_ts = ir::Tensor(merge_sort_buffer_name,
                                          pair_type,
                                          {ir::Expr(128)},
                                          {ir::Expr(128)},
                                          ir::FunctionRef());
        merge_buffer_ts->WithBuffer("shared",
                                    merge_sort_buffer_name + "_buffer");

        auto merge_ptr_buffer_name = cinn::UniqName("merge_ptr");
        auto merge_ptr_ts = ir::Tensor(merge_ptr_buffer_name,
                                       common::Int(32),
                                       shape_wo_topk_axis,
                                       std::move(shape_wo_topk_axis),
                                       ir::FunctionRef());
        merge_ptr_ts->WithBuffer("global", merge_ptr_buffer_name + "_buffer");
        // TODO(heqianyue): check, merge_ptr_ts is not initialized to 0

        // We call the grid reduce version by default, and in the optim pass, we
        // will decide which function to call, depending on the relative size of
        // grid/block and tensor shape. If we do not need grid reduce, merge_ptr
        // will be replaced by a shared memory ptr
        auto idx = CallExternWithWriteArgs(block_reduce_func,
                                           {local_sorted,
                                            Expr(argidx_buffer_size),
                                            Expr(topk),
                                            offset,
                                            ptr_offset,
                                            stride},
                                           {merge_buffer_ts, merge_ptr_ts});
        return idx;
      },
      "topk_index_ret");

  auto res_val = Compute(
      output_shape,
      [=](const std::vector<Expr> &indices) {
        auto topk_axis_indices = ir::Load::Make(A, res_idx);
        std::vector<Expr> tensor_idx(indices.begin(), indices.end());
        tensor_idx[axis] = topk_axis_indices;

        return ir::Load::Make(A, topk_axis_indices);
      },
      "topk_value_ret");

  return {res_val, res_idx};
}

std::shared_ptr<framework::OpStrategy> StrategyForTopKSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  PADDLE_ENFORCE_GE(
      attr_store.count("largest"),
      1,
      ::common::errors::InvalidArgument(
          "The attr_store doesn't have the attribute of 'largest'."));
  PADDLE_ENFORCE_GE(attr_store.count("k"),
                    1,
                    ::common::errors::InvalidArgument(
                        "The attr_store doesn't have the attribute of 'k'."));
  int topk = std::get<int>(attr_store.at("k"));
  bool largest = std::get<bool>(attr_store.at("largest"));

  framework::CINNCompute topk_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_NE(
        args.empty(),
        true,
        ::common::errors::InvalidArgument(
            "The input argument of TopK compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      1U,
                      ::common::errors::InvalidArgument(
                          "The input arguments' size of TopK should be 1"));
    Expr A = pack_args[0];
    PADDLE_ENFORCE_NOT_NULL(
        A.as_tensor(),
        ::common::errors::InvalidArgument(
            "Required Input must be a tensor. Please check."));
    PADDLE_ENFORCE_NE(output_shapes.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The output shape of TopK is empty! Please check."));
    auto tensor_A = A.as_tensor_ref();

    std::string tensor_name = pack_args[0].operator std::string();
    auto out = TopK(tensor_A, target, axis, topk, largest, tensor_name);
    std::vector<CINNValue> res;
    PADDLE_ENFORCE_EQ(
        res.size(),
        2,
        ::common::errors::InvalidArgument(
            "The output size of TopK is unexpected (expect: %d, actual: %lu).",
            2,
            res.size()));

    res.push_back(CINNValue(out.at(0)));
    res.push_back(CINNValue(out.at(1)));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(topk_compute, "strategy.topk", 1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(top_ops) {
  CINN_REGISTER_OP(topk)
      .describe("TopK.")
      .set_num_inputs(1)
      .set_num_outputs(2)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForTopKSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kReduction)
      .set_support_level(4);

  return true;
}
