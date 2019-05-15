/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/target_util.h"

#include "absl/meta/type_traits.h"
#include "absl/types/variant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Operator.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

#include <type_traits>

namespace xla {
namespace gpu {

namespace {

using IntrinsicOrDeviceFunction =
    absl::variant<llvm::Intrinsic::ID, const string>;
using IntrinsicOrString = absl::variant<int, const string>;
// Wrapper structure for carrying information about the intrinsic ids or the
// device function names for NVPTX/AMDGPU platforms.
struct TargetFunctionInfo {
  IntrinsicOrDeviceFunction nvptx_function;
  IntrinsicOrDeviceFunction amdgpu_function;
};

struct TargetFunctionInfoVisitor {
  absl::Span<llvm::Value* const> operands;
  llvm::IRBuilder<>* builder;
  absl::Span<llvm::Type* const> overloaded_types;
  llvm::Module* module;
  llvm::Triple target_triple;
  TargetFunctionInfoVisitor(absl::Span<llvm::Value* const> operands,
                            absl::Span<llvm::Type* const> overloaded_types,
                            llvm::IRBuilder<>* b)
      : operands(operands), overloaded_types(overloaded_types), builder(b) {
    module = builder->GetInsertBlock()->getModule();
    llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  }

  TargetFunctionInfoVisitor() {}
  llvm::Value* operator()(const std::string& s) const {
    LOG(FATAL) << "Unexpected function provided for " << target_triple.str();
    return nullptr;
  }
  llvm::Value* operator()(const llvm::Intrinsic::ID llvm_intrinsic_id) const {
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
        module, llvm_intrinsic_id, llvm_ir::AsArrayRef(overloaded_types));
    return builder->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
  }
};

// Gets the llvm intrinsic id or name of the device function on different
// platforms (NVPTX, AMDGPU) that corresponds to the TargetFunctionID that is
// provided.
TargetFunctionInfo GetTargetFunctionInfo(TargetFunctionID function_id) {
  switch (function_id) {
    case TargetFunctionID::kThreadIdx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
              llvm::Intrinsic::amdgcn_workitem_id_x};
    }
    case TargetFunctionID::kThreadIdy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
              llvm::Intrinsic::amdgcn_workitem_id_y};
    }
    case TargetFunctionID::kThreadIdz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
              llvm::Intrinsic::amdgcn_workitem_id_z};
    }
    case TargetFunctionID::kBlockIdx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
              llvm::Intrinsic::amdgcn_workgroup_id_x};
    }
    case TargetFunctionID::kBlockIdy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
              llvm::Intrinsic::amdgcn_workgroup_id_y};
    }
    case TargetFunctionID::kBlockIdz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
              llvm::Intrinsic::amdgcn_workgroup_id_z};
    }
    case TargetFunctionID::kBarrierId: {
      return {llvm::Intrinsic::nvvm_barrier0,
              llvm::Intrinsic::amdgcn_s_barrier};
    }
  }
}
}  // namespace

// Helper function to emit call to intrinsic or device function.
llvm::Value* EmitCallToTargetFunctionHelper(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  TargetFunctionInfo gpu_info = GetTargetFunctionInfo(function_id);
  IntrinsicOrDeviceFunction* gpu_function;
  if ((target_triple.getArch() == llvm::Triple::nvptx) ||
      (target_triple.getArch() == llvm::Triple::nvptx64)) {
    gpu_function = &gpu_info.nvptx_function;
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    gpu_function = &gpu_info.amdgpu_function;
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
  return (absl::visit(TargetFunctionInfoVisitor{operands, overloaded_types, b},
                      *gpu_function));
}

llvm::Value* EmitCallToTargetFunction(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
  return (EmitCallToTargetFunctionHelper(function_id, operands, {},
                                         PRIMITIVE_TYPE_INVALID, {},
                                         overloaded_types, b));
}
}  // namespace gpu
}  // namespace xla
