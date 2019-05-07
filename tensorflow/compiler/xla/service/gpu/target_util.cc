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
// Provide helper routine for obtaining  gpu target information useful
// for llvm IR contruction.

#include "tensorflow/compiler/xla/service/gpu/target_util.h"

#include "absl/types/variant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/MDBuilder.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {
namespace {
// Utility functions to obtain NVPTX/AMDGPU specific information.

using absl::StrAppend;
using IntrinsicOrDeviceFunction =
    absl::variant<llvm::Intrinsic::ID, const string>;

// Wrapper structure for carrying information about the intrinsic ids or the
// device function for NVPTX/AMDGPU platforms.
struct TargetFunctionInfo {
  IntrinsicOrDeviceFunction nvptx_function;
  IntrinsicOrDeviceFunction amdgpu_function;
};

// Gets the llvm intrinsic id or name of the device function on different
// platforms (NVPTX, AMDGPU) that corresponds to the TargetFunctionID that is
// provided.
struct TargetFunctionInfo GetTargetFunctionInfo(TargetFunctionID function_id) {
  switch (function_id) {
    case TargetFunctionID::kShflDownF32: {
      return {llvm::Intrinsic::nvvm_shfl_sync_down_f32, "__ockl_readuplane"};
    }
    case TargetFunctionID::kShflDownI32: {
      return {llvm::Intrinsic::nvvm_shfl_sync_down_i32, "__ockl_readuplane"};
    }
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

// Helper function to emit call to AMDGPU shfl function
llvm::Value* EmitAMDGPUShfl(
    const string& callee_name, absl::Span<llvm::Value* const> operands,
    PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  std::vector<llvm::Value*> converted_operands;
  // AMD device function only makes use of 2nd and 3rd operands. Also, the
  // device function only accepts integer arguments. For F32 arguments,
  // conversions need to be generated.
  if (output_type == F32) {
    converted_operands.push_back(b->CreateBitCast(
        operands[1], llvm_ir::PrimitiveTypeToIrType(S32, module)));
  } else if (output_type == S32) {
    converted_operands.push_back(operands[1]);
  } else {
    LOG(FATAL) << "Unimplemented type for AMDGPU shfl function.";
  }
  converted_operands.push_back(operands[2]);
  std::vector<llvm::Type*> ir_input_types(
      2, llvm_ir::PrimitiveTypeToIrType(S32, module));
  llvm::Type* ir_output_type =
      llvm_ir::PrimitiveTypeToIrType(output_type, module);
  llvm::FunctionType* callee_type =
      llvm::FunctionType::get(ir_output_type,  // Return type.
                              ir_input_types,  // Parameter types.
                              false);

  string munged_callee = callee_name;
  StrAppend(&munged_callee, "_i32");
  llvm::FunctionCallee shfl_call = module->getOrInsertFunction(llvm_ir::AsStringRef(munged_callee), callee_type);
#if 0 
  llvm::Function* callee = llvm::dyn_cast<llvm::Function>(
          module->getOrInsertFunction(llvm_ir::AsStringRef(munged_callee),
                                callee_type)
          .getCallee());
#endif 
  llvm::Function* callee = llvm::dyn_cast<llvm::Function>(shfl_call.getCallee());
  for (auto attribute : attributes) {
    callee->addFnAttr(attribute);
  }
  llvm::Value* result =
      b->CreateCall(shfl_call, llvm_ir::AsArrayRef(converted_operands));
  if (output_type == F32) {
    return (
        b->CreateBitCast(result, llvm::Type::getFloatTy(module->getContext())));
  } else {
    return (result);
  }
}

llvm::Value* EmitCallToTargetFunctionHelper(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  struct TargetFunctionInfo gpu_info = GetTargetFunctionInfo(function_id);
  IntrinsicOrDeviceFunction* gpu_function;
  if ((target_triple.getArch() == llvm::Triple::nvptx) ||
      (target_triple.getArch() == llvm::Triple::nvptx64)) {
    gpu_function = &gpu_info.nvptx_function;
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    gpu_function = &gpu_info.amdgpu_function;
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
  if (auto llvm_intrinsic_id =
          absl::get_if<llvm::Intrinsic::ID>(gpu_function)) {
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
        module, *llvm_intrinsic_id, llvm_ir::AsArrayRef(overloaded_types));
    return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
  } else if (auto callee_name = absl::get_if<const string>(gpu_function)) {
    std::vector<llvm::Type*> ir_input_types;
    for (PrimitiveType input_type : input_types) {
      ir_input_types.push_back(
          llvm_ir::PrimitiveTypeToIrType(input_type, module));
    }
    if (target_triple.getArch() == llvm::Triple::amdgcn &&
        (function_id == TargetFunctionID::kShflDownF32 ||
         function_id == TargetFunctionID::kShflDownI32)) {
      return EmitAMDGPUShfl(*callee_name, operands, output_type, attributes, b);
    }
    llvm::FunctionType* callee_type = llvm::FunctionType::get(
        llvm_ir::PrimitiveTypeToIrType(output_type,
                                       module),  // Return type.
        ir_input_types,                          // Parameter types.
        false);                                  // No variadic arguments.

    string munged_callee = *callee_name;
    switch (output_type) {
      case S32:
        StrAppend(&munged_callee, "_i32");
        break;
      case S64:
        StrAppend(&munged_callee, "_i64");
        break;
      case F32:
        StrAppend(&munged_callee, "_f32");
        break;
      case F64:
        StrAppend(&munged_callee, "_f64");
        break;
      default:
        LOG(FATAL) << "Bad Type " << PrimitiveType_Name(output_type) << "\n";
    }
    // Declares the callee if it is not declared already.
    llvm::FunctionCallee shfl_call = module->getOrInsertFunction(llvm_ir::AsStringRef(munged_callee), callee_type);
    llvm::Value* result = b->CreateCall(shfl_call, llvm_ir::AsArrayRef(operands));
#if 0 
    llvm::Function* callee = llvm::dyn_cast<llvm::Function>(
        b->GetInsertBlock()
            ->getModule()
            ->getOrInsertFunction(llvm_ir::AsStringRef(munged_callee),
                                  callee_type)
            .getCallee());
    for (auto attribute : attributes) {
      callee->addFnAttr(attribute);
    }
    llvm::Value* result = b->CreateCall(callee, llvm_ir::AsArrayRef(operands));
#endif 
    return result;
  }
}

llvm::Value* EmitCallToTargetFunction(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
  VLOG(2) << "Inside EmitCallToTargetFunctiob -intrinsic";
  return (EmitCallToTargetFunctionHelper(function_id, operands, {},
                                         PRIMITIVE_TYPE_INVALID, {},
                                         overloaded_types, b));
}

llvm::Value* EmitCallToTargetFunction(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    llvm::IRBuilder<>* b) {
  VLOG(2) << "Inside EmitCallToTargetFunctiob - function";
  return (EmitCallToTargetFunctionHelper(function_id, operands, input_types,
                                         output_type, attributes, {}, b));
}

llvm::Value* EmitCallToTargetFunction(
    struct TargetFunctionCallInfo function_info) {
  VLOG(2) << "Inside EmitCallToTargetFunctiob - combined";
  return (EmitCallToTargetFunctionHelper(
      function_info.function_id, function_info.operands,
      function_info.input_types, function_info.output_type,
      function_info.attributes, function_info.overloaded_types,
      function_info.b));
}

}  // namespace gpu
}  // namespace xla
