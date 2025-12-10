/**
 * @file compiler.cpp
 * @brief Unified Kernel Compiler implementation
 */

#include "compute/compiler.hpp"
#include <cstring>

namespace CaptionEngine {

std::pair<std::optional<ComputeKernel>, CompileError>
KernelCompiler::compile_kernel(std::string_view name, std::string_view source,
                               Target target) {
  if (source.empty()) {
    return {std::nullopt, CompileError::EmptySource};
  }

  ComputeKernel kernel;
  kernel.name = std::string(name);
  // Default workgroup size (can be parsed from source in future)
  kernel.workgroup_size = {16, 16, 1};

  switch (target) {
  case Target::WGSL: {
    // For WGSL, "bytecode" is just the source string.
    // We store it as a byte vector including null terminator for safety?
    // Or just bytes. emdawnwebgpu wrapper expects char*.
    kernel.format = ComputeKernel::Format::WGSL;
    kernel.bytecode.resize(source.size() + 1);
    std::memcpy(kernel.bytecode.data(), source.data(), source.size());
    kernel.bytecode[source.size()] = '\0'; // Null terminate
    break;
  }
  case Target::SPIR_V: {
    // Not implemented yet - would require shaderc or similar
    return {std::nullopt, CompileError::UnsupportedTarget};
  }
  default:
    return {std::nullopt, CompileError::UnsupportedTarget};
  }

  return {std::move(kernel), CompileError::None};
}

} // namespace CaptionEngine
