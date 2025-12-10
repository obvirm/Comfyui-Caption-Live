#pragma once

#include "compute/types.hpp"
#include <optional>
#include <string>
#include <string_view>
#include <vector>


namespace CaptionEngine {

enum class CompileError {
  None,
  EmptySource,
  UnsupportedTarget,
  CompilationFailed,
  InvalidBytecode
};

class KernelCompiler {
public:
  enum class Target {
    SPIR_V, // Vulkan/OpenGL/WebGPU (via Shim)
    WGSL,   // WebGPU Native
    PTX,    // NVIDIA CUDA
    MSL,    // Apple Metal
    HLSL,   // DirectX 12
    GLSL    // OpenGL
  };

  /**
   * @brief Compile/Prepare a compute kernel for a specific target
   *
   * @param name Name of the kernel (for debug/cache)
   * @param source Source code (WGSL, GLSL, HLSL, or CUDA C++)
   * @param target Target format
   * @return ComputeKernel or Error
   */
  [[nodiscard]] static std::pair<std::optional<ComputeKernel>, CompileError>
  compile_kernel(std::string_view name, std::string_view source, Target target);
};

} // namespace CaptionEngine
