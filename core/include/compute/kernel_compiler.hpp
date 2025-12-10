/**
 * @file kernel_compiler.hpp
 * @brief Cross-platform compute kernel compiler
 */

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>


namespace CaptionEngine {
namespace Compute {

/// Compile target for kernels
enum class CompileTarget {
  SPIRV, ///< Vulkan/OpenGL/WebGPU (SPIR-V)
  PTX,   ///< NVIDIA CUDA (PTX)
  MSL,   ///< Apple Metal Shading Language
  HLSL,  ///< DirectX 12 High-Level Shading Language
  GLSL,  ///< OpenGL/WebGL GLSL
  WGSL   ///< WebGPU Shading Language
};

/// Shader stage type
enum class ShaderStage {
  Compute,
  Vertex,
  Fragment,
  Geometry,
  TessControl,
  TessEval
};

/// Compilation error types
enum class CompileError {
  None,
  SyntaxError,
  SemanticError,
  UnsupportedFeature,
  TargetNotSupported,
  InternalError
};

/// Compilation warning
struct CompileWarning {
  std::string message;
  uint32_t line;
  uint32_t column;
};

/// Compilation result
struct CompileResult {
  std::vector<uint8_t> bytecode;
  std::vector<CompileWarning> warnings;
  std::string entry_point;
  ShaderStage stage;

  [[nodiscard]] bool success() const noexcept { return !bytecode.empty(); }
};

/// Shader parameter reflection
struct ShaderParameter {
  std::string name;
  std::string type;
  uint32_t binding;
  uint32_t set;
  uint32_t offset;
  uint32_t size;
  bool is_buffer;
  bool is_texture;
  bool is_sampler;
};

/// Shader reflection data
struct ShaderReflection {
  std::string entry_point;
  ShaderStage stage;
  std::vector<ShaderParameter> parameters;
  std::array<uint32_t, 3> workgroup_size;
  uint32_t push_constant_size;
};

/**
 * @brief Unified kernel compiler for all backends
 */
class KernelCompiler {
public:
  KernelCompiler();
  ~KernelCompiler();

  /// Compile from GLSL source
  [[nodiscard]] std::expected<CompileResult, CompileError> compile_glsl(
      std::string_view source, ShaderStage stage, CompileTarget target,
      const std::vector<std::pair<std::string, std::string>> &defines = {});

  /// Compile from HLSL source
  [[nodiscard]] std::expected<CompileResult, CompileError>
  compile_hlsl(std::string_view source, ShaderStage stage,
               CompileTarget target);

  /// Cross-compile SPIR-V to target
  [[nodiscard]] std::expected<CompileResult, CompileError>
  cross_compile(std::span<const uint8_t> spirv, CompileTarget target);

  /// Get shader reflection
  [[nodiscard]] std::expected<ShaderReflection, CompileError>
  reflect(std::span<const uint8_t> bytecode, CompileTarget source_format);

  /// Optimize bytecode
  [[nodiscard]] std::expected<std::vector<uint8_t>, CompileError>
  optimize(std::span<const uint8_t> bytecode, CompileTarget format,
           uint32_t optimization_level = 2);

  /// Validate bytecode
  [[nodiscard]] bool validate(std::span<const uint8_t> bytecode,
                              CompileTarget format);

  /// Get supported targets
  [[nodiscard]] std::vector<CompileTarget> supported_targets() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Pre-compiled kernel cache
 */
class KernelCache {
public:
  struct CacheKey {
    std::string source_hash;
    CompileTarget target;
    std::string defines_hash;

    bool operator==(const CacheKey &) const = default;
  };

  /// Get cached kernel
  [[nodiscard]] const std::vector<uint8_t> *get(const CacheKey &key) const;

  /// Store kernel in cache
  void put(const CacheKey &key, std::vector<uint8_t> bytecode);

  /// Clear cache
  void clear();

  /// Save cache to disk
  bool save(const std::string &path) const;

  /// Load cache from disk
  bool load(const std::string &path);

  /// Get cache size in bytes
  [[nodiscard]] size_t size_bytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace Compute
} // namespace CaptionEngine
