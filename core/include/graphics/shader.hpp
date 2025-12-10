#pragma once
/**
 * @file shader.hpp
 * @brief Cross-platform shader compilation and management
 */

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace CaptionEngine::Graphics {

/// Shader stage type
enum class ShaderStage { Vertex, Fragment, Compute };

/// Shader language
enum class ShaderLanguage {
  WGSL,  // WebGPU Shading Language
  GLSL,  // OpenGL Shading Language
  HLSL,  // High-Level Shading Language
  SPIRV, // SPIR-V bytecode
  MSL    // Metal Shading Language
};

/// Uniform type for shader parameters
enum class UniformType {
  Float,
  Float2,
  Float3,
  Float4,
  Mat3,
  Mat4,
  Int,
  UInt,
  Sampler2D
};

/// Uniform binding descriptor
struct UniformBinding {
  std::string name;
  UniformType type;
  uint32_t binding;
  uint32_t set = 0;
};

/// Shader reflection data
struct ShaderReflection {
  std::vector<UniformBinding> uniforms;
  std::vector<std::string> textures;
  uint32_t workgroup_size[3] = {1, 1, 1}; // For compute shaders
};

/// Compiled shader module
struct ShaderModule {
  std::vector<uint8_t> bytecode; // SPIR-V or native bytecode
  std::string source;            // Original source (for debugging)
  ShaderStage stage;
  ShaderLanguage language;
  ShaderReflection reflection;
  std::string entry_point = "main";
};

/// Shader compilation error
struct ShaderError {
  std::string message;
  uint32_t line = 0;
  uint32_t column = 0;
};

/// Shader compilation result
using ShaderResult = std::variant<ShaderModule, ShaderError>;

/**
 * @brief Cross-platform shader compiler
 *
 * Compiles shaders from various source languages to
 * target-specific formats (SPIR-V, WGSL, MSL, etc.)
 */
class ShaderCompiler {
public:
  /// Target platform for compilation
  enum class Target {
    SPIRV, // Vulkan/OpenGL
    WGSL,  // WebGPU
    MSL,   // Metal
    DXIL   // DirectX 12
  };

  /// Compile shader from source
  [[nodiscard]] static ShaderResult compile(const std::string &source,
                                            ShaderStage stage,
                                            ShaderLanguage source_lang,
                                            Target target);

  /// Compile WGSL to target (most common path)
  [[nodiscard]] static ShaderResult
  compile_wgsl(const std::string &source, ShaderStage stage, Target target);

  /// Validate shader without full compilation
  [[nodiscard]] static std::optional<ShaderError>
  validate(const std::string &source, ShaderLanguage lang);

  /// Get reflection data from compiled shader
  [[nodiscard]] static ShaderReflection reflect(const ShaderModule &module);
};

/// Built-in shader library
namespace Shaders {

/// Get text rendering vertex shader (WGSL)
[[nodiscard]] std::string text_vertex_wgsl();

/// Get text rendering fragment shader (WGSL)
[[nodiscard]] std::string text_fragment_wgsl();

/// Get sprite batch vertex shader (WGSL)
[[nodiscard]] std::string sprite_vertex_wgsl();

/// Get sprite batch fragment shader (WGSL)
[[nodiscard]] std::string sprite_fragment_wgsl();

/// Get blur compute shader (WGSL)
[[nodiscard]] std::string blur_compute_wgsl();

/// Get chromatic aberration compute shader (WGSL)
[[nodiscard]] std::string chromatic_aberration_wgsl();

} // namespace Shaders

} // namespace CaptionEngine::Graphics
