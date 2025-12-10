#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>


namespace CaptionEngine {

// Hardware-agnostic parameter type
enum class ParamType {
  Float,
  Int,
  UInt,
  Vector2,
  Vector3,
  Vector4,
  Matrix4x4,
  Buffer, // BufferHandle
  Texture // TextureHandle (future)
};

// Parameter descriptor
struct Parameter {
  std::string name;
  ParamType type;
  size_t offset; // Offset in uniform buffer or binding index
};

// Compute Dimensions
struct WorkGroupSize {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// Unified Kernel Representation
struct ComputeKernel {
  std::string name;
  std::vector<uint8_t> bytecode; // SPIR-V, WGSL (text), or PTX
  std::vector<Parameter> parameters;
  WorkGroupSize workgroup_size;

  // Shader source format
  enum class Format { SPIRV, WGSL, PTX, MSL, HLSL, GLSL } format;
};

} // namespace CaptionEngine
