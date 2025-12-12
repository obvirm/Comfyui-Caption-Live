#pragma once
/**
 * @file api.hpp
 * @brief Graphics API abstraction layer for multi-backend rendering
 *
 * Provides unified interface for graphics operations across:
 * - WebGPU (Browser WASM)
 * - Vulkan (Cross-platform Native)
 * - CUDA (NVIDIA Native)
 * - Metal (macOS/iOS)
 */

#include "gpu/backend.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace CaptionEngine::Graphics {

/// Texture handle type
using TextureHandle = uint64_t;
/// Shader handle type
using ShaderHandle = uint64_t;
/// Framebuffer handle type
using FramebufferHandle = uint64_t;

/// Invalid handle constant
constexpr uint64_t INVALID_HANDLE = 0;

/// Texture format enum
enum class TextureFormat {
  RGBA8_UNORM,     // 8-bit RGBA normalized
  RGBA8_SRGB,      // sRGB RGBA
  RGBA16_FLOAT,    // 16-bit float RGBA (HDR)
  RGBA32_FLOAT,    // 32-bit float RGBA (compute)
  R8_UNORM,        // Single channel (alpha/mask)
  DEPTH24_STENCIL8 // Depth buffer
};

/// Texture usage flags
enum class TextureUsage : uint32_t {
  None = 0,
  Sampled = 1 << 0,      // Can be sampled in shader
  Storage = 1 << 1,      // Can be written in compute
  RenderTarget = 1 << 2, // Can be render target
  CopySrc = 1 << 3,      // Can be copy source
  CopyDst = 1 << 4       // Can be copy destination
};

inline TextureUsage operator|(TextureUsage a, TextureUsage b) {
  return static_cast<TextureUsage>(static_cast<uint32_t>(a) |
                                   static_cast<uint32_t>(b));
}

/// Texture descriptor
struct TextureDescriptor {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  TextureFormat format = TextureFormat::RGBA8_UNORM;
  TextureUsage usage = TextureUsage::Sampled | TextureUsage::CopyDst;
  bool generate_mipmaps = false;
};

/// Blend mode for compositing
enum class BlendMode {
  None,     // No blending (overwrite)
  Alpha,    // Standard alpha
  Additive, // Add colors
  Multiply, // Multiply colors
  Screen    // Screen blend
};

/// Vertex attribute format
enum class VertexFormat { Float2, Float3, Float4, UByte4_Norm };

/// Quad vertex for sprite rendering
struct QuadVertex {
  float x, y;         // Position
  float u, v;         // Texture coords
  uint8_t r, g, b, a; // Vertex color
};

/**
 * @brief Graphics API abstraction interface
 *
 * Provides unified graphics operations for rendering.
 * Create via GraphicsAPI::create() factory method.
 */
class GraphicsAPI {
public:
  virtual ~GraphicsAPI() = default;

  /// Get API name (e.g., "WebGPU", "Vulkan", "CUDA")
  [[nodiscard]] virtual std::string name() const = 0;

  // --- Texture Operations ---

  /// Create texture from descriptor
  [[nodiscard]] virtual TextureHandle
  create_texture(const TextureDescriptor &desc) = 0;

  /// Create texture from raw pixel data
  [[nodiscard]] virtual TextureHandle
  create_texture_from_data(const TextureDescriptor &desc,
                           std::span<const uint8_t> data) = 0;

  /// Destroy texture
  virtual void destroy_texture(TextureHandle handle) = 0;

  /// Upload data to texture
  virtual void upload_texture(TextureHandle handle,
                              std::span<const uint8_t> data) = 0;

  /// Download texture data
  [[nodiscard]] virtual std::vector<uint8_t>
  download_texture(TextureHandle handle) = 0;

  // --- Shader Operations ---

  /// Load shader from source (WGSL, GLSL, HLSL depending on backend)
  [[nodiscard]] virtual ShaderHandle
  load_shader(const std::string &source, const std::string &entry_point) = 0;

  /// Destroy shader
  virtual void destroy_shader(ShaderHandle handle) = 0;

  // --- Framebuffer Operations ---

  /// Create framebuffer (render target)
  [[nodiscard]] virtual FramebufferHandle
  create_framebuffer(uint32_t width, uint32_t height, TextureFormat format) = 0;

  /// Destroy framebuffer
  virtual void destroy_framebuffer(FramebufferHandle handle) = 0;

  /// Begin rendering to framebuffer
  virtual void begin_render_pass(FramebufferHandle target) = 0;

  /// End render pass
  virtual void end_render_pass() = 0;

  /// Get framebuffer texture handle
  [[nodiscard]] virtual TextureHandle
  framebuffer_texture(FramebufferHandle handle) = 0;

  // --- Draw Operations ---

  /// Clear current render target
  virtual void clear(float r, float g, float b, float a) = 0;

  /// Draw textured quad
  virtual void draw_quad(TextureHandle texture, float x, float y, float width,
                         float height, float u0, float v0, float u1, float v1,
                         BlendMode blend = BlendMode::Alpha) = 0;

  /// Draw batch of quads (optimized)
  virtual void draw_quads(TextureHandle texture,
                          std::span<const QuadVertex> vertices,
                          BlendMode blend = BlendMode::Alpha) = 0;

  /// Submit all queued commands
  virtual void submit() = 0;

  /// Wait for GPU to complete
  virtual void wait_idle() = 0;

  // --- Factory ---

  /// Create best available graphics API
  [[nodiscard]] static std::unique_ptr<GraphicsAPI> create();

  /// Create specific graphics API by name
  [[nodiscard]] static std::unique_ptr<GraphicsAPI>
  create(const std::string &api_name);
};

} // namespace CaptionEngine::Graphics
