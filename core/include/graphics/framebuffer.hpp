#pragma once
/**
 * @file framebuffer.hpp
 * @brief Framebuffer and render target management
 */

#include "graphics/api.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace CaptionEngine::Graphics {

/// Render pass load operation
enum class LoadOp {
  Load,    // Preserve existing content
  Clear,   // Clear to specified color
  DontCare // Undefined (optimization hint)
};

/// Render pass store operation
enum class StoreOp {
  Store,   // Write results to texture
  DontCare // Discard results (optimization, e.g., MSAA resolve)
};

/// Clear value for render pass
struct ClearValue {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 1.0f;
  float depth = 1.0f;
  uint32_t stencil = 0;
};

/// Color attachment descriptor
struct ColorAttachment {
  TextureHandle texture = INVALID_HANDLE;
  LoadOp load_op = LoadOp::Clear;
  StoreOp store_op = StoreOp::Store;
  ClearValue clear;

  // Optional resolve target for MSAA
  TextureHandle resolve_target = INVALID_HANDLE;
};

/// Depth/stencil attachment descriptor
struct DepthStencilAttachment {
  TextureHandle texture = INVALID_HANDLE;
  LoadOp depth_load_op = LoadOp::Clear;
  StoreOp depth_store_op = StoreOp::DontCare;
  LoadOp stencil_load_op = LoadOp::Clear;
  StoreOp stencil_store_op = StoreOp::DontCare;
  ClearValue clear;
  bool read_only = false;
};

/// Render pass descriptor
struct RenderPassDescriptor {
  std::vector<ColorAttachment> color_attachments;
  std::optional<DepthStencilAttachment> depth_stencil;
  std::string label; // Debug label
};

/// Framebuffer descriptor
struct FramebufferDescriptor {
  uint32_t width = 1920;
  uint32_t height = 1080;
  TextureFormat color_format = TextureFormat::RGBA8_UNORM;
  bool has_depth = false;
  uint32_t sample_count = 1; // 1 = no MSAA, 4 = 4x MSAA
  std::string label;
};

/**
 * @brief Framebuffer wrapper with automatic resource management
 */
class Framebuffer {
public:
  Framebuffer(GraphicsAPI &api, const FramebufferDescriptor &desc);
  ~Framebuffer();

  // Non-copyable, movable
  Framebuffer(const Framebuffer &) = delete;
  Framebuffer &operator=(const Framebuffer &) = delete;
  Framebuffer(Framebuffer &&) noexcept;
  Framebuffer &operator=(Framebuffer &&) noexcept;

  /// Get underlying handle
  [[nodiscard]] FramebufferHandle handle() const;

  /// Get color texture
  [[nodiscard]] TextureHandle color_texture() const;

  /// Get depth texture (if present)
  [[nodiscard]] TextureHandle depth_texture() const;

  /// Get dimensions
  [[nodiscard]] uint32_t width() const;
  [[nodiscard]] uint32_t height() const;

  /// Begin rendering to this framebuffer
  void begin(const ClearValue &clear = {});

  /// End rendering
  void end();

  /// Resize framebuffer (recreates textures)
  void resize(uint32_t width, uint32_t height);

  /// Read pixels back to CPU (slow, blocking)
  [[nodiscard]] std::vector<uint8_t> read_pixels() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Double-buffered swap chain for smooth animation
 */
class SwapChain {
public:
  SwapChain(GraphicsAPI &api, uint32_t width, uint32_t height,
            uint32_t buffer_count = 2);
  ~SwapChain();

  /// Get current back buffer
  [[nodiscard]] Framebuffer &current();

  /// Swap buffers
  void present();

  /// Resize swap chain
  void resize(uint32_t width, uint32_t height);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace CaptionEngine::Graphics
