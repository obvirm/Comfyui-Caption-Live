#pragma once
/**
 * @file backend.hpp
 * @brief GPU Backend abstraction layer
 *
 * Unified interface for Vulkan, WebGPU, Metal, and CUDA backends.
 * All rendering goes through this interface for cross-platform consistency.
 */

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>


namespace CaptionEngine {
namespace GPU {

// Forward declarations
class Buffer;
class Texture;
class Shader;
class Pipeline;
class CommandBuffer;

/// GPU Backend type enumeration
enum class BackendType {
  Auto,      ///< Auto-detect best available
  Vulkan,    ///< Vulkan 1.3 (Desktop/Android)
  WebGPU,    ///< WebGPU (Browser/Dawn)
  Metal,     ///< Metal (Apple)
  DirectX12, ///< DirectX 12 (Windows)
  CUDA       ///< CUDA compute-only (NVIDIA)
};

/// Texture format
enum class TextureFormat {
  RGBA8,   ///< 8-bit per channel RGBA
  RGBA16F, ///< 16-bit float per channel
  RGBA32F, ///< 32-bit float per channel
  R8,      ///< Single channel 8-bit (SDF)
  R32F     ///< Single channel 32-bit float
};

/// Buffer usage flags
enum class BufferUsage : uint32_t {
  Vertex = 1 << 0,
  Index = 1 << 1,
  Uniform = 1 << 2,
  Storage = 1 << 3, ///< Compute shader read/write
  CopySrc = 1 << 4,
  CopyDst = 1 << 5
};

inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
  return static_cast<BufferUsage>(static_cast<uint32_t>(a) |
                                  static_cast<uint32_t>(b));
}

/// Shader stage
enum class ShaderStage { Vertex, Fragment, Compute };

/// Error type for GPU operations
struct GPUError {
  std::string message;
  int code = 0;
};

template <typename T> using GPUResult = std::expected<T, GPUError>;

/**
 * @brief GPU Buffer handle
 */
class Buffer {
public:
  virtual ~Buffer() = default;

  virtual size_t size() const = 0;
  virtual BufferUsage usage() const = 0;

  /// Write data to buffer
  virtual void write(std::span<const uint8_t> data, size_t offset = 0) = 0;

  /// Read data from buffer (for debugging)
  virtual std::vector<uint8_t> read() = 0;
};

/**
 * @brief GPU Texture handle
 */
class Texture {
public:
  virtual ~Texture() = default;

  virtual uint32_t width() const = 0;
  virtual uint32_t height() const = 0;
  virtual TextureFormat format() const = 0;

  /// Upload pixel data
  virtual void upload(std::span<const uint8_t> data) = 0;

  /// Download pixel data
  virtual std::vector<uint8_t> download() = 0;
};

/**
 * @brief Compiled shader module
 */
class Shader {
public:
  virtual ~Shader() = default;
  virtual ShaderStage stage() const = 0;
};

/**
 * @brief Render or compute pipeline
 */
class Pipeline {
public:
  virtual ~Pipeline() = default;
  virtual bool isCompute() const = 0;
};

/**
 * @brief Command buffer for recording GPU commands
 */
class CommandBuffer {
public:
  virtual ~CommandBuffer() = default;

  /// Begin recording
  virtual void begin() = 0;

  /// End recording
  virtual void end() = 0;

  /// Set compute pipeline
  virtual void setComputePipeline(Pipeline *pipeline) = 0;

  /// Set render pipeline
  virtual void setRenderPipeline(Pipeline *pipeline) = 0;

  /// Bind buffer to slot
  virtual void bindBuffer(uint32_t binding, Buffer *buffer) = 0;

  /// Bind texture to slot
  virtual void bindTexture(uint32_t binding, Texture *texture) = 0;

  /// Dispatch compute shader
  virtual void dispatch(uint32_t groupsX, uint32_t groupsY,
                        uint32_t groupsZ) = 0;

  /// Draw call
  virtual void draw(uint32_t vertexCount, uint32_t instanceCount = 1) = 0;

  /// Copy texture to buffer (for readback)
  virtual void copyTextureToBuffer(Texture *src, Buffer *dst) = 0;
};

/**
 * @brief Main GPU Backend interface
 *
 * Implement this for each graphics API (Vulkan, WebGPU, etc.)
 */
class Backend {
public:
  virtual ~Backend() = default;

  /// Get backend type
  virtual BackendType type() const = 0;

  /// Get backend name (e.g., "Vulkan 1.3", "WebGPU Dawn")
  virtual std::string name() const = 0;

  /// Check if backend is ready
  virtual bool isReady() const = 0;

  // --- Resource Creation ---

  /// Create GPU buffer
  virtual GPUResult<std::unique_ptr<Buffer>>
  createBuffer(size_t size, BufferUsage usage) = 0;

  /// Create texture
  virtual GPUResult<std::unique_ptr<Texture>>
  createTexture(uint32_t width, uint32_t height, TextureFormat format) = 0;

  /// Create shader from WGSL source
  virtual GPUResult<std::unique_ptr<Shader>>
  createShaderWGSL(const std::string &source, ShaderStage stage,
                   const std::string &entryPoint = "main") = 0;

  /// Create shader from SPIR-V binary
  virtual GPUResult<std::unique_ptr<Shader>>
  createShaderSPIRV(std::span<const uint32_t> spirv, ShaderStage stage,
                    const std::string &entryPoint = "main") = 0;

  /// Create compute pipeline
  virtual GPUResult<std::unique_ptr<Pipeline>>
  createComputePipeline(Shader *computeShader) = 0;

  /// Create render pipeline
  virtual GPUResult<std::unique_ptr<Pipeline>>
  createRenderPipeline(Shader *vertexShader, Shader *fragmentShader,
                       TextureFormat outputFormat = TextureFormat::RGBA8) = 0;

  // --- Command Execution ---

  /// Create command buffer
  virtual GPUResult<std::unique_ptr<CommandBuffer>> createCommandBuffer() = 0;

  /// Submit command buffer for execution
  virtual void submit(CommandBuffer *cmd) = 0;

  /// Wait for all GPU work to complete
  virtual void waitIdle() = 0;

  // --- Factory ---

  /// Create best available backend
  static std::unique_ptr<Backend>
  create(BackendType preferred = BackendType::Auto);
};

} // namespace GPU
} // namespace CaptionEngine
