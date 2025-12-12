#pragma once
/**
 * @file backend.hpp
 * @brief GPU Backend abstraction layer
 *
 * Unified interface for Vulkan, WebGPU, Metal, and CUDA backends.
 * All rendering goes through this interface for cross-platform consistency.
 */

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// C++23 expected fallback
#if __has_include(<expected>) && __cplusplus >= 202302L
#include <expected>
#define CE_HAS_STD_EXPECTED 1
#else
#define CE_HAS_STD_EXPECTED 0
#endif

// span may not be available in all C++20 implementations
#if __has_include(<span>) && __cplusplus > 201703L
#include <span>
#else
// Minimal span fallback
namespace std {
template <typename T> class span {
public:
  span() : data_(nullptr), size_(0) {}
  span(T *d, size_t s) : data_(d), size_(s) {}

  // Allow construction from container (mutable)
  template <typename Container>
  span(Container &c) : data_(c.data()), size_(size_t(c.size())) {}

  // Allow construction from const container (if T is const)
  template <typename Container>
  span(const Container &c) : data_(c.data()), size_(size_t(c.size())) {}

  T *data() const { return data_; }
  size_t size() const { return size_; }

  static constexpr size_t npos = -1;
  span subspan(size_t offset, size_t count = npos) const {
    if (count == npos)
      return span(data_ + offset, size_ - offset);
    return span(data_ + offset, count);
  }

  using iterator = T *;
  using const_iterator = const T *;

  T &operator[](size_t idx) const { return data_[idx]; }

  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }
  const_iterator cbegin() const { return data_; }
  const_iterator cend() const { return data_ + size_; }

private:
  T *data_;
  size_t size_;
};
} // namespace std
#endif

// Fix for libc++ rebind_pointer_t error with std::unique_ptr
#include <memory>
namespace std {
template <typename T, typename D> struct pointer_traits<unique_ptr<T, D>> {
  using pointer = unique_ptr<T, D>;
  using element_type = T;
  using difference_type = ptrdiff_t;

  template <typename U> using rebind = unique_ptr<U>;

  static pointer pointer_to(element_type &r) { return pointer(addressof(r)); }
};
} // namespace std

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

/// Result type for GPU operations (C++20 compatible)
// Simplified GPUResult implementation to avoid libc++ bugs
template <typename E> struct unexpected {
  E err;
  explicit unexpected(E e) : err(std::move(e)) {}
};

template <typename T> class GPUResult {
public:
  // Success constructor
  GPUResult(T value) : value_(std::move(value)), has_value_(true) {}

  // Error constructors
  GPUResult(GPUError error) : error_(std::move(error)), has_value_(false) {}
  GPUResult(unexpected<GPUError> u)
      : error_(std::move(u.err)), has_value_(false) {}

  // Move constructor
  GPUResult(GPUResult &&other) noexcept : has_value_(other.has_value_) {
    if (has_value_) {
      new (&value_) T(std::move(other.value_));
    } else {
      new (&error_) GPUError(std::move(other.error_));
    }
  }

  // Move assignment
  GPUResult &operator=(GPUResult &&other) noexcept {
    if (this != &other) {
      destroy();
      has_value_ = other.has_value_;
      if (has_value_) {
        new (&value_) T(std::move(other.value_));
      } else {
        new (&error_) GPUError(std::move(other.error_));
      }
    }
    return *this;
  }

  // Destructor
  ~GPUResult() { destroy(); }

  // Delete copy operations (unique_ptr is not copyable)
  GPUResult(const GPUResult &) = delete;
  GPUResult &operator=(const GPUResult &) = delete;

  bool has_value() const { return has_value_; }
  explicit operator bool() const { return has_value_; }

  T &value() { return value_; }
  const T &value() const { return value_; }
  T &operator*() { return value(); }
  const T &operator*() const { return value(); }
  T *operator->() { return &value_; }
  const T *operator->() const { return &value_; }

  GPUError &error() { return error_; }
  const GPUError &error() const { return error_; }

private:
  void destroy() {
    if (has_value_) {
      value_.~T();
    } else {
      error_.~GPUError();
    }
  }

  union {
    T value_;
    GPUError error_;
  };
  bool has_value_;
};

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
