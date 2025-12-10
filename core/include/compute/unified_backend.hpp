/**
 * @file unified_backend.hpp
 * @brief Unified compute backend with auto-detection
 */

#pragma once

// Prevent Windows min/max macro conflicts
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace CaptionEngine {
namespace Compute {

/// Backend capability flags
enum class BackendCapability : uint32_t {
  None = 0,
  Compute = 1 << 0,
  Graphics = 1 << 1,
  AsyncCompute = 1 << 2,
  SharedMemory = 1 << 3,
  AtomicOperations = 1 << 4,
  Float16 = 1 << 5,
  Float64 = 1 << 6,
  Int64Atomics = 1 << 7,
  SubgroupOperations = 1 << 8,
  RayTracing = 1 << 9
};

inline BackendCapability operator|(BackendCapability a, BackendCapability b) {
  return static_cast<BackendCapability>(static_cast<uint32_t>(a) |
                                        static_cast<uint32_t>(b));
}

inline bool has_capability(BackendCapability caps, BackendCapability flag) {
  return (static_cast<uint32_t>(caps) & static_cast<uint32_t>(flag)) != 0;
}

/// Backend info structure
struct BackendInfo {
  std::string name;
  std::string vendor;
  std::string driver_version;
  uint64_t total_memory;
  uint64_t available_memory;
  uint32_t compute_units;
  uint32_t max_workgroup_size;
  std::array<uint32_t, 3> max_workgroup_count;
  BackendCapability capabilities;
  bool is_discrete;
  bool is_integrated;
};

/// Buffer usage flags
enum class BufferUsage : uint32_t {
  Storage = 1 << 0,
  Uniform = 1 << 1,
  Vertex = 1 << 2,
  Index = 1 << 3,
  Indirect = 1 << 4,
  TransferSrc = 1 << 5,
  TransferDst = 1 << 6
};

inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
  return static_cast<BufferUsage>(static_cast<uint32_t>(a) |
                                  static_cast<uint32_t>(b));
}

/// Memory type
enum class MemoryType {
  DeviceLocal, ///< GPU only, fastest
  HostVisible, ///< CPU readable/writable
  HostCached,  ///< CPU cached
  Unified      ///< Shared CPU/GPU memory
};

/// Buffer handle
using BufferHandle = uint64_t;
constexpr BufferHandle InvalidBuffer = 0;

/// Kernel handle
using KernelHandle = uint64_t;
constexpr KernelHandle InvalidKernel = 0;

/**
 * @brief Unified compute backend interface
 *
 * Provides a common interface for all GPU compute backends
 * (CUDA, Vulkan, Metal, WebGPU, DirectX 12, CPU fallback)
 */
class UnifiedBackend {
public:
  virtual ~UnifiedBackend() = default;

  /// Get backend info
  [[nodiscard]] virtual BackendInfo get_info() const = 0;

  /// Initialize backend
  virtual bool initialize() = 0;

  /// Shutdown backend
  virtual void shutdown() = 0;

  // === Buffer Management ===

  /// Create buffer
  [[nodiscard]] virtual BufferHandle
  create_buffer(size_t size, BufferUsage usage, MemoryType memory_type) = 0;

  /// Destroy buffer
  virtual void destroy_buffer(BufferHandle handle) = 0;

  /// Upload data to buffer
  virtual void upload(BufferHandle handle, std::span<const uint8_t> data,
                      size_t offset = 0) = 0;

  /// Download data from buffer
  virtual void download(BufferHandle handle, std::span<uint8_t> data,
                        size_t offset = 0) = 0;

  /// Copy between buffers
  virtual void copy(BufferHandle src, BufferHandle dst, size_t size,
                    size_t src_offset = 0, size_t dst_offset = 0) = 0;

  /// Map buffer for CPU access
  [[nodiscard]] virtual void *map(BufferHandle handle, size_t offset = 0,
                                  size_t size = 0) = 0;

  /// Unmap buffer
  virtual void unmap(BufferHandle handle) = 0;

  // === Kernel Management ===

  /// Create kernel from bytecode
  [[nodiscard]] virtual KernelHandle
  create_kernel(std::span<const uint8_t> bytecode,
                std::string_view entry_point) = 0;

  /// Destroy kernel
  virtual void destroy_kernel(KernelHandle handle) = 0;

  /// Bind buffer to kernel
  virtual void bind_buffer(KernelHandle kernel, uint32_t binding,
                           BufferHandle buffer) = 0;

  /// Set push constants
  virtual void set_push_constants(KernelHandle kernel,
                                  std::span<const uint8_t> data) = 0;

  /// Dispatch compute
  virtual void dispatch(KernelHandle kernel, uint32_t groups_x,
                        uint32_t groups_y, uint32_t groups_z) = 0;

  /// Dispatch indirect
  virtual void dispatch_indirect(KernelHandle kernel,
                                 BufferHandle indirect_buffer,
                                 size_t offset = 0) = 0;

  // === Synchronization ===

  /// Insert memory barrier
  virtual void barrier() = 0;

  /// Wait for all operations to complete
  virtual void synchronize() = 0;

  /// Begin recording commands
  virtual void begin_recording() = 0;

  /// End recording and submit
  virtual void end_recording() = 0;

  // === Factory ===

  /// Create best available backend
  [[nodiscard]] static std::unique_ptr<UnifiedBackend> create_best();

  /// Create specific backend type
  [[nodiscard]] static std::unique_ptr<UnifiedBackend>
  create(const std::string &type);

  /// Get available backend types
  [[nodiscard]] static std::vector<std::string> available_backends();
};

/**
 * @brief CPU fallback compute backend
 */
class CPUBackend : public UnifiedBackend {
public:
  CPUBackend();
  ~CPUBackend() override;

  BackendInfo get_info() const override;
  bool initialize() override;
  void shutdown() override;

  BufferHandle create_buffer(size_t size, BufferUsage usage,
                             MemoryType memory_type) override;
  void destroy_buffer(BufferHandle handle) override;
  void upload(BufferHandle handle, std::span<const uint8_t> data,
              size_t offset) override;
  void download(BufferHandle handle, std::span<uint8_t> data,
                size_t offset) override;
  void copy(BufferHandle src, BufferHandle dst, size_t size, size_t src_offset,
            size_t dst_offset) override;
  void *map(BufferHandle handle, size_t offset, size_t size) override;
  void unmap(BufferHandle handle) override;

  KernelHandle create_kernel(std::span<const uint8_t> bytecode,
                             std::string_view entry_point) override;
  void destroy_kernel(KernelHandle handle) override;
  void bind_buffer(KernelHandle kernel, uint32_t binding,
                   BufferHandle buffer) override;
  void set_push_constants(KernelHandle kernel,
                          std::span<const uint8_t> data) override;
  void dispatch(KernelHandle kernel, uint32_t groups_x, uint32_t groups_y,
                uint32_t groups_z) override;
  void dispatch_indirect(KernelHandle kernel, BufferHandle indirect_buffer,
                         size_t offset) override;

  void barrier() override;
  void synchronize() override;
  void begin_recording() override;
  void end_recording() override;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace Compute
} // namespace CaptionEngine
