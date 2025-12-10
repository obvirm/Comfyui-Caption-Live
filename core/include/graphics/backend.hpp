#pragma once
/**
 * @file backend.hpp
 * @brief Compute backend abstraction for GPU/CPU rendering
 */

#include "compute/types.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace CaptionEngine {

/// GPU buffer handle
using BufferHandle = uint64_t;

/// Memory type for buffer allocation
enum class MemoryType {
  DeviceLocal, // GPU memory (fastest for compute)
  HostVisible, // CPU accessible (for uploads)
  HostCached   // CPU cached (for downloads)
};

/**
 * @brief Abstract compute backend interface
 *
 * Unified interface for GPU compute across different platforms.
 */
class ComputeBackend {
public:
  virtual ~ComputeBackend() = default;

  /// Get backend name
  [[nodiscard]] virtual std::string name() const = 0;

  /// Check if backend supports compute shaders
  [[nodiscard]] virtual bool supports_compute() const = 0;

  /// Create GPU buffer
  [[nodiscard]] virtual BufferHandle create_buffer(size_t size,
                                                   MemoryType type) = 0;

  /// Destroy buffer
  virtual void destroy_buffer(BufferHandle handle) = 0;

  /// Upload data to buffer
  virtual void upload_buffer(BufferHandle handle,
                             std::span<const uint8_t> data) = 0;

  /// Download data from buffer
  [[nodiscard]] virtual std::vector<uint8_t>
  download_buffer(BufferHandle handle, size_t size) = 0;

  /// Dispatch compute shader (Unified Interface)
  virtual void dispatch_compute(std::string_view shader_name,
                                std::span<BufferHandle> buffers,
                                WorkGroupSize workgroups) = 0;

  /// Register a compiled kernel
  virtual bool register_kernel(const ComputeKernel &kernel) = 0;

  /// Wait for GPU operations to complete
  virtual void synchronize() = 0;

  /// Factory: create best available backend
  [[nodiscard]] static std::unique_ptr<ComputeBackend> create_best();

  /// Factory: create specific backend
  [[nodiscard]] static std::unique_ptr<ComputeBackend>
  create(const std::string &backend_name);
};

/**
 * @brief CPU fallback compute backend
 */
class CPUBackend : public ComputeBackend {
public:
  CPUBackend();
  ~CPUBackend() override;

  [[nodiscard]] std::string name() const override { return "CPU"; }
  [[nodiscard]] bool supports_compute() const override { return true; }

  [[nodiscard]] BufferHandle create_buffer(size_t size,
                                           MemoryType type) override;
  void destroy_buffer(BufferHandle handle) override;
  void upload_buffer(BufferHandle handle,
                     std::span<const uint8_t> data) override;
  [[nodiscard]] std::vector<uint8_t> download_buffer(BufferHandle handle,
                                                     size_t size) override;
  void dispatch_compute(std::string_view shader_name,
                        std::span<BufferHandle> buffers,
                        WorkGroupSize workgroups) override;
  bool register_kernel(const ComputeKernel &kernel) override;
  void synchronize() override;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief WebGPU compute backend
 */
class WebGPUBackend : public ComputeBackend {
public:
  WebGPUBackend();
  ~WebGPUBackend() override;

  [[nodiscard]] std::string name() const override { return "WebGPU"; }
  [[nodiscard]] bool supports_compute() const override { return true; }

  [[nodiscard]] BufferHandle create_buffer(size_t size,
                                           MemoryType type) override;
  void destroy_buffer(BufferHandle handle) override;
  void upload_buffer(BufferHandle handle,
                     std::span<const uint8_t> data) override;
  [[nodiscard]] std::vector<uint8_t> download_buffer(BufferHandle handle,
                                                     size_t size) override;
  void dispatch_compute(std::string_view shader_name,
                        std::span<BufferHandle> buffers,
                        WorkGroupSize workgroups) override;
  bool register_kernel(const ComputeKernel &kernel) override;
  void synchronize() override;

  // Initialize WebGPU device (async in JS, blocking here for setup)
  bool initialize();

private:
  void on_adapter_ready(void *adapter_handle,
                        void *userdata); // userdata is generic
  void on_device_ready(void *device_handle, void *userdata);

  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace CaptionEngine
