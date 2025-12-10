#pragma once
/**
 * @file webgpu.hpp
 * @brief WebGPU compute backend for browser and Dawn native
 */

#include "graphics/backend.hpp"

#if defined(__EMSCRIPTEN__) || defined(HAS_DAWN)

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <webgpu/webgpu.h>
#else
// Dawn native headers
#include <dawn/webgpu.h>
#endif

namespace CaptionEngine {

/**
 * @brief WebGPU compute backend
 *
 * Provides GPU compute via WebGPU API.
 * - In browser: Uses native WebGPU
 * - On native: Uses Google Dawn implementation
 */
class WebGPUBackendExt : public ComputeBackend {
public:
  /// WebGPU configuration
  struct Config {
    bool enable_timestamps = false;             // Enable query timestamps
    size_t max_buffer_size = 256 * 1024 * 1024; // 256MB default
    std::string preferred_adapter;              // Preferred GPU adapter name
  };

  explicit WebGPUBackendExt(const Config &config = {});
  ~WebGPUBackendExt() override;

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

  // --- WebGPU-specific Methods ---

  /// Initialize async (required for browser)
  /// Returns true when initialization complete
  [[nodiscard]] bool initialize_async();

  /// Check if device is ready
  [[nodiscard]] bool is_ready() const;

  /// Get WebGPU device
  [[nodiscard]] WGPUDevice wgpu_device() const;

  /// Get WebGPU queue
  [[nodiscard]] WGPUQueue wgpu_queue() const;

  /// Create shader module from WGSL source
  [[nodiscard]] WGPUShaderModule create_shader_module(const std::string &wgsl);

  /// Create compute pipeline
  [[nodiscard]] WGPUComputePipeline
  create_compute_pipeline(WGPUShaderModule shader,
                          const std::string &entry_point);

  /// Get adapter info string
  [[nodiscard]] std::string adapter_info() const;

  /// Get supported limits
  [[nodiscard]] WGPULimits device_limits() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/// Check if WebGPU is available
[[nodiscard]] bool webgpu_available();

/// Get WebGPU adapter names
[[nodiscard]] std::vector<std::string> webgpu_adapter_names();

} // namespace CaptionEngine

#else

// Stub when WebGPU not available
namespace CaptionEngine {
inline bool webgpu_available() { return false; }
inline std::vector<std::string> webgpu_adapter_names() { return {}; }
} // namespace CaptionEngine

#endif // __EMSCRIPTEN__ || HAS_DAWN
