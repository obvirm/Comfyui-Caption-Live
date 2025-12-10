#pragma once
/**
 * @file cuda.hpp
 * @brief CUDA compute backend for NVIDIA GPUs
 */

#include "graphics/backend.hpp"

#if defined(HAS_CUDA) || defined(__CUDACC__)

#include <cuda_runtime.h>

namespace CaptionEngine {

/**
 * @brief CUDA compute backend for NVIDIA GPUs
 *
 * Provides high-performance compute on NVIDIA hardware.
 * Requires CUDA toolkit and compatible GPU.
 */
class CUDABackend : public ComputeBackend {
public:
  CUDABackend();
  ~CUDABackend() override;

  [[nodiscard]] std::string name() const override { return "CUDA"; }
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

  // --- CUDA-specific Methods ---

  /// Get CUDA device ID
  [[nodiscard]] int device_id() const;

  /// Get CUDA device properties
  [[nodiscard]] cudaDeviceProp device_properties() const;

  /// Get available GPU memory in bytes
  [[nodiscard]] size_t available_memory() const;

  /// Get current CUDA stream
  [[nodiscard]] cudaStream_t stream() const;

  /// Launch kernel with custom grid/block size
  void launch_kernel(const std::string &kernel_name, dim3 grid_size,
                     dim3 block_size, size_t shared_memory,
                     std::span<void *> args);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/// Check if CUDA is available on this system
[[nodiscard]] bool cuda_available();

/// Get number of CUDA devices
[[nodiscard]] int cuda_device_count();

/// Get best CUDA device (highest compute capability)
[[nodiscard]] int cuda_best_device();

} // namespace CaptionEngine

#else

// Stub when CUDA not available
namespace CaptionEngine {
inline bool cuda_available() { return false; }
inline int cuda_device_count() { return 0; }
inline int cuda_best_device() { return -1; }
} // namespace CaptionEngine

#endif // HAS_CUDA
