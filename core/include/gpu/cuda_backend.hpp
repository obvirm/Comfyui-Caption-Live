#pragma once
/**
 * @file cuda_backend.hpp
 * @brief CUDA compute backend for NVIDIA GPUs
 *
 * High-performance compute-only backend for GPU-accelerated effects.
 * Works alongside Vulkan for rendering + CUDA for compute.
 */

#include "gpu/backend.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Stub types for non-CUDA compilation
typedef void *cudaStream_t;
typedef int cudaError_t;
#endif

namespace CaptionEngine {
namespace GPU {

/**
 * @brief CUDA compute backend (NVIDIA only)
 *
 * Features:
 * - High-performance compute kernels
 * - Tensor Core acceleration (if available)
 * - Interop with Vulkan for rendering
 * - Async stream execution
 */
class CUDABackend : public Backend {
public:
  CUDABackend();
  ~CUDABackend() override;

  // Backend interface
  BackendType type() const override { return BackendType::CUDA; }
  std::string name() const override;
  bool isReady() const override;

  // Resource creation
  GPUResult<std::unique_ptr<Buffer>> createBuffer(size_t size,
                                                  BufferUsage usage) override;

  GPUResult<std::unique_ptr<Texture>>
  createTexture(uint32_t width, uint32_t height, TextureFormat format) override;

  // CUDA doesn't use shaders in the same way - use kernels instead
  GPUResult<std::unique_ptr<Shader>>
  createShaderWGSL(const std::string &source, ShaderStage stage,
                   const std::string &entryPoint) override;

  GPUResult<std::unique_ptr<Shader>>
  createShaderSPIRV(std::span<const uint32_t> spirv, ShaderStage stage,
                    const std::string &entryPoint) override;

  GPUResult<std::unique_ptr<Pipeline>>
  createComputePipeline(Shader *computeShader) override;

  GPUResult<std::unique_ptr<Pipeline>>
  createRenderPipeline(Shader *vertexShader, Shader *fragmentShader,
                       TextureFormat outputFormat) override;

  // Command execution
  GPUResult<std::unique_ptr<CommandBuffer>> createCommandBuffer() override;
  void submit(CommandBuffer *cmd) override;
  void waitIdle() override;

  // CUDA-specific functionality
  int deviceId() const;
  cudaStream_t stream() const;

  /// Check compute capability
  int computeCapabilityMajor() const;
  int computeCapabilityMinor() const;

  /// Check if Tensor Cores available
  bool hasTensorCores() const;

  /// Get available VRAM in bytes
  size_t availableMemory() const;
  size_t totalMemory() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

// ============================================================================
// CUDA Kernel Helpers
// ============================================================================

/// Launch configuration for CUDA kernels
struct CUDALaunchConfig {
  dim3 grid;
  dim3 block;
  size_t sharedMemory = 0;
  cudaStream_t stream = nullptr;
};

/// Calculate optimal block size for kernel
inline dim3 calcBlockSize(uint32_t width, uint32_t height, uint32_t blockX = 16,
                          uint32_t blockY = 16) {
  return dim3(blockX, blockY, 1);
}

/// Calculate grid size for given dimensions and block size
inline dim3 calcGridSize(uint32_t width, uint32_t height, dim3 block) {
  return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              1);
}

} // namespace GPU
} // namespace CaptionEngine
