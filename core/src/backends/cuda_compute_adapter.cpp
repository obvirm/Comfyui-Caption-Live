/**
 * @file cuda_compute_adapter.cpp
 * @brief Adapter bridging GPU::CUDABackend to ComputeBackend interface
 *
 * This adapter allows CUDA to be used through the unified ComputeBackend
 * interface, enabling seamless backend switching between CPU, WebGPU, and CUDA.
 */

#include "graphics/backend.hpp"
#include "platform/platform.hpp"

#if CE_HAS_CUDA

#include "cuda/cuda_kernels.hpp"
#include "gpu/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>

namespace CaptionEngine {

/**
 * @brief CUDA Compute Backend Adapter
 *
 * Bridges the GPU::CUDABackend to the ComputeBackend interface used by
 * the main engine.
 */
class CUDAComputeBackend : public ComputeBackend {
public:
  CUDAComputeBackend();
  ~CUDAComputeBackend() override;

  // ComputeBackend interface
  [[nodiscard]] std::string name() const override;
  [[nodiscard]] bool supports_compute() const override { return initialized_; }

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

  // CUDA-specific
  bool initialize();
  cudaStream_t stream() const { return stream_; }

  // Effect dispatch helpers
  void dispatch_text_glow(uint32_t *output, const uint32_t *input, int width,
                          int height, float glow_radius, uint32_t glow_color,
                          float glow_intensity);

  void dispatch_glitch(uint32_t *output, const uint32_t *input, int width,
                       int height, float intensity, float time, uint32_t seed);

  void dispatch_blur(uint32_t *output, const uint32_t *input, uint32_t *temp,
                     int width, int height, float sigma);

  void dispatch_composite(uint32_t *output, const uint32_t *bg,
                          const uint32_t *fg, int width, int height,
                          float opacity);

private:
  struct BufferInfo {
    void *device_ptr = nullptr;
    size_t size = 0;
    MemoryType type = MemoryType::DeviceLocal;
  };

  std::unordered_map<BufferHandle, BufferInfo> buffers_;
  BufferHandle next_handle_ = 1;

  cudaStream_t stream_ = nullptr;
  int device_id_ = -1;
  bool initialized_ = false;

  std::string device_name_;
  int compute_major_ = 0;
  int compute_minor_ = 0;
  size_t total_memory_ = 0;
};

// ============================================================================
// Implementation
// ============================================================================

CUDAComputeBackend::CUDAComputeBackend() { initialize(); }

CUDAComputeBackend::~CUDAComputeBackend() {
  // Free all buffers
  for (auto &[handle, info] : buffers_) {
    if (info.device_ptr) {
      cudaFree(info.device_ptr);
    }
  }
  buffers_.clear();

  // Destroy stream
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  if (device_id_ >= 0) {
    cudaDeviceReset();
  }
}

bool CUDAComputeBackend::initialize() {
  if (initialized_)
    return true;

  // Get device count
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "❌ CUDA: No CUDA-capable devices found" << std::endl;
    return false;
  }

  // Select best device (highest compute capability)
  int best_device = 0;
  int best_compute = 0;

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    int compute = prop.major * 10 + prop.minor;
    if (compute > best_compute) {
      best_compute = compute;
      best_device = i;
    }
  }

  // Set device
  err = cudaSetDevice(best_device);
  if (err != cudaSuccess) {
    std::cerr << "❌ CUDA: Failed to set device: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  device_id_ = best_device;

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, best_device);

  compute_major_ = prop.major;
  compute_minor_ = prop.minor;
  total_memory_ = prop.totalGlobalMem;
  device_name_ = prop.name;

  // Create stream
  err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    std::cerr << "❌ CUDA: Failed to create stream: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  initialized_ = true;

  std::cout << "✅ CUDA Compute Backend initialized: " << device_name_
            << std::endl;
  std::cout << "   Compute: " << compute_major_ << "." << compute_minor_
            << std::endl;
  std::cout << "   VRAM: " << (total_memory_ / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "   Tensor Cores: " << (compute_major_ >= 7 ? "Yes" : "No")
            << std::endl;

  return true;
}

std::string CUDAComputeBackend::name() const {
  if (!initialized_)
    return "CUDA (not initialized)";
  return "CUDA " + std::to_string(compute_major_) + "." +
         std::to_string(compute_minor_) + " - " + device_name_;
}

BufferHandle CUDAComputeBackend::create_buffer(size_t size, MemoryType type) {
  if (!initialized_)
    return 0;

  void *device_ptr = nullptr;
  cudaError_t err = cudaMalloc(&device_ptr, size);

  if (err != cudaSuccess) {
    std::cerr << "❌ CUDA: Failed to allocate buffer: "
              << cudaGetErrorString(err) << std::endl;
    return 0;
  }

  // Zero-initialize
  cudaMemset(device_ptr, 0, size);

  BufferHandle handle = next_handle_++;
  buffers_[handle] = {device_ptr, size, type};

  return handle;
}

void CUDAComputeBackend::destroy_buffer(BufferHandle handle) {
  auto it = buffers_.find(handle);
  if (it != buffers_.end()) {
    if (it->second.device_ptr) {
      cudaFree(it->second.device_ptr);
    }
    buffers_.erase(it);
  }
}

void CUDAComputeBackend::upload_buffer(BufferHandle handle,
                                       std::span<const uint8_t> data) {
  auto it = buffers_.find(handle);
  if (it == buffers_.end() || !it->second.device_ptr)
    return;

  size_t copy_size = std::min(data.size(), it->second.size);
  cudaMemcpyAsync(it->second.device_ptr, data.data(), copy_size,
                  cudaMemcpyHostToDevice, stream_);
}

std::vector<uint8_t> CUDAComputeBackend::download_buffer(BufferHandle handle,
                                                         size_t size) {
  auto it = buffers_.find(handle);
  if (it == buffers_.end() || !it->second.device_ptr) {
    return {};
  }

  size_t copy_size = std::min(size, it->second.size);
  std::vector<uint8_t> data(copy_size);

  cudaMemcpyAsync(data.data(), it->second.device_ptr, copy_size,
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  return data;
}

void CUDAComputeBackend::dispatch_compute(std::string_view shader_name,
                                          std::span<BufferHandle> buffers,
                                          WorkGroupSize workgroups) {
  if (!initialized_)
    return;

  // Map shader names to CUDA kernel launches
  // This is where we bridge the abstract compute interface to concrete kernels

  if (shader_name == "text_glow" && buffers.size() >= 2) {
    auto *output = static_cast<uint32_t *>(buffers_[buffers[0]].device_ptr);
    auto *input =
        static_cast<const uint32_t *>(buffers_[buffers[1]].device_ptr);

    int width = workgroups.x * 16; // Assuming 16x16 workgroup size
    int height = workgroups.y * 16;

    // Default glow parameters - in practice these would come from uniforms
    CUDA::launch_text_glow(output, input, width, height, 10.0f, 0xFF00FFFF,
                           0.8f, stream_);
  } else if (shader_name == "composite" && buffers.size() >= 3) {
    auto *output = static_cast<uint32_t *>(buffers_[buffers[0]].device_ptr);
    auto *bg = static_cast<const uint32_t *>(buffers_[buffers[1]].device_ptr);
    auto *fg = static_cast<const uint32_t *>(buffers_[buffers[2]].device_ptr);

    int width = workgroups.x * 16;
    int height = workgroups.y * 16;

    CUDA::launch_composite(output, bg, fg, width, height, 1.0f, stream_);
  } else if (shader_name == "blur" && buffers.size() >= 3) {
    auto *output = static_cast<uint32_t *>(buffers_[buffers[0]].device_ptr);
    auto *input =
        static_cast<const uint32_t *>(buffers_[buffers[1]].device_ptr);
    auto *temp = static_cast<uint32_t *>(buffers_[buffers[2]].device_ptr);

    int width = workgroups.x * 16;
    int height = workgroups.y * 16;

    CUDA::launch_blur(output, input, temp, width, height, 5.0f, stream_);
  } else if (shader_name == "glitch" && buffers.size() >= 2) {
    auto *output = static_cast<uint32_t *>(buffers_[buffers[0]].device_ptr);
    auto *input =
        static_cast<const uint32_t *>(buffers_[buffers[1]].device_ptr);

    int width = workgroups.x * 16;
    int height = workgroups.y * 16;

    CUDA::launch_glitch(output, input, width, height, 0.5f, 0.0f, 12345,
                        stream_);
  }

  // Add more kernel mappings as needed
}

bool CUDAComputeBackend::register_kernel(const ComputeKernel &kernel) {
  // For CUDA, kernels are compiled at build time
  // This could be used for JIT compilation of PTX in the future
  if (kernel.format == ComputeKernel::Format::PTX) {
    // TODO: Implement PTX JIT compilation
    std::cout << "📝 CUDA: Registered kernel: " << kernel.name << std::endl;
    return true;
  }
  return false;
}

void CUDAComputeBackend::synchronize() {
  if (stream_) {
    cudaStreamSynchronize(stream_);
  }
}

// ============================================================================
// Effect Dispatch Helpers (direct kernel calls)
// ============================================================================

void CUDAComputeBackend::dispatch_text_glow(uint32_t *output,
                                            const uint32_t *input, int width,
                                            int height, float glow_radius,
                                            uint32_t glow_color,
                                            float glow_intensity) {
  if (!initialized_)
    return;
  CUDA::launch_text_glow(output, input, width, height, glow_radius, glow_color,
                         glow_intensity, stream_);
}

void CUDAComputeBackend::dispatch_glitch(uint32_t *output,
                                         const uint32_t *input, int width,
                                         int height, float intensity,
                                         float time, uint32_t seed) {
  if (!initialized_)
    return;
  CUDA::launch_glitch(output, input, width, height, intensity, time, seed,
                      stream_);
}

void CUDAComputeBackend::dispatch_blur(uint32_t *output, const uint32_t *input,
                                       uint32_t *temp, int width, int height,
                                       float sigma) {
  if (!initialized_)
    return;
  CUDA::launch_blur(output, input, temp, width, height, sigma, stream_);
}

void CUDAComputeBackend::dispatch_composite(uint32_t *output,
                                            const uint32_t *bg,
                                            const uint32_t *fg, int width,
                                            int height, float opacity) {
  if (!initialized_)
    return;
  CUDA::launch_composite(output, bg, fg, width, height, opacity, stream_);
}

// ============================================================================
// Factory Function Registration
// ============================================================================

// This function is called from cpu_backend.cpp factory
std::unique_ptr<ComputeBackend> create_cuda_backend() {
  auto backend = std::make_unique<CUDAComputeBackend>();
  if (backend->supports_compute()) {
    return backend;
  }
  return nullptr;
}

} // namespace CaptionEngine

#else // !CE_HAS_CUDA

// When CUDA is not available, we still need to provide the factory function
// but it just returns nullptr

namespace CaptionEngine {

// Stub when CUDA not available
std::unique_ptr<ComputeBackend> create_cuda_backend() { return nullptr; }

} // namespace CaptionEngine

#endif // CE_HAS_CUDA
