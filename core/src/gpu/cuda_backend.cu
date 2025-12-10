/**
 * @file cuda_backend.cu
 * @brief CUDA compute backend implementation
 *
 * High-performance GPU compute for NVIDIA hardware.
 */

#include "gpu/cuda_backend.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include <cstring>
#include <iostream>
#include <vector>


namespace CaptionEngine {
namespace GPU {

// ============================================================================
// CUDA Error Handling
// ============================================================================

#ifdef __CUDACC__
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "❌ CUDA error: " << cudaGetErrorString(err) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_RET(call, ret)                                              \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "❌ CUDA error: " << cudaGetErrorString(err) << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return ret;                                                              \
    }                                                                          \
  } while (0)
#endif

// ============================================================================
// CUDA Buffer Implementation
// ============================================================================

class CUDABuffer : public Buffer {
public:
  CUDABuffer(size_t size, BufferUsage usage) : size_(size), usage_(usage) {
#ifdef __CUDACC__
    cudaError_t err = cudaMalloc(&devicePtr_, size);
    if (err != cudaSuccess) {
      std::cerr << "❌ Failed to allocate CUDA buffer: "
                << cudaGetErrorString(err) << std::endl;
      devicePtr_ = nullptr;
    }
#endif
  }

  ~CUDABuffer() override {
#ifdef __CUDACC__
    if (devicePtr_)
      cudaFree(devicePtr_);
#endif
  }

  size_t size() const override { return size_; }
  BufferUsage usage() const override { return usage_; }

  void write(std::span<const uint8_t> data, size_t offset) override {
#ifdef __CUDACC__
    if (devicePtr_) {
      cudaMemcpy(static_cast<uint8_t *>(devicePtr_) + offset, data.data(),
                 data.size(), cudaMemcpyHostToDevice);
    }
#endif
  }

  std::vector<uint8_t> read() override {
    std::vector<uint8_t> data(size_);
#ifdef __CUDACC__
    if (devicePtr_) {
      cudaMemcpy(data.data(), devicePtr_, size_, cudaMemcpyDeviceToHost);
    }
#endif
    return data;
  }

  void *devicePtr() const { return devicePtr_; }

private:
  void *devicePtr_ = nullptr;
  size_t size_;
  BufferUsage usage_;
};

// ============================================================================
// CUDA Texture Implementation
// ============================================================================

class CUDATexture : public Texture {
public:
  CUDATexture(uint32_t w, uint32_t h, TextureFormat fmt)
      : width_(w), height_(h), format_(fmt) {
#ifdef __CUDACC__
    size_t pitch;
    cudaError_t err =
        cudaMallocPitch(&devicePtr_, &pitch, w * bytesPerPixel(), h);
    if (err != cudaSuccess) {
      std::cerr << "❌ Failed to allocate CUDA texture: "
                << cudaGetErrorString(err) << std::endl;
      devicePtr_ = nullptr;
    }
    pitch_ = pitch;
#endif
  }

  ~CUDATexture() override {
#ifdef __CUDACC__
    if (devicePtr_)
      cudaFree(devicePtr_);
#endif
  }

  uint32_t width() const override { return width_; }
  uint32_t height() const override { return height_; }
  TextureFormat format() const override { return format_; }

  void upload(std::span<const uint8_t> data) override {
#ifdef __CUDACC__
    if (devicePtr_) {
      cudaMemcpy2D(devicePtr_, pitch_, data.data(), width_ * bytesPerPixel(),
                   width_ * bytesPerPixel(), height_, cudaMemcpyHostToDevice);
    }
#endif
  }

  std::vector<uint8_t> download() override {
    std::vector<uint8_t> data(width_ * height_ * bytesPerPixel());
#ifdef __CUDACC__
    if (devicePtr_) {
      cudaMemcpy2D(data.data(), width_ * bytesPerPixel(), devicePtr_, pitch_,
                   width_ * bytesPerPixel(), height_, cudaMemcpyDeviceToHost);
    }
#endif
    return data;
  }

  void *devicePtr() const { return devicePtr_; }
  size_t pitch() const { return pitch_; }

private:
  uint32_t bytesPerPixel() const {
    switch (format_) {
    case TextureFormat::RGBA8:
      return 4;
    case TextureFormat::RGBA16F:
      return 8;
    case TextureFormat::RGBA32F:
      return 16;
    case TextureFormat::R8:
      return 1;
    case TextureFormat::R32F:
      return 4;
    default:
      return 4;
    }
  }

  void *devicePtr_ = nullptr;
  size_t pitch_ = 0;
  uint32_t width_, height_;
  TextureFormat format_;
};

// ============================================================================
// CUDA Command Buffer Implementation
// ============================================================================

class CUDACommandBuffer : public CommandBuffer {
public:
  CUDACommandBuffer(cudaStream_t stream) : stream_(stream) {}

  void begin() override {
    // CUDA commands are immediate, no explicit begin
  }

  void end() override {
    // Nothing to do
  }

  void setComputePipeline(Pipeline *) override {
    // CUDA doesn't have pipelines like Vulkan
  }

  void setRenderPipeline(Pipeline *) override {
    // CUDA is compute-only
  }

  void bindBuffer(uint32_t binding, Buffer *buffer) override {
    boundBuffers_[binding] = static_cast<CUDABuffer *>(buffer);
  }

  void bindTexture(uint32_t binding, Texture *texture) override {
    boundTextures_[binding] = static_cast<CUDATexture *>(texture);
  }

  void dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) override {
    // Store dispatch info - actual kernel launch done separately
    lastDispatch_ = {groupsX, groupsY, groupsZ};
  }

  void draw(uint32_t, uint32_t) override {
    // CUDA is compute-only, no draw calls
  }

  void copyTextureToBuffer(Texture *src, Buffer *dst) override {
#ifdef __CUDACC__
    auto *cudaSrc = static_cast<CUDATexture *>(src);
    auto *cudaDst = static_cast<CUDABuffer *>(dst);

    cudaMemcpy2DAsync(cudaDst->devicePtr(), cudaSrc->width() * 4,
                      cudaSrc->devicePtr(), cudaSrc->pitch(),
                      cudaSrc->width() * 4, cudaSrc->height(),
                      cudaMemcpyDeviceToDevice, stream_);
#endif
  }

  cudaStream_t stream() const { return stream_; }

  CUDABuffer *getBuffer(uint32_t binding) const {
    auto it = boundBuffers_.find(binding);
    return it != boundBuffers_.end() ? it->second : nullptr;
  }

  CUDATexture *getTexture(uint32_t binding) const {
    auto it = boundTextures_.find(binding);
    return it != boundTextures_.end() ? it->second : nullptr;
  }

private:
  cudaStream_t stream_;
  std::unordered_map<uint32_t, CUDABuffer *> boundBuffers_;
  std::unordered_map<uint32_t, CUDATexture *> boundTextures_;
  struct {
    uint32_t x, y, z;
  } lastDispatch_ = {0, 0, 0};
};

// ============================================================================
// CUDA Backend Implementation
// ============================================================================

struct CUDABackend::Impl {
  int deviceId = -1;
  cudaStream_t stream = nullptr;
  bool ready = false;

  // Device properties
  int computeMajor = 0;
  int computeMinor = 0;
  size_t totalMem = 0;
  std::string deviceName;
};

CUDABackend::CUDABackend() : pimpl_(std::make_unique<Impl>()) {
#ifdef __CUDACC__
  // Get device count
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  if (err != cudaSuccess || deviceCount == 0) {
    std::cerr << "❌ No CUDA-capable devices found" << std::endl;
    return;
  }

  // Select best device (highest compute capability)
  int bestDevice = 0;
  int bestCompute = 0;

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    int compute = prop.major * 10 + prop.minor;
    if (compute > bestCompute) {
      bestCompute = compute;
      bestDevice = i;
    }
  }

  // Set device
  err = cudaSetDevice(bestDevice);
  if (err != cudaSuccess) {
    std::cerr << "❌ Failed to set CUDA device: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }

  pimpl_->deviceId = bestDevice;

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, bestDevice);

  pimpl_->computeMajor = prop.major;
  pimpl_->computeMinor = prop.minor;
  pimpl_->totalMem = prop.totalGlobalMem;
  pimpl_->deviceName = prop.name;

  // Create stream
  err = cudaStreamCreate(&pimpl_->stream);
  if (err != cudaSuccess) {
    std::cerr << "❌ Failed to create CUDA stream: " << cudaGetErrorString(err)
              << std::endl;
    return;
  }

  pimpl_->ready = true;

  std::cout << "✅ CUDA backend initialized: " << prop.name << std::endl;
  std::cout << "   Compute: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "   VRAM: " << (prop.totalGlobalMem / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "   Tensor Cores: " << (hasTensorCores() ? "Yes" : "No")
            << std::endl;
#else
  std::cerr << "❌ CUDA not available (compiled without CUDA support)"
            << std::endl;
#endif
}

CUDABackend::~CUDABackend() {
#ifdef __CUDACC__
  if (pimpl_->stream) {
    cudaStreamSynchronize(pimpl_->stream);
    cudaStreamDestroy(pimpl_->stream);
  }
  if (pimpl_->deviceId >= 0) {
    cudaDeviceReset();
  }
#endif
}

std::string CUDABackend::name() const {
  if (!pimpl_->ready)
    return "CUDA (not available)";
  return "CUDA " + std::to_string(pimpl_->computeMajor) + "." +
         std::to_string(pimpl_->computeMinor) + " - " + pimpl_->deviceName;
}

bool CUDABackend::isReady() const { return pimpl_->ready; }

int CUDABackend::deviceId() const { return pimpl_->deviceId; }

cudaStream_t CUDABackend::stream() const { return pimpl_->stream; }

int CUDABackend::computeCapabilityMajor() const { return pimpl_->computeMajor; }

int CUDABackend::computeCapabilityMinor() const { return pimpl_->computeMinor; }

bool CUDABackend::hasTensorCores() const {
  // Tensor Cores available on Volta (7.0) and later
  return pimpl_->computeMajor >= 7;
}

size_t CUDABackend::totalMemory() const { return pimpl_->totalMem; }

size_t CUDABackend::availableMemory() const {
#ifdef __CUDACC__
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return free;
#else
  return 0;
#endif
}

GPUResult<std::unique_ptr<Buffer>>
CUDABackend::createBuffer(size_t size, BufferUsage usage) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"CUDA not ready"});
  }
  return std::make_unique<CUDABuffer>(size, usage);
}

GPUResult<std::unique_ptr<Texture>>
CUDABackend::createTexture(uint32_t width, uint32_t height,
                           TextureFormat format) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"CUDA not ready"});
  }
  return std::make_unique<CUDATexture>(width, height, format);
}

GPUResult<std::unique_ptr<Shader>>
CUDABackend::createShaderWGSL(const std::string &, ShaderStage,
                              const std::string &) {
  return std::unexpected(GPUError{"CUDA uses kernels, not shaders"});
}

GPUResult<std::unique_ptr<Shader>>
CUDABackend::createShaderSPIRV(std::span<const uint32_t>, ShaderStage,
                               const std::string &) {
  return std::unexpected(GPUError{"CUDA uses kernels, not SPIR-V"});
}

GPUResult<std::unique_ptr<Pipeline>>
CUDABackend::createComputePipeline(Shader *) {
  return std::unexpected(GPUError{"CUDA uses direct kernel launches"});
}

GPUResult<std::unique_ptr<Pipeline>>
CUDABackend::createRenderPipeline(Shader *, Shader *, TextureFormat) {
  return std::unexpected(
      GPUError{"CUDA is compute-only, use Vulkan for rendering"});
}

GPUResult<std::unique_ptr<CommandBuffer>> CUDABackend::createCommandBuffer() {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"CUDA not ready"});
  }
  return std::make_unique<CUDACommandBuffer>(pimpl_->stream);
}

void CUDABackend::submit(CommandBuffer *) {
  // CUDA commands are immediate, nothing to submit
}

void CUDABackend::waitIdle() {
#ifdef __CUDACC__
  if (pimpl_->stream) {
    cudaStreamSynchronize(pimpl_->stream);
  }
#endif
}

} // namespace GPU
} // namespace CaptionEngine

// ============================================================================
// CUDA Kernels for Caption Effects
// ============================================================================

#ifdef __CUDACC__

namespace CaptionEngine {
namespace Kernels {

/// SDF generation kernel
__global__ void generateSDF(const uint8_t *__restrict__ input,
                            float *__restrict__ output, int width, int height,
                            float spread) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  bool inside = input[idx] > 127;

  float minDist = spread;
  int searchRadius = static_cast<int>(ceilf(spread));

  for (int dy = -searchRadius; dy <= searchRadius; dy++) {
    for (int dx = -searchRadius; dx <= searchRadius; dx++) {
      int sx = x + dx;
      int sy = y + dy;

      if (sx < 0 || sx >= width || sy < 0 || sy >= height)
        continue;

      bool sampleInside = input[sy * width + sx] > 127;

      if (sampleInside != inside) {
        float dist = sqrtf(static_cast<float>(dx * dx + dy * dy));
        minDist = fminf(minDist, dist);
      }
    }
  }

  // Normalize to [0, 1]
  float sdfValue;
  if (inside) {
    sdfValue = 0.5f + (minDist / spread) * 0.5f;
  } else {
    sdfValue = 0.5f - (minDist / spread) * 0.5f;
  }

  output[idx] = fmaxf(0.0f, fminf(1.0f, sdfValue));
}

/// Alpha compositing kernel (Porter-Duff over)
__global__ void compositeOver(const float4 *__restrict__ backdrop,
                              const float4 *__restrict__ source,
                              float4 *__restrict__ output, int width,
                              int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  float4 bg = backdrop[idx];
  float4 fg = source[idx];

  float srcA = fg.w;
  float dstA = bg.w;
  float outA = srcA + dstA * (1.0f - srcA);

  if (outA < 0.001f) {
    output[idx] = make_float4(0, 0, 0, 0);
    return;
  }

  float invOutA = 1.0f / outA;
  output[idx] =
      make_float4((fg.x * srcA + bg.x * dstA * (1.0f - srcA)) * invOutA,
                  (fg.y * srcA + bg.y * dstA * (1.0f - srcA)) * invOutA,
                  (fg.z * srcA + bg.z * dstA * (1.0f - srcA)) * invOutA, outA);
}

/// Box blur kernel (for glow effects)
__global__ void boxBlur(const float4 *__restrict__ input,
                        float4 *__restrict__ output, int width, int height,
                        int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float4 sum = make_float4(0, 0, 0, 0);
  int count = 0;

  for (int dy = -radius; dy <= radius; dy++) {
    for (int dx = -radius; dx <= radius; dx++) {
      int sx = min(max(x + dx, 0), width - 1);
      int sy = min(max(y + dy, 0), height - 1);

      float4 sample = input[sy * width + sx];
      sum.x += sample.x;
      sum.y += sample.y;
      sum.z += sample.z;
      sum.w += sample.w;
      count++;
    }
  }

  float invCount = 1.0f / static_cast<float>(count);
  output[y * width + x] = make_float4(sum.x * invCount, sum.y * invCount,
                                      sum.z * invCount, sum.w * invCount);
}

} // namespace Kernels
} // namespace CaptionEngine

#endif // __CUDACC__
