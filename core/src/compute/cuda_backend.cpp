/**
 * @file cuda_backend.cpp
 * @brief CUDA compute backend implementation for NVIDIA GPUs
 */

#if defined(CAPTION_HAS_CUDA) || defined(__CUDACC__)

#include "compute/cuda.hpp"
#include "compute/unified_backend.hpp"
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


namespace CaptionEngine {
namespace Compute {

// ============================================================
// CUDA Error Checking
// ============================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err) + " at " + __FILE__ +   \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

// ============================================================
// CUDA Backend Implementation
// ============================================================

struct CUDABackend::Impl {
  int device_id = 0;
  cudaDeviceProp device_props{};
  cudaStream_t stream = nullptr;

  struct Buffer {
    void *device_ptr = nullptr;
    size_t size = 0;
    MemoryType type = MemoryType::DeviceLocal;
    void *host_ptr = nullptr; // For unified memory
  };

  std::unordered_map<BufferHandle, Buffer> buffers;
  BufferHandle next_buffer_id = 1;

  std::unordered_map<std::string, CUfunction> kernels;
  std::mutex mutex;

  bool initialized = false;

  BackendInfo info;
};

CUDABackend::CUDABackend() : pimpl_(std::make_unique<Impl>()) {
  // Find best device
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  // Select device with highest compute capability
  int best_device = 0;
  int best_score = 0;

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);

    int score = props.major * 10 + props.minor;
    if (score > best_score) {
      best_score = score;
      best_device = i;
    }
  }

  pimpl_->device_id = best_device;
  CUDA_CHECK(cudaSetDevice(best_device));
  CUDA_CHECK(cudaGetDeviceProperties(&pimpl_->device_props, best_device));

  // Create stream
  CUDA_CHECK(cudaStreamCreate(&pimpl_->stream));

  // Fill backend info
  pimpl_->info.name = pimpl_->device_props.name;
  pimpl_->info.vendor = "NVIDIA";
  pimpl_->info.driver_version = std::to_string(pimpl_->device_props.major) +
                                "." +
                                std::to_string(pimpl_->device_props.minor);
  pimpl_->info.total_memory = pimpl_->device_props.totalGlobalMem;

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  pimpl_->info.available_memory = free_mem;

  pimpl_->info.compute_units = pimpl_->device_props.multiProcessorCount;
  pimpl_->info.max_workgroup_size = pimpl_->device_props.maxThreadsPerBlock;
  pimpl_->info.max_workgroup_count = {
      static_cast<uint32_t>(pimpl_->device_props.maxGridSize[0]),
      static_cast<uint32_t>(pimpl_->device_props.maxGridSize[1]),
      static_cast<uint32_t>(pimpl_->device_props.maxGridSize[2])};

  pimpl_->info.capabilities =
      BackendCapability::Compute | BackendCapability::AsyncCompute |
      BackendCapability::SharedMemory | BackendCapability::AtomicOperations |
      BackendCapability::Float16 | BackendCapability::Float64;

  pimpl_->info.is_discrete = true;
  pimpl_->info.is_integrated = false;
  pimpl_->initialized = true;
}

CUDABackend::~CUDABackend() { shutdown(); }

BackendInfo CUDABackend::get_info() const { return pimpl_->info; }

bool CUDABackend::initialize() { return pimpl_->initialized; }

void CUDABackend::shutdown() {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  // Synchronize before cleanup
  if (pimpl_->stream) {
    cudaStreamSynchronize(pimpl_->stream);
  }

  // Free all buffers
  for (auto &[handle, buffer] : pimpl_->buffers) {
    if (buffer.type == MemoryType::Unified && buffer.host_ptr) {
      cudaFreeHost(buffer.host_ptr);
    } else if (buffer.device_ptr) {
      cudaFree(buffer.device_ptr);
    }
  }
  pimpl_->buffers.clear();

  // Destroy stream
  if (pimpl_->stream) {
    cudaStreamDestroy(pimpl_->stream);
    pimpl_->stream = nullptr;
  }

  pimpl_->initialized = false;
}

BufferHandle CUDABackend::create_buffer(size_t size, BufferUsage usage,
                                        MemoryType memory_type) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  BufferHandle handle = pimpl_->next_buffer_id++;
  Impl::Buffer buffer;
  buffer.size = size;
  buffer.type = memory_type;

  switch (memory_type) {
  case MemoryType::DeviceLocal:
    CUDA_CHECK(cudaMalloc(&buffer.device_ptr, size));
    break;

  case MemoryType::HostVisible:
  case MemoryType::HostCached:
    // Use pinned host memory for faster transfers
    CUDA_CHECK(cudaMallocHost(&buffer.host_ptr, size));
    CUDA_CHECK(cudaMalloc(&buffer.device_ptr, size));
    break;

  case MemoryType::Unified:
    // Use unified memory
    CUDA_CHECK(cudaMallocManaged(&buffer.device_ptr, size));
    buffer.host_ptr = buffer.device_ptr;
    break;
  }

  pimpl_->buffers[handle] = buffer;
  return handle;
}

void CUDABackend::destroy_buffer(BufferHandle handle) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    auto &buffer = it->second;

    if (buffer.type == MemoryType::Unified) {
      cudaFree(buffer.device_ptr);
    } else {
      if (buffer.host_ptr) {
        cudaFreeHost(buffer.host_ptr);
      }
      if (buffer.device_ptr) {
        cudaFree(buffer.device_ptr);
      }
    }

    pimpl_->buffers.erase(it);
  }
}

void CUDABackend::upload(BufferHandle handle, std::span<const uint8_t> data,
                         size_t offset) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    auto &buffer = it->second;
    size_t copy_size = std::min(data.size(), buffer.size - offset);

    if (buffer.type == MemoryType::Unified) {
      // Direct copy for unified memory
      std::memcpy(static_cast<uint8_t *>(buffer.device_ptr) + offset,
                  data.data(), copy_size);
    } else if (buffer.host_ptr) {
      // Copy to pinned then async transfer
      std::memcpy(static_cast<uint8_t *>(buffer.host_ptr) + offset, data.data(),
                  copy_size);
      CUDA_CHECK(
          cudaMemcpyAsync(static_cast<uint8_t *>(buffer.device_ptr) + offset,
                          static_cast<uint8_t *>(buffer.host_ptr) + offset,
                          copy_size, cudaMemcpyHostToDevice, pimpl_->stream));
    } else {
      // Direct device copy
      CUDA_CHECK(cudaMemcpyAsync(
          static_cast<uint8_t *>(buffer.device_ptr) + offset, data.data(),
          copy_size, cudaMemcpyHostToDevice, pimpl_->stream));
    }
  }
}

void CUDABackend::download(BufferHandle handle, std::span<uint8_t> data,
                           size_t offset) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    auto &buffer = it->second;
    size_t copy_size = std::min(data.size(), buffer.size - offset);

    if (buffer.type == MemoryType::Unified) {
      // Sync first then direct copy
      cudaStreamSynchronize(pimpl_->stream);
      std::memcpy(data.data(),
                  static_cast<uint8_t *>(buffer.device_ptr) + offset,
                  copy_size);
    } else {
      // Async copy to host
      CUDA_CHECK(cudaMemcpyAsync(
          data.data(), static_cast<uint8_t *>(buffer.device_ptr) + offset,
          copy_size, cudaMemcpyDeviceToHost, pimpl_->stream));
      cudaStreamSynchronize(pimpl_->stream);
    }
  }
}

void CUDABackend::copy(BufferHandle src, BufferHandle dst, size_t size,
                       size_t src_offset, size_t dst_offset) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto src_it = pimpl_->buffers.find(src);
  auto dst_it = pimpl_->buffers.find(dst);

  if (src_it != pimpl_->buffers.end() && dst_it != pimpl_->buffers.end()) {
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<uint8_t *>(dst_it->second.device_ptr) + dst_offset,
        static_cast<uint8_t *>(src_it->second.device_ptr) + src_offset, size,
        cudaMemcpyDeviceToDevice, pimpl_->stream));
  }
}

void *CUDABackend::map(BufferHandle handle, size_t offset, size_t size) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    auto &buffer = it->second;

    if (buffer.type == MemoryType::Unified) {
      // Unified memory is always mapped
      cudaStreamSynchronize(pimpl_->stream);
      return static_cast<uint8_t *>(buffer.device_ptr) + offset;
    } else if (buffer.host_ptr) {
      // Sync device to host
      CUDA_CHECK(cudaMemcpyAsync(buffer.host_ptr, buffer.device_ptr,
                                 buffer.size, cudaMemcpyDeviceToHost,
                                 pimpl_->stream));
      cudaStreamSynchronize(pimpl_->stream);
      return static_cast<uint8_t *>(buffer.host_ptr) + offset;
    }
  }
  return nullptr;
}

void CUDABackend::unmap(BufferHandle handle) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    auto &buffer = it->second;

    if (buffer.type != MemoryType::Unified && buffer.host_ptr) {
      // Sync host to device
      CUDA_CHECK(cudaMemcpyAsync(buffer.device_ptr, buffer.host_ptr,
                                 buffer.size, cudaMemcpyHostToDevice,
                                 pimpl_->stream));
    }
  }
}

KernelHandle CUDABackend::create_kernel(std::span<const uint8_t> bytecode,
                                        std::string_view entry_point) {
  // CUDA kernels are loaded via cuModuleLoad from PTX
  // For now, return invalid handle - kernels are pre-registered
  return InvalidKernel;
}

void CUDABackend::destroy_kernel(KernelHandle handle) {
  // No-op for now
}

void CUDABackend::bind_buffer(KernelHandle kernel, uint32_t binding,
                              BufferHandle buffer) {
  // CUDA uses explicit argument passing, not binding
}

void CUDABackend::set_push_constants(KernelHandle kernel,
                                     std::span<const uint8_t> data) {
  // CUDA uses explicit argument passing
}

void CUDABackend::dispatch(KernelHandle kernel, uint32_t groups_x,
                           uint32_t groups_y, uint32_t groups_z) {
  // Kernel dispatch is handled via launch_kernel
}

void CUDABackend::dispatch_indirect(KernelHandle kernel,
                                    BufferHandle indirect_buffer,
                                    size_t offset) {
  // Not directly supported in CUDA
}

void CUDABackend::barrier() {
  CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream));
}

void CUDABackend::synchronize() {
  CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream));
}

void CUDABackend::begin_recording() {
  // CUDA uses implicit command recording
}

void CUDABackend::end_recording() {
  // CUDA uses implicit submission
}

// ============================================================
// CUDA-specific Methods
// ============================================================

int CUDABackend::device_id() const { return pimpl_->device_id; }

cudaDeviceProp CUDABackend::device_properties() const {
  return pimpl_->device_props;
}

size_t CUDABackend::available_memory() const {
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  return free_mem;
}

cudaStream_t CUDABackend::stream() const { return pimpl_->stream; }

void *CUDABackend::get_device_ptr(BufferHandle handle) const {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    return it->second.device_ptr;
  }
  return nullptr;
}

// ============================================================
// Utility Functions
// ============================================================

bool cuda_available() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

int cuda_device_count() {
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}

int cuda_best_device() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (device_count == 0)
    return -1;

  int best_device = 0;
  int best_score = 0;

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);

    int score = props.major * 1000 + props.minor * 100 +
                static_cast<int>(props.totalGlobalMem / (1024 * 1024 * 1024));
    if (score > best_score) {
      best_score = score;
      best_device = i;
    }
  }

  return best_device;
}

} // namespace Compute
} // namespace CaptionEngine

#endif // CAPTION_HAS_CUDA
