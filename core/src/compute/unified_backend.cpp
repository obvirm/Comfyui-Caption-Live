/**
 * @file unified_backend.cpp
 * @brief Unified compute backend implementation with auto-detection
 */

// Must define NOMINMAX before Windows.h to prevent min/max macro conflicts
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "compute/unified_backend.hpp"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace CaptionEngine {
namespace Compute {

// ============================================================
// CPU Backend Implementation
// ============================================================

struct CPUBackend::Impl {
  struct Buffer {
    std::vector<uint8_t> data;
    BufferUsage usage;
    MemoryType memory_type;
    void *mapped_ptr = nullptr;
  };

  std::unordered_map<BufferHandle, Buffer> buffers;
  BufferHandle next_buffer_id = 1;
  std::mutex mutex;

  BackendInfo info;
  bool initialized = false;
};

CPUBackend::CPUBackend() : pimpl_(std::make_unique<Impl>()) {
  // Initialize backend info
  pimpl_->info.name = "CPU Fallback";
  pimpl_->info.vendor = "System";
  pimpl_->info.driver_version = "1.0";

  // Get system memory
#if defined(_WIN32)
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);
  pimpl_->info.total_memory = memInfo.ullTotalPhys;
  pimpl_->info.available_memory = memInfo.ullAvailPhys;
#elif defined(__linux__)
  struct sysinfo info;
  sysinfo(&info);
  pimpl_->info.total_memory = info.totalram * info.mem_unit;
  pimpl_->info.available_memory = info.freeram * info.mem_unit;
#else
  pimpl_->info.total_memory = 8ULL * 1024 * 1024 * 1024; // Default 8GB
  pimpl_->info.available_memory = 4ULL * 1024 * 1024 * 1024;
#endif

  // Get CPU core count
  pimpl_->info.compute_units =
      std::max(1u, std::thread::hardware_concurrency());
  pimpl_->info.max_workgroup_size = 1024;
  pimpl_->info.max_workgroup_count = {65535, 65535, 65535};
  pimpl_->info.capabilities =
      BackendCapability::Compute | BackendCapability::Float64;
  pimpl_->info.is_discrete = false;
  pimpl_->info.is_integrated = true;
}

CPUBackend::~CPUBackend() { shutdown(); }

BackendInfo CPUBackend::get_info() const { return pimpl_->info; }

bool CPUBackend::initialize() {
  pimpl_->initialized = true;
  return true;
}

void CPUBackend::shutdown() {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);
  pimpl_->buffers.clear();
  pimpl_->initialized = false;
}

BufferHandle CPUBackend::create_buffer(size_t size, BufferUsage usage,
                                       MemoryType memory_type) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  BufferHandle handle = pimpl_->next_buffer_id++;

  Impl::Buffer buffer;
  buffer.data.resize(size, 0);
  buffer.usage = usage;
  buffer.memory_type = memory_type;

  pimpl_->buffers[handle] = std::move(buffer);
  return handle;
}

void CPUBackend::destroy_buffer(BufferHandle handle) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);
  pimpl_->buffers.erase(handle);
}

void CPUBackend::upload(BufferHandle handle, std::span<const uint8_t> data,
                        size_t offset) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    size_t copy_size = std::min(data.size(), it->second.data.size() - offset);
    std::memcpy(it->second.data.data() + offset, data.data(), copy_size);
  }
}

void CPUBackend::download(BufferHandle handle, std::span<uint8_t> data,
                          size_t offset) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    size_t copy_size = std::min(data.size(), it->second.data.size() - offset);
    std::memcpy(data.data(), it->second.data.data() + offset, copy_size);
  }
}

void CPUBackend::copy(BufferHandle src, BufferHandle dst, size_t size,
                      size_t src_offset, size_t dst_offset) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto src_it = pimpl_->buffers.find(src);
  auto dst_it = pimpl_->buffers.find(dst);

  if (src_it != pimpl_->buffers.end() && dst_it != pimpl_->buffers.end()) {
    size_t actual_size =
        std::min({size, src_it->second.data.size() - src_offset,
                  dst_it->second.data.size() - dst_offset});
    std::memcpy(dst_it->second.data.data() + dst_offset,
                src_it->second.data.data() + src_offset, actual_size);
  }
}

void *CPUBackend::map(BufferHandle handle, size_t offset, size_t /*size*/) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    it->second.mapped_ptr = it->second.data.data() + offset;
    return it->second.mapped_ptr;
  }
  return nullptr;
}

void CPUBackend::unmap(BufferHandle handle) {
  std::lock_guard<std::mutex> lock(pimpl_->mutex);

  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    it->second.mapped_ptr = nullptr;
  }
}

KernelHandle CPUBackend::create_kernel(std::span<const uint8_t> /*bytecode*/,
                                       std::string_view /*entry_point*/) {
  // CPU backend doesn't support compiled kernels
  return InvalidKernel;
}

void CPUBackend::destroy_kernel(KernelHandle /*handle*/) {
  // No-op
}

void CPUBackend::bind_buffer(KernelHandle /*kernel*/, uint32_t /*binding*/,
                             BufferHandle /*buffer*/) {
  // No-op
}

void CPUBackend::set_push_constants(KernelHandle /*kernel*/,
                                    std::span<const uint8_t> /*data*/) {
  // No-op
}

void CPUBackend::dispatch(KernelHandle /*kernel*/, uint32_t /*groups_x*/,
                          uint32_t /*groups_y*/, uint32_t /*groups_z*/) {
  // No-op - CPU uses different execution model
}

void CPUBackend::dispatch_indirect(KernelHandle /*kernel*/,
                                   BufferHandle /*indirect_buffer*/,
                                   size_t /*offset*/) {
  // No-op
}

void CPUBackend::barrier() {
  // No-op for CPU
}

void CPUBackend::synchronize() {
  // No-op - CPU operations are synchronous
}

void CPUBackend::begin_recording() {
  // No-op
}

void CPUBackend::end_recording() {
  // No-op
}

// ============================================================
// Factory Methods
// ============================================================

std::unique_ptr<UnifiedBackend> UnifiedBackend::create_best() {
  auto backends = available_backends();

  // Priority: CUDA > Vulkan > Metal > DirectX12 > WebGPU > CPU
  const std::vector<std::string> priority = {"cuda", "vulkan", "metal",
                                             "dx12", "webgpu", "cpu"};

  for (const auto &pref : priority) {
    if (std::find(backends.begin(), backends.end(), pref) != backends.end()) {
      return create(pref);
    }
  }

  // Fallback to CPU
  return std::make_unique<CPUBackend>();
}

std::unique_ptr<UnifiedBackend>
UnifiedBackend::create(const std::string &type) {
  if (type == "cpu" || type == "CPU") {
    return std::make_unique<CPUBackend>();
  }

  // For other backends, return CPU fallback for now
  // Real implementations would check availability and create appropriate
  // backend
  return std::make_unique<CPUBackend>();
}

std::vector<std::string> UnifiedBackend::available_backends() {
  std::vector<std::string> backends;

  // Always available
  backends.push_back("cpu");

  // TODO: Check for CUDA, Vulkan, Metal, etc.
  // This would require platform-specific detection

  return backends;
}

} // namespace Compute
} // namespace CaptionEngine
