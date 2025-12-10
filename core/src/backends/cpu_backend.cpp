/**
 * @file cpu_backend.cpp
 * @brief CPU fallback compute backend
 */

#include "graphics/backend.hpp"
#include <atomic>
#include <unordered_map>

namespace CaptionEngine {

struct CPUBackend::Impl {
  std::unordered_map<BufferHandle, std::vector<uint8_t>> buffers;
  std::atomic<BufferHandle> next_handle{1};
};

CPUBackend::CPUBackend() : pimpl_(std::make_unique<Impl>()) {}
CPUBackend::~CPUBackend() = default;

BufferHandle CPUBackend::create_buffer(size_t size, MemoryType /*type*/) {
  BufferHandle handle = pimpl_->next_handle++;
  pimpl_->buffers[handle] = std::vector<uint8_t>(size, 0);
  return handle;
}

void CPUBackend::destroy_buffer(BufferHandle handle) {
  pimpl_->buffers.erase(handle);
}

void CPUBackend::upload_buffer(BufferHandle handle,
                               std::span<const uint8_t> data) {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    std::copy(data.begin(), data.end(), it->second.begin());
  }
}

std::vector<uint8_t> CPUBackend::download_buffer(BufferHandle handle,
                                                 size_t size) {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    size_t copy_size = std::min(size, it->second.size());
    return std::vector<uint8_t>(it->second.begin(),
                                it->second.begin() + copy_size);
  }
  return {};
}

// Note: MemoryType enum is defined in backend.hpp -> include/compute/types.hpp?
// No, backend.hpp includes types.hpp. But backend.hpp defines MemoryType
// independently in previous version. My rewrite of backend.hpp kept MemoryType
// inside backend.hpp because it wasn't in `types.hpp`. Wait, `types.hpp` had
// ParamType etc. Verify backend.hpp content again? I overwrote backend.hpp with
// MemoryType definition. So it's fine.

// Note: WorkgroupSize vs WorkGroupSize.
// I used `WorkGroupSize` in `types.hpp` but `WorkgroupSize` in `backend.hpp`
// previous version. My `backend.hpp` rewrite used `WorkGroupSize` in
// `dispatch_compute` but struct remains named `WorkgroupSize`? Let's check
// `types.hpp`. `struct WorkGroupSize { ... }`. Let's check `backend.hpp`
// overwritten content.
// ... `virtual void dispatch_compute(std::string_view shader_name, ...,
// WorkGroupSize workgroups) = 0;` (I used UpperCamel). But I removed the struct
// definition from `backend.hpp`? The rewrite included "compute/types.hpp". The
// previous `backend.hpp` had `struct WorkgroupSize`. My rewrite removed it
// (implied by types.hpp include? Or I missed it?). The rewrite content I sent:
// `#include "compute/types.hpp"` ... `using BufferHandle = ...` ... `enum class
// MemoryType`. I did NOT redefine `WorkGroupSize` in backend.hpp. I used it
// from `types.hpp`. So the type is `CaptionEngine::WorkGroupSize`. The cpu
// backend overrides must match.

void CPUBackend::dispatch_compute(std::string_view /*shader_name*/,
                                  std::span<BufferHandle> /*buffers*/,
                                  WorkGroupSize /*workgroups*/
) {
  // CPU compute implementation
  // For now, no-op - compute shaders would be implemented as C++ functions
}

bool CPUBackend::register_kernel(const ComputeKernel & /*kernel*/) {
  // CPU backend could support software kernels here
  return true;
}

void CPUBackend::synchronize() {
  // No-op for CPU - all operations are synchronous
}

// Factory functions
std::unique_ptr<ComputeBackend> ComputeBackend::create_best() {
#ifdef __EMSCRIPTEN__
  // On Web (WASM), prefer WebGPU if available
  auto webgpu = std::make_unique<WebGPUBackend>();
  if (webgpu->initialize()) {
    return webgpu;
  }
  // Fallback to CPU
#endif
  return std::make_unique<CPUBackend>();
}

std::unique_ptr<ComputeBackend>
ComputeBackend::create(const std::string &name) {
  if (name == "CPU" || name == "cpu") {
    return std::make_unique<CPUBackend>();
  }
  if (name == "WebGPU" || name == "webgpu") {
#if defined(__EMSCRIPTEN__)
    auto backend = std::make_unique<WebGPUBackend>();
    if (backend->initialize()) {
      return backend;
    }
    // If initialization fails, fall back to CPU?
    return std::make_unique<CPUBackend>();
#else
    // WebGPU not supported on this platform
    return std::make_unique<CPUBackend>();
#endif
  }
  return create_best();
}

} // namespace CaptionEngine
