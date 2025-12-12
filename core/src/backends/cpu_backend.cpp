/**
 * @file cpu_backend.cpp
 * @brief CPU fallback compute backend implementation
 */

#include "compute/unified_backend.hpp"
#include <algorithm> // for std::min, std::copy
#include <cstring>
#include <iostream>
#include <unordered_map>


namespace CaptionEngine {
namespace Compute {

struct CPUBackend::Impl {
  struct Buffer {
    std::vector<uint8_t> data;
    BufferUsage usage;
    MemoryType type;
  };
  std::unordered_map<BufferHandle, Buffer> buffers;
  BufferHandle next_buffer_id = 1;
};

CPUBackend::CPUBackend() : pimpl_(std::make_unique<Impl>()) {}
CPUBackend::~CPUBackend() = default;

BackendInfo CPUBackend::get_info() const {
  return {"CPU Fallback",
          "Generic",
          "1.0",
          0, // Memory info not tracked
          0,
          1,
          1,
          {1, 1, 1},
          BackendCapability::None,
          false,
          false};
}

bool CPUBackend::initialize() { return true; }
void CPUBackend::shutdown() {}

BufferHandle CPUBackend::create_buffer(size_t size, BufferUsage usage,
                                       MemoryType type) {
  auto id = pimpl_->next_buffer_id++;
  Impl::Buffer buf;
  buf.data.resize(size);
  buf.usage = usage;
  buf.type = type;
  pimpl_->buffers[id] = std::move(buf);
  return id;
}

void CPUBackend::destroy_buffer(BufferHandle handle) {
  pimpl_->buffers.erase(handle);
}

void CPUBackend::upload(BufferHandle handle, std::span<const uint8_t> data,
                        size_t offset) {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    if (offset + data.size() <= it->second.data.size()) {
      std::memcpy(it->second.data.data() + offset, data.data(), data.size());
    }
  }
}

void CPUBackend::download(BufferHandle handle, std::span<uint8_t> data,
                          size_t offset) {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    size_t copy_size = std::min(data.size(), it->second.data.size() - offset);
    std::memcpy(data.data(), it->second.data.data() + offset, copy_size);
  }
}

void CPUBackend::copy(BufferHandle src, BufferHandle dst, size_t size,
                      size_t src_offset, size_t dst_offset) {
  auto src_it = pimpl_->buffers.find(src);
  auto dst_it = pimpl_->buffers.find(dst);
  if (src_it != pimpl_->buffers.end() && dst_it != pimpl_->buffers.end()) {
    // Simple copy validation
    if (src_offset + size <= src_it->second.data.size() &&
        dst_offset + size <= dst_it->second.data.size()) {
      std::memcpy(dst_it->second.data.data() + dst_offset,
                  src_it->second.data.data() + src_offset, size);
    }
  }
}

void *CPUBackend::map(BufferHandle handle, size_t offset, size_t size) {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    return it->second.data.data() + offset;
  }
  return nullptr;
}

void CPUBackend::unmap(BufferHandle /*handle*/) {}

// Kernel stubs
KernelHandle CPUBackend::create_kernel(std::span<const uint8_t>,
                                       std::string_view) {
  return 0;
}
void CPUBackend::destroy_kernel(KernelHandle) {}
void CPUBackend::bind_buffer(KernelHandle, uint32_t, BufferHandle) {}
void CPUBackend::set_push_constants(KernelHandle, std::span<const uint8_t>) {}
void CPUBackend::dispatch(KernelHandle, uint32_t, uint32_t, uint32_t) {}
void CPUBackend::dispatch_indirect(KernelHandle, BufferHandle, size_t) {}

void CPUBackend::barrier() {}
void CPUBackend::synchronize() {}
void CPUBackend::begin_recording() {}
void CPUBackend::end_recording() {}

} // namespace Compute
} // namespace CaptionEngine
