/**
 * @file sycl.hpp
 * @brief SYCL/DPC++ backend for Intel/AMD GPUs
 */

#pragma once

#include "unified_backend.hpp"

#if defined(CAPTION_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace CaptionEngine {
namespace Compute {

/**
 * @brief SYCL backend for cross-vendor GPU compute
 *
 * Supports Intel oneAPI, AMD ROCm via hipSYCL, and other SYCL implementations.
 */
class SYCLBackend : public UnifiedBackend {
public:
  SYCLBackend();
  ~SYCLBackend() override;

  /// Check if SYCL is available on this system
  [[nodiscard]] static bool is_available();

  /// Get available SYCL devices
  [[nodiscard]] static std::vector<std::string> get_devices();

  BackendInfo get_info() const override;
  bool initialize() override;
  void shutdown() override;

  BufferHandle create_buffer(size_t size, BufferUsage usage,
                             MemoryType memory_type) override;
  void destroy_buffer(BufferHandle handle) override;
  void upload(BufferHandle handle, std::span<const uint8_t> data,
              size_t offset) override;
  void download(BufferHandle handle, std::span<uint8_t> data,
                size_t offset) override;
  void copy(BufferHandle src, BufferHandle dst, size_t size, size_t src_offset,
            size_t dst_offset) override;
  void *map(BufferHandle handle, size_t offset, size_t size) override;
  void unmap(BufferHandle handle) override;

  KernelHandle create_kernel(std::span<const uint8_t> bytecode,
                             std::string_view entry_point) override;
  void destroy_kernel(KernelHandle handle) override;
  void bind_buffer(KernelHandle kernel, uint32_t binding,
                   BufferHandle buffer) override;
  void set_push_constants(KernelHandle kernel,
                          std::span<const uint8_t> data) override;
  void dispatch(KernelHandle kernel, uint32_t groups_x, uint32_t groups_y,
                uint32_t groups_z) override;
  void dispatch_indirect(KernelHandle kernel, BufferHandle indirect_buffer,
                         size_t offset) override;

  void barrier() override;
  void synchronize() override;
  void begin_recording() override;
  void end_recording() override;

#if defined(CAPTION_HAS_SYCL)
  /// Get SYCL queue (for advanced usage)
  [[nodiscard]] sycl::queue &get_queue();

  /// Get SYCL device
  [[nodiscard]] const sycl::device &get_device() const;
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Helper for SYCL kernel execution
 */
template <typename KernelFunc>
void execute_sycl_kernel(SYCLBackend &backend,
                         const std::array<size_t, 3> &global_size,
                         const std::array<size_t, 3> &local_size,
                         KernelFunc &&kernel) {
#if defined(CAPTION_HAS_SYCL)
  backend.get_queue().submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(global_size[0], global_size[1], global_size[2]),
            sycl::range<3>(local_size[0], local_size[1], local_size[2])),
        std::forward<KernelFunc>(kernel));
  });
#else
  (void)backend;
  (void)global_size;
  (void)local_size;
  (void)kernel;
  throw std::runtime_error("SYCL not available");
#endif
}

} // namespace Compute
} // namespace CaptionEngine
