/**
 * @file metal_backend.mm
 * @brief Metal compute backend implementation for macOS/iOS
 */

#if defined(HAS_METAL) && defined(__APPLE__)

#include "graphics/backend.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>
#include <unordered_map>
#include <vector>


namespace CaptionEngine {

class MetalBackend : public ComputeBackend {
public:
  MetalBackend() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      throw std::runtime_error("Metal not available");
    }

    command_queue_ = [device_ newCommandQueue];
  }

  ~MetalBackend() override {
    synchronize();

    for (auto &[handle, buffer] : buffers_) {
      [buffer release];
    }

    for (auto &[name, pipeline] : pipelines_) {
      [pipeline release];
    }

    [command_queue_ release];
    [device_ release];
  }

  [[nodiscard]] std::string name() const override { return "Metal"; }
  [[nodiscard]] bool supports_compute() const override { return true; }

  [[nodiscard]] BufferHandle create_buffer(size_t size,
                                           MemoryType type) override {
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (type == MemoryType::DeviceOnly) {
      options = MTLResourceStorageModePrivate;
    }

    id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:options];
    if (!buffer)
      return 0;

    BufferHandle handle = next_buffer_id_++;
    buffers_[handle] = buffer;
    buffer_sizes_[handle] = size;
    return handle;
  }

  void destroy_buffer(BufferHandle handle) override {
    auto it = buffers_.find(handle);
    if (it != buffers_.end()) {
      [it->second release];
      buffers_.erase(it);
      buffer_sizes_.erase(handle);
    }
  }

  void upload_buffer(BufferHandle handle,
                     std::span<const uint8_t> data) override {
    auto it = buffers_.find(handle);
    if (it == buffers_.end())
      return;

    memcpy([it->second contents], data.data(), data.size());
  }

  [[nodiscard]] std::vector<uint8_t> download_buffer(BufferHandle handle,
                                                     size_t size) override {
    auto it = buffers_.find(handle);
    if (it == buffers_.end())
      return {};

    std::vector<uint8_t> result(size);
    memcpy(result.data(), [it->second contents], size);
    return result;
  }

  void dispatch_compute(std::string_view shader_name,
                        std::span<BufferHandle> buffers,
                        WorkGroupSize workgroups) override {
    auto pit = pipelines_.find(std::string(shader_name));
    if (pit == pipelines_.end())
      return;

    id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];

    [encoder setComputePipelineState:pit->second];

    for (size_t i = 0; i < buffers.size(); ++i) {
      auto bit = buffers_.find(buffers[i]);
      if (bit != buffers_.end()) {
        [encoder setBuffer:bit->second offset:0 atIndex:i];
      }
    }

    MTLSize grid_size = MTLSizeMake(workgroups.x, workgroups.y, workgroups.z);
    MTLSize threadgroup_size = MTLSizeMake(16, 16, 1);

    [encoder dispatchThreadgroups:grid_size
            threadsPerThreadgroup:threadgroup_size];

    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }

  bool register_kernel(const ComputeKernel &kernel) override {
    if (kernel.format != ComputeKernel::Format::MSL) {
      return false;
    }

    NSString *source = [[NSString alloc] initWithBytes:kernel.bytecode.data()
                                                length:kernel.bytecode.size()
                                              encoding:NSUTF8StringEncoding];

    NSError *error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:source
                                                   options:nil
                                                     error:&error];
    [source release];

    if (!library || error) {
      return false;
    }

    NSString *functionName =
        [NSString stringWithUTF8String:kernel.name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    [library release];

    if (!function) {
      return false;
    }

    id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:function error:&error];
    [function release];

    if (!pipeline || error) {
      return false;
    }

    pipelines_[kernel.name] = pipeline;
    return true;
  }

  void synchronize() override {
    // Metal command buffers are synchronous when using waitUntilCompleted
  }

  // Metal-specific
  id<MTLDevice> metal_device() const { return device_; }
  id<MTLCommandQueue> command_queue() const { return command_queue_; }

private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;

  std::unordered_map<BufferHandle, id<MTLBuffer>> buffers_;
  std::unordered_map<BufferHandle, size_t> buffer_sizes_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;

  BufferHandle next_buffer_id_ = 1;
};

// Factory function
std::unique_ptr<ComputeBackend> create_metal_backend() {
  try {
    return std::make_unique<MetalBackend>();
  } catch (...) {
    return nullptr;
  }
}

bool metal_available() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device) {
    [device release];
    return true;
  }
  return false;
}

} // namespace CaptionEngine

#endif // HAS_METAL && __APPLE__
