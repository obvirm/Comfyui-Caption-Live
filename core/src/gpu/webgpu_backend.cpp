/**
 * @file webgpu_backend.cpp
 * @brief WebGPU backend implementation
 *
 * Works with both:
 * - Emscripten WebGPU (browser)
 * - Dawn (native desktop)
 */

#include "gpu/webgpu_backend.hpp"
#include <iostream>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

namespace CaptionEngine {
namespace GPU {

// ============================================================================
// WebGPU Buffer Implementation
// ============================================================================

class WebGPUBuffer : public Buffer {
public:
  WebGPUBuffer(WGPUDevice device, size_t size, BufferUsage usage)
      : device_(device), size_(size), usage_(usage) {

    WGPUBufferDescriptor desc = {};
    desc.size = size;
    desc.usage = translateUsage(usage);
    desc.mappedAtCreation = false;

    buffer_ = wgpuDeviceCreateBuffer(device, &desc);
  }

  ~WebGPUBuffer() override {
    if (buffer_) {
      wgpuBufferDestroy(buffer_);
      wgpuBufferRelease(buffer_);
    }
  }

  size_t size() const override { return size_; }
  BufferUsage usage() const override { return usage_; }

  void write(std::span<const uint8_t> data, size_t offset) override {
    wgpuQueueWriteBuffer(queue_, buffer_, offset, data.data(), data.size());
  }

  std::vector<uint8_t> read() override {
    // TODO: Implement async buffer read
    return {};
  }

  WGPUBuffer handle() const { return buffer_; }
  void setQueue(WGPUQueue queue) { queue_ = queue; }

private:
  WGPUBufferUsage translateUsage(BufferUsage usage) {
    WGPUBufferUsage flags = WGPUBufferUsage_None;
    auto u = static_cast<uint32_t>(usage);
    if (u & static_cast<uint32_t>(BufferUsage::Vertex))
      flags |= WGPUBufferUsage_Vertex;
    if (u & static_cast<uint32_t>(BufferUsage::Index))
      flags |= WGPUBufferUsage_Index;
    if (u & static_cast<uint32_t>(BufferUsage::Uniform))
      flags |= WGPUBufferUsage_Uniform;
    if (u & static_cast<uint32_t>(BufferUsage::Storage))
      flags |= WGPUBufferUsage_Storage;
    if (u & static_cast<uint32_t>(BufferUsage::CopySrc))
      flags |= WGPUBufferUsage_CopySrc;
    if (u & static_cast<uint32_t>(BufferUsage::CopyDst))
      flags |= WGPUBufferUsage_CopyDst;
    return flags;
  }

  WGPUDevice device_;
  WGPUQueue queue_ = nullptr;
  WGPUBuffer buffer_ = nullptr;
  size_t size_;
  BufferUsage usage_;
};

// ============================================================================
// WebGPU Texture Implementation
// ============================================================================

class WebGPUTexture : public Texture {
public:
  WebGPUTexture(WGPUDevice device, uint32_t w, uint32_t h, TextureFormat fmt)
      : device_(device), width_(w), height_(h), format_(fmt) {

    WGPUTextureDescriptor desc = {};
    desc.size = {w, h, 1};
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.dimension = WGPUTextureDimension_2D;
    desc.format = translateFormat(fmt);
    desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst |
                 WGPUTextureUsage_RenderAttachment;

    texture_ = wgpuDeviceCreateTexture(device, &desc);

    WGPUTextureViewDescriptor viewDesc = {};
    view_ = wgpuTextureCreateView(texture_, &viewDesc);
  }

  ~WebGPUTexture() override {
    if (view_)
      wgpuTextureViewRelease(view_);
    if (texture_) {
      wgpuTextureDestroy(texture_);
      wgpuTextureRelease(texture_);
    }
  }

  uint32_t width() const override { return width_; }
  uint32_t height() const override { return height_; }
  TextureFormat format() const override { return format_; }

  void upload(std::span<const uint8_t> data) override {
    // TODO: Implement texture upload using Emscripten WebGPU API
    // WGPUImageCopyTexture may be renamed in newer Dawn/Emscripten versions
    (void)data;
  }

  std::vector<uint8_t> download() override {
    // TODO: Implement texture readback
    return {};
  }

  WGPUTexture handle() const { return texture_; }
  WGPUTextureView view() const { return view_; }
  void setQueue(WGPUQueue queue) { queue_ = queue; }

private:
  WGPUTextureFormat translateFormat(TextureFormat fmt) {
    switch (fmt) {
    case TextureFormat::RGBA8:
      return WGPUTextureFormat_RGBA8Unorm;
    case TextureFormat::RGBA16F:
      return WGPUTextureFormat_RGBA16Float;
    case TextureFormat::RGBA32F:
      return WGPUTextureFormat_RGBA32Float;
    case TextureFormat::R8:
      return WGPUTextureFormat_R8Unorm;
    case TextureFormat::R32F:
      return WGPUTextureFormat_R32Float;
    default:
      return WGPUTextureFormat_RGBA8Unorm;
    }
  }

  uint32_t bytesPerPixel() {
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

  WGPUDevice device_;
  WGPUQueue queue_ = nullptr;
  WGPUTexture texture_ = nullptr;
  WGPUTextureView view_ = nullptr;
  uint32_t width_, height_;
  TextureFormat format_;
};

// ============================================================================
// WebGPU Shader Implementation
// ============================================================================

class WebGPUShader : public Shader {
public:
  WebGPUShader(WGPUDevice device, const std::string &source, ShaderStage stage,
               const std::string &entryPoint)
      : stage_(stage), entryPoint_(entryPoint) {

    WGPUShaderSourceWGSL wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = {source.c_str(), WGPU_STRLEN};

    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = reinterpret_cast<WGPUChainedStruct *>(&wgslDesc);
    desc.label = {entryPoint.c_str(), WGPU_STRLEN};

    module_ = wgpuDeviceCreateShaderModule(device, &desc);
  }

  ~WebGPUShader() override {
    if (module_)
      wgpuShaderModuleRelease(module_);
  }

  ShaderStage stage() const override { return stage_; }
  WGPUShaderModule handle() const { return module_; }
  const std::string &entryPoint() const { return entryPoint_; }

private:
  WGPUShaderModule module_ = nullptr;
  ShaderStage stage_;
  std::string entryPoint_;
};

// ============================================================================
// WebGPU Pipeline Implementation
// ============================================================================

class WebGPUPipeline : public Pipeline {
public:
  WebGPUPipeline(WGPURenderPipeline renderPipeline)
      : renderPipeline_(renderPipeline), isCompute_(false) {}

  WebGPUPipeline(WGPUComputePipeline computePipeline)
      : computePipeline_(computePipeline), isCompute_(true) {}

  ~WebGPUPipeline() override {
    if (renderPipeline_)
      wgpuRenderPipelineRelease(renderPipeline_);
    if (computePipeline_)
      wgpuComputePipelineRelease(computePipeline_);
  }

  bool isCompute() const override { return isCompute_; }

  WGPURenderPipeline renderHandle() const { return renderPipeline_; }
  WGPUComputePipeline computeHandle() const { return computePipeline_; }

private:
  WGPURenderPipeline renderPipeline_ = nullptr;
  WGPUComputePipeline computePipeline_ = nullptr;
  bool isCompute_;
};

// ============================================================================
// WebGPU Command Buffer Implementation
// ============================================================================

class WebGPUCommandBuffer : public CommandBuffer {
public:
  WebGPUCommandBuffer(WGPUDevice device) : device_(device) {}

  ~WebGPUCommandBuffer() override {
    if (encoder_)
      wgpuCommandEncoderRelease(encoder_);
  }

  void begin() override {
    WGPUCommandEncoderDescriptor desc = {};
    encoder_ = wgpuDeviceCreateCommandEncoder(device_, &desc);
  }

  void end() override {
    // Commands are finalized in submit()
  }

  void setComputePipeline(Pipeline *pipeline) override {
    currentComputePipeline_ = static_cast<WebGPUPipeline *>(pipeline);
  }

  void setRenderPipeline(Pipeline *pipeline) override {
    currentRenderPipeline_ = static_cast<WebGPUPipeline *>(pipeline);
  }

  void bindBuffer(uint32_t binding, Buffer *buffer) override {
    // Store for bind group creation
    boundBuffers_[binding] = static_cast<WebGPUBuffer *>(buffer);
  }

  void bindTexture(uint32_t binding, Texture *texture) override {
    boundTextures_[binding] = static_cast<WebGPUTexture *>(texture);
  }

  void dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) override {
    if (!encoder_ || !currentComputePipeline_)
      return;

    WGPUComputePassDescriptor passDesc = {};
    WGPUComputePassEncoder pass =
        wgpuCommandEncoderBeginComputePass(encoder_, &passDesc);

    wgpuComputePassEncoderSetPipeline(pass,
                                      currentComputePipeline_->computeHandle());
    // TODO: Set bind groups
    wgpuComputePassEncoderDispatchWorkgroups(pass, groupsX, groupsY, groupsZ);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
  }

  void draw(uint32_t vertexCount, uint32_t instanceCount) override {
    // TODO: Implement render pass
  }

  void copyTextureToBuffer(Texture *src, Buffer *dst) override {
    // TODO: Implement
  }

  WGPUCommandEncoder encoder() const { return encoder_; }

  WGPUCommandBuffer finish() {
    WGPUCommandBufferDescriptor desc = {};
    return wgpuCommandEncoderFinish(encoder_, &desc);
  }

private:
  WGPUDevice device_;
  WGPUCommandEncoder encoder_ = nullptr;
  WebGPUPipeline *currentComputePipeline_ = nullptr;
  WebGPUPipeline *currentRenderPipeline_ = nullptr;
  std::unordered_map<uint32_t, WebGPUBuffer *> boundBuffers_;
  std::unordered_map<uint32_t, WebGPUTexture *> boundTextures_;
};

// ============================================================================
// WebGPU Backend Implementation
// ============================================================================

struct WebGPUBackend::Impl {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  bool ready = false;
};

WebGPUBackend::WebGPUBackend() : pimpl_(std::make_unique<Impl>()) {
  std::cerr << "WebGPU backend stubbed for debugging." << std::endl;
}

WebGPUBackend::~WebGPUBackend() {
  if (pimpl_->queue)
    wgpuQueueRelease(pimpl_->queue);
  if (pimpl_->device)
    wgpuDeviceRelease(pimpl_->device);
  if (pimpl_->adapter)
    wgpuAdapterRelease(pimpl_->adapter);
  if (pimpl_->instance)
    wgpuInstanceRelease(pimpl_->instance);
}

std::string WebGPUBackend::name() const {
#ifdef __EMSCRIPTEN__
  return "WebGPU (Browser)";
#else
  return "WebGPU (Dawn)";
#endif
}

bool WebGPUBackend::isReady() const { return pimpl_->ready; }

GPUResult<std::unique_ptr<Buffer>>
WebGPUBackend::createBuffer(size_t size, BufferUsage usage) {
  return unexpected(GPUError{"Debug build"});
}

GPUResult<std::unique_ptr<Texture>>
WebGPUBackend::createTexture(uint32_t width, uint32_t height,
                             TextureFormat format) {
  return unexpected(GPUError{"Debug build"});
}

GPUResult<std::unique_ptr<Shader>>
WebGPUBackend::createShaderWGSL(const std::string &source, ShaderStage stage,
                                const std::string &entryPoint) {
  return unexpected(GPUError{"Debug build"});
}

GPUResult<std::unique_ptr<Shader>>
WebGPUBackend::createShaderSPIRV(std::span<const uint32_t> spirv,
                                 ShaderStage stage,
                                 const std::string &entryPoint) {
  return unexpected(GPUError{"Debug build"});
}

GPUResult<std::unique_ptr<Pipeline>>
WebGPUBackend::createComputePipeline(Shader *computeShader) {
  return unexpected(GPUError{"Debug build"});
}

GPUResult<std::unique_ptr<Pipeline>> WebGPUBackend::createRenderPipeline(
    Shader *vertexShader, Shader *fragmentShader, TextureFormat outputFormat) {
  return unexpected(GPUError{"Debug build"});
}

GPUResult<std::unique_ptr<CommandBuffer>> WebGPUBackend::createCommandBuffer() {
  return unexpected(GPUError{"Debug build"});
}

void WebGPUBackend::submit(CommandBuffer *cmd) {
  auto *webgpuCmd = static_cast<WebGPUCommandBuffer *>(cmd);
  WGPUCommandBuffer commandBuffer = webgpuCmd->finish();
  wgpuQueueSubmit(pimpl_->queue, 1, &commandBuffer);
  wgpuCommandBufferRelease(commandBuffer);
}

void WebGPUBackend::waitIdle() {
  // WebGPU doesn't have explicit wait, work is submitted asynchronously
#ifndef __EMSCRIPTEN__
  // Dawn might have device tick
#endif
}

// ============================================================================
// Backend Factory
// ============================================================================

std::unique_ptr<Backend> Backend::create(BackendType preferred) {
  // For now, only WebGPU is implemented
  return std::make_unique<WebGPUBackend>();
}

} // namespace GPU
} // namespace CaptionEngine
