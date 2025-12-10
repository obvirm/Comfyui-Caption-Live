#pragma once
/**
 * @file webgpu_backend.hpp
 * @brief WebGPU backend implementation (Browser + Dawn)
 */

#include "gpu/backend.hpp"

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#else
// Dawn (native WebGPU)
#include <dawn/webgpu.h>
#endif

namespace CaptionEngine {
namespace GPU {

/**
 * @brief WebGPU implementation of GPU Backend
 *
 * Works in:
 * - Browser (via Emscripten WebGPU)
 * - Native (via Google Dawn library)
 */
class WebGPUBackend : public Backend {
public:
  WebGPUBackend();
  ~WebGPUBackend() override;

  // Backend interface
  BackendType type() const override { return BackendType::WebGPU; }
  std::string name() const override;
  bool isReady() const override;

  // Resource creation
  GPUResult<std::unique_ptr<Buffer>> createBuffer(size_t size,
                                                  BufferUsage usage) override;

  GPUResult<std::unique_ptr<Texture>>
  createTexture(uint32_t width, uint32_t height, TextureFormat format) override;

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

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace GPU
} // namespace CaptionEngine
