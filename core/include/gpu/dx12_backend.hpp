#pragma once
/**
 * @file dx12_backend.hpp
 * @brief DirectX 12 GPU Backend for Windows Native
 *
 * Provides compute shader acceleration on Windows 10+ without requiring
 * third-party APIs like Vulkan or CUDA.
 *
 * Features:
 * - DirectX 12 compute pipeline
 * - UAV (Unordered Access View) for read/write buffers
 * - HLSL shader compilation via D3DCompile
 * - Fence-based synchronization
 */

#ifdef _WIN32

// Windows headers MUST come first
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>


// Now include our headers
#include "gpu/backend.hpp"

// Link libraries
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

namespace CaptionEngine {
namespace GPU {

using Microsoft::WRL::ComPtr;

/**
 * @brief DirectX 12 Buffer implementation
 */
class DX12Buffer : public Buffer {
public:
  DX12Buffer(ComPtr<ID3D12Resource> resource, size_t size, BufferUsage usage);
  ~DX12Buffer() override;

  size_t size() const override { return size_; }
  BufferUsage usage() const override { return usage_; }

  void write(std::span<const uint8_t> data, size_t offset = 0) override;
  std::vector<uint8_t> read() override;

  ID3D12Resource *resource() const { return resource_.Get(); }
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddress() const {
    return resource_->GetGPUVirtualAddress();
  }

private:
  ComPtr<ID3D12Resource> resource_;
  size_t size_;
  BufferUsage usage_;
  void *mapped_ptr_ = nullptr;
};

/**
 * @brief DirectX 12 Texture implementation
 */
class DX12Texture : public Texture {
public:
  DX12Texture(ComPtr<ID3D12Resource> resource, uint32_t width, uint32_t height,
              TextureFormat format);
  ~DX12Texture() override;

  uint32_t width() const override { return width_; }
  uint32_t height() const override { return height_; }
  TextureFormat format() const override { return format_; }

  void upload(std::span<const uint8_t> data) override;
  std::vector<uint8_t> download() override;

  ID3D12Resource *resource() const { return resource_.Get(); }

private:
  ComPtr<ID3D12Resource> resource_;
  uint32_t width_;
  uint32_t height_;
  TextureFormat format_;
};

/**
 * @brief DirectX 12 Shader (compiled DXIL)
 */
class DX12Shader : public Shader {
public:
  DX12Shader(ComPtr<ID3DBlob> bytecode, ShaderStage stage);
  ~DX12Shader() override;

  ShaderStage stage() const override { return stage_; }

  ID3DBlob *bytecode() const { return bytecode_.Get(); }
  D3D12_SHADER_BYTECODE shaderBytecode() const {
    return {bytecode_->GetBufferPointer(), bytecode_->GetBufferSize()};
  }

private:
  ComPtr<ID3DBlob> bytecode_;
  ShaderStage stage_;
};

/**
 * @brief DirectX 12 Compute Pipeline (PSO)
 */
class DX12Pipeline : public Pipeline {
public:
  DX12Pipeline(ComPtr<ID3D12PipelineState> pso,
               ComPtr<ID3D12RootSignature> rootSig, bool isCompute);
  ~DX12Pipeline() override;

  bool isCompute() const override { return isCompute_; }

  ID3D12PipelineState *pso() const { return pso_.Get(); }
  ID3D12RootSignature *rootSignature() const { return rootSig_.Get(); }

private:
  ComPtr<ID3D12PipelineState> pso_;
  ComPtr<ID3D12RootSignature> rootSig_;
  bool isCompute_;
};

/**
 * @brief DirectX 12 Command Buffer (Command List)
 */
class DX12CommandBuffer : public CommandBuffer {
public:
  DX12CommandBuffer(ComPtr<ID3D12GraphicsCommandList> cmdList,
                    ComPtr<ID3D12CommandAllocator> allocator,
                    ID3D12Device *device);
  ~DX12CommandBuffer() override;

  void begin() override;
  void end() override;

  void setComputePipeline(Pipeline *pipeline) override;
  void setRenderPipeline(Pipeline *pipeline) override;
  void bindBuffer(uint32_t binding, Buffer *buffer) override;
  void bindTexture(uint32_t binding, Texture *texture) override;
  void dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) override;
  void draw(uint32_t vertexCount, uint32_t instanceCount = 1) override;
  void copyTextureToBuffer(Texture *src, Buffer *dst) override;

  ID3D12GraphicsCommandList *commandList() const { return cmdList_.Get(); }
  ID3D12CommandAllocator *allocator() const { return allocator_.Get(); }

  void reset();

private:
  ComPtr<ID3D12GraphicsCommandList> cmdList_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ID3D12Device *device_;
  DX12Pipeline *currentPipeline_ = nullptr;
};

/**
 * @brief DirectX 12 GPU Backend
 *
 * Compute-focused backend for Windows 10+.
 * Uses DirectX 12 for GPU compute operations.
 */
class DX12Backend : public Backend {
public:
  DX12Backend();
  ~DX12Backend() override;

  // Backend interface
  BackendType type() const override { return BackendType::DirectX12; }
  std::string name() const override;
  bool isReady() const override;

  // Resource creation
  GPUResult<std::unique_ptr<Buffer>> createBuffer(size_t size,
                                                  BufferUsage usage) override;

  GPUResult<std::unique_ptr<Texture>>
  createTexture(uint32_t width, uint32_t height, TextureFormat format) override;

  GPUResult<std::unique_ptr<Shader>>
  createShaderWGSL(const std::string &source, ShaderStage stage,
                   const std::string &entryPoint = "main") override;

  GPUResult<std::unique_ptr<Shader>>
  createShaderSPIRV(std::span<const uint32_t> spirv, ShaderStage stage,
                    const std::string &entryPoint = "main") override;

  /// Create shader from HLSL source (DX12-specific)
  GPUResult<std::unique_ptr<Shader>>
  createShaderHLSL(const std::string &source, ShaderStage stage,
                   const std::string &entryPoint = "main");

  GPUResult<std::unique_ptr<Pipeline>>
  createComputePipeline(Shader *computeShader) override;

  GPUResult<std::unique_ptr<Pipeline>> createRenderPipeline(
      Shader *vertexShader, Shader *fragmentShader,
      TextureFormat outputFormat = TextureFormat::RGBA8) override;

  // Command execution
  GPUResult<std::unique_ptr<CommandBuffer>> createCommandBuffer() override;
  void submit(CommandBuffer *cmd) override;
  void waitIdle() override;

  // DX12-specific accessors
  ID3D12Device *device() const;
  ID3D12CommandQueue *commandQueue() const;
  IDXGIAdapter1 *adapter() const;

  // Feature support
  bool supportsShaderModel6() const;
  bool supportsRaytracing() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  bool initialize();
  ComPtr<ID3D12RootSignature> createComputeRootSignature();
};

} // namespace GPU
} // namespace CaptionEngine

#endif // _WIN32
