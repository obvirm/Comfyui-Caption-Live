/**
 * @file dx12_backend.cpp
 * @brief DirectX 12 GPU Backend Implementation
 *
 * Compute-focused backend for Windows 10+ native GPU acceleration.
 */

#ifdef _WIN32

#include "gpu/dx12_backend.hpp"
#include "platform/platform.hpp"

#include <d3dcompiler.h>
#include <stdexcept>
#include <string>

#pragma comment(lib, "d3dcompiler.lib")

namespace CaptionEngine {
namespace GPU {

// =============================================================================
// Helper Functions
// =============================================================================

static DXGI_FORMAT toDXGIFormat(TextureFormat format) {
  switch (format) {
  case TextureFormat::RGBA8:
    return DXGI_FORMAT_R8G8B8A8_UNORM;
  case TextureFormat::RGBA16F:
    return DXGI_FORMAT_R16G16B16A16_FLOAT;
  case TextureFormat::RGBA32F:
    return DXGI_FORMAT_R32G32B32A32_FLOAT;
  case TextureFormat::R8:
    return DXGI_FORMAT_R8_UNORM;
  case TextureFormat::R32F:
    return DXGI_FORMAT_R32_FLOAT;
  default:
    return DXGI_FORMAT_R8G8B8A8_UNORM;
  }
}

static uint32_t bytesPerPixel(TextureFormat format) {
  switch (format) {
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

static const char *shaderTarget(ShaderStage stage) {
  switch (stage) {
  case ShaderStage::Vertex:
    return "vs_5_1";
  case ShaderStage::Fragment:
    return "ps_5_1";
  case ShaderStage::Compute:
    return "cs_5_1";
  default:
    return "cs_5_1";
  }
}

// =============================================================================
// DX12Buffer Implementation
// =============================================================================

DX12Buffer::DX12Buffer(ComPtr<ID3D12Resource> resource, size_t size,
                       BufferUsage usage)
    : resource_(std::move(resource)), size_(size), usage_(usage) {
  // Map for CPU access if needed
  D3D12_RANGE readRange = {0, 0};
  if (SUCCEEDED(resource_->Map(0, &readRange, &mapped_ptr_))) {
    // Mapped successfully
  }
}

DX12Buffer::~DX12Buffer() {
  if (mapped_ptr_) {
    resource_->Unmap(0, nullptr);
  }
}

void DX12Buffer::write(std::span<const uint8_t> data, size_t offset) {
  if (mapped_ptr_ && offset + data.size() <= size_) {
    memcpy(static_cast<uint8_t *>(mapped_ptr_) + offset, data.data(),
           data.size());
  }
}

std::vector<uint8_t> DX12Buffer::read() {
  std::vector<uint8_t> result(size_);
  if (mapped_ptr_) {
    memcpy(result.data(), mapped_ptr_, size_);
  }
  return result;
}

// =============================================================================
// DX12Texture Implementation
// =============================================================================

DX12Texture::DX12Texture(ComPtr<ID3D12Resource> resource, uint32_t width,
                         uint32_t height, TextureFormat format)
    : resource_(std::move(resource)), width_(width), height_(height),
      format_(format) {}

DX12Texture::~DX12Texture() = default;

void DX12Texture::upload(std::span<const uint8_t> data) {
  // TODO: Implement upload via staging buffer
  // For now, this is a placeholder
}

std::vector<uint8_t> DX12Texture::download() {
  // TODO: Implement download via readback buffer
  return {};
}

// =============================================================================
// DX12Shader Implementation
// =============================================================================

DX12Shader::DX12Shader(ComPtr<ID3DBlob> bytecode, ShaderStage stage)
    : bytecode_(std::move(bytecode)), stage_(stage) {}

DX12Shader::~DX12Shader() = default;

// =============================================================================
// DX12Pipeline Implementation
// =============================================================================

DX12Pipeline::DX12Pipeline(ComPtr<ID3D12PipelineState> pso,
                           ComPtr<ID3D12RootSignature> rootSig, bool isCompute)
    : pso_(std::move(pso)), rootSig_(std::move(rootSig)),
      isCompute_(isCompute) {}

DX12Pipeline::~DX12Pipeline() = default;

// =============================================================================
// DX12CommandBuffer Implementation
// =============================================================================

DX12CommandBuffer::DX12CommandBuffer(ComPtr<ID3D12GraphicsCommandList> cmdList,
                                     ComPtr<ID3D12CommandAllocator> allocator,
                                     ID3D12Device *device)
    : cmdList_(std::move(cmdList)), allocator_(std::move(allocator)),
      device_(device) {}

DX12CommandBuffer::~DX12CommandBuffer() = default;

void DX12CommandBuffer::begin() {
  allocator_->Reset();
  cmdList_->Reset(allocator_.Get(), nullptr);
}

void DX12CommandBuffer::end() { cmdList_->Close(); }

void DX12CommandBuffer::setComputePipeline(Pipeline *pipeline) {
  auto *dx12Pipeline = static_cast<DX12Pipeline *>(pipeline);
  currentPipeline_ = dx12Pipeline;
  cmdList_->SetComputeRootSignature(dx12Pipeline->rootSignature());
  cmdList_->SetPipelineState(dx12Pipeline->pso());
}

void DX12CommandBuffer::setRenderPipeline(Pipeline *pipeline) {
  auto *dx12Pipeline = static_cast<DX12Pipeline *>(pipeline);
  currentPipeline_ = dx12Pipeline;
  cmdList_->SetGraphicsRootSignature(dx12Pipeline->rootSignature());
  cmdList_->SetPipelineState(dx12Pipeline->pso());
}

void DX12CommandBuffer::bindBuffer(uint32_t binding, Buffer *buffer) {
  auto *dx12Buffer = static_cast<DX12Buffer *>(buffer);
  if (currentPipeline_ && currentPipeline_->isCompute()) {
    cmdList_->SetComputeRootUnorderedAccessView(binding,
                                                dx12Buffer->gpuAddress());
  } else {
    cmdList_->SetGraphicsRootShaderResourceView(binding,
                                                dx12Buffer->gpuAddress());
  }
}

void DX12CommandBuffer::bindTexture(uint32_t binding, Texture *texture) {
  // TODO: Implement texture binding via descriptor heap
}

void DX12CommandBuffer::dispatch(uint32_t groupsX, uint32_t groupsY,
                                 uint32_t groupsZ) {
  cmdList_->Dispatch(groupsX, groupsY, groupsZ);
}

void DX12CommandBuffer::draw(uint32_t vertexCount, uint32_t instanceCount) {
  cmdList_->DrawInstanced(vertexCount, instanceCount, 0, 0);
}

void DX12CommandBuffer::copyTextureToBuffer(Texture *src, Buffer *dst) {
  // TODO: Implement texture to buffer copy
}

void DX12CommandBuffer::reset() {
  allocator_->Reset();
  cmdList_->Reset(allocator_.Get(), nullptr);
}

// =============================================================================
// DX12Backend Implementation
// =============================================================================

struct DX12Backend::Impl {
  ComPtr<IDXGIFactory6> factory;
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> commandQueue;
  ComPtr<ID3D12Fence> fence;
  UINT64 fenceValue = 0;
  HANDLE fenceEvent = nullptr;

  std::string adapterName;
  bool ready = false;

  // Descriptor heaps
  ComPtr<ID3D12DescriptorHeap> srvUavHeap;
  UINT srvUavDescriptorSize = 0;

  ~Impl() {
    if (fenceEvent) {
      CloseHandle(fenceEvent);
    }
  }
};

DX12Backend::DX12Backend() : pimpl_(std::make_unique<Impl>()) {
  pimpl_->ready = initialize();
}

DX12Backend::~DX12Backend() { waitIdle(); }

bool DX12Backend::initialize() {
  HRESULT hr;

  // Enable debug layer in debug builds
#if defined(_DEBUG) || defined(CE_DEBUG)
  ComPtr<ID3D12Debug> debugController;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
    debugController->EnableDebugLayer();
  }
#endif

  // Create DXGI factory
  UINT factoryFlags = 0;
#if defined(_DEBUG) || defined(CE_DEBUG)
  factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

  hr = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&pimpl_->factory));
  if (FAILED(hr)) {
    return false;
  }

  // Find best adapter
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0; pimpl_->factory->EnumAdapterByGpuPreference(
                       i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                       IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    // Skip software adapters
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      continue;
    }

    // Check if adapter supports D3D12
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                    __uuidof(ID3D12Device), nullptr))) {
      pimpl_->adapter = adapter;

      // Convert adapter name to string
      char name[128];
      size_t converted;
      wcstombs_s(&converted, name, sizeof(name), desc.Description, _TRUNCATE);
      pimpl_->adapterName = name;
      break;
    }
  }

  if (!pimpl_->adapter) {
    return false;
  }

  // Create device
  hr = D3D12CreateDevice(pimpl_->adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                         IID_PPV_ARGS(&pimpl_->device));
  if (FAILED(hr)) {
    return false;
  }

  // Create command queue
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

  hr = pimpl_->device->CreateCommandQueue(&queueDesc,
                                          IID_PPV_ARGS(&pimpl_->commandQueue));
  if (FAILED(hr)) {
    return false;
  }

  // Create fence
  hr = pimpl_->device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                   IID_PPV_ARGS(&pimpl_->fence));
  if (FAILED(hr)) {
    return false;
  }

  pimpl_->fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!pimpl_->fenceEvent) {
    return false;
  }

  // Create SRV/UAV descriptor heap
  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.NumDescriptors = 1024;
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

  hr = pimpl_->device->CreateDescriptorHeap(&heapDesc,
                                            IID_PPV_ARGS(&pimpl_->srvUavHeap));
  if (FAILED(hr)) {
    return false;
  }

  pimpl_->srvUavDescriptorSize =
      pimpl_->device->GetDescriptorHandleIncrementSize(
          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  return true;
}

std::string DX12Backend::name() const {
  if (pimpl_->ready) {
    return "DirectX 12 - " + pimpl_->adapterName;
  }
  return "DirectX 12 (not initialized)";
}

bool DX12Backend::isReady() const { return pimpl_->ready; }

ID3D12Device *DX12Backend::device() const { return pimpl_->device.Get(); }

ID3D12CommandQueue *DX12Backend::commandQueue() const {
  return pimpl_->commandQueue.Get();
}

IDXGIAdapter1 *DX12Backend::adapter() const { return pimpl_->adapter.Get(); }

bool DX12Backend::supportsShaderModel6() const {
  if (!pimpl_->device)
    return false;

  D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_0};
  return SUCCEEDED(pimpl_->device->CheckFeatureSupport(
             D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))) &&
         shaderModel.HighestShaderModel >= D3D_SHADER_MODEL_6_0;
}

bool DX12Backend::supportsRaytracing() const {
  if (!pimpl_->device)
    return false;

  D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5 = {};
  return SUCCEEDED(pimpl_->device->CheckFeatureSupport(
             D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5))) &&
         opts5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
}

// Resource Creation

GPUResult<std::unique_ptr<Buffer>>
DX12Backend::createBuffer(size_t size, BufferUsage usage) {
  if (!pimpl_->ready) {
    return unexpected(GPUError{"DX12 not initialized", -1});
  }

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_UPLOAD; // CPU accessible

  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Width = size;
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  resourceDesc.SampleDesc.Count = 1;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  // Set flags based on usage
  if (static_cast<uint32_t>(usage) &
      static_cast<uint32_t>(BufferUsage::Storage)) {
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    // UAV requires default heap
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
  }

  ComPtr<ID3D12Resource> resource;
  HRESULT hr = pimpl_->device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource));

  if (FAILED(hr)) {
    return unexpected(
        GPUError{"Failed to create DX12 buffer", static_cast<int>(hr)});
  }

  return std::make_unique<DX12Buffer>(std::move(resource), size, usage);
}

GPUResult<std::unique_ptr<Texture>>
DX12Backend::createTexture(uint32_t width, uint32_t height,
                           TextureFormat format) {
  if (!pimpl_->ready) {
    return unexpected(GPUError{"DX12 not initialized", -1});
  }

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  resourceDesc.Width = width;
  resourceDesc.Height = height;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.Format = toDXGIFormat(format);
  resourceDesc.SampleDesc.Count = 1;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  ComPtr<ID3D12Resource> resource;
  HRESULT hr = pimpl_->device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource));

  if (FAILED(hr)) {
    return unexpected(
        GPUError{"Failed to create DX12 texture", static_cast<int>(hr)});
  }

  return std::make_unique<DX12Texture>(std::move(resource), width, height,
                                       format);
}

GPUResult<std::unique_ptr<Shader>>
DX12Backend::createShaderWGSL(const std::string &source, ShaderStage stage,
                              const std::string &entryPoint) {
  // WGSL is not directly supported by DX12
  // Would need to cross-compile WGSL -> HLSL -> DXIL
  return unexpected(
      GPUError{"WGSL shaders not supported in DX12 backend. Use HLSL.", -1});
}

GPUResult<std::unique_ptr<Shader>>
DX12Backend::createShaderSPIRV(std::span<const uint32_t> spirv,
                               ShaderStage stage,
                               const std::string &entryPoint) {
  // SPIR-V is not directly supported by DX12
  // Would need SPIRV-Cross to convert to HLSL
  return unexpected(
      GPUError{"SPIR-V shaders not supported in DX12 backend. Use HLSL.", -1});
}

GPUResult<std::unique_ptr<Shader>>
DX12Backend::createShaderHLSL(const std::string &source, ShaderStage stage,
                              const std::string &entryPoint) {
  if (!pimpl_->ready) {
    return unexpected(GPUError{"DX12 not initialized", -1});
  }

  ComPtr<ID3DBlob> shaderBlob;
  ComPtr<ID3DBlob> errorBlob;

  UINT compileFlags = 0;
#if defined(_DEBUG) || defined(CE_DEBUG)
  compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  compileFlags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

  HRESULT hr = D3DCompile(source.c_str(), source.size(), nullptr, nullptr,
                          nullptr, entryPoint.c_str(), shaderTarget(stage),
                          compileFlags, 0, &shaderBlob, &errorBlob);

  if (FAILED(hr)) {
    std::string errorMsg = "HLSL compilation failed";
    if (errorBlob) {
      errorMsg +=
          ": " + std::string(static_cast<char *>(errorBlob->GetBufferPointer()),
                             errorBlob->GetBufferSize());
    }
    return unexpected(GPUError{errorMsg, static_cast<int>(hr)});
  }

  return std::make_unique<DX12Shader>(std::move(shaderBlob), stage);
}

ComPtr<ID3D12RootSignature> DX12Backend::createComputeRootSignature() {
  // Create a simple root signature for compute shaders
  // 4 UAV slots (u0-u3) and 1 CBV slot (b0)
  D3D12_ROOT_PARAMETER params[5] = {};

  // UAV slots
  for (int i = 0; i < 4; ++i) {
    params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[i].Descriptor.ShaderRegister = i;
    params[i].Descriptor.RegisterSpace = 0;
    params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  }

  // CBV slot
  params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
  params[4].Descriptor.ShaderRegister = 0;
  params[4].Descriptor.RegisterSpace = 0;
  params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
  rootSigDesc.NumParameters = 5;
  rootSigDesc.pParameters = params;
  rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ComPtr<ID3DBlob> signature;
  ComPtr<ID3DBlob> error;
  HRESULT hr = D3D12SerializeRootSignature(
      &rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
  if (FAILED(hr)) {
    return nullptr;
  }

  ComPtr<ID3D12RootSignature> rootSig;
  hr = pimpl_->device->CreateRootSignature(0, signature->GetBufferPointer(),
                                           signature->GetBufferSize(),
                                           IID_PPV_ARGS(&rootSig));
  if (FAILED(hr)) {
    return nullptr;
  }

  return rootSig;
}

GPUResult<std::unique_ptr<Pipeline>>
DX12Backend::createComputePipeline(Shader *computeShader) {
  if (!pimpl_->ready) {
    return unexpected(GPUError{"DX12 not initialized", -1});
  }

  auto *dx12Shader = static_cast<DX12Shader *>(computeShader);
  if (!dx12Shader || dx12Shader->stage() != ShaderStage::Compute) {
    return unexpected(GPUError{"Invalid compute shader", -1});
  }

  auto rootSig = createComputeRootSignature();
  if (!rootSig) {
    return unexpected(GPUError{"Failed to create root signature", -1});
  }

  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
  psoDesc.pRootSignature = rootSig.Get();
  psoDesc.CS = dx12Shader->shaderBytecode();

  ComPtr<ID3D12PipelineState> pso;
  HRESULT hr =
      pimpl_->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));

  if (FAILED(hr)) {
    return unexpected(
        GPUError{"Failed to create compute pipeline", static_cast<int>(hr)});
  }

  return std::make_unique<DX12Pipeline>(std::move(pso), std::move(rootSig),
                                        true);
}

GPUResult<std::unique_ptr<Pipeline>>
DX12Backend::createRenderPipeline(Shader *vertexShader, Shader *fragmentShader,
                                  TextureFormat outputFormat) {
  // Render pipeline would require more setup (input layout, blend state, etc.)
  // For now, focus on compute
  return unexpected(
      GPUError{"Render pipeline not yet implemented in DX12 backend", -1});
}

GPUResult<std::unique_ptr<CommandBuffer>> DX12Backend::createCommandBuffer() {
  if (!pimpl_->ready) {
    return unexpected(GPUError{"DX12 not initialized", -1});
  }

  ComPtr<ID3D12CommandAllocator> allocator;
  HRESULT hr = pimpl_->device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));

  if (FAILED(hr)) {
    return unexpected(
        GPUError{"Failed to create command allocator", static_cast<int>(hr)});
  }

  ComPtr<ID3D12GraphicsCommandList> cmdList;
  hr = pimpl_->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&cmdList));

  if (FAILED(hr)) {
    return unexpected(
        GPUError{"Failed to create command list", static_cast<int>(hr)});
  }

  cmdList->Close(); // Start closed, will be reset on begin()

  return std::make_unique<DX12CommandBuffer>(
      std::move(cmdList), std::move(allocator), pimpl_->device.Get());
}

void DX12Backend::submit(CommandBuffer *cmd) {
  if (!pimpl_->ready || !cmd)
    return;

  auto *dx12Cmd = static_cast<DX12CommandBuffer *>(cmd);
  ID3D12CommandList *cmdLists[] = {dx12Cmd->commandList()};
  pimpl_->commandQueue->ExecuteCommandLists(1, cmdLists);
}

void DX12Backend::waitIdle() {
  if (!pimpl_->ready)
    return;

  const UINT64 fenceVal = ++pimpl_->fenceValue;
  pimpl_->commandQueue->Signal(pimpl_->fence.Get(), fenceVal);

  if (pimpl_->fence->GetCompletedValue() < fenceVal) {
    pimpl_->fence->SetEventOnCompletion(fenceVal, pimpl_->fenceEvent);
    WaitForSingleObject(pimpl_->fenceEvent, INFINITE);
  }
}

} // namespace GPU
} // namespace CaptionEngine

#endif // _WIN32
