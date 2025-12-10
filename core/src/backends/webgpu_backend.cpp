/**
 * @file webgpu_backend.cpp
 * @brief WebGPU compute backend implementation (Emscripten via emdawnwebgpu)
 */

#include "graphics/backend.hpp"
#include <iostream>
#include <unordered_map>


#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <iostream>
#include <webgpu/webgpu.h>     // C API
#include <webgpu/webgpu_cpp.h> // C++ Wrapper

namespace CaptionEngine {

struct WebGPUBackend::Impl {
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;

  bool ready = false;

  // Registry
  std::unordered_map<std::string, wgpu::ComputePipeline> pipelines;
};

WebGPUBackend::WebGPUBackend() : pimpl_(std::make_unique<Impl>()) {}

WebGPUBackend::~WebGPUBackend() = default;

// Embedded Basic Shader for MVP
const char *BASIC_WGSL = R"(
struct FrameUniforms {
    width: u32,
    height: u32,
    time: f32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= uniforms.width || y >= uniforms.height) { return; }
    
    let u = f32(x) / f32(uniforms.width);
    let v = f32(y) / f32(uniforms.height);
    
    let r = u32(u * 255.0);
    let g = u32(v * 255.0);
    let b = u32((sin(uniforms.time) * 0.5 + 0.5) * 255.0);
    let a = 255u;
    
    let color = r | (g << 8) | (b << 16) | (a << 24);
    outputBuffer[y * uniforms.width + x] = color;
}
)";

// Callback logic (internal)
void WebGPUBackend::on_device_ready(void *device_handle, void * /*userdata*/) {
  if (!device_handle)
    return;

  WGPUDevice c_device = static_cast<WGPUDevice>(device_handle);
  pimpl_->device = wgpu::Device::Acquire(c_device);
  pimpl_->queue = pimpl_->device.GetQueue();
  pimpl_->ready = true;
  std::cout << "✅ WebGPU Device acquired!" << std::endl;
}

void WebGPUBackend::on_adapter_ready(void *adapter_handle,
                                     void * /*userdata*/) {
  if (!adapter_handle)
    return;

  WGPUAdapter c_adapter = static_cast<WGPUAdapter>(adapter_handle);
  pimpl_->adapter = wgpu::Adapter::Acquire(c_adapter);

  wgpu::DeviceDescriptor deviceDesc = {};

  auto callback = [](wgpu::RequestDeviceStatus status, wgpu::Device device,
                     char const *message, WebGPUBackend *backend) {
    if (status == wgpu::RequestDeviceStatus::Success) {
      // Transfer ownership of 'device' handle to on_device_ready
      // MoveToCHandle releases strict ownership from wrapper but returns
      // handle.
      backend->on_device_ready(device.MoveToCHandle(), nullptr);
    } else {
      std::cerr << "❌ WebGPU RequestDevice failed: "
                << (message ? message : "Unknown") << std::endl;
    }
  };

  pimpl_->adapter.RequestDevice(
      &deviceDesc, wgpu::CallbackMode::AllowSpontaneous, callback, this);
}

bool WebGPUBackend::initialize() {
  wgpu::InstanceDescriptor desc = {};
  pimpl_->instance = wgpu::CreateInstance(&desc);

  if (!pimpl_->instance) {
    std::cerr << "❌ Failed to create WebGPU instance" << std::endl;
    return false;
  }

  wgpu::RequestAdapterOptions options = {};
  options.powerPreference = wgpu::PowerPreference::HighPerformance;

  auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                     char const *message, WebGPUBackend *backend) {
    if (status == wgpu::RequestAdapterStatus::Success) {
      backend->on_adapter_ready(adapter.MoveToCHandle(), nullptr);
    } else {
      std::cout << "WebGPU Adapter request failed." << std::endl;
    }
  };

  pimpl_->instance.RequestAdapter(
      &options, wgpu::CallbackMode::AllowSpontaneous, callback, this);

  return true;
}

BufferHandle WebGPUBackend::create_buffer(size_t size, MemoryType type) {
  if (!pimpl_->ready)
    return 0;

  wgpu::BufferDescriptor desc = {};
  desc.size = size;

  if (type == MemoryType::DeviceLocal) {
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
                 wgpu::BufferUsage::CopySrc;
  } else if (type == MemoryType::HostVisible) {
    desc.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
  } else if (type == MemoryType::HostCached) {
    desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  }

  wgpu::Buffer buffer = pimpl_->device.CreateBuffer(&desc);
  // Return handle, but don't release.
  // MoveToCHandle prevents destructor from releasing.
  return reinterpret_cast<BufferHandle>(buffer.MoveToCHandle());
}

void WebGPUBackend::destroy_buffer(BufferHandle handle) {
  if (!handle)
    return;
  WGPUBuffer c_buffer = reinterpret_cast<WGPUBuffer>(handle);
  wgpuBufferRelease(c_buffer);
}

void WebGPUBackend::upload_buffer(BufferHandle handle,
                                  std::span<const uint8_t> data) {
  if (!pimpl_->ready)
    return;
  WGPUBuffer c_buffer = reinterpret_cast<WGPUBuffer>(handle);
  // Acquire -> Write -> Detach
  // Assuming wrapper WriteBuffer uses c_buffer.
  // If we use C++ wrapper for WriteBuffer? No, Queue::WriteBuffer takes
  // wgpu::Buffer.
  wgpu::Buffer buffer = wgpu::Buffer::Acquire(c_buffer);
  pimpl_->queue.WriteBuffer(buffer, 0, data.data(), data.size());
  buffer.MoveToCHandle(); // Detach to prevent double release
}

std::vector<uint8_t> WebGPUBackend::download_buffer(BufferHandle, size_t) {
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

// In Impl:
// std::unordered_map<std::string, wgpu::ComputePipeline> pipelines;

void WebGPUBackend::dispatch_compute(std::string_view shader_name,
                                     std::span<BufferHandle> buffers,
                                     WorkGroupSize workgroups) {
  if (!pimpl_->ready || buffers.empty())
    return;

  // Find pipeline
  std::string name(shader_name);
  auto it = pimpl_->pipelines.find(name);

  wgpu::ComputePipeline pipeline;
  if (it != pimpl_->pipelines.end()) {
    pipeline = it->second;
  } else {
    // Fallback for "basic" if not registered (Legacy support/Testing)
    if (shader_name == "basic") {
      wgpu::ShaderSourceWGSL wgslDesc = {};
      wgslDesc.code = BASIC_WGSL;
      wgpu::ShaderModuleDescriptor shaderDesc = {};
      shaderDesc.nextInChain = &wgslDesc;
      auto shaderModule = pimpl_->device.CreateShaderModule(&shaderDesc);
      wgpu::ComputePipelineDescriptor pipelineDesc = {};
      pipelineDesc.compute.module = shaderModule;
      pipelineDesc.compute.entryPoint = "main";
      pipeline = pimpl_->device.CreateComputePipeline(&pipelineDesc);
    } else {
      std::cerr << "❌ Shader not found: " << shader_name << std::endl;
      return;
    }
  }

  // 3. Create Bind Group
  if (buffers.size() < 2)
    return;

  // ... (Binding logic remains same) ...
  // Helper to acquire and detach later
  auto uniformBuffer =
      wgpu::Buffer::Acquire(reinterpret_cast<WGPUBuffer>(buffers[0]));
  auto storageBuffer =
      wgpu::Buffer::Acquire(reinterpret_cast<WGPUBuffer>(buffers[1]));

  auto bindGroupLayout = pipeline.GetBindGroupLayout(0);

  std::vector<wgpu::BindGroupEntry> entries(2);
  entries[0].binding = 0;
  entries[0].buffer = uniformBuffer;
  entries[0].size = wgpu::kWholeSize;

  entries[1].binding = 1;
  entries[1].buffer = storageBuffer;
  entries[1].size = wgpu::kWholeSize;

  wgpu::BindGroupDescriptor bindGroupDesc = {};
  bindGroupDesc.layout = bindGroupLayout;
  bindGroupDesc.entryCount = entries.size();
  bindGroupDesc.entries = entries.data();

  auto bindGroup = pimpl_->device.CreateBindGroup(&bindGroupDesc);

  // Detach buffers to prevent release
  uniformBuffer.MoveToCHandle();
  storageBuffer.MoveToCHandle();

  // 4. Encode Commands
  wgpu::CommandEncoder encoder = pimpl_->device.CreateCommandEncoder();
  wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.DispatchWorkgroups(workgroups.x, workgroups.y, workgroups.z);
  pass.End();

  wgpu::CommandBuffer commands = encoder.Finish();
  pimpl_->queue.Submit(1, &commands);
}

bool WebGPUBackend::register_kernel(const ComputeKernel &kernel) {
  if (!pimpl_->ready)
    return false;

  // Only support WGSL for now
  if (kernel.format != ComputeKernel::Format::WGSL)
    return false;

  // Create Shader Module
  wgpu::ShaderSourceWGSL wgslDesc = {};
  // kernel.bytecode is the source string + null terminator
  wgslDesc.code = reinterpret_cast<const char *>(kernel.bytecode.data());

  wgpu::ShaderModuleDescriptor shaderDesc = {};
  shaderDesc.nextInChain = &wgslDesc;

  // Label it
  shaderDesc.label = kernel.name.c_str();

  auto shaderModule = pimpl_->device.CreateShaderModule(&shaderDesc);

  wgpu::ComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.compute.module = shaderModule;
  pipelineDesc.compute.entryPoint = "main";
  pipelineDesc.label = kernel.name.c_str();

  auto pipeline = pimpl_->device.CreateComputePipeline(&pipelineDesc);

  pimpl_->pipelines[kernel.name] = pipeline;
  std::cout << "✅ Registered Kernel: " << kernel.name << std::endl;
  return true;
}

void WebGPUBackend::synchronize() {}

} // namespace CaptionEngine

#else

// Stub for non-Emscripten builds
namespace CaptionEngine {
class WebGPUBackend::Impl {};
WebGPUBackend::WebGPUBackend() = default;
WebGPUBackend::~WebGPUBackend() = default;
bool WebGPUBackend::initialize() { return false; }
BufferHandle WebGPUBackend::create_buffer(size_t, MemoryType) { return 0; }
void WebGPUBackend::destroy_buffer(BufferHandle) {}
void WebGPUBackend::upload_buffer(BufferHandle, std::span<const uint8_t>) {}
std::vector<uint8_t> WebGPUBackend::download_buffer(BufferHandle, size_t) {
  return {};
}
void WebGPUBackend::dispatch_compute(std::string_view, std::span<BufferHandle>,
                                     WorkGroupSize) {}
void WebGPUBackend::synchronize() {}
void WebGPUBackend::on_adapter_ready(void *, void *) {}
void WebGPUBackend::on_device_ready(void *, void *) {}
} // namespace CaptionEngine

#endif
