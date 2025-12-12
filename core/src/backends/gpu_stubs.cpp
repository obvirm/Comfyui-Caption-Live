/**
 * @file gpu_stubs.cpp
 * @brief Stub implementations for GPU backends when SDKs are not available
 *
 * This file provides empty/throwing implementations so the library can compile
 * and link even when CUDA, Vulkan, or WebGPU SDKs are not installed.
 */

#include "gpu/cuda_backend.hpp"
#include "gpu/vulkan_backend.hpp"
#include "gpu/webgpu_backend.hpp"
#include "text/sdf_generator.hpp"
#include "text/sdf_text_renderer.hpp"

#include <stdexcept>
#include <vector>

namespace CaptionEngine {
namespace GPU {

// ============================================================================
// Vulkan Backend Stub (when CE_HAS_VULKAN == 0)
// ============================================================================
#if CE_HAS_VULKAN == 0

struct VulkanBackend::Impl {};

VulkanBackend::VulkanBackend() : pimpl_(nullptr) {}
VulkanBackend::~VulkanBackend() = default;

std::string VulkanBackend::name() const { return "Vulkan (not available)"; }
bool VulkanBackend::isReady() const { return false; }

GPUResult<std::unique_ptr<Buffer>> VulkanBackend::createBuffer(size_t,
                                                               BufferUsage) {
  return GPUError{"Vulkan not available"};
}

GPUResult<std::unique_ptr<Texture>>
VulkanBackend::createTexture(uint32_t, uint32_t, TextureFormat) {
  return GPUError{"Vulkan not available"};
}

GPUResult<std::unique_ptr<Shader>>
VulkanBackend::createShaderWGSL(const std::string &, ShaderStage,
                                const std::string &) {
  return GPUError{"Vulkan not available"};
}

GPUResult<std::unique_ptr<Shader>>
VulkanBackend::createShaderSPIRV(std::span<const uint32_t>, ShaderStage,
                                 const std::string &) {
  return GPUError{"Vulkan not available"};
}

GPUResult<std::unique_ptr<Pipeline>>
VulkanBackend::createComputePipeline(Shader *) {
  return GPUError{"Vulkan not available"};
}

GPUResult<std::unique_ptr<Pipeline>>
VulkanBackend::createRenderPipeline(Shader *, Shader *, TextureFormat) {
  return GPUError{"Vulkan not available"};
}

GPUResult<std::unique_ptr<CommandBuffer>> VulkanBackend::createCommandBuffer() {
  return GPUError{"Vulkan not available"};
}

void VulkanBackend::submit(CommandBuffer *) {}
void VulkanBackend::waitIdle() {}

VkDevice VulkanBackend::device() const { return nullptr; }
VkPhysicalDevice VulkanBackend::physicalDevice() const { return nullptr; }
VkQueue VulkanBackend::graphicsQueue() const { return nullptr; }
VkQueue VulkanBackend::computeQueue() const { return nullptr; }
VkCommandPool VulkanBackend::commandPool() const { return nullptr; }

#endif // CE_HAS_VULKAN == 0

// ============================================================================
// WebGPU Backend Stub (when CE_HAS_WEBGPU == 0)
// ============================================================================
#if CE_HAS_WEBGPU == 0

struct WebGPUBackend::Impl {};

WebGPUBackend::WebGPUBackend() : pimpl_(nullptr) {}
WebGPUBackend::~WebGPUBackend() = default;

std::string WebGPUBackend::name() const { return "WebGPU (not available)"; }
bool WebGPUBackend::isReady() const { return false; }

GPUResult<std::unique_ptr<Buffer>> WebGPUBackend::createBuffer(size_t,
                                                               BufferUsage) {
  return GPUError{"WebGPU not available"};
}

GPUResult<std::unique_ptr<Texture>>
WebGPUBackend::createTexture(uint32_t, uint32_t, TextureFormat) {
  return GPUError{"WebGPU not available"};
}

GPUResult<std::unique_ptr<Shader>>
WebGPUBackend::createShaderWGSL(const std::string &, ShaderStage,
                                const std::string &) {
  return GPUError{"WebGPU not available"};
}

GPUResult<std::unique_ptr<Shader>>
WebGPUBackend::createShaderSPIRV(std::span<const uint32_t>, ShaderStage,
                                 const std::string &) {
  return GPUError{"WebGPU not available"};
}

GPUResult<std::unique_ptr<Pipeline>>
WebGPUBackend::createComputePipeline(Shader *) {
  return GPUError{"WebGPU not available"};
}

GPUResult<std::unique_ptr<Pipeline>>
WebGPUBackend::createRenderPipeline(Shader *, Shader *, TextureFormat) {
  return GPUError{"WebGPU not available"};
}

GPUResult<std::unique_ptr<CommandBuffer>> WebGPUBackend::createCommandBuffer() {
  return GPUError{"WebGPU not available"};
}

void WebGPUBackend::submit(CommandBuffer *) {}
void WebGPUBackend::waitIdle() {}

#endif // CE_HAS_WEBGPU == 0

// ============================================================================
// CUDA Backend Stub (when CE_HAS_CUDA == 0)
// ============================================================================
#if CE_HAS_CUDA == 0

struct CUDABackend::Impl {};

CUDABackend::CUDABackend() : pimpl_(nullptr) {}
CUDABackend::~CUDABackend() = default;

std::string CUDABackend::name() const { return "CUDA (not available)"; }
bool CUDABackend::isReady() const { return false; }

GPUResult<std::unique_ptr<Buffer>> CUDABackend::createBuffer(size_t,
                                                             BufferUsage) {
  return GPUError{"CUDA not available"};
}

GPUResult<std::unique_ptr<Texture>>
CUDABackend::createTexture(uint32_t, uint32_t, TextureFormat) {
  return GPUError{"CUDA not available"};
}

GPUResult<std::unique_ptr<Shader>>
CUDABackend::createShaderWGSL(const std::string &, ShaderStage,
                              const std::string &) {
  return GPUError{"CUDA not available"};
}

GPUResult<std::unique_ptr<Shader>>
CUDABackend::createShaderSPIRV(std::span<const uint32_t>, ShaderStage,
                               const std::string &) {
  return GPUError{"CUDA not available"};
}

GPUResult<std::unique_ptr<Pipeline>>
CUDABackend::createComputePipeline(Shader *) {
  return GPUError{"CUDA not available"};
}

GPUResult<std::unique_ptr<Pipeline>>
CUDABackend::createRenderPipeline(Shader *, Shader *, TextureFormat) {
  return GPUError{"CUDA not available"};
}

GPUResult<std::unique_ptr<CommandBuffer>> CUDABackend::createCommandBuffer() {
  return GPUError{"CUDA not available"};
}

void CUDABackend::submit(CommandBuffer *) {}
void CUDABackend::waitIdle() {}

#endif // CE_HAS_CUDA == 0

} // namespace GPU

// ============================================================================
// Text Renderer Stubs (when CE_HAS_FREETYPE == 0)
// ============================================================================
namespace Text {

#if !defined(CE_HAS_FREETYPE) || CE_HAS_FREETYPE == 0

// SDFAtlas Stubs
struct SDFAtlas::Impl {};

SDFAtlas::SDFAtlas(GPU::Backend *, const FontConfig &) : pimpl_(nullptr) {}
SDFAtlas::~SDFAtlas() = default;

bool SDFAtlas::loadFromMemory(const uint8_t *, size_t) { return false; }
bool SDFAtlas::loadFromFile(const std::string &) { return false; }
const GlyphMetrics *SDFAtlas::getGlyph(uint32_t) const { return nullptr; }
GPU::Texture *SDFAtlas::getTexture() const { return nullptr; }
float SDFAtlas::lineHeight() const { return 0.0f; }

// SDFTextRenderer Stubs
struct SDFTextRenderer::Impl {};

SDFTextRenderer::SDFTextRenderer(GPU::Backend *) : pimpl_(nullptr) {}
SDFTextRenderer::~SDFTextRenderer() = default;

void SDFTextRenderer::setFont(SDFAtlas *) {}
void SDFTextRenderer::beginFrame() {}
void SDFTextRenderer::drawText(const std::string &, glm::vec2,
                               const TextStyle &) {}
void SDFTextRenderer::drawTextInBox(const std::string &, glm::vec2, glm::vec2,
                                    const TextStyle &) {}
void SDFTextRenderer::endFrame(GPU::CommandBuffer *, GPU::Texture *) {}
float SDFTextRenderer::measureWidth(const std::string &, float) { return 0.0f; }

// BoxRenderer Stubs
struct BoxRenderer::Impl {};
BoxRenderer::BoxRenderer(GPU::Backend *) : pimpl_(nullptr) {}
BoxRenderer::~BoxRenderer() = default;
void BoxRenderer::drawBox(GPU::CommandBuffer *, glm::vec2, glm::vec2, glm::vec4,
                          float, float) {}

#endif // CE_HAS_FREETYPE == 0

} // namespace Text
} // namespace CaptionEngine
