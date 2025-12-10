#pragma once
/**
 * @file vulkan_backend.hpp
 * @brief Vulkan 1.3 backend implementation
 *
 * High-performance desktop rendering with compute shader support.
 */

#include "gpu/backend.hpp"

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.h>

namespace CaptionEngine {
namespace GPU {

/**
 * @brief Vulkan implementation of GPU Backend
 *
 * Features:
 * - Vulkan 1.3 with dynamic rendering
 * - Compute shader support
 * - SPIR-V shader compilation
 * - Headless rendering (no window required)
 */
class VulkanBackend : public Backend {
public:
  VulkanBackend();
  ~VulkanBackend() override;

  // Backend interface
  BackendType type() const override { return BackendType::Vulkan; }
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

  // Vulkan-specific accessors
  VkDevice device() const;
  VkPhysicalDevice physicalDevice() const;
  VkQueue graphicsQueue() const;
  VkQueue computeQueue() const;
  VkCommandPool commandPool() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace GPU
} // namespace CaptionEngine
