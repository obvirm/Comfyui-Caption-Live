#pragma once
/**
 * @file vulkan.hpp
 * @brief Vulkan compute backend for cross-platform GPU acceleration
 */

#include "graphics/backend.hpp"

#if defined(HAS_VULKAN)

#include <vulkan/vulkan.h>

namespace CaptionEngine {

/**
 * @brief Vulkan compute backend
 *
 * Provides cross-platform GPU compute using Vulkan API.
 * Works on Windows, Linux, and Android.
 */
class VulkanBackend : public ComputeBackend {
public:
  /// Vulkan configuration
  struct Config {
    bool enable_validation = false; // Enable validation layers
    bool prefer_discrete = true;    // Prefer discrete GPU over integrated
    std::string app_name = "CaptionEngine";
    uint32_t app_version = 1;
  };

  explicit VulkanBackend(const Config &config = {});
  ~VulkanBackend() override;

  [[nodiscard]] std::string name() const override { return "Vulkan"; }
  [[nodiscard]] bool supports_compute() const override { return true; }

  [[nodiscard]] BufferHandle create_buffer(size_t size,
                                           MemoryType type) override;
  void destroy_buffer(BufferHandle handle) override;
  void upload_buffer(BufferHandle handle,
                     std::span<const uint8_t> data) override;
  [[nodiscard]] std::vector<uint8_t> download_buffer(BufferHandle handle,
                                                     size_t size) override;
  void dispatch_compute(std::string_view shader_name,
                        std::span<BufferHandle> buffers,
                        WorkGroupSize workgroups) override;
  bool register_kernel(const ComputeKernel &kernel) override;
  void synchronize() override;

  // --- Vulkan-specific Methods ---

  /// Get Vulkan instance
  [[nodiscard]] VkInstance instance() const;

  /// Get Vulkan physical device
  [[nodiscard]] VkPhysicalDevice physical_device() const;

  /// Get Vulkan logical device
  [[nodiscard]] VkDevice device() const;

  /// Get compute queue
  [[nodiscard]] VkQueue compute_queue() const;

  /// Get compute queue family index
  [[nodiscard]] uint32_t compute_queue_family() const;

  /// Get device properties
  [[nodiscard]] VkPhysicalDeviceProperties device_properties() const;

  /// Create shader module from SPIR-V
  [[nodiscard]] VkShaderModule
  create_shader_module(std::span<const uint32_t> spirv);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/// Check if Vulkan is available
[[nodiscard]] bool vulkan_available();

/// Get Vulkan version
[[nodiscard]] uint32_t vulkan_version();

/// List available Vulkan devices
[[nodiscard]] std::vector<std::string> vulkan_device_names();

} // namespace CaptionEngine

#else

// Stub when Vulkan not available
namespace CaptionEngine {
inline bool vulkan_available() { return false; }
inline uint32_t vulkan_version() { return 0; }
inline std::vector<std::string> vulkan_device_names() { return {}; }
} // namespace CaptionEngine

#endif // HAS_VULKAN
