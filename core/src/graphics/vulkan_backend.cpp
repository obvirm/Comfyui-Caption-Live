/**
 * @file vulkan_backend.cpp
 * @brief Vulkan compute backend implementation
 */

#include "compute/vulkan.hpp"

#if defined(HAS_VULKAN)

#include <stdexcept>
#include <vector>

namespace CaptionEngine {

struct VulkanBackend::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  uint32_t compute_queue_family = 0;
  VkPhysicalDeviceProperties properties{};
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

  struct BufferInfo {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
  };
  std::unordered_map<BufferHandle, BufferInfo> buffers;
  BufferHandle next_buffer_id = 1;

  struct PipelineInfo {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
  };
  std::unordered_map<std::string, PipelineInfo> pipelines;
};

VulkanBackend::VulkanBackend(const Config &config)
    : pimpl_(std::make_unique<Impl>()) {

  // Create Vulkan instance
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = config.app_name.c_str();
  app_info.applicationVersion = config.app_version;
  app_info.pEngineName = "CaptionEngine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char *> layers;
  if (config.enable_validation) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  create_info.ppEnabledLayerNames = layers.data();

  if (vkCreateInstance(&create_info, nullptr, &pimpl_->instance) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance");
  }

  // Enumerate physical devices
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(pimpl_->instance, &device_count, nullptr);

  if (device_count == 0) {
    throw std::runtime_error("No Vulkan-capable devices found");
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(pimpl_->instance, &device_count, devices.data());

  // Select best device
  for (const auto &device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    if (config.prefer_discrete &&
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      pimpl_->physical_device = device;
      pimpl_->properties = props;
      break;
    }
  }

  if (pimpl_->physical_device == VK_NULL_HANDLE) {
    pimpl_->physical_device = devices[0];
    vkGetPhysicalDeviceProperties(pimpl_->physical_device, &pimpl_->properties);
  }

  // Find compute queue family
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pimpl_->physical_device,
                                           &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      pimpl_->physical_device, &queue_family_count, queue_families.data());

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      pimpl_->compute_queue_family = i;
      break;
    }
  }

  // Create logical device
  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = pimpl_->compute_queue_family;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_create_info{};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;

  if (vkCreateDevice(pimpl_->physical_device, &device_create_info, nullptr,
                     &pimpl_->device) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan device");
  }

  vkGetDeviceQueue(pimpl_->device, pimpl_->compute_queue_family, 0,
                   &pimpl_->compute_queue);

  // Create command pool
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = pimpl_->compute_queue_family;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  vkCreateCommandPool(pimpl_->device, &pool_info, nullptr,
                      &pimpl_->command_pool);
}

VulkanBackend::~VulkanBackend() {
  if (pimpl_) {
    synchronize();

    // Cleanup pipelines
    for (auto &[name, pipeline] : pimpl_->pipelines) {
      if (pipeline.pipeline)
        vkDestroyPipeline(pimpl_->device, pipeline.pipeline, nullptr);
      if (pipeline.layout)
        vkDestroyPipelineLayout(pimpl_->device, pipeline.layout, nullptr);
      if (pipeline.desc_layout)
        vkDestroyDescriptorSetLayout(pimpl_->device, pipeline.desc_layout,
                                     nullptr);
    }

    // Cleanup buffers
    for (auto &[handle, info] : pimpl_->buffers) {
      if (info.buffer)
        vkDestroyBuffer(pimpl_->device, info.buffer, nullptr);
      if (info.memory)
        vkFreeMemory(pimpl_->device, info.memory, nullptr);
    }

    if (pimpl_->command_pool)
      vkDestroyCommandPool(pimpl_->device, pimpl_->command_pool, nullptr);
    if (pimpl_->descriptor_pool)
      vkDestroyDescriptorPool(pimpl_->device, pimpl_->descriptor_pool, nullptr);
    if (pimpl_->device)
      vkDestroyDevice(pimpl_->device, nullptr);
    if (pimpl_->instance)
      vkDestroyInstance(pimpl_->instance, nullptr);
  }
}

BufferHandle VulkanBackend::create_buffer(size_t size, MemoryType type) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  Impl::BufferInfo info;
  info.size = size;

  if (vkCreateBuffer(pimpl_->device, &buffer_info, nullptr, &info.buffer) !=
      VK_SUCCESS) {
    return 0;
  }

  // Allocate memory
  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(pimpl_->device, info.buffer, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;

  // Find suitable memory type
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(pimpl_->physical_device, &mem_props);

  VkMemoryPropertyFlags required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (type == MemoryType::HostVisible) {
    required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((mem_reqs.memoryTypeBits & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & required_flags) ==
            required_flags) {
      alloc_info.memoryTypeIndex = i;
      break;
    }
  }

  if (vkAllocateMemory(pimpl_->device, &alloc_info, nullptr, &info.memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(pimpl_->device, info.buffer, nullptr);
    return 0;
  }

  vkBindBufferMemory(pimpl_->device, info.buffer, info.memory, 0);

  BufferHandle handle = pimpl_->next_buffer_id++;
  pimpl_->buffers[handle] = info;
  return handle;
}

void VulkanBackend::destroy_buffer(BufferHandle handle) {
  auto it = pimpl_->buffers.find(handle);
  if (it != pimpl_->buffers.end()) {
    vkDestroyBuffer(pimpl_->device, it->second.buffer, nullptr);
    vkFreeMemory(pimpl_->device, it->second.memory, nullptr);
    pimpl_->buffers.erase(it);
  }
}

void VulkanBackend::upload_buffer(BufferHandle handle,
                                  std::span<const uint8_t> data) {
  auto it = pimpl_->buffers.find(handle);
  if (it == pimpl_->buffers.end())
    return;

  void *mapped;
  vkMapMemory(pimpl_->device, it->second.memory, 0, data.size(), 0, &mapped);
  std::memcpy(mapped, data.data(), data.size());
  vkUnmapMemory(pimpl_->device, it->second.memory);
}

std::vector<uint8_t> VulkanBackend::download_buffer(BufferHandle handle,
                                                    size_t size) {
  auto it = pimpl_->buffers.find(handle);
  if (it == pimpl_->buffers.end())
    return {};

  std::vector<uint8_t> result(size);
  void *mapped;
  vkMapMemory(pimpl_->device, it->second.memory, 0, size, 0, &mapped);
  std::memcpy(result.data(), mapped, size);
  vkUnmapMemory(pimpl_->device, it->second.memory);
  return result;
}

void VulkanBackend::dispatch_compute(std::string_view shader_name,
                                     std::span<BufferHandle> buffers,
                                     WorkGroupSize workgroups) {
  // Implementation would create command buffer, bind pipeline, dispatch
  // Simplified for now
  (void)shader_name;
  (void)buffers;
  (void)workgroups;
}

bool VulkanBackend::register_kernel(const ComputeKernel &kernel) {
  if (kernel.format != ComputeKernel::Format::SPIRV) {
    return false;
  }

  // Create shader module from SPIR-V
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = kernel.bytecode.size();
  create_info.pCode =
      reinterpret_cast<const uint32_t *>(kernel.bytecode.data());

  VkShaderModule shader_module;
  if (vkCreateShaderModule(pimpl_->device, &create_info, nullptr,
                           &shader_module) != VK_SUCCESS) {
    return false;
  }

  // Create compute pipeline
  // ... (simplified)

  vkDestroyShaderModule(pimpl_->device, shader_module, nullptr);
  return true;
}

void VulkanBackend::synchronize() { vkQueueWaitIdle(pimpl_->compute_queue); }

VkInstance VulkanBackend::instance() const { return pimpl_->instance; }
VkPhysicalDevice VulkanBackend::physical_device() const {
  return pimpl_->physical_device;
}
VkDevice VulkanBackend::device() const { return pimpl_->device; }
VkQueue VulkanBackend::compute_queue() const { return pimpl_->compute_queue; }
uint32_t VulkanBackend::compute_queue_family() const {
  return pimpl_->compute_queue_family;
}
VkPhysicalDeviceProperties VulkanBackend::device_properties() const {
  return pimpl_->properties;
}

VkShaderModule
VulkanBackend::create_shader_module(std::span<const uint32_t> spirv) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv.size() * sizeof(uint32_t);
  create_info.pCode = spirv.data();

  VkShaderModule module;
  vkCreateShaderModule(pimpl_->device, &create_info, nullptr, &module);
  return module;
}

// Utility functions
bool vulkan_available() {
  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

  VkInstance instance;
  if (vkCreateInstance(&create_info, nullptr, &instance) == VK_SUCCESS) {
    vkDestroyInstance(instance, nullptr);
    return true;
  }
  return false;
}

uint32_t vulkan_version() { return VK_API_VERSION_1_3; }

std::vector<std::string> vulkan_device_names() {
  std::vector<std::string> names;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

  VkInstance instance;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    return names;
  }

  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);

  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance, &count, devices.data());

  for (const auto &device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    names.push_back(props.deviceName);
  }

  vkDestroyInstance(instance, nullptr);
  return names;
}

} // namespace CaptionEngine

#else // !HAS_VULKAN

// Stub implementations when Vulkan is not available
#include "compute/vulkan.hpp"
#include <stdexcept>

namespace CaptionEngine {

struct VulkanBackend::Impl {};

VulkanBackend::VulkanBackend(const Config &) : pimpl_(nullptr) {
  throw std::runtime_error("Vulkan backend not available - SDK not installed");
}

VulkanBackend::~VulkanBackend() = default;

BufferHandle VulkanBackend::create_buffer(size_t, MemoryType) { return 0; }
void VulkanBackend::destroy_buffer(BufferHandle) {}
void VulkanBackend::upload_buffer(BufferHandle, std::span<const uint8_t>) {}
std::vector<uint8_t> VulkanBackend::download_buffer(BufferHandle, size_t) {
  return {};
}
void VulkanBackend::dispatch_compute(std::string_view, std::span<BufferHandle>,
                                     WorkGroupSize) {}
bool VulkanBackend::register_kernel(const ComputeKernel &) { return false; }
void VulkanBackend::synchronize() {}

bool vulkan_available() { return false; }
uint32_t vulkan_version() { return 0; }
std::vector<std::string> vulkan_device_names() { return {}; }

} // namespace CaptionEngine

#endif // HAS_VULKAN
