/**
 * @file vulkan_backend.cpp
 * @brief Vulkan 1.3 backend implementation
 *
 * Headless Vulkan rendering for ComfyUI backend processing.
 */

#include "gpu/vulkan_backend.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>


namespace CaptionEngine {
namespace GPU {

// ============================================================================
// Validation Layer Configuration
// ============================================================================

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION = false;
#else
constexpr bool ENABLE_VALIDATION = true;
#endif

const std::vector<const char *> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> DEVICE_EXTENSIONS = {
    // No swapchain needed for headless rendering
};

// ============================================================================
// Vulkan Buffer Implementation
// ============================================================================

class VulkanBuffer : public Buffer {
public:
  VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice, size_t size,
               BufferUsage usage)
      : device_(device), size_(size), usage_(usage) {

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = translateUsage(usage);
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer_) != VK_SUCCESS) {
      std::cerr << "❌ Failed to create Vulkan buffer" << std::endl;
      return;
    }

    // Get memory requirements
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buffer_, &memReqs);

    // Find suitable memory type
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex =
        findMemoryType(memReqs.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       memProps);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
      std::cerr << "❌ Failed to allocate Vulkan buffer memory" << std::endl;
      vkDestroyBuffer(device, buffer_, nullptr);
      buffer_ = VK_NULL_HANDLE;
      return;
    }

    vkBindBufferMemory(device, buffer_, memory_, 0);
  }

  ~VulkanBuffer() override {
    if (memory_ != VK_NULL_HANDLE)
      vkFreeMemory(device_, memory_, nullptr);
    if (buffer_ != VK_NULL_HANDLE)
      vkDestroyBuffer(device_, buffer_, nullptr);
  }

  size_t size() const override { return size_; }
  BufferUsage usage() const override { return usage_; }

  void write(std::span<const uint8_t> data, size_t offset) override {
    void *mapped;
    vkMapMemory(device_, memory_, offset, data.size(), 0, &mapped);
    std::memcpy(mapped, data.data(), data.size());
    vkUnmapMemory(device_, memory_);
  }

  std::vector<uint8_t> read() override {
    std::vector<uint8_t> data(size_);
    void *mapped;
    vkMapMemory(device_, memory_, 0, size_, 0, &mapped);
    std::memcpy(data.data(), mapped, size_);
    vkUnmapMemory(device_, memory_);
    return data;
  }

  VkBuffer handle() const { return buffer_; }

private:
  VkBufferUsageFlags translateUsage(BufferUsage usage) {
    VkBufferUsageFlags flags = 0;
    auto u = static_cast<uint32_t>(usage);
    if (u & static_cast<uint32_t>(BufferUsage::Vertex))
      flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (u & static_cast<uint32_t>(BufferUsage::Index))
      flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (u & static_cast<uint32_t>(BufferUsage::Uniform))
      flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (u & static_cast<uint32_t>(BufferUsage::Storage))
      flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (u & static_cast<uint32_t>(BufferUsage::CopySrc))
      flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (u & static_cast<uint32_t>(BufferUsage::CopyDst))
      flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return flags;
  }

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties,
                          const VkPhysicalDeviceMemoryProperties &memProps) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    return 0;
  }

  VkDevice device_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  size_t size_;
  BufferUsage usage_;
};

// ============================================================================
// Vulkan Texture Implementation
// ============================================================================

class VulkanTexture : public Texture {
public:
  VulkanTexture(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t w,
                uint32_t h, TextureFormat fmt)
      : device_(device), width_(w), height_(h), format_(fmt) {

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {w, h, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = translateFormat(fmt);
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image_) != VK_SUCCESS) {
      std::cerr << "❌ Failed to create Vulkan image" << std::endl;
      return;
    }

    // Allocate image memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, image_, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memProps);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
      std::cerr << "❌ Failed to allocate Vulkan image memory" << std::endl;
      vkDestroyImage(device, image_, nullptr);
      image_ = VK_NULL_HANDLE;
      return;
    }

    vkBindImageMemory(device, image_, memory_, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = translateFormat(fmt);
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device, &viewInfo, nullptr, &imageView_);
  }

  ~VulkanTexture() override {
    if (imageView_ != VK_NULL_HANDLE)
      vkDestroyImageView(device_, imageView_, nullptr);
    if (memory_ != VK_NULL_HANDLE)
      vkFreeMemory(device_, memory_, nullptr);
    if (image_ != VK_NULL_HANDLE)
      vkDestroyImage(device_, image_, nullptr);
  }

  uint32_t width() const override { return width_; }
  uint32_t height() const override { return height_; }
  TextureFormat format() const override { return format_; }

  void upload(std::span<const uint8_t> data) override {
    // TODO: Implement staging buffer upload
  }

  std::vector<uint8_t> download() override {
    // TODO: Implement staging buffer download
    return {};
  }

  VkImage handle() const { return image_; }
  VkImageView view() const { return imageView_; }

private:
  VkFormat translateFormat(TextureFormat fmt) {
    switch (fmt) {
    case TextureFormat::RGBA8:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case TextureFormat::RGBA16F:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case TextureFormat::RGBA32F:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case TextureFormat::R8:
      return VK_FORMAT_R8_UNORM;
    case TextureFormat::R32F:
      return VK_FORMAT_R32_SFLOAT;
    default:
      return VK_FORMAT_R8G8B8A8_UNORM;
    }
  }

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties,
                          const VkPhysicalDeviceMemoryProperties &memProps) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    return 0;
  }

  VkDevice device_;
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageView imageView_ = VK_NULL_HANDLE;
  uint32_t width_, height_;
  TextureFormat format_;
};

// ============================================================================
// Vulkan Shader Implementation
// ============================================================================

class VulkanShader : public Shader {
public:
  VulkanShader(VkDevice device, std::span<const uint32_t> spirv,
               ShaderStage stage, const std::string &entryPoint)
      : device_(device), stage_(stage), entryPoint_(entryPoint) {

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    if (vkCreateShaderModule(device, &createInfo, nullptr, &module_) !=
        VK_SUCCESS) {
      std::cerr << "❌ Failed to create Vulkan shader module" << std::endl;
    }
  }

  ~VulkanShader() override {
    if (module_ != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device_, module_, nullptr);
    }
  }

  ShaderStage stage() const override { return stage_; }
  VkShaderModule handle() const { return module_; }
  const std::string &entryPoint() const { return entryPoint_; }

  VkShaderStageFlagBits vkStage() const {
    switch (stage_) {
    case ShaderStage::Vertex:
      return VK_SHADER_STAGE_VERTEX_BIT;
    case ShaderStage::Fragment:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case ShaderStage::Compute:
      return VK_SHADER_STAGE_COMPUTE_BIT;
    default:
      return VK_SHADER_STAGE_VERTEX_BIT;
    }
  }

private:
  VkDevice device_;
  VkShaderModule module_ = VK_NULL_HANDLE;
  ShaderStage stage_;
  std::string entryPoint_;
};

// ============================================================================
// Vulkan Pipeline Implementation
// ============================================================================

class VulkanPipeline : public Pipeline {
public:
  VulkanPipeline(VkDevice device, VkPipeline pipeline, VkPipelineLayout layout,
                 bool isCompute)
      : device_(device), pipeline_(pipeline), layout_(layout),
        isCompute_(isCompute) {}

  ~VulkanPipeline() override {
    if (pipeline_ != VK_NULL_HANDLE)
      vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device_, layout_, nullptr);
  }

  bool isCompute() const override { return isCompute_; }
  VkPipeline handle() const { return pipeline_; }
  VkPipelineLayout layout() const { return layout_; }

private:
  VkDevice device_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  bool isCompute_;
};

// ============================================================================
// Vulkan Command Buffer Implementation
// ============================================================================

class VulkanCommandBuffer : public CommandBuffer {
public:
  VulkanCommandBuffer(VkDevice device, VkCommandPool pool)
      : device_(device), pool_(pool) {

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(device, &allocInfo, &cmd_);
  }

  ~VulkanCommandBuffer() override {
    if (cmd_ != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, pool_, 1, &cmd_);
    }
  }

  void begin() override {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &beginInfo);
  }

  void end() override { vkEndCommandBuffer(cmd_); }

  void setComputePipeline(Pipeline *pipeline) override {
    auto *vkPipeline = static_cast<VulkanPipeline *>(pipeline);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      vkPipeline->handle());
    currentLayout_ = vkPipeline->layout();
  }

  void setRenderPipeline(Pipeline *pipeline) override {
    auto *vkPipeline = static_cast<VulkanPipeline *>(pipeline);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vkPipeline->handle());
    currentLayout_ = vkPipeline->layout();
  }

  void bindBuffer(uint32_t binding, Buffer *buffer) override {
    // TODO: Create and bind descriptor sets
  }

  void bindTexture(uint32_t binding, Texture *texture) override {
    // TODO: Create and bind descriptor sets
  }

  void dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) override {
    vkCmdDispatch(cmd_, groupsX, groupsY, groupsZ);
  }

  void draw(uint32_t vertexCount, uint32_t instanceCount) override {
    vkCmdDraw(cmd_, vertexCount, instanceCount, 0, 0);
  }

  void copyTextureToBuffer(Texture *src, Buffer *dst) override {
    // TODO: Implement image to buffer copy
  }

  VkCommandBuffer handle() const { return cmd_; }

private:
  VkDevice device_;
  VkCommandPool pool_;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkPipelineLayout currentLayout_ = VK_NULL_HANDLE;
};

// ============================================================================
// Vulkan Backend Implementation
// ============================================================================

struct VulkanBackend::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue computeQueue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  uint32_t graphicsFamily = 0;
  uint32_t computeFamily = 0;
  bool ready = false;

  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *userData) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cerr << "🔶 Vulkan: " << data->pMessage << std::endl;
  }
  return VK_FALSE;
}

VulkanBackend::VulkanBackend() : pimpl_(std::make_unique<Impl>()) {
  // 1. Create Instance
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "CaptionEngine";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "CaptionEngine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char *> extensions;
  if (ENABLE_VALIDATION) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  if (ENABLE_VALIDATION) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(VALIDATION_LAYERS.size());
    createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
  }

  if (vkCreateInstance(&createInfo, nullptr, &pimpl_->instance) != VK_SUCCESS) {
    std::cerr << "❌ Failed to create Vulkan instance" << std::endl;
    return;
  }

  // 2. Setup debug messenger
  if (ENABLE_VALIDATION) {
    VkDebugUtilsMessengerCreateInfoEXT debugInfo = {};
    debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debugInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        pimpl_->instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) {
      func(pimpl_->instance, &debugInfo, nullptr, &pimpl_->debugMessenger);
    }
  }

  // 3. Pick physical device
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(pimpl_->instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    std::cerr << "❌ No Vulkan-capable GPU found" << std::endl;
    return;
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(pimpl_->instance, &deviceCount, devices.data());

  // Prefer discrete GPU
  for (const auto &device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      pimpl_->physicalDevice = device;
      std::cout << "✅ Selected GPU: " << props.deviceName << std::endl;
      break;
    }
  }

  if (pimpl_->physicalDevice == VK_NULL_HANDLE) {
    pimpl_->physicalDevice = devices[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pimpl_->physicalDevice, &props);
    std::cout << "✅ Selected GPU: " << props.deviceName << std::endl;
  }

  // 4. Find queue families
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pimpl_->physicalDevice,
                                           &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      pimpl_->physicalDevice, &queueFamilyCount, queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      pimpl_->graphicsFamily = i;
    }
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      pimpl_->computeFamily = i;
    }
  }

  // 5. Create logical device
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  float queuePriority = 1.0f;

  VkDeviceQueueCreateInfo graphicsQueueInfo = {};
  graphicsQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  graphicsQueueInfo.queueFamilyIndex = pimpl_->graphicsFamily;
  graphicsQueueInfo.queueCount = 1;
  graphicsQueueInfo.pQueuePriorities = &queuePriority;
  queueCreateInfos.push_back(graphicsQueueInfo);

  if (pimpl_->computeFamily != pimpl_->graphicsFamily) {
    VkDeviceQueueCreateInfo computeQueueInfo = graphicsQueueInfo;
    computeQueueInfo.queueFamilyIndex = pimpl_->computeFamily;
    queueCreateInfos.push_back(computeQueueInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures = {};

  VkDeviceCreateInfo deviceInfo = {};
  deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  deviceInfo.pQueueCreateInfos = queueCreateInfos.data();
  deviceInfo.pEnabledFeatures = &deviceFeatures;
  deviceInfo.enabledExtensionCount =
      static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
  deviceInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

  if (vkCreateDevice(pimpl_->physicalDevice, &deviceInfo, nullptr,
                     &pimpl_->device) != VK_SUCCESS) {
    std::cerr << "❌ Failed to create Vulkan device" << std::endl;
    return;
  }

  vkGetDeviceQueue(pimpl_->device, pimpl_->graphicsFamily, 0,
                   &pimpl_->graphicsQueue);
  vkGetDeviceQueue(pimpl_->device, pimpl_->computeFamily, 0,
                   &pimpl_->computeQueue);

  // 6. Create command pool
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = pimpl_->graphicsFamily;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(pimpl_->device, &poolInfo, nullptr,
                          &pimpl_->commandPool) != VK_SUCCESS) {
    std::cerr << "❌ Failed to create command pool" << std::endl;
    return;
  }

  pimpl_->ready = true;
  std::cout << "✅ Vulkan backend initialized" << std::endl;
}

VulkanBackend::~VulkanBackend() {
  if (pimpl_->device)
    vkDeviceWaitIdle(pimpl_->device);

  if (pimpl_->commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(pimpl_->device, pimpl_->commandPool, nullptr);
  }
  if (pimpl_->device != VK_NULL_HANDLE) {
    vkDestroyDevice(pimpl_->device, nullptr);
  }
  if (pimpl_->debugMessenger != VK_NULL_HANDLE) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        pimpl_->instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func)
      func(pimpl_->instance, pimpl_->debugMessenger, nullptr);
  }
  if (pimpl_->instance != VK_NULL_HANDLE) {
    vkDestroyInstance(pimpl_->instance, nullptr);
  }
}

std::string VulkanBackend::name() const {
  if (!pimpl_->physicalDevice)
    return "Vulkan (not initialized)";

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pimpl_->physicalDevice, &props);
  return std::string("Vulkan 1.3 - ") + props.deviceName;
}

bool VulkanBackend::isReady() const { return pimpl_->ready; }

GPUResult<std::unique_ptr<Buffer>>
VulkanBackend::createBuffer(size_t size, BufferUsage usage) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"Vulkan not ready"});
  }
  return std::make_unique<VulkanBuffer>(pimpl_->device, pimpl_->physicalDevice,
                                        size, usage);
}

GPUResult<std::unique_ptr<Texture>>
VulkanBackend::createTexture(uint32_t width, uint32_t height,
                             TextureFormat format) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"Vulkan not ready"});
  }
  return std::make_unique<VulkanTexture>(pimpl_->device, pimpl_->physicalDevice,
                                         width, height, format);
}

GPUResult<std::unique_ptr<Shader>>
VulkanBackend::createShaderWGSL(const std::string &source, ShaderStage stage,
                                const std::string &entryPoint) {
  // Vulkan doesn't support WGSL directly, need SPIR-V
  return std::unexpected(GPUError{"WGSL not supported in Vulkan, use SPIR-V"});
}

GPUResult<std::unique_ptr<Shader>>
VulkanBackend::createShaderSPIRV(std::span<const uint32_t> spirv,
                                 ShaderStage stage,
                                 const std::string &entryPoint) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"Vulkan not ready"});
  }
  return std::make_unique<VulkanShader>(pimpl_->device, spirv, stage,
                                        entryPoint);
}

GPUResult<std::unique_ptr<Pipeline>>
VulkanBackend::createComputePipeline(Shader *computeShader) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"Vulkan not ready"});
  }

  auto *shader = static_cast<VulkanShader *>(computeShader);

  // Create pipeline layout (empty for now)
  VkPipelineLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

  VkPipelineLayout layout;
  if (vkCreatePipelineLayout(pimpl_->device, &layoutInfo, nullptr, &layout) !=
      VK_SUCCESS) {
    return std::unexpected(GPUError{"Failed to create pipeline layout"});
  }

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shader->handle();
  pipelineInfo.stage.pName = shader->entryPoint().c_str();
  pipelineInfo.layout = layout;

  VkPipeline pipeline;
  if (vkCreateComputePipelines(pimpl_->device, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &pipeline) != VK_SUCCESS) {
    vkDestroyPipelineLayout(pimpl_->device, layout, nullptr);
    return std::unexpected(GPUError{"Failed to create compute pipeline"});
  }

  return std::make_unique<VulkanPipeline>(pimpl_->device, pipeline, layout,
                                          true);
}

GPUResult<std::unique_ptr<Pipeline>> VulkanBackend::createRenderPipeline(
    Shader *vertexShader, Shader *fragmentShader, TextureFormat outputFormat) {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"Vulkan not ready"});
  }

  // TODO: Implement render pipeline creation with dynamic rendering
  return std::unexpected(GPUError{"Render pipeline not yet implemented"});
}

GPUResult<std::unique_ptr<CommandBuffer>> VulkanBackend::createCommandBuffer() {
  if (!pimpl_->ready) {
    return std::unexpected(GPUError{"Vulkan not ready"});
  }
  return std::make_unique<VulkanCommandBuffer>(pimpl_->device,
                                               pimpl_->commandPool);
}

void VulkanBackend::submit(CommandBuffer *cmd) {
  auto *vkCmd = static_cast<VulkanCommandBuffer *>(cmd);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  VkCommandBuffer cmdHandle = vkCmd->handle();
  submitInfo.pCommandBuffers = &cmdHandle;

  vkQueueSubmit(pimpl_->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
}

void VulkanBackend::waitIdle() {
  if (pimpl_->device) {
    vkDeviceWaitIdle(pimpl_->device);
  }
}

VkDevice VulkanBackend::device() const { return pimpl_->device; }
VkPhysicalDevice VulkanBackend::physicalDevice() const {
  return pimpl_->physicalDevice;
}
VkQueue VulkanBackend::graphicsQueue() const { return pimpl_->graphicsQueue; }
VkQueue VulkanBackend::computeQueue() const { return pimpl_->computeQueue; }
VkCommandPool VulkanBackend::commandPool() const { return pimpl_->commandPool; }

} // namespace GPU
} // namespace CaptionEngine
