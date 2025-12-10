/**
 * @file render_pipeline.cpp
 * @brief Unified Render Pipeline implementation
 *
 * Connects WebGPU, Vulkan, and CUDA backends into single rendering API.
 */

#include "engine/render_pipeline.hpp"
#include "text/sdf_generator.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>


// JSON parsing (simplified)
#include <regex>
#include <sstream>


namespace CaptionEngine {

// ============================================================================
// Scene Template Parsing
// ============================================================================

SceneTemplate SceneTemplate::fromJSON(const std::string &json) {
  SceneTemplate scene;

  // Simple regex-based parsing for key fields
  // In production, use nlohmann/json or rapidjson

  std::regex widthRe(R"("width"\s*:\s*(\d+))");
  std::regex heightRe(R"("height"\s*:\s*(\d+))");
  std::regex durationRe(R"("duration"\s*:\s*([\d.]+))");
  std::regex contentRe(R"("content"\s*:\s*"([^"]*)") ");
      std::regex fontSizeRe(R"("font_size"\s*:\s*([\d.]+))");
  std::regex colorRe(R"("color"\s*:\s*"([^"]*)") ");
      std::regex posXRe(R"("x"\s*:\s*([\d.]+))");
  std::regex posYRe(R"("y"\s*:\s*([\d.]+))");
  std::regex animTypeRe(R"("type"\s*:\s*"([^"]*)") ");

      std::smatch match;

  // Parse canvas
  if (std::regex_search(json, match, widthRe)) {
    scene.target.width = std::stoul(match[1]);
  }
  if (std::regex_search(json, match, heightRe)) {
    scene.target.height = std::stoul(match[1]);
  }
  if (std::regex_search(json, match, durationRe)) {
    scene.duration = std::stof(match[1]);
  }

  // Find layers section
  size_t layersStart = json.find("\"layers\"");
  if (layersStart != std::string::npos) {
    std::string layersSection = json.substr(layersStart);

    CaptionLayer layer;

    if (std::regex_search(layersSection, match, contentRe)) {
      layer.text = match[1];
    }
    if (std::regex_search(layersSection, match, fontSizeRe)) {
      layer.textStyle.fontSize = std::stof(match[1]);
    }
    if (std::regex_search(layersSection, match, posXRe)) {
      layer.position.x = std::stof(match[1]);
    }
    if (std::regex_search(layersSection, match, posYRe)) {
      layer.position.y = std::stof(match[1]);
    }

    // Animation type
    size_t animStart = layersSection.find("\"animation\"");
    if (animStart != std::string::npos) {
      std::string animSection = layersSection.substr(animStart);
      if (std::regex_search(animSection, match, animTypeRe)) {
        layer.animation.type = match[1];
      }
    }

    if (!layer.text.empty()) {
      scene.layers.push_back(layer);
    }
  }

  return scene;
}

std::string SceneTemplate::toJSON() const {
  std::ostringstream ss;
  ss << "{";
  ss << "\"canvas\":{\"width\":" << target.width
     << ",\"height\":" << target.height << "},";
  ss << "\"duration\":" << duration << ",";
  ss << "\"layers\":[";

  for (size_t i = 0; i < layers.size(); i++) {
    if (i > 0)
      ss << ",";
    const auto &layer = layers[i];
    ss << "{";
    ss << "\"type\":\"text\",";
    ss << "\"content\":\"" << layer.text << "\",";
    ss << "\"position\":{\"x\":" << layer.position.x
       << ",\"y\":" << layer.position.y << "},";
    ss << "\"style\":{\"font_size\":" << layer.textStyle.fontSize << "}";
    ss << "}";
  }

  ss << "]}";
  return ss.str();
}

// ============================================================================
// Checksum Calculation
// ============================================================================

uint64_t RenderPipeline::calculateChecksum(const FrameOutput &frame) {
  // FNV-1a hash
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < frame.pixels.size(); i++) {
    hash ^= frame.pixels[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool RenderPipeline::framesMatch(const FrameOutput &a, const FrameOutput &b) {
  if (a.width != b.width || a.height != b.height)
    return false;
  if (a.pixels.size() != b.pixels.size())
    return false;
  return std::memcmp(a.pixels.data(), b.pixels.data(), a.pixels.size()) == 0;
}

// ============================================================================
// Render Pipeline Implementation
// ============================================================================

struct RenderPipeline::Impl {
  // Backends
  std::unique_ptr<GPU::Backend> primaryBackend;
  std::unique_ptr<GPU::CUDABackend>
      cudaBackend; // Optional compute acceleration

  // Resources
  std::unique_ptr<GPU::Texture> renderTarget;
  std::unique_ptr<GPU::Texture> stagingTexture;
  std::unique_ptr<GPU::Buffer> outputBuffer;

  // Text rendering
  std::unique_ptr<Text::SDFGenerator> sdfGenerator;
  std::unique_ptr<Text::SDFAtlas> fontAtlas;
  std::unique_ptr<Text::SDFTextRenderer> textRenderer;

  // State
  RenderTarget targetConfig;
  bool initialized = false;
  bool fontLoaded = false;
};

RenderPipeline::RenderPipeline() : RenderPipeline(GPU::BackendType::Auto) {}

RenderPipeline::RenderPipeline(GPU::BackendType preferredBackend)
    : pimpl_(std::make_unique<Impl>()) {

  std::cout << "🎬 Initializing Unified Render Pipeline..." << std::endl;

  // Try to initialize backends in order of preference
  if (preferredBackend == GPU::BackendType::Auto) {
    // Try Vulkan first (most capable)
    pimpl_->primaryBackend = std::make_unique<GPU::VulkanBackend>();
    if (!pimpl_->primaryBackend->isReady()) {
      std::cout << "⚠️ Vulkan not available, trying WebGPU..." << std::endl;
      pimpl_->primaryBackend = std::make_unique<GPU::WebGPUBackend>();
    }
  } else if (preferredBackend == GPU::BackendType::Vulkan) {
    pimpl_->primaryBackend = std::make_unique<GPU::VulkanBackend>();
  } else if (preferredBackend == GPU::BackendType::WebGPU) {
    pimpl_->primaryBackend = std::make_unique<GPU::WebGPUBackend>();
  }

  if (pimpl_->primaryBackend && pimpl_->primaryBackend->isReady()) {
    std::cout << "✅ Primary backend: " << pimpl_->primaryBackend->name()
              << std::endl;
  }

  // Try to add CUDA acceleration
  pimpl_->cudaBackend = std::make_unique<GPU::CUDABackend>();
  if (pimpl_->cudaBackend->isReady()) {
    std::cout << "✅ CUDA acceleration: " << pimpl_->cudaBackend->name()
              << std::endl;
  } else {
    pimpl_->cudaBackend.reset();
    std::cout << "ℹ️ CUDA acceleration not available" << std::endl;
  }

  // Initialize SDF generator
  pimpl_->sdfGenerator =
      std::make_unique<Text::SDFGenerator>(pimpl_->primaryBackend.get());
}

RenderPipeline::~RenderPipeline() = default;

bool RenderPipeline::initialize(const RenderTarget &target) {
  if (!pimpl_->primaryBackend || !pimpl_->primaryBackend->isReady()) {
    std::cerr << "❌ No GPU backend available" << std::endl;
    return false;
  }

  pimpl_->targetConfig = target;

  // Create render target texture
  auto texResult = pimpl_->primaryBackend->createTexture(
      target.width, target.height, target.format);
  if (!texResult) {
    std::cerr << "❌ Failed to create render target: "
              << texResult.error().message << std::endl;
    return false;
  }
  pimpl_->renderTarget = std::move(*texResult);

  // Create output buffer for readback
  size_t bufferSize = target.width * target.height * 4; // RGBA
  auto bufResult = pimpl_->primaryBackend->createBuffer(
      bufferSize, GPU::BufferUsage::CopyDst | GPU::BufferUsage::Storage);
  if (!bufResult) {
    std::cerr << "❌ Failed to create output buffer" << std::endl;
    return false;
  }
  pimpl_->outputBuffer = std::move(*bufResult);

  // Initialize text renderer
  pimpl_->textRenderer =
      std::make_unique<Text::SDFTextRenderer>(pimpl_->primaryBackend.get());

  pimpl_->initialized = true;
  std::cout << "✅ Render pipeline initialized: " << target.width << "x"
            << target.height << std::endl;

  return true;
}

bool RenderPipeline::loadFont(const uint8_t *data, size_t size) {
  if (!pimpl_->sdfGenerator)
    return false;

  if (!pimpl_->sdfGenerator->loadFont(data, size)) {
    return false;
  }

  // Generate SDF atlas
  Text::SDFParams params;
  params.fontSize = 64;
  params.sdfSpread = 8.0f;
  params.atlasSize = 2048;
  params.useGPU = pimpl_->cudaBackend != nullptr;

  auto atlasResult = pimpl_->sdfGenerator->generateStandardAtlas(params);

  // Create atlas texture
  if (pimpl_->primaryBackend) {
    // Store atlas data - would create texture here
    pimpl_->fontLoaded = true;
    std::cout << "✅ Font loaded with " << atlasResult.glyphs.size()
              << " glyphs" << std::endl;
  }

  return pimpl_->fontLoaded;
}

bool RenderPipeline::loadFontFile(const std::string &path) {
  if (!pimpl_->sdfGenerator)
    return false;
  return pimpl_->sdfGenerator->loadFontFile(path);
}

FrameOutput RenderPipeline::renderFrame(const SceneTemplate &scene,
                                        const FrameTiming &timing) {
  FrameOutput output;
  output.width = pimpl_->targetConfig.width;
  output.height = pimpl_->targetConfig.height;
  output.timestamp = timing.currentTime;
  output.pixels.resize(output.width * output.height * 4, 0);

  if (!pimpl_->initialized) {
    std::cerr << "❌ Pipeline not initialized" << std::endl;
    return output;
  }

  // Create command buffer
  auto cmdResult = pimpl_->primaryBackend->createCommandBuffer();
  if (!cmdResult) {
    std::cerr << "❌ Failed to create command buffer" << std::endl;
    return output;
  }
  auto &cmd = *cmdResult;

  cmd->begin();

  // Clear to transparent
  // (In full implementation, would issue clear command)

  // Render each layer
  for (const auto &layer : scene.layers) {
    renderCaptionLayer(layer, timing, cmd.get());
  }

  // Copy render target to output buffer
  cmd->copyTextureToBuffer(pimpl_->renderTarget.get(),
                           pimpl_->outputBuffer.get());

  cmd->end();

  // Submit and wait
  pimpl_->primaryBackend->submit(cmd.get());
  pimpl_->primaryBackend->waitIdle();

  // Read back pixels
  output.pixels = pimpl_->outputBuffer->read();
  output.checksum = calculateChecksum(output);

  return output;
}

FrameOutput RenderPipeline::renderFrameComposite(const SceneTemplate &scene,
                                                 const FrameTiming &timing,
                                                 const uint8_t *inputImage,
                                                 uint32_t inputWidth,
                                                 uint32_t inputHeight) {
  FrameOutput output;
  output.width = pimpl_->targetConfig.width;
  output.height = pimpl_->targetConfig.height;
  output.timestamp = timing.currentTime;

  if (!pimpl_->initialized) {
    std::cerr << "❌ Pipeline not initialized" << std::endl;
    output.pixels.resize(output.width * output.height * 4, 0);
    return output;
  }

  // Copy input image to render target
  if (inputImage && inputWidth == output.width &&
      inputHeight == output.height) {
    std::span<const uint8_t> inputSpan(inputImage,
                                       inputWidth * inputHeight * 4);
    pimpl_->renderTarget->upload(inputSpan);
  }

  // Render caption layers on top
  auto cmdResult = pimpl_->primaryBackend->createCommandBuffer();
  if (!cmdResult) {
    output.pixels.resize(output.width * output.height * 4, 0);
    return output;
  }
  auto &cmd = *cmdResult;

  cmd->begin();

  for (const auto &layer : scene.layers) {
    renderCaptionLayer(layer, timing, cmd.get());
  }

  cmd->copyTextureToBuffer(pimpl_->renderTarget.get(),
                           pimpl_->outputBuffer.get());
  cmd->end();

  pimpl_->primaryBackend->submit(cmd.get());
  pimpl_->primaryBackend->waitIdle();

  output.pixels = pimpl_->outputBuffer->read();
  output.checksum = calculateChecksum(output);

  return output;
}

std::vector<FrameOutput> RenderPipeline::renderSequence(
    const SceneTemplate &scene,
    std::function<void(uint32_t, uint32_t)> progressCallback) {
  std::vector<FrameOutput> frames;

  if (!pimpl_->initialized) {
    std::cerr << "❌ Pipeline not initialized" << std::endl;
    return frames;
  }

  float fps = pimpl_->targetConfig.fps;
  uint32_t totalFrames = static_cast<uint32_t>(scene.duration * fps);

  frames.reserve(totalFrames);

  for (uint32_t i = 0; i < totalFrames; i++) {
    FrameTiming timing;
    timing.frameIndex = i;
    timing.currentTime = static_cast<double>(i) / fps;
    timing.duration = scene.duration;
    timing.deltaTime = 1.0f / fps;

    frames.push_back(renderFrame(scene, timing));

    if (progressCallback) {
      progressCallback(i + 1, totalFrames);
    }
  }

  return frames;
}

GPU::BackendType RenderPipeline::activeBackend() const {
  if (!pimpl_->primaryBackend)
    return GPU::BackendType::Auto;
  return pimpl_->primaryBackend->type();
}

std::string RenderPipeline::backendName() const {
  if (!pimpl_->primaryBackend)
    return "None";
  return pimpl_->primaryBackend->name();
}

bool RenderPipeline::hasCUDAAcceleration() const {
  return pimpl_->cudaBackend && pimpl_->cudaBackend->isReady();
}

bool RenderPipeline::isReady() const { return pimpl_->initialized; }

// ============================================================================
// Private Helper Methods
// ============================================================================

void RenderPipeline::renderCaptionLayer(const CaptionLayer &layer,
                                        const FrameTiming &timing,
                                        GPU::CommandBuffer *cmd) {
  // Calculate pixel position
  float x = layer.position.x * pimpl_->targetConfig.width;
  float y = layer.position.y * pimpl_->targetConfig.height;

  // Handle animation
  std::string displayText = layer.text;
  bool showHighlight = false;
  size_t highlightStart = 0, highlightEnd = 0;

  if (layer.animation.type == "box" ||
      layer.animation.type == "box_highlight") {
    // Find active segment
    for (size_t i = 0; i < layer.animation.segments.size(); i++) {
      const auto &seg = layer.animation.segments[i];
      if (timing.currentTime >= seg.start && timing.currentTime < seg.end) {
        showHighlight = true;
        // Calculate highlight bounds based on segment
        break;
      }
    }
  } else if (layer.animation.type == "typewriter") {
    // Progressive reveal
    double progress = timing.currentTime / timing.duration;
    size_t chars = static_cast<size_t>(progress * layer.text.length());
    displayText = layer.text.substr(0, chars);
  }

  // Render highlight box if needed
  if (showHighlight && pimpl_->textRenderer) {
    // Would render box behind text here
  }

  // Render text
  if (pimpl_->textRenderer && pimpl_->fontLoaded) {
    pimpl_->textRenderer->drawText(displayText, {x, y}, layer.textStyle);
  }
}

// ============================================================================
// Global Pipeline Instance
// ============================================================================

static std::unique_ptr<RenderPipeline> g_pipeline;

RenderPipeline &getGlobalPipeline() {
  if (!g_pipeline) {
    g_pipeline = std::make_unique<RenderPipeline>();
  }
  return *g_pipeline;
}

} // namespace CaptionEngine
