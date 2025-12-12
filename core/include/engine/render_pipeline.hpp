#pragma once
/**
 * @file render_pipeline.hpp
 * @brief Unified Render Pipeline - Connects all GPU backends
 *
 * Single entry point for rendering that automatically selects
 * the best available backend (CUDA > Vulkan > WebGPU).
 */

#include "gpu/backend.hpp"
#include "text/types.hpp" // Shared TextStyle, TextAlign
#include <glm/glm.hpp>

// Forward declarations - avoid including all backend headers
namespace CaptionEngine {
namespace GPU {
class VulkanBackend;
class WebGPUBackend;
class CUDABackend;
} // namespace GPU
namespace Text {
class SDFTextRenderer;
class SDFAtlas;
} // namespace Text
} // namespace CaptionEngine

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace CaptionEngine {

/// Render target configuration
struct RenderTarget {
  uint32_t width = 1920;
  uint32_t height = 1080;
  GPU::TextureFormat format = GPU::TextureFormat::RGBA8;
  float fps = 60.0f;
};

/// Frame timing information
struct FrameTiming {
  double currentTime = 0.0; ///< Current time in seconds
  double duration = 0.0;    ///< Total duration in seconds
  uint32_t frameIndex = 0;  ///< Current frame number
  float deltaTime = 0.0f;   ///< Time since last frame
};

/// Caption segment for animation
struct CaptionSegment {
  std::string text;
  double start = 0.0;
  double end = 0.0;
};

/// Caption layer data
struct CaptionLayer {
  std::string text;
  glm::vec2 position = {0.5f, 0.8f}; ///< Normalized [0-1]
  TextStyle textStyle;

  // Animation
  struct Animation {
    std::string type = "none"; ///< box, typewriter, bounce, etc.
    std::vector<CaptionSegment> segments;
    glm::vec4 highlightColor = {0.22f, 0.90f, 0.37f, 1.0f}; // #39E55F
    float boxRadius = 8.0f;
    float boxPadding = 8.0f;
  } animation;
};

/// Full scene template
struct SceneTemplate {
  RenderTarget target;
  float duration = 5.0f;
  std::vector<CaptionLayer> layers;

  /// Parse from JSON
  static SceneTemplate fromJSON(const std::string &json);

  /// Serialize to JSON
  std::string toJSON() const;
};

/// Frame output
struct FrameOutput {
  std::vector<uint8_t> pixels; ///< RGBA pixel data
  uint32_t width = 0;
  uint32_t height = 0;
  double timestamp = 0.0;
  uint64_t checksum = 0; ///< For validation
};

/**
 * @brief Unified Render Pipeline
 *
 * Orchestrates all GPU backends for consistent rendering:
 * - Automatically selects best backend
 * - Provides identical output across platforms
 * - Supports compute acceleration when available
 */
class RenderPipeline {
public:
  /// Create pipeline with automatic backend selection
  RenderPipeline();

  /// Create pipeline with specific backend
  explicit RenderPipeline(GPU::BackendType preferredBackend);

  ~RenderPipeline();

  // --- Configuration ---

  /// Initialize with render target
  bool initialize(const RenderTarget &target);

  /// Load font for text rendering
  bool loadFont(const uint8_t *data, size_t size);
  bool loadFontFile(const std::string &path);

  // --- Rendering ---

  /// Render a single frame
  FrameOutput renderFrame(const SceneTemplate &scene,
                          const FrameTiming &timing);

  /// Render frame with input image for compositing
  FrameOutput renderFrameComposite(const SceneTemplate &scene,
                                   const FrameTiming &timing,
                                   const uint8_t *inputImage, ///< RGBA pixels
                                   uint32_t inputWidth, uint32_t inputHeight);

  /// Render all frames of a scene
  std::vector<FrameOutput> renderSequence(
      const SceneTemplate &scene,
      std::function<void(uint32_t frame, uint32_t total)> progressCallback =
          nullptr);

  // --- Backend Info ---

  /// Get active backend type
  GPU::BackendType activeBackend() const;

  /// Get backend name
  std::string backendName() const;

  /// Check if CUDA acceleration is available
  bool hasCUDAAcceleration() const;

  /// Check if ready to render
  bool isReady() const;

  // --- Validation ---

  /// Calculate frame checksum for validation
  static uint64_t calculateChecksum(const FrameOutput &frame);

  /// Compare two frames for pixel-perfect match
  static bool framesMatch(const FrameOutput &a, const FrameOutput &b);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

  // Private helper
  void renderCaptionLayer(const CaptionLayer &layer, const FrameTiming &timing,
                          GPU::CommandBuffer *cmd);
};

/**
 * @brief Global pipeline instance
 *
 * Singleton for easy access from Python bindings.
 */
RenderPipeline &getGlobalPipeline();

} // namespace CaptionEngine
