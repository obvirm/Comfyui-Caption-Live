#pragma once
/**
 * @file engine.hpp
 * @brief Core Caption Engine interface
 */

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace CaptionEngine {

// Forward declarations
class Template;
class Renderer;
class ComputeBackend;

/// Frame data container
struct FrameData {
  std::vector<uint8_t> pixels; // RGBA format
  uint32_t width = 0;
  uint32_t height = 0;
  double timestamp = 0.0;
};

/// Render quality levels
enum class Quality {
  Low,    // 480p, fast
  Medium, // 720p, balanced
  High,   // 1080p, quality
  Ultra   // 4K, maximum
};

/// Available compute backends
enum class BackendType {
  Auto,   // Auto-detect best
  CPU,    // Software fallback
  WebGPU, // Browser compute
  Vulkan, // Cross-platform GPU
  CUDA,   // NVIDIA GPU
  Metal   // Apple GPU
};

/// Engine configuration
struct EngineConfig {
  BackendType backend = BackendType::Auto;
  Quality quality = Quality::High;
  bool deterministic = true;
  uint32_t max_texture_size = 4096;
  size_t memory_limit_mb = 512;
};

/**
 * @brief Main Caption Engine
 *
 * Unified rendering engine for caption effects across all platforms.
 * Supports CPU, WebGPU, Vulkan, CUDA, and Metal backends.
 */
class Engine {
public:
  /// Create engine with default configuration
  Engine();

  /// Create engine with custom configuration
  explicit Engine(const EngineConfig &config);

  /// Destructor
  ~Engine();

  // Move-only semantics
  Engine(Engine &&) noexcept;
  Engine &operator=(Engine &&) noexcept;
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  /**
   * @brief Render a single frame
   * @param template_json JSON template string
   * @param time Current timestamp in seconds
   * @return RGBA pixel data
   */
  [[nodiscard]] FrameData render_frame(const std::string &template_json,
                                       double time);

  /**
   * @brief Render frame with input image compositing
   * @param template_json JSON template string
   * @param time Current timestamp
   * @param input_image Input image to composite on
   * @return Composited RGBA pixel data
   */
  [[nodiscard]] FrameData
  render_frame_composite(const std::string &template_json, double time,
                         const FrameData &input_image);

  /**
   * @brief Export frame as PNG bytes
   * @param frame Frame data to encode
   * @return PNG encoded bytes
   */
  [[nodiscard]] std::vector<uint8_t> export_png(const FrameData &frame);

  /**
   * @brief Get current backend type
   */
  [[nodiscard]] BackendType current_backend() const;

  /**
   * @brief Validate deterministic frame hash
   * @param frame Frame to validate
   * @param expected_hash Expected hash value
   * @return True if frame matches expected hash
   */
  [[nodiscard]] bool validate_frame(const FrameData &frame,
                                    uint64_t expected_hash) const;

  /**
   * @brief Compute deterministic hash for frame
   */
  [[nodiscard]] uint64_t compute_frame_hash(const FrameData &frame) const;

  /**
   * @brief Test GPU backend functionality
   * @return True if dispatch successful
   */
  [[nodiscard]] bool test_gpu_compute();

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace CaptionEngine
