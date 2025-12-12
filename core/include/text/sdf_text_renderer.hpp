#pragma once
/**
 * @file sdf_text_renderer.hpp
 * @brief GPU-accelerated SDF text rendering
 *
 * Uses Signed Distance Fields for crisp text at any resolution.
 * All rendering happens on GPU - no CPU text rasterization.
 */

#include "gpu/backend.hpp"
#include "text/types.hpp" // Shared TextStyle, TextAlign
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

namespace CaptionEngine {
namespace Text {

/// Glyph metrics (also defined in sdf_generator.hpp - use that one)
// Forward declaration - full definition in sdf_generator.hpp
struct GlyphMetrics;

/// Font configuration
struct FontConfig {
  float sdfSpread = 4.0f;    ///< SDF spread in pixels
  uint32_t atlasSize = 2048; ///< Atlas texture size
  uint32_t fontSize = 64;    ///< Base font size for SDF generation
};

/**
 * @brief SDF Font Atlas
 *
 * Contains pre-generated SDF for all glyphs in a font.
 */
class SDFAtlas {
public:
  SDFAtlas(GPU::Backend *gpu, const FontConfig &config);
  ~SDFAtlas();

  /// Load font from embedded data
  bool loadFromMemory(const uint8_t *data, size_t size);

  /// Load font from file
  bool loadFromFile(const std::string &path);

  /// Get glyph metrics
  const GlyphMetrics *getGlyph(uint32_t codepoint) const;

  /// Get SDF texture
  GPU::Texture *getTexture() const;

  /// Get line height
  float lineHeight() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

// Use shared TextStyle and TextAlign from text/types.hpp

/**
 * @brief GPU Text Renderer
 *
 * Renders text using SDF for maximum quality.
 * 100% GPU - identical output in WASM and native.
 */
class SDFTextRenderer {
public:
  SDFTextRenderer(GPU::Backend *gpu);
  ~SDFTextRenderer();

  /// Set font atlas to use
  void setFont(SDFAtlas *atlas);

  /// Begin frame (clears vertex buffer)
  void beginFrame();

  /// Add text to render
  void drawText(const std::string &text, glm::vec2 position,
                const TextStyle &style);

  /// Add text with bounding box
  void drawTextInBox(const std::string &text, glm::vec2 boxPos,
                     glm::vec2 boxSize, const TextStyle &style);

  /// End frame and render to target
  void endFrame(GPU::CommandBuffer *cmd, GPU::Texture *target);

  /// Measure text width
  float measureWidth(const std::string &text, float fontSize);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Highlight box renderer
 *
 * Renders rounded rectangle behind highlighted text.
 */
class BoxRenderer {
public:
  BoxRenderer(GPU::Backend *gpu);
  ~BoxRenderer();

  /// Draw rounded box
  void drawBox(GPU::CommandBuffer *cmd, glm::vec2 position, glm::vec2 size,
               glm::vec4 color, float cornerRadius, float padding = 0.0f);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace Text
} // namespace CaptionEngine
