#pragma once
/**
 * @file renderer.hpp
 * @brief Core rendering interface
 */

#include "engine/template.hpp"
#include <cstdint>
#include <memory>
#include <vector>


namespace CaptionEngine {

/// RGBA image buffer
struct ImageBuffer {
  std::vector<uint8_t> data;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 4; // RGBA

  /// Create empty buffer with dimensions
  static ImageBuffer create(uint32_t w, uint32_t h);

  /// Get pixel at (x, y)
  [[nodiscard]] uint8_t *pixel(uint32_t x, uint32_t y);
  [[nodiscard]] const uint8_t *pixel(uint32_t x, uint32_t y) const;

  /// Fill with color
  void fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);

  /// Clear to transparent
  void clear();
};

/// Effect calculation state
struct EffectState {
  std::optional<size_t> active_segment;
  double segment_progress = 0.0;
  float scale = 1.0f;
  bool show_box = false;
  std::optional<std::string> visible_text;
};

/**
 * @brief Core renderer for caption frames
 */
class Renderer {
public:
  Renderer();
  ~Renderer();

  /**
   * @brief Render a frame at given time
   * @param tmpl Template to render
   * @param time Current time in seconds
   * @return Rendered RGBA image buffer
   */
  [[nodiscard]] ImageBuffer render_frame(const Template &tmpl, double time);

  /**
   * @brief Render frame with input image
   * @param tmpl Template to render
   * @param time Current time
   * @param input Input image to composite on
   * @return Composited image buffer
   */
  [[nodiscard]] ImageBuffer render_frame(const Template &tmpl, double time,
                                         const ImageBuffer &input);

private:
  // Render individual layers
  void render_text_layer(ImageBuffer &img, const TextLayer &layer, double time);
  void render_image_layer(ImageBuffer &img, const ImageLayer &layer,
                          double time);

  // Text rendering
  void draw_text(ImageBuffer &img, const std::string &text, float x, float y,
                 float font_size, uint32_t color);

  void draw_text_with_stroke(ImageBuffer &img, const std::string &text, float x,
                             float y, float font_size, uint32_t text_color,
                             uint32_t stroke_color, float stroke_width);

  // Box highlight
  void draw_rounded_rect(ImageBuffer &img, int x, int y, uint32_t w, uint32_t h,
                         float radius, uint32_t color);

  // Effect calculations
  EffectState calculate_effect(const Animation &anim, double time);

  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace CaptionEngine
