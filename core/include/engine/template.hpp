#pragma once
/**
 * @file template.hpp
 * @brief Template data structures for caption definitions
 */

#include <optional>
#include <string>
#include <variant>
#include <vector>


namespace CaptionEngine {

/// Canvas dimensions
struct Canvas {
  uint32_t width = 1920;
  uint32_t height = 1080;
};

/// Position in normalized coordinates (0.0-1.0)
struct Position {
  double x = 0.5;
  double y = 0.5;
};

/// Text styling options
struct TextStyle {
  float font_size = 50.0f;
  std::string font_family = "Inter";
  std::string color = "#FFFFFF";
  std::optional<std::string> stroke_color;
  float stroke_width = 0.0f;
  uint16_t font_weight = 900;
};

/// Timed text segment
struct Segment {
  std::string text;
  double start = 0.0;
  double end = 1.0;
};

/// Box highlight animation
struct BoxHighlightAnimation {
  std::vector<Segment> segments;
  std::string box_color = "#39E55F";
  float box_radius = 8.0f;
  float box_padding = 8.0f;
};

/// Typewriter animation
struct TypewriterAnimation {
  std::vector<Segment> segments;
  float chars_per_second = 20.0f;
};

/// Bounce animation
struct BounceAnimation {
  std::vector<Segment> segments;
  float scale = 1.25f;
  float duration = 0.15f;
};

/// Colored word animation
struct ColoredAnimation {
  std::vector<Segment> segments;
  std::string active_color = "#39E55F";
};

/// Animation type variant
using Animation =
    std::variant<std::monostate, BoxHighlightAnimation, TypewriterAnimation,
                 BounceAnimation, ColoredAnimation>;

/// Text layer definition
struct TextLayer {
  std::string content;
  TextStyle style;
  Position position;
  Animation animation;
};

/// Image layer definition
struct ImageLayer {
  std::string src; // Base64 or path
  Position position;
  std::optional<double> width;
  std::optional<double> height;
};

/// Layer type variant
using Layer = std::variant<TextLayer, ImageLayer>;

/// Root template structure
struct Template {
  Canvas canvas;
  double duration = 5.0;
  double fps = 60.0;
  std::vector<Layer> layers;

  /// Parse template from JSON string
  static Template from_json(const std::string &json);

  /// Serialize to JSON string
  [[nodiscard]] std::string to_json() const;
};

} // namespace CaptionEngine
