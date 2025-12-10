/**
 * @file template_parser.cpp
 * @brief JSON template parser implementation
 */

#include "engine/template.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace CaptionEngine {

using json = nlohmann::json;

/// Parse position from JSON
Position parse_position(const json &j) {
  Position pos;
  pos.x = j.value("x", 0.5);
  pos.y = j.value("y", 0.5);
  return pos;
}

/// Parse text style from JSON
TextStyle parse_text_style(const json &j) {
  TextStyle style;
  style.font_family = j.value("font_family", std::string("Inter"));
  style.font_size = j.value("font_size", 50.0f);
  style.font_weight = j.value("font_weight", static_cast<uint16_t>(900));
  style.color = j.value("color", std::string("#FFFFFF"));
  if (j.contains("stroke_color")) {
    style.stroke_color = j["stroke_color"].get<std::string>();
  }
  style.stroke_width = j.value("stroke_width", 0.0f);
  return style;
}

/// Parse segment from JSON
Segment parse_segment(const json &j) {
  Segment seg;
  seg.text = j.value("text", std::string(""));
  seg.start = j.value("start", 0.0);
  seg.end = j.value("end", 1.0);
  return seg;
}

/// Parse segments array from JSON
std::vector<Segment> parse_segments(const json &j) {
  std::vector<Segment> segments;
  if (j.is_array()) {
    for (const auto &seg : j) {
      segments.push_back(parse_segment(seg));
    }
  }
  return segments;
}

/// Parse animation from JSON
Animation parse_animation(const json &j) {
  if (!j.contains("type")) {
    return std::monostate{};
  }

  std::string type = j.value("type", std::string("none"));

  if (type == "box_highlight") {
    BoxHighlightAnimation anim;
    if (j.contains("segments")) {
      anim.segments = parse_segments(j["segments"]);
    }
    anim.box_color = j.value("box_color", std::string("#39E55F"));
    anim.box_radius = j.value("box_radius", 8.0f);
    anim.box_padding = j.value("box_padding", 8.0f);
    return anim;
  } else if (type == "typewriter") {
    TypewriterAnimation anim;
    if (j.contains("segments")) {
      anim.segments = parse_segments(j["segments"]);
    }
    anim.chars_per_second = j.value("chars_per_second", 20.0f);
    return anim;
  } else if (type == "bounce") {
    BounceAnimation anim;
    if (j.contains("segments")) {
      anim.segments = parse_segments(j["segments"]);
    }
    anim.scale = j.value("scale", 1.25f);
    anim.duration = j.value("duration", 0.15f);
    return anim;
  } else if (type == "colored") {
    ColoredAnimation anim;
    if (j.contains("segments")) {
      anim.segments = parse_segments(j["segments"]);
    }
    anim.active_color = j.value("active_color", std::string("#39E55F"));
    return anim;
  }

  return std::monostate{};
}

/// Parse text layer from JSON
TextLayer parse_text_layer(const json &j) {
  TextLayer layer;
  layer.content = j.value("content", std::string(""));

  if (j.contains("style")) {
    layer.style = parse_text_style(j["style"]);
  }

  if (j.contains("position")) {
    layer.position = parse_position(j["position"]);
  }

  if (j.contains("animation")) {
    layer.animation = parse_animation(j["animation"]);
  }

  return layer;
}

/// Parse image layer from JSON
ImageLayer parse_image_layer(const json &j) {
  ImageLayer layer;
  layer.src = j.value("src", std::string(""));

  if (j.contains("position")) {
    layer.position = parse_position(j["position"]);
  }

  if (j.contains("width")) {
    layer.width = j["width"].get<double>();
  }
  if (j.contains("height")) {
    layer.height = j["height"].get<double>();
  }

  return layer;
}

/// Main template parser
Template Template::from_json(const std::string &json_str) {
  Template tmpl;

  try {
    json j = json::parse(json_str);

    // Canvas settings
    if (j.contains("canvas")) {
      const auto &canvas = j["canvas"];
      tmpl.canvas.width = canvas.value("width", 1920u);
      tmpl.canvas.height = canvas.value("height", 1080u);
    }

    // Duration and FPS
    tmpl.duration = j.value("duration", 5.0);
    tmpl.fps = j.value("fps", 60.0);

    // Parse layers
    if (j.contains("layers")) {
      for (const auto &layer : j["layers"]) {
        std::string layer_type = layer.value("type", std::string("text"));

        if (layer_type == "text") {
          tmpl.layers.push_back(parse_text_layer(layer));
        } else if (layer_type == "image") {
          tmpl.layers.push_back(parse_image_layer(layer));
        }
      }
    }

    // Legacy format: text_layers
    if (j.contains("text_layers")) {
      for (const auto &layer : j["text_layers"]) {
        tmpl.layers.push_back(parse_text_layer(layer));
      }
    }

  } catch (const json::exception &e) {
    throw std::runtime_error(std::string("Template parse error: ") + e.what());
  }

  return tmpl;
}

/// Serialize template to JSON
std::string Template::to_json() const {
  json j;

  j["duration"] = duration;
  j["fps"] = fps;

  j["canvas"] = {{"width", canvas.width}, {"height", canvas.height}};

  // Serialize layers
  j["layers"] = json::array();
  for (const auto &layer : layers) {
    json lj;

    if (std::holds_alternative<TextLayer>(layer)) {
      const auto &tl = std::get<TextLayer>(layer);
      lj["type"] = "text";
      lj["content"] = tl.content;

      lj["position"] = {{"x", tl.position.x}, {"y", tl.position.y}};

      lj["style"] = {{"font_family", tl.style.font_family},
                     {"font_size", tl.style.font_size},
                     {"font_weight", tl.style.font_weight},
                     {"color", tl.style.color},
                     {"stroke_width", tl.style.stroke_width}};

      if (tl.style.stroke_color) {
        lj["style"]["stroke_color"] = *tl.style.stroke_color;
      }

    } else if (std::holds_alternative<ImageLayer>(layer)) {
      const auto &il = std::get<ImageLayer>(layer);
      lj["type"] = "image";
      lj["src"] = il.src;

      lj["position"] = {{"x", il.position.x}, {"y", il.position.y}};

      if (il.width)
        lj["width"] = *il.width;
      if (il.height)
        lj["height"] = *il.height;
    }

    j["layers"].push_back(lj);
  }

  return j.dump(2);
}

} // namespace CaptionEngine
