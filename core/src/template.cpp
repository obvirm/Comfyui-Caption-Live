/**
 * @file template.cpp
 * @brief Template parsing and serialization
 */

#include "engine/template.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace CaptionEngine {

// JSON parsing helpers
namespace {

Position parse_position(const json &j) {
  Position pos;
  pos.x = j.value("x", 0.5);
  pos.y = j.value("y", 0.5);
  return pos;
}

TextStyle parse_text_style(const json &j) {
  TextStyle style;
  style.font_size = j.value("font_size", 50.0f);
  style.font_family = j.value("font_family", "Inter");
  style.color = j.value("color", "#FFFFFF");
  if (j.contains("stroke_color")) {
    style.stroke_color = j["stroke_color"].get<std::string>();
  }
  style.stroke_width = j.value("stroke_width", 0.0f);
  style.font_weight = j.value("font_weight", 900);
  return style;
}

std::vector<Segment> parse_segments(const json &arr) {
  std::vector<Segment> segments;
  for (const auto &s : arr) {
    Segment seg;
    seg.text = s.value("text", "");
    seg.start = s.value("start", 0.0);
    seg.end = s.value("end", 1.0);
    segments.push_back(std::move(seg));
  }
  return segments;
}

Animation parse_animation(const json &j) {
  if (j.is_null()) {
    return std::monostate{};
  }

  std::string type = j.value("type", "");

  if (type == "box_highlight") {
    BoxHighlightAnimation anim;
    anim.segments = parse_segments(j["segments"]);
    anim.box_color = j.value("box_color", "#39E55F");
    anim.box_radius = j.value("box_radius", 8.0f);
    anim.box_padding = j.value("box_padding", 8.0f);
    return anim;
  }

  if (type == "typewriter") {
    TypewriterAnimation anim;
    if (j.contains("segments")) {
      anim.segments = parse_segments(j["segments"]);
    }
    anim.chars_per_second = j.value("chars_per_second", 20.0f);
    return anim;
  }

  if (type == "bounce") {
    BounceAnimation anim;
    anim.segments = parse_segments(j["segments"]);
    anim.scale = j.value("scale", 1.25f);
    anim.duration = j.value("bounce_duration", 0.15f);
    return anim;
  }

  if (type == "colored") {
    ColoredAnimation anim;
    anim.segments = parse_segments(j["segments"]);
    anim.active_color = j.value("active_color", "#39E55F");
    return anim;
  }

  return std::monostate{};
}

Layer parse_layer(const json &j) {
  std::string type = j.value("type", "text");

  if (type == "text") {
    TextLayer layer;
    layer.content = j.value("content", "");
    layer.style = parse_text_style(j.value("style", json::object()));
    layer.position = parse_position(j.value("position", json::object()));
    layer.animation = parse_animation(j.value("animation", json()));
    return layer;
  }

  if (type == "image") {
    ImageLayer layer;
    layer.src = j.value("src", "");
    layer.position = parse_position(j.value("position", json::object()));
    if (j.contains("size")) {
      layer.width = j["size"].value("width", 1.0);
      layer.height = j["size"].value("height", 1.0);
    }
    return layer;
  }

  throw std::runtime_error("Unknown layer type: " + type);
}

} // anonymous namespace

Template Template::from_json(const std::string &json_str) {
  json j = json::parse(json_str);

  Template tmpl;

  // Canvas
  if (j.contains("canvas")) {
    tmpl.canvas.width = j["canvas"].value("width", 1920u);
    tmpl.canvas.height = j["canvas"].value("height", 1080u);
  }

  // Duration and FPS
  tmpl.duration = j.value("duration", 5.0);
  tmpl.fps = j.value("fps", 60.0);

  // Layers
  if (j.contains("layers")) {
    for (const auto &layer_json : j["layers"]) {
      tmpl.layers.push_back(parse_layer(layer_json));
    }
  }

  return tmpl;
}

std::string Template::to_json() const {
  json j;

  j["canvas"] = {{"width", canvas.width}, {"height", canvas.height}};

  j["duration"] = duration;
  j["fps"] = fps;

  j["layers"] = json::array();
  for (const auto &layer : layers) {
    // TODO: Serialize layers
  }

  return j.dump(2);
}

} // namespace CaptionEngine
