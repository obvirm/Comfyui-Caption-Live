/**
 * @file renderer.cpp
 * @brief Core rendering implementation
 */

#include "engine/renderer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace CaptionEngine {

// Forward declaration for text renderer (defined in text_renderer.cpp)
void draw_text_to_buffer(ImageBuffer &img, const std::string &text, float x,
                         float y, float font_size, uint32_t color);

// Helper: Parse hex color to RGBA
static uint32_t parse_hex_color(const std::string &hex) {
  if (hex.empty())
    return 0xFFFFFFFF;

  std::string h = hex;
  if (h[0] == '#')
    h = h.substr(1);

  uint32_t r = 255, g = 255, b = 255, a = 255;

  if (h.length() >= 6) {
    r = std::stoul(h.substr(0, 2), nullptr, 16);
    g = std::stoul(h.substr(2, 2), nullptr, 16);
    b = std::stoul(h.substr(4, 2), nullptr, 16);
  }
  if (h.length() >= 8) {
    a = std::stoul(h.substr(6, 2), nullptr, 16);
  }

  return (r << 24) | (g << 16) | (b << 8) | a;
}

// Helper: Find active segment for given time
static std::optional<std::pair<size_t, double>>
find_active_segment(const std::vector<Segment> &segments, double time) {
  for (size_t i = 0; i < segments.size(); ++i) {
    if (time >= segments[i].start && time < segments[i].end) {
      double progress =
          (time - segments[i].start) / (segments[i].end - segments[i].start);
      return std::make_pair(i, progress);
    }
  }
  return std::nullopt;
}

// ImageBuffer implementation
ImageBuffer ImageBuffer::create(uint32_t w, uint32_t h) {
  ImageBuffer buf;
  buf.width = w;
  buf.height = h;
  buf.channels = 4;
  buf.data.resize(w * h * 4, 0);
  return buf;
}

uint8_t *ImageBuffer::pixel(uint32_t x, uint32_t y) {
  if (x >= width || y >= height)
    return nullptr;
  return &data[(y * width + x) * channels];
}

const uint8_t *ImageBuffer::pixel(uint32_t x, uint32_t y) const {
  if (x >= width || y >= height)
    return nullptr;
  return &data[(y * width + x) * channels];
}

void ImageBuffer::fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  for (size_t i = 0; i < data.size(); i += 4) {
    data[i + 0] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = a;
  }
}

void ImageBuffer::clear() { std::fill(data.begin(), data.end(), 0); }

// Renderer implementation
struct Renderer::Impl {
  // Font data would be loaded here
};

Renderer::Renderer() : pimpl_(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;

ImageBuffer Renderer::render_frame(const Template &tmpl, double time) {
  ImageBuffer img = ImageBuffer::create(tmpl.canvas.width, tmpl.canvas.height);
  img.clear();

  for (const auto &layer : tmpl.layers) {
    if (auto *text = std::get_if<TextLayer>(&layer)) {
      render_text_layer(img, *text, time);
    } else if (auto *image = std::get_if<ImageLayer>(&layer)) {
      render_image_layer(img, *image, time);
    }
  }

  return img;
}

ImageBuffer Renderer::render_frame(const Template &tmpl, double time,
                                   const ImageBuffer &input) {
  ImageBuffer img = input;

  for (const auto &layer : tmpl.layers) {
    if (auto *text = std::get_if<TextLayer>(&layer)) {
      render_text_layer(img, *text, time);
    } else if (auto *image = std::get_if<ImageLayer>(&layer)) {
      render_image_layer(img, *image, time);
    }
  }

  return img;
}

void Renderer::render_text_layer(ImageBuffer &img, const TextLayer &layer,
                                 double time) {
  EffectState state = calculate_effect(layer.animation, time);

  uint32_t text_color = parse_hex_color(layer.style.color);
  uint32_t stroke_color = layer.style.stroke_color
                              ? parse_hex_color(*layer.style.stroke_color)
                              : 0x000000FF;

  float x = static_cast<float>(layer.position.x * img.width);
  float y = static_cast<float>(layer.position.y * img.height);

  // Handle box highlight
  if (auto *box_anim = std::get_if<BoxHighlightAnimation>(&layer.animation)) {
    if (state.show_box && state.active_segment) {
      uint32_t box_color = parse_hex_color(box_anim->box_color);
      std::cout << "🎨 Box color: " << box_anim->box_color << " -> 0x"
                << std::hex << box_color << std::dec << std::endl;
      float word_width = layer.style.font_size * 3;
      float word_height = layer.style.font_size;

      int box_x = static_cast<int>(x - word_width / 2 - box_anim->box_padding);
      int box_y = static_cast<int>(y - word_height / 2 - box_anim->box_padding);

      draw_rounded_rect(
          img, box_x, box_y,
          static_cast<uint32_t>(word_width + box_anim->box_padding * 2),
          static_cast<uint32_t>(word_height + box_anim->box_padding * 2),
          box_anim->box_radius, box_color);
    }
  }

  // Adjust Y to center text vertically (draw_text expects top-left, we have
  // center)
  float text_draw_y = y - layer.style.font_size / 2.0f;

  if (layer.style.stroke_width > 0) {
    draw_text_with_stroke(img, layer.content, x, text_draw_y,
                          layer.style.font_size, text_color, stroke_color,
                          layer.style.stroke_width);
  } else {
    draw_text(img, layer.content, x, text_draw_y, layer.style.font_size,
              text_color);
  }
}

void Renderer::render_image_layer(ImageBuffer & /*img*/,
                                  const ImageLayer & /*layer*/,
                                  double /*time*/) {
  // TODO: Implement
}

void Renderer::draw_text(ImageBuffer &img, const std::string &text, float x,
                         float y, float font_size, uint32_t color) {
  draw_text_to_buffer(img, text, x, y, font_size, color);
}

void Renderer::draw_text_with_stroke(ImageBuffer &img, const std::string &text,
                                     float x, float y, float font_size,
                                     uint32_t text_color, uint32_t stroke_color,
                                     float stroke_width) {
  int radius = static_cast<int>(std::ceil(stroke_width));

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      if (dx * dx + dy * dy > radius * radius)
        continue;
      draw_text(img, text, x + dx, y + dy, font_size, stroke_color);
    }
  }

  draw_text(img, text, x, y, font_size, text_color);
}

void Renderer::draw_rounded_rect(ImageBuffer &img, int x, int y, uint32_t w,
                                 uint32_t h, float radius, uint32_t color) {
  int r = static_cast<int>(std::min({radius, w / 2.0f, h / 2.0f}));

  uint8_t cr = (color >> 24) & 0xFF;
  uint8_t cg = (color >> 16) & 0xFF;
  uint8_t cb = (color >> 8) & 0xFF;
  uint8_t ca = color & 0xFF;

  for (uint32_t dy = 0; dy < h; ++dy) {
    for (uint32_t dx = 0; dx < w; ++dx) {
      int px = x + static_cast<int>(dx);
      int py = y + static_cast<int>(dy);

      if (px < 0 || py < 0 || px >= static_cast<int>(img.width) ||
          py >= static_cast<int>(img.height)) {
        continue;
      }

      bool in_rect = true;

      if (dx < static_cast<uint32_t>(r) && dy < static_cast<uint32_t>(r)) {
        int cx = r - static_cast<int>(dx);
        int cy = r - static_cast<int>(dy);
        in_rect = (cx * cx + cy * cy <= r * r);
      } else if (dx >= w - r && dy < static_cast<uint32_t>(r)) {
        int cx = static_cast<int>(dx) - static_cast<int>(w - r - 1);
        int cy = r - static_cast<int>(dy);
        in_rect = (cx * cx + cy * cy <= r * r);
      } else if (dx < static_cast<uint32_t>(r) && dy >= h - r) {
        int cx = r - static_cast<int>(dx);
        int cy = static_cast<int>(dy) - static_cast<int>(h - r - 1);
        in_rect = (cx * cx + cy * cy <= r * r);
      } else if (dx >= w - r && dy >= h - r) {
        int cx = static_cast<int>(dx) - static_cast<int>(w - r - 1);
        int cy = static_cast<int>(dy) - static_cast<int>(h - r - 1);
        in_rect = (cx * cx + cy * cy <= r * r);
      }

      if (in_rect) {
        uint8_t *pixel = img.pixel(px, py);
        if (pixel) {
          float alpha = ca / 255.0f;
          pixel[0] = static_cast<uint8_t>(cr * alpha + pixel[0] * (1 - alpha));
          pixel[1] = static_cast<uint8_t>(cg * alpha + pixel[1] * (1 - alpha));
          pixel[2] = static_cast<uint8_t>(cb * alpha + pixel[2] * (1 - alpha));
          pixel[3] = std::max(pixel[3], ca);
        }
      }
    }
  }
}

EffectState Renderer::calculate_effect(const Animation &anim, double time) {
  EffectState state;

  if (auto *box = std::get_if<BoxHighlightAnimation>(&anim)) {
    auto active = find_active_segment(box->segments, time);
    if (active) {
      state.active_segment = active->first;
      state.segment_progress = active->second;
      state.show_box = true;

      if (active->second < 0.1) {
        float t = static_cast<float>(active->second / 0.1);
        state.scale = 0.8f + 0.2f * t;
      }
    }
  } else if (auto *typewriter = std::get_if<TypewriterAnimation>(&anim)) {
    double elapsed = time;
    int chars = static_cast<int>(elapsed * typewriter->chars_per_second);
    (void)chars; // TODO: Set visible_text
  } else if (auto *bounce = std::get_if<BounceAnimation>(&anim)) {
    auto active = find_active_segment(bounce->segments, time);
    if (active && active->second < bounce->duration) {
      float t = static_cast<float>(active->second / bounce->duration);
      state.scale = 1.0f + (bounce->scale - 1.0f) * std::sin(t * 3.14159f);
    }
  } else if (auto *colored = std::get_if<ColoredAnimation>(&anim)) {
    auto active = find_active_segment(colored->segments, time);
    if (active) {
      state.active_segment = active->first;
      state.segment_progress = active->second;
    }
  }

  return state;
}

} // namespace CaptionEngine
