/**
 * @file skia_text_renderer.cpp
 * @brief High-quality text rendering using Google Skia
 *
 * Provides anti-aliased text rendering with proper stroke outlines
 * using Skia's SkCanvas API. Falls back to stb_truetype if Skia unavailable.
 */

#ifdef CE_HAS_SKIA

#include "core/embedded_font.hpp"
#include "engine/renderer.hpp"

// Skia headers - manual source build uses include/core/ prefix
// CMakeLists adds E:/skia to include dirs, so we use include/core/...
#include "include/core/SkCanvas.h"
#include "include/core/SkData.h"
#include "include/core/SkFont.h"
#include "include/core/SkFontMgr.h"
#include "include/core/SkPaint.h"
#include "include/core/SkPath.h"
#include "include/core/SkSurface.h"
#include "include/core/SkTextBlob.h"
#include "include/core/SkTypeface.h"

#ifdef _WIN32
#include "include/ports/SkTypeface_win.h"
#endif

#include <iostream>
#include <memory>
#include <string>

namespace CaptionEngine {

// Global Skia state
static sk_sp<SkTypeface> g_skia_typeface = nullptr;
static bool g_skia_initialized = false;
static std::string g_skia_font_path;

/**
 * Initialize Skia with embedded font or file
 */
bool skia_init_font(const std::string &font_path) {
  if (g_skia_initialized && g_skia_font_path == font_path) {
    return true;
  }

  // ...

  // ...

  sk_sp<SkFontMgr> fontMgr;
#ifdef _WIN32
  fontMgr = SkFontMgr_New_DirectWrite();
#else
  fontMgr = SkFontMgr::RefDefault();
#endif
  if (!fontMgr) {
    std::cerr << "❌ Skia: Failed to get font manager" << std::endl;
    return false;
  }

  if (!font_path.empty()) {
    // Load from file
    g_skia_typeface = fontMgr->makeFromFile(font_path.c_str());
    if (g_skia_typeface) {
      g_skia_initialized = true;
      g_skia_font_path = font_path;
      std::cout << "✅ Skia: Loaded font from file: " << font_path << std::endl;
      return true;
    }
    std::cerr << "⚠️ Skia: Failed to load font from file, using embedded"
              << std::endl;
  }

  // Load embedded font
  sk_sp<SkData> fontData =
      SkData::MakeWithoutCopy(EMBEDDED_FONT, EMBEDDED_FONT_SIZE);
  g_skia_typeface = fontMgr->makeFromData(fontData, 0);

  if (g_skia_typeface) {
    g_skia_initialized = true;
    g_skia_font_path = "";
    std::cout << "✅ Skia: Loaded embedded font (" << EMBEDDED_FONT_SIZE
              << " bytes)" << std::endl;
    return true;
  }

  std::cerr << "❌ Skia: Failed to load any font" << std::endl;
  return false;
}

/**
 * Measure text width using Skia
 */
float skia_measure_text_width(const std::string &text, float font_size) {
  if (!g_skia_initialized && !skia_init_font("")) {
    return text.length() * font_size * 0.6f; // Fallback
  }

  SkFont font(g_skia_typeface, font_size);
  return font.measureText(text.c_str(), text.length(), SkTextEncoding::kUTF8);
}

/**
 * Convert RGBA uint32 to SkColor
 */
static SkColor uint32_to_skcolor(uint32_t color) {
  uint8_t r = (color >> 24) & 0xFF;
  uint8_t g = (color >> 16) & 0xFF;
  uint8_t b = (color >> 8) & 0xFF;
  uint8_t a = color & 0xFF;
  return SkColorSetARGB(a, r, g, b);
}

/**
 * Draw text to ImageBuffer using Skia
 */
void skia_draw_text_to_buffer(ImageBuffer &img, const std::string &text,
                              float x, float y, float font_size,
                              uint32_t color) {
  if (!g_skia_initialized && !skia_init_font("")) {
    std::cerr << "❌ Skia: Cannot draw text (font not initialized)"
              << std::endl;
    return;
  }

  // Create Skia surface from ImageBuffer
  SkImageInfo info = SkImageInfo::Make(
      img.width, img.height, kRGBA_8888_SkColorType, kUnpremul_SkAlphaType);

  sk_sp<SkSurface> surface =
      SkSurfaces::WrapPixels(info, img.data.data(), img.width * 4);
  if (!surface) {
    std::cerr << "❌ Skia: Failed to create surface" << std::endl;
    return;
  }

  SkCanvas *canvas = surface->getCanvas();

  // Setup font
  SkFont font(g_skia_typeface, font_size);
  font.setEdging(SkFont::Edging::kSubpixelAntiAlias);
  font.setSubpixel(true);

  // Setup paint
  SkPaint paint;
  paint.setColor(uint32_to_skcolor(color));
  paint.setAntiAlias(true);

  // Center text
  float textWidth =
      font.measureText(text.c_str(), text.length(), SkTextEncoding::kUTF8);
  float drawX = x - textWidth / 2.0f;

  // Draw text
  canvas->drawString(text.c_str(), drawX, y, font, paint);
}

/**
 * Draw text with stroke using Skia
 * High-quality anti-aliased stroke using SkPaint::kStroke_Style
 */
void skia_draw_text_with_stroke(ImageBuffer &img, const std::string &text,
                                float x, float y, float font_size,
                                uint32_t text_color, uint32_t stroke_color,
                                float stroke_width) {
  if (!g_skia_initialized && !skia_init_font("")) {
    std::cerr << "❌ Skia: Cannot draw text with stroke (font not initialized)"
              << std::endl;
    return;
  }

  // Create Skia surface from ImageBuffer
  SkImageInfo info = SkImageInfo::Make(
      img.width, img.height, kRGBA_8888_SkColorType, kUnpremul_SkAlphaType);

  sk_sp<SkSurface> surface =
      SkSurfaces::WrapPixels(info, img.data.data(), img.width * 4);
  if (!surface) {
    std::cerr << "❌ Skia: Failed to create surface for stroke" << std::endl;
    return;
  }

  SkCanvas *canvas = surface->getCanvas();

  // Setup font with subpixel anti-aliasing
  SkFont font(g_skia_typeface, font_size);
  font.setEdging(SkFont::Edging::kSubpixelAntiAlias);
  font.setSubpixel(true);

  // Center text
  float textWidth =
      font.measureText(text.c_str(), text.length(), SkTextEncoding::kUTF8);
  float drawX = x - textWidth / 2.0f;

  // Create text blob for efficient multi-pass rendering
  sk_sp<SkTextBlob> blob = SkTextBlob::MakeFromString(text.c_str(), font);
  if (!blob) {
    std::cerr << "❌ Skia: Failed to create text blob" << std::endl;
    return;
  }

  // Draw stroke first (behind text)
  SkPaint strokePaint;
  strokePaint.setColor(uint32_to_skcolor(stroke_color));
  strokePaint.setAntiAlias(true);
  strokePaint.setStyle(SkPaint::kStroke_Style);
  strokePaint.setStrokeWidth(
      stroke_width * 2.0f); // Skia stroke is centered, double for visible width
  strokePaint.setStrokeJoin(SkPaint::kRound_Join);
  strokePaint.setStrokeCap(SkPaint::kRound_Cap);

  canvas->drawTextBlob(blob, drawX, y, strokePaint);

  // Draw fill (main text) on top
  SkPaint fillPaint;
  fillPaint.setColor(uint32_to_skcolor(text_color));
  fillPaint.setAntiAlias(true);
  fillPaint.setStyle(SkPaint::kFill_Style);

  canvas->drawTextBlob(blob, drawX, y, fillPaint);
}

// ============================================================================
// Wrapper functions to replace stb_truetype when Skia is available
// ============================================================================

// These will be called from renderer.cpp when CE_HAS_SKIA is defined
extern "C" {

bool skia_load_font_from_file(const char *font_path) {
  return skia_init_font(font_path ? font_path : "");
}

float skia_measure_text(const char *text, float font_size) {
  return skia_measure_text_width(text, font_size);
}

void skia_draw_text(uint8_t *buffer, int width, int height, const char *text,
                    float x, float y, float font_size, uint32_t color) {
  ImageBuffer img;
  img.width = width;
  img.height = height;
  img.channels = 4;
  img.data.assign(buffer, buffer + (width * height * 4));

  skia_draw_text_to_buffer(img, text, x, y, font_size, color);

  // Copy back
  std::memcpy(buffer, img.data.data(), width * height * 4);
}

void skia_draw_text_stroke(uint8_t *buffer, int width, int height,
                           const char *text, float x, float y, float font_size,
                           uint32_t text_color, uint32_t stroke_color,
                           float stroke_width) {
  ImageBuffer img;
  img.width = width;
  img.height = height;
  img.channels = 4;
  img.data.assign(buffer, buffer + (width * height * 4));

  skia_draw_text_with_stroke(img, text, x, y, font_size, text_color,
                             stroke_color, stroke_width);

  // Copy back
  std::memcpy(buffer, img.data.data(), width * height * 4);
}

} // extern "C"

} // namespace CaptionEngine

#endif // CE_HAS_SKIA
