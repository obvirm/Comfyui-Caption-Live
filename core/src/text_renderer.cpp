/**
 * @file text_renderer.cpp
 * @brief Text rendering using stb_truetype with EMBEDDED font
 *
 * UNIFIED ENGINE: Same code compiles to WASM and .pyd
 * Font is embedded in binary - no file I/O needed!
 */

#include "core/embedded_font.hpp" // Embedded Roboto-Bold font data
#include "engine/renderer.hpp"


#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace CaptionEngine {

// Font state
static stbtt_fontinfo g_font_info;
static bool g_font_initialized = false;

// Initialize font from EMBEDDED data (guaranteed to work everywhere)
static bool init_font() {
  if (g_font_initialized)
    return true;

  std::cout << "� Initializing EMBEDDED font (" << EMBEDDED_FONT_SIZE
            << " bytes)..." << std::endl;

  // Use embedded font data directly - no file I/O needed!
  if (stbtt_InitFont(&g_font_info, EMBEDDED_FONT, 0)) {
    g_font_initialized = true;
    std::cout << "✅ Embedded font initialized successfully!" << std::endl;
    return true;
  }

  std::cerr << "❌ Failed to initialize embedded font!" << std::endl;
  return false;
}

// Measure text width
float measure_text_width(const std::string &text, float font_size) {
  if (!init_font()) {
    // Fallback: approximate width
    return text.length() * font_size * 0.6f;
  }

  float scale = stbtt_ScaleForPixelHeight(&g_font_info, font_size);
  float width = 0;

  for (size_t i = 0; i < text.length(); ++i) {
    int advance, lsb;
    stbtt_GetCodepointHMetrics(&g_font_info, text[i], &advance, &lsb);
    width += advance * scale;

    // Kerning
    if (i + 1 < text.length()) {
      width +=
          stbtt_GetCodepointKernAdvance(&g_font_info, text[i], text[i + 1]) *
          scale;
    }
  }

  return width;
}

// Blend pixel with alpha compositing
static void blend_pixel(uint8_t *dst, uint8_t r, uint8_t g, uint8_t b,
                        uint8_t a) {
  if (a == 0)
    return;

  float alpha = a / 255.0f;
  float inv_alpha = 1.0f - alpha;

  dst[0] = static_cast<uint8_t>(r * alpha + dst[0] * inv_alpha);
  dst[1] = static_cast<uint8_t>(g * alpha + dst[1] * inv_alpha);
  dst[2] = static_cast<uint8_t>(b * alpha + dst[2] * inv_alpha);
  dst[3] = std::max(dst[3], a);
}

// Draw text to image buffer
void draw_text_to_buffer(ImageBuffer &img, const std::string &text, float x,
                         float y, float font_size, uint32_t color) {
  if (!init_font()) {
    std::cerr << "❌ Skipping text draw (font init failed): " << text
              << std::endl;
    return;
  }

  float scale = stbtt_ScaleForPixelHeight(&g_font_info, font_size);

  int ascent, descent, line_gap;
  stbtt_GetFontVMetrics(&g_font_info, &ascent, &descent, &line_gap);

  float baseline = y + ascent * scale;
  float cursor_x = x - measure_text_width(text, font_size) / 2.0f; // Center

  uint8_t r = (color >> 24) & 0xFF;
  uint8_t g = (color >> 16) & 0xFF;
  uint8_t b = (color >> 8) & 0xFF;
  uint8_t a = color & 0xFF;

  for (size_t i = 0; i < text.length(); ++i) {
    int c = text[i];

    // Get glyph metrics
    int advance, lsb;
    stbtt_GetCodepointHMetrics(&g_font_info, c, &advance, &lsb);

    // Get glyph bitmap
    int x0, y0, x1, y1;
    stbtt_GetCodepointBitmapBox(&g_font_info, c, scale, scale, &x0, &y0, &x1,
                                &y1);

    int glyph_w = x1 - x0;
    int glyph_h = y1 - y0;

    if (glyph_w > 0 && glyph_h > 0) {
      std::vector<uint8_t> bitmap(glyph_w * glyph_h);
      stbtt_MakeCodepointBitmap(&g_font_info, bitmap.data(), glyph_w, glyph_h,
                                glyph_w, scale, scale, c);

      // Blit to image
      int px = static_cast<int>(cursor_x + lsb * scale + x0);
      int py = static_cast<int>(baseline + y0);

      for (int gy = 0; gy < glyph_h; ++gy) {
        for (int gx = 0; gx < glyph_w; ++gx) {
          int img_x = px + gx;
          int img_y = py + gy;

          if (img_x >= 0 && img_x < static_cast<int>(img.width) && img_y >= 0 &&
              img_y < static_cast<int>(img.height)) {

            uint8_t coverage = bitmap[gy * glyph_w + gx];
            uint8_t final_alpha = static_cast<uint8_t>((coverage * a) / 255);

            uint8_t *pixel = img.pixel(img_x, img_y);
            blend_pixel(pixel, r, g, b, final_alpha);
          }
        }
      }
    }

    // Advance cursor
    cursor_x += advance * scale;

    // Kerning
    if (i + 1 < text.length()) {
      cursor_x +=
          stbtt_GetCodepointKernAdvance(&g_font_info, c, text[i + 1]) * scale;
    }
  }
}

} // namespace CaptionEngine
