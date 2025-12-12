/**
 * @file text_renderer.cpp
 * @brief Text rendering using stb_truetype with EMBEDDED font
 *
 * UNIFIED ENGINE: Same code compiles to WASM and .pyd
 * Font is embedded in binary - no file I/O needed!
 */

#include "core/embedded_font.hpp" // Embedded Roboto-Bold font data
#include "engine/renderer.hpp"
#include "gpu/backend.hpp" // Fix for libc++ rebind_pointer_t


#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace CaptionEngine {

// Font state
static stbtt_fontinfo g_font_info;
static bool g_font_initialized = false;
static std::vector<uint8_t> g_font_data; // Buffer for file-loaded font
static std::string g_current_font_path;  // Currently loaded font path

// Load font from file
bool load_font_from_file(const std::string &font_path) {
  if (font_path.empty()) {
    return false;
  }

  // If same font already loaded, skip
  if (g_font_initialized && g_current_font_path == font_path) {
    return true;
  }

  std::cout << "🔤 Loading font from file: " << font_path << std::endl;

  // Read font file
  FILE *f = fopen(font_path.c_str(), "rb");
  if (!f) {
    std::cerr << "❌ Cannot open font file: " << font_path << std::endl;
    return false;
  }

  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);

  g_font_data.resize(size);
  size_t read = fread(g_font_data.data(), 1, size, f);
  fclose(f);

  if (read != size) {
    std::cerr << "❌ Failed to read font file: " << font_path << std::endl;
    g_font_data.clear();
    return false;
  }

  // Initialize font
  int offset = stbtt_GetFontOffsetForIndex(g_font_data.data(), 0);
  if (offset < 0) {
    std::cerr << "❌ Invalid font file format: " << font_path << std::endl;
    g_font_data.clear();
    return false;
  }

  if (stbtt_InitFont(&g_font_info, g_font_data.data(), offset)) {
    g_font_initialized = true;
    g_current_font_path = font_path;
    std::cout << "✅ Font loaded from file successfully! (" << size << " bytes)"
              << std::endl;
    return true;
  }

  std::cerr << "❌ Failed to initialize font from file!" << std::endl;
  g_font_data.clear();
  return false;
}

// Initialize font from EMBEDDED data (fallback)
static bool init_font() {
  if (g_font_initialized)
    return true;

  std::cout << "🔤 Initializing EMBEDDED font (" << EMBEDDED_FONT_SIZE
            << " bytes)..." << std::endl;

  // Debug: Print first few bytes to verify data
  std::cout << "   First 4 bytes: 0x" << std::hex << (int)EMBEDDED_FONT[0]
            << " 0x" << (int)EMBEDDED_FONT[1] << " 0x" << (int)EMBEDDED_FONT[2]
            << " 0x" << (int)EMBEDDED_FONT[3] << std::dec << std::endl;

  // Get the font offset for index 0 (first font in collection/file)
  int offset = stbtt_GetFontOffsetForIndex(EMBEDDED_FONT, 0);
  std::cout << "   Font offset for index 0: " << offset << std::endl;

  if (offset < 0) {
    std::cerr << "❌ Invalid font data (offset < 0)!" << std::endl;
    return false;
  }

  // Use embedded font data directly - no file I/O needed!
  int result = stbtt_InitFont(&g_font_info, EMBEDDED_FONT, offset);
  std::cout << "   stbtt_InitFont result: " << result << std::endl;

  if (result) {
    g_font_initialized = true;
    g_current_font_path = ""; // embedded font
    std::cout << "✅ Embedded font initialized successfully! (offset=" << offset
              << ")" << std::endl;
    return true;
  }

  std::cerr << "❌ Failed to initialize embedded font!" << std::endl;
  std::cerr << "   EMBEDDED_FONT pointer: " << (void *)EMBEDDED_FONT
            << std::endl;
  std::cerr << "   EMBEDDED_FONT_SIZE: " << EMBEDDED_FONT_SIZE << std::endl;
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
