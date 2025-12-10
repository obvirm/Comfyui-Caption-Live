/**
 * @file sdf_generator.cpp
 * @brief SDF generator implementation with FreeType + GPU compute
 */

#include "text/sdf_generator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace CaptionEngine {
namespace Text {

SDFGenerator::SDFGenerator(GPU::Backend *gpu) : gpu_(gpu) {
  // Initialize FreeType
  FT_Error error = FT_Init_FreeType(&ftLibrary_);
  if (error) {
    std::cerr << "❌ Failed to initialize FreeType: " << error << std::endl;
  } else {
    std::cout << "✅ FreeType initialized" << std::endl;
  }

  // Initialize GPU SDF compute pipeline if GPU available
  if (gpu_ && gpu_->isReady()) {
    auto shaderResult = gpu_->createShaderWGSL(
        SDF_COMPUTE_SHADER, GPU::ShaderStage::Compute, "main");

    if (shaderResult) {
      sdfComputeShader_ = std::move(*shaderResult);

      auto pipelineResult =
          gpu_->createComputePipeline(sdfComputeShader_.get());
      if (pipelineResult) {
        sdfPipeline_ = std::move(*pipelineResult);
        std::cout << "✅ GPU SDF compute pipeline created" << std::endl;
      }
    }
  }
}

SDFGenerator::~SDFGenerator() {
  if (ftFace_)
    FT_Done_Face(ftFace_);
  if (ftLibrary_)
    FT_Done_FreeType(ftLibrary_);
}

bool SDFGenerator::loadFont(const uint8_t *data, size_t size) {
  if (!ftLibrary_)
    return false;

  if (ftFace_) {
    FT_Done_Face(ftFace_);
    ftFace_ = nullptr;
  }

  FT_Error error = FT_New_Memory_Face(ftLibrary_, data, size, 0, &ftFace_);
  if (error) {
    std::cerr << "❌ Failed to load font from memory: " << error << std::endl;
    return false;
  }

  std::cout << "✅ Font loaded: " << ftFace_->family_name << " "
            << ftFace_->style_name << std::endl;
  return true;
}

bool SDFGenerator::loadFontFile(const std::string &path) {
  if (!ftLibrary_)
    return false;

  if (ftFace_) {
    FT_Done_Face(ftFace_);
    ftFace_ = nullptr;
  }

  FT_Error error = FT_New_Face(ftLibrary_, path.c_str(), 0, &ftFace_);
  if (error) {
    std::cerr << "❌ Failed to load font file: " << path << std::endl;
    return false;
  }

  std::cout << "✅ Font loaded: " << ftFace_->family_name << std::endl;
  return true;
}

GlyphData SDFGenerator::generateGlyph(uint32_t codepoint,
                                      const SDFParams &params) {
  GlyphData result;
  result.codepoint = codepoint;

  if (!ftFace_)
    return result;

  // Set font size (render at higher res for SDF quality)
  FT_Set_Pixel_Sizes(ftFace_, 0, params.fontSize);

  // Load glyph
  FT_UInt glyphIndex = FT_Get_Char_Index(ftFace_, codepoint);
  if (FT_Load_Glyph(ftFace_, glyphIndex, FT_LOAD_DEFAULT)) {
    return result;
  }

  // Render to bitmap
  if (FT_Render_Glyph(ftFace_->glyph, FT_RENDER_MODE_NORMAL)) {
    return result;
  }

  FT_Bitmap &bitmap = ftFace_->glyph->bitmap;

  // Copy bitmap data
  result.width = bitmap.width;
  result.height = bitmap.rows;
  result.bitmap.resize(bitmap.width * bitmap.rows);

  for (uint32_t y = 0; y < bitmap.rows; y++) {
    for (uint32_t x = 0; x < bitmap.width; x++) {
      result.bitmap[y * bitmap.width + x] = bitmap.buffer[y * bitmap.pitch + x];
    }
  }

  // Metrics
  result.bearingX = ftFace_->glyph->bitmap_left;
  result.bearingY = ftFace_->glyph->bitmap_top;
  result.advance = ftFace_->glyph->advance.x >> 6; // 26.6 fixed point

  return result;
}

std::vector<uint8_t>
SDFGenerator::generateSDFCPU(const std::vector<uint8_t> &bitmap, uint32_t width,
                             uint32_t height, float spread) {
  std::vector<uint8_t> sdf(width * height);

  int searchRadius = static_cast<int>(std::ceil(spread));

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      uint8_t pixel = bitmap[y * width + x];
      bool inside = pixel > 127;

      float minDist = spread;

      // Search for nearest edge
      for (int dy = -searchRadius; dy <= searchRadius; dy++) {
        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
          int sx = static_cast<int>(x) + dx;
          int sy = static_cast<int>(y) + dy;

          if (sx < 0 || sx >= static_cast<int>(width) || sy < 0 ||
              sy >= static_cast<int>(height)) {
            continue;
          }

          uint8_t samplePixel = bitmap[sy * width + sx];
          bool sampleInside = samplePixel > 127;

          if (sampleInside != inside) {
            float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            minDist = std::min(minDist, dist);
          }
        }
      }

      // Normalize to [0, 255]
      float normalized;
      if (inside) {
        normalized = 0.5f + (minDist / spread) * 0.5f;
      } else {
        normalized = 0.5f - (minDist / spread) * 0.5f;
      }

      sdf[y * width + x] =
          static_cast<uint8_t>(std::clamp(normalized, 0.0f, 1.0f) * 255.0f);
    }
  }

  return sdf;
}

std::vector<uint8_t>
SDFGenerator::generateSDFGPU(const std::vector<uint8_t> &bitmap, uint32_t width,
                             uint32_t height, float spread) {
  if (!gpu_ || !sdfPipeline_) {
    // Fall back to CPU
    return generateSDFCPU(bitmap, width, height, spread);
  }

  // TODO: Implement GPU path
  // For now, use CPU fallback
  return generateSDFCPU(bitmap, width, height, spread);
}

SDFGenerator::AtlasResult
SDFGenerator::generateAtlas(const std::string &charset,
                            const SDFParams &params) {
  AtlasResult result;
  result.width = params.atlasSize;
  result.height = params.atlasSize;
  result.pixels.resize(params.atlasSize * params.atlasSize, 0);

  if (!ftFace_)
    return result;

  // Set font size
  FT_Set_Pixel_Sizes(ftFace_, 0, params.fontSize);
  result.lineHeight = static_cast<float>(ftFace_->size->metrics.height >> 6);

  // Pack glyphs into atlas
  uint32_t penX = params.padding;
  uint32_t penY = params.padding;
  uint32_t rowHeight = 0;

  for (char c : charset) {
    GlyphData glyph = generateGlyph(static_cast<uint32_t>(c), params);

    if (glyph.bitmap.empty())
      continue;

    // Add SDF padding
    uint32_t sdfPadding = static_cast<uint32_t>(params.sdfSpread);
    uint32_t glyphWidth = glyph.width + sdfPadding * 2;
    uint32_t glyphHeight = glyph.height + sdfPadding * 2;

    // Check if we need new row
    if (penX + glyphWidth + params.padding > params.atlasSize) {
      penX = params.padding;
      penY += rowHeight + params.padding;
      rowHeight = 0;
    }

    // Check if atlas is full
    if (penY + glyphHeight + params.padding > params.atlasSize) {
      std::cerr << "⚠️ Atlas full, stopping at character: " << c << std::endl;
      break;
    }

    // Generate SDF for this glyph
    // First, pad the bitmap
    std::vector<uint8_t> paddedBitmap(glyphWidth * glyphHeight, 0);
    for (uint32_t y = 0; y < glyph.height; y++) {
      for (uint32_t x = 0; x < glyph.width; x++) {
        paddedBitmap[(y + sdfPadding) * glyphWidth + (x + sdfPadding)] =
            glyph.bitmap[y * glyph.width + x];
      }
    }

    // Generate SDF
    std::vector<uint8_t> sdf =
        params.useGPU ? generateSDFGPU(paddedBitmap, glyphWidth, glyphHeight,
                                       params.sdfSpread)
                      : generateSDFCPU(paddedBitmap, glyphWidth, glyphHeight,
                                       params.sdfSpread);

    // Copy to atlas
    for (uint32_t y = 0; y < glyphHeight; y++) {
      for (uint32_t x = 0; x < glyphWidth; x++) {
        uint32_t atlasX = penX + x;
        uint32_t atlasY = penY + y;
        result.pixels[atlasY * params.atlasSize + atlasX] =
            sdf[y * glyphWidth + x];
      }
    }

    // Store glyph metrics
    GlyphMetrics metrics;
    metrics.atlasX = penX;
    metrics.atlasY = penY;
    metrics.width = glyphWidth;
    metrics.height = glyphHeight;
    metrics.bearingX = glyph.bearingX - static_cast<float>(sdfPadding);
    metrics.bearingY = glyph.bearingY + static_cast<float>(sdfPadding);
    metrics.advance = glyph.advance;

    result.glyphs[static_cast<uint32_t>(c)] = metrics;

    // Advance pen position
    penX += glyphWidth + params.padding;
    rowHeight = std::max(rowHeight, glyphHeight);
  }

  std::cout << "✅ Generated SDF atlas: " << result.glyphs.size() << " glyphs, "
            << result.width << "x" << result.height << std::endl;

  return result;
}

SDFGenerator::AtlasResult
SDFGenerator::generateStandardAtlas(const SDFParams &params) {
  // Standard ASCII + extended characters
  std::string charset;

  // ASCII printable (32-126)
  for (char c = 32; c < 127; c++) {
    charset += c;
  }

  // Common symbols and extended
  charset += "©®™°±²³¼½¾×÷";
  charset += "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ";
  charset += "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ";

  return generateAtlas(charset, params);
}

} // namespace Text
} // namespace CaptionEngine
