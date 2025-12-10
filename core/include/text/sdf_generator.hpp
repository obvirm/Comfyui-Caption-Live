#pragma once
/**
 * @file sdf_generator.hpp
 * @brief GPU-accelerated SDF (Signed Distance Field) generation
 *
 * Generates SDF textures from font glyphs for high-quality text rendering.
 */

#include "gpu/backend.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace CaptionEngine {
namespace Text {

/// Glyph data for SDF generation
struct GlyphData {
  uint32_t codepoint;
  std::vector<uint8_t> bitmap; ///< Grayscale bitmap
  uint32_t width, height;
  float bearingX, bearingY;
  float advance;
};

/// SDF generation parameters
struct SDFParams {
  uint32_t fontSize = 64;    ///< Base font size for rendering
  float sdfSpread = 8.0f;    ///< SDF spread in pixels
  uint32_t padding = 4;      ///< Padding between glyphs
  uint32_t atlasSize = 2048; ///< Atlas texture size
  bool useGPU = true;        ///< Use GPU for SDF generation
};

/**
 * @brief SDF Generator using FreeType + GPU compute
 *
 * Pipeline:
 * 1. FreeType renders high-res glyph bitmaps
 * 2. GPU compute shader generates SDF from bitmaps
 * 3. Glyphs are packed into atlas texture
 */
class SDFGenerator {
public:
  SDFGenerator(GPU::Backend *gpu = nullptr);
  ~SDFGenerator();

  /// Load font from memory (embedded font data)
  bool loadFont(const uint8_t *data, size_t size);

  /// Load font from file
  bool loadFontFile(const std::string &path);

  /// Generate SDF for a single glyph
  GlyphData generateGlyph(uint32_t codepoint, const SDFParams &params);

  /// Generate SDF atlas for character range
  struct AtlasResult {
    std::vector<uint8_t> pixels; ///< SDF pixel data (R8)
    uint32_t width, height;
    std::unordered_map<uint32_t, GlyphMetrics> glyphs;
    float lineHeight;
  };

  AtlasResult
  generateAtlas(const std::string &charset, ///< Characters to include
                const SDFParams &params);

  /// Generate atlas for common ASCII + extended
  AtlasResult generateStandardAtlas(const SDFParams &params);

private:
  /// CPU SDF generation (fallback)
  std::vector<uint8_t> generateSDFCPU(const std::vector<uint8_t> &bitmap,
                                      uint32_t width, uint32_t height,
                                      float spread);

  /// GPU SDF generation (fast)
  std::vector<uint8_t> generateSDFGPU(const std::vector<uint8_t> &bitmap,
                                      uint32_t width, uint32_t height,
                                      float spread);

  GPU::Backend *gpu_;
  FT_Library ftLibrary_ = nullptr;
  FT_Face ftFace_ = nullptr;

  // GPU resources for SDF compute
  std::unique_ptr<GPU::Shader> sdfComputeShader_;
  std::unique_ptr<GPU::Pipeline> sdfPipeline_;
};

// ============================================================================
// SDF Compute Shader (WGSL source)
// ============================================================================

inline const char *SDF_COMPUTE_SHADER = R"(
// SDF Generation Compute Shader
// Input: Grayscale glyph bitmap
// Output: Signed Distance Field

struct SDFParams {
    width: u32,
    height: u32,
    spread: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> params: SDFParams;
@group(0) @binding(1) var inputBitmap: texture_2d<f32>;
@group(0) @binding(2) var outputSDF: texture_storage_2d<r8unorm, write>;

// Calculate distance to nearest edge
fn findNearestEdge(coord: vec2<i32>, inside: bool) -> f32 {
    let maxDist = params.spread;
    var minDist = maxDist;
    
    let searchRadius = i32(ceil(maxDist));
    
    for (var dy = -searchRadius; dy <= searchRadius; dy++) {
        for (var dx = -searchRadius; dx <= searchRadius; dx++) {
            let sampleCoord = coord + vec2<i32>(dx, dy);
            
            // Bounds check
            if (sampleCoord.x < 0 || sampleCoord.x >= i32(params.width) ||
                sampleCoord.y < 0 || sampleCoord.y >= i32(params.height)) {
                continue;
            }
            
            let sampleValue = textureLoad(inputBitmap, sampleCoord, 0).r;
            let sampleInside = sampleValue > 0.5;
            
            // Found an edge (different from current pixel)
            if (sampleInside != inside) {
                let dist = length(vec2<f32>(f32(dx), f32(dy)));
                minDist = min(minDist, dist);
            }
        }
    }
    
    return minDist;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(gid.xy);
    
    if (u32(coord.x) >= params.width || u32(coord.y) >= params.height) {
        return;
    }
    
    let pixelValue = textureLoad(inputBitmap, coord, 0).r;
    let inside = pixelValue > 0.5;
    
    let dist = findNearestEdge(coord, inside);
    
    // Normalize to [0, 1] range
    // 0.5 = edge, < 0.5 = outside, > 0.5 = inside
    var sdfValue: f32;
    if (inside) {
        sdfValue = 0.5 + (dist / params.spread) * 0.5;
    } else {
        sdfValue = 0.5 - (dist / params.spread) * 0.5;
    }
    
    sdfValue = clamp(sdfValue, 0.0, 1.0);
    
    textureStore(outputSDF, coord, vec4<f32>(sdfValue, 0.0, 0.0, 1.0));
}
)";

} // namespace Text
} // namespace CaptionEngine
