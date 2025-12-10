#pragma once
/**
 * @file texture.hpp
 * @brief Texture management and atlas utilities
 */

#include "graphics/api.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace CaptionEngine::Graphics {

/// Rectangle in pixel coordinates
struct Rect {
  int32_t x = 0;
  int32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

/// UV rectangle in normalized coordinates [0,1]
struct UVRect {
  float u0 = 0.0f;
  float v0 = 0.0f;
  float u1 = 1.0f;
  float v1 = 1.0f;
};

/// Image data container
struct ImageData {
  std::vector<uint8_t> pixels; // RGBA format
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 4;

  /// Create from raw pointer (copies data)
  static ImageData from_raw(const uint8_t *data, uint32_t w, uint32_t h,
                            uint32_t channels = 4);

  /// Load from file path
  static std::optional<ImageData> load_file(const std::string &path);

  /// Load from memory (PNG, JPEG, etc.)
  static std::optional<ImageData> load_memory(std::span<const uint8_t> data);

  /// Save to PNG file
  bool save_png(const std::string &path) const;

  /// Resize image (bilinear interpolation)
  [[nodiscard]] ImageData resize(uint32_t new_width, uint32_t new_height) const;

  /// Extract sub-image
  [[nodiscard]] ImageData sub_image(const Rect &rect) const;

  /// Flip vertically
  void flip_y();

  /// Premultiply alpha
  void premultiply_alpha();
};

/**
 * @brief Texture atlas for efficient batching
 *
 * Packs multiple small textures into one large texture
 * for single-draw-call rendering.
 */
class TextureAtlas {
public:
  /// Atlas configuration
  struct Config {
    uint32_t width = 4096;
    uint32_t height = 4096;
    uint32_t padding = 2; // Pixels between entries
    bool power_of_two = true;
  };

  explicit TextureAtlas(Config config = {});
  ~TextureAtlas();

  /// Add image to atlas, returns region ID
  [[nodiscard]] std::optional<uint32_t> add(const std::string &name,
                                            const ImageData &image);

  /// Get UV rect for a named region
  [[nodiscard]] std::optional<UVRect> get_uv(const std::string &name) const;

  /// Get UV rect by ID
  [[nodiscard]] std::optional<UVRect> get_uv(uint32_t id) const;

  /// Build atlas texture (call after adding all images)
  [[nodiscard]] ImageData build() const;

  /// Get number of regions
  [[nodiscard]] size_t region_count() const;

  /// Check if name exists
  [[nodiscard]] bool contains(const std::string &name) const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Texture cache with LRU eviction
 */
class TextureCache {
public:
  explicit TextureCache(size_t max_memory_mb = 256);
  ~TextureCache();

  /// Get or create texture from image data
  [[nodiscard]] TextureHandle get_or_create(GraphicsAPI &api,
                                            const std::string &key,
                                            const ImageData &data);

  /// Get existing texture (nullptr if not cached)
  [[nodiscard]] std::optional<TextureHandle> get(const std::string &key) const;

  /// Remove texture from cache
  void remove(const std::string &key);

  /// Clear all cached textures
  void clear();

  /// Get current memory usage in bytes
  [[nodiscard]] size_t memory_usage() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace CaptionEngine::Graphics
