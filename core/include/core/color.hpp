#pragma once
/**
 * @file color.hpp
 * @brief Header-only color utilities for compile-time optimization
 *
 * Provides RGBA color manipulation, hex parsing, and blending functions.
 * All functions are constexpr or inline for compiler optimizations.
 */

#include <cstdint>
#include <string_view>

namespace CaptionEngine::Color {

// ============================================================================
// RGBA Color Structure
// ============================================================================

/**
 * @brief RGBA color with 8-bit components
 *
 * Memory layout is R, G, B, A (matches most GPU texture formats)
 */
struct RGBA {
  uint8_t r, g, b, a;

  /// Default constructor (white, opaque)
  constexpr RGBA() noexcept : r(255), g(255), b(255), a(255) {}

  /// Component constructor
  constexpr RGBA(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_ = 255) noexcept
      : r(r_), g(g_), b(b_), a(a_) {}

  /// Pack to 32-bit value (ABGR format for GPU)
  [[nodiscard]] constexpr uint32_t to_packed_abgr() const noexcept {
    return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(r);
  }

  /// Pack to 32-bit value (RGBA format)
  [[nodiscard]] constexpr uint32_t to_packed_rgba() const noexcept {
    return (static_cast<uint32_t>(r) << 24) | (static_cast<uint32_t>(g) << 16) |
           (static_cast<uint32_t>(b) << 8) | static_cast<uint32_t>(a);
  }

  /// Unpack from ABGR format
  [[nodiscard]] static constexpr RGBA
  from_packed_abgr(uint32_t packed) noexcept {
    return {static_cast<uint8_t>(packed & 0xFF),
            static_cast<uint8_t>((packed >> 8) & 0xFF),
            static_cast<uint8_t>((packed >> 16) & 0xFF),
            static_cast<uint8_t>((packed >> 24) & 0xFF)};
  }

  /// Unpack from RGBA format
  [[nodiscard]] static constexpr RGBA
  from_packed_rgba(uint32_t packed) noexcept {
    return {static_cast<uint8_t>((packed >> 24) & 0xFF),
            static_cast<uint8_t>((packed >> 16) & 0xFF),
            static_cast<uint8_t>((packed >> 8) & 0xFF),
            static_cast<uint8_t>(packed & 0xFF)};
  }

  /// Create from float components [0, 1]
  [[nodiscard]] static constexpr RGBA from_floats(float r_, float g_, float b_,
                                                  float a_ = 1.0f) noexcept {
    auto clamp01 = [](float v) constexpr {
      return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    };
    return {static_cast<uint8_t>(clamp01(r_) * 255.0f),
            static_cast<uint8_t>(clamp01(g_) * 255.0f),
            static_cast<uint8_t>(clamp01(b_) * 255.0f),
            static_cast<uint8_t>(clamp01(a_) * 255.0f)};
  }

  /// Convert to float components [0, 1]
  [[nodiscard]] constexpr void to_floats(float &r_, float &g_, float &b_,
                                         float &a_) const noexcept {
    constexpr float inv255 = 1.0f / 255.0f;
    r_ = r * inv255;
    g_ = g * inv255;
    b_ = b * inv255;
    a_ = a * inv255;
  }

  /// Equality comparison
  [[nodiscard]] constexpr bool operator==(const RGBA &other) const noexcept {
    return r == other.r && g == other.g && b == other.b && a == other.a;
  }
};

// ============================================================================
// Predefined Colors
// ============================================================================

namespace Colors {
constexpr RGBA White{255, 255, 255, 255};
constexpr RGBA Black{0, 0, 0, 255};
constexpr RGBA Transparent{0, 0, 0, 0};
constexpr RGBA Red{255, 0, 0, 255};
constexpr RGBA Green{0, 255, 0, 255};
constexpr RGBA Blue{0, 0, 255, 255};
constexpr RGBA Yellow{255, 255, 0, 255};
constexpr RGBA Cyan{0, 255, 255, 255};
constexpr RGBA Magenta{255, 0, 255, 255};
constexpr RGBA Orange{255, 165, 0, 255};
constexpr RGBA Purple{128, 0, 128, 255};
constexpr RGBA Pink{255, 192, 203, 255};
constexpr RGBA Gray{128, 128, 128, 255};
constexpr RGBA LightGray{192, 192, 192, 255};
constexpr RGBA DarkGray{64, 64, 64, 255};

// TikTok style colors
constexpr RGBA TikTokPink{254, 44, 85, 255};
constexpr RGBA TikTokCyan{37, 244, 238, 255};
constexpr RGBA HighlightGreen{57, 229, 95, 255};
} // namespace Colors

// ============================================================================
// Hex Color Parsing
// ============================================================================

/**
 * @brief Parse hex color string
 * @param hex Color string (#RGB, #RGBA, #RRGGBB, #RRGGBBAA)
 * @return Parsed RGBA color, or white if invalid
 */
[[nodiscard]] inline RGBA parse_hex(std::string_view hex) noexcept {
  if (hex.empty()) {
    return Colors::White;
  }

  // Remove leading '#' if present
  if (hex[0] == '#') {
    hex.remove_prefix(1);
  }

  // Parse hex value
  uint32_t val = 0;
  for (char c : hex) {
    val *= 16;
    if (c >= '0' && c <= '9') {
      val += c - '0';
    } else if (c >= 'a' && c <= 'f') {
      val += c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      val += c - 'A' + 10;
    } else {
      return Colors::White; // Invalid character
    }
  }

  switch (hex.size()) {
  case 3: // #RGB -> #RRGGBB
    return {static_cast<uint8_t>(((val >> 8) & 0xF) * 17),
            static_cast<uint8_t>(((val >> 4) & 0xF) * 17),
            static_cast<uint8_t>((val & 0xF) * 17), 255};

  case 4: // #RGBA -> #RRGGBBAA
    return {static_cast<uint8_t>(((val >> 12) & 0xF) * 17),
            static_cast<uint8_t>(((val >> 8) & 0xF) * 17),
            static_cast<uint8_t>(((val >> 4) & 0xF) * 17),
            static_cast<uint8_t>((val & 0xF) * 17)};

  case 6: // #RRGGBB
    return {static_cast<uint8_t>((val >> 16) & 0xFF),
            static_cast<uint8_t>((val >> 8) & 0xFF),
            static_cast<uint8_t>(val & 0xFF), 255};

  case 8: // #RRGGBBAA
    return {static_cast<uint8_t>((val >> 24) & 0xFF),
            static_cast<uint8_t>((val >> 16) & 0xFF),
            static_cast<uint8_t>((val >> 8) & 0xFF),
            static_cast<uint8_t>(val & 0xFF)};

  default:
    return Colors::White;
  }
}

// ============================================================================
// Color Blending
// ============================================================================

/**
 * @brief Alpha blend (Porter-Duff "over" operator)
 * @param fg Foreground color (on top)
 * @param bg Background color (underneath)
 * @return Blended color
 */
[[nodiscard]] constexpr RGBA blend_over(RGBA fg, RGBA bg) noexcept {
  // Fast path for common cases
  if (fg.a == 255) {
    return fg;
  }
  if (fg.a == 0) {
    return bg;
  }

  // Alpha compositing
  float fa = fg.a / 255.0f;
  float ba = bg.a / 255.0f;
  float oa = fa + ba * (1.0f - fa);

  if (oa < 0.001f) {
    return Colors::Transparent;
  }

  float inv_oa = 1.0f / oa;
  float fg_contrib = fa * inv_oa;
  float bg_contrib = ba * (1.0f - fa) * inv_oa;

  return {static_cast<uint8_t>(fg.r * fg_contrib + bg.r * bg_contrib),
          static_cast<uint8_t>(fg.g * fg_contrib + bg.g * bg_contrib),
          static_cast<uint8_t>(fg.b * fg_contrib + bg.b * bg_contrib),
          static_cast<uint8_t>(oa * 255.0f)};
}

/**
 * @brief Lerp between two colors
 */
[[nodiscard]] constexpr RGBA lerp(RGBA a, RGBA b, float t) noexcept {
  auto lerp_u8 = [](uint8_t a, uint8_t b, float t) constexpr -> uint8_t {
    return static_cast<uint8_t>(a + (b - a) * t);
  };
  return {lerp_u8(a.r, b.r, t), lerp_u8(a.g, b.g, t), lerp_u8(a.b, b.b, t),
          lerp_u8(a.a, b.a, t)};
}

/**
 * @brief Multiply color (darkening)
 */
[[nodiscard]] constexpr RGBA multiply(RGBA a, RGBA b) noexcept {
  return {static_cast<uint8_t>((a.r * b.r) / 255),
          static_cast<uint8_t>((a.g * b.g) / 255),
          static_cast<uint8_t>((a.b * b.b) / 255),
          static_cast<uint8_t>((a.a * b.a) / 255)};
}

/**
 * @brief Screen blend (lightening)
 */
[[nodiscard]] constexpr RGBA screen(RGBA a, RGBA b) noexcept {
  return {static_cast<uint8_t>(255 - ((255 - a.r) * (255 - b.r)) / 255),
          static_cast<uint8_t>(255 - ((255 - a.g) * (255 - b.g)) / 255),
          static_cast<uint8_t>(255 - ((255 - a.b) * (255 - b.b)) / 255),
          static_cast<uint8_t>(255 - ((255 - a.a) * (255 - b.a)) / 255)};
}

/**
 * @brief Apply alpha to color (premultiply)
 */
[[nodiscard]] constexpr RGBA premultiply(RGBA c) noexcept {
  float a = c.a / 255.0f;
  return {static_cast<uint8_t>(c.r * a), static_cast<uint8_t>(c.g * a),
          static_cast<uint8_t>(c.b * a), c.a};
}

/**
 * @brief Set alpha of color
 */
[[nodiscard]] constexpr RGBA with_alpha(RGBA c, uint8_t alpha) noexcept {
  return {c.r, c.g, c.b, alpha};
}

/**
 * @brief Set alpha of color (float version)
 */
[[nodiscard]] constexpr RGBA with_alpha(RGBA c, float alpha) noexcept {
  auto clamped = alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);
  return {c.r, c.g, c.b, static_cast<uint8_t>(clamped * 255.0f)};
}

} // namespace CaptionEngine::Color
