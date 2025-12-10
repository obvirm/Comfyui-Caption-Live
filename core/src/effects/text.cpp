/**
 * @file text.cpp
 * @brief Text rendering effects (bounce, typewriter, colored, karaoke)
 */

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace CaptionEngine {
namespace Effects {

/// Easing functions for smooth animations
namespace Easing {

inline float linear(float t) { return t; }

inline float ease_in_quad(float t) { return t * t; }

inline float ease_out_quad(float t) { return t * (2.0f - t); }

inline float ease_in_out_quad(float t) {
  return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t;
}

inline float ease_out_elastic(float t) {
  if (t == 0.0f || t == 1.0f)
    return t;
  constexpr float p = 0.3f;
  constexpr float s = p / 4.0f;
  constexpr float pi = 3.14159265358979f;
  return std::pow(2.0f, -10.0f * t) * std::sin((t - s) * (2.0f * pi) / p) +
         1.0f;
}

inline float ease_out_bounce(float t) {
  constexpr float n1 = 7.5625f;
  constexpr float d1 = 2.75f;

  if (t < 1.0f / d1) {
    return n1 * t * t;
  } else if (t < 2.0f / d1) {
    t -= 1.5f / d1;
    return n1 * t * t + 0.75f;
  } else if (t < 2.5f / d1) {
    t -= 2.25f / d1;
    return n1 * t * t + 0.9375f;
  } else {
    t -= 2.625f / d1;
    return n1 * t * t + 0.984375f;
  }
}

} // namespace Easing

/**
 * @brief Bounce animation effect
 */
struct BounceEffect {
  float scale = 1.25f;
  float duration = 0.15f;

  float calculate_scale(float segment_progress) const {
    if (segment_progress > duration) {
      return 1.0f;
    }
    float t = segment_progress / duration;
    float bounce = Easing::ease_out_elastic(t);
    return 1.0f + (scale - 1.0f) * (1.0f - bounce);
  }
};

/**
 * @brief Typewriter animation effect
 */
struct TypewriterEffect {
  float chars_per_second = 20.0f;
  bool show_cursor = true;
  char cursor_char = '|';
  float cursor_blink_rate = 0.5f;

  std::string get_visible_text(const std::string &full_text,
                               float segment_progress) const {
    size_t visible_chars =
        static_cast<size_t>(segment_progress * chars_per_second);
    size_t max_chars = full_text.size();

    if (visible_chars > max_chars) {
      visible_chars = max_chars;
    }

    std::string result = full_text.substr(0, visible_chars);

    if (show_cursor && visible_chars < max_chars) {
      float blink =
          std::fmod(segment_progress * cursor_blink_rate * 2.0f, 1.0f);
      if (blink < 0.5f) {
        result += cursor_char;
      }
    }

    return result;
  }
};

/**
 * @brief Colored word animation effect
 */
struct ColoredWordEffect {
  uint32_t active_color = 0xFF5FE539;
  uint32_t inactive_color = 0xFFFFFFFF;
  float fade_duration = 0.1f;

  uint32_t get_word_color(bool is_active, float transition_progress) const {
    if (transition_progress >= fade_duration || is_active) {
      return is_active ? active_color : inactive_color;
    }

    float t = transition_progress / fade_duration;
    t = Easing::ease_out_quad(t);

    auto lerp_channel = [](uint8_t a, uint8_t b, float t_val) -> uint8_t {
      return static_cast<uint8_t>(a + static_cast<float>(b - a) * t_val);
    };

    uint8_t r = lerp_channel((inactive_color >> 0) & 0xFF,
                             (active_color >> 0) & 0xFF, t);
    uint8_t g = lerp_channel((inactive_color >> 8) & 0xFF,
                             (active_color >> 8) & 0xFF, t);
    uint8_t b = lerp_channel((inactive_color >> 16) & 0xFF,
                             (active_color >> 16) & 0xFF, t);
    uint8_t a = lerp_channel((inactive_color >> 24) & 0xFF,
                             (active_color >> 24) & 0xFF, t);

    return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(r);
  }
};

/**
 * @brief Box highlight animation effect
 */
struct BoxHighlightEffect {
  uint32_t box_color = 0xFF5FE539;
  float box_radius = 8.0f;
  float box_padding = 8.0f;
  float animation_duration = 0.2f;

  float get_box_visibility(float segment_progress) const {
    if (segment_progress >= animation_duration) {
      return 1.0f;
    }
    return Easing::ease_out_quad(segment_progress / animation_duration);
  }

  float get_box_scale(float segment_progress) const {
    float visibility = get_box_visibility(segment_progress);
    return 0.8f + 0.2f * visibility;
  }
};

/**
 * @brief Karaoke-style progress effect
 */
struct KaraokeEffect {
  uint32_t sung_color = 0xFF00FFFF;
  uint32_t unsung_color = 0xFFFFFFFF;

  float get_fill_percentage(float word_progress) const {
    if (word_progress < 0.0f)
      return 0.0f;
    if (word_progress > 1.0f)
      return 1.0f;
    return word_progress;
  }
};

/**
 * @brief Wave animation effect
 */
struct WaveEffect {
  float amplitude = 5.0f;
  float frequency = 4.0f;
  float speed = 2.0f;

  float get_y_offset(int char_index, float time) const {
    constexpr float pi = 3.14159265358979f;
    float phase = static_cast<float>(char_index) / frequency + time * speed;
    return std::sin(phase * 2.0f * pi) * amplitude;
  }
};

/**
 * @brief Shake/vibrate effect - simplified without RNG dependency
 */
struct ShakeEffect {
  float intensity = 3.0f;
  float frequency = 30.0f;
  uint64_t seed = 12345;

  std::pair<float, float> get_offset(float time) const {
    // Simple deterministic pseudo-random using seed + time
    uint64_t frame = static_cast<uint64_t>(time * frequency);
    uint64_t hash = (seed ^ frame) * 0x517cc1b727220a95ULL;
    hash ^= hash >> 32;

    // Extract two floats from hash
    float x =
        (static_cast<float>((hash >> 0) & 0xFFFF) / 65535.0f * 2.0f - 1.0f) *
        intensity;
    float y =
        (static_cast<float>((hash >> 16) & 0xFFFF) / 65535.0f * 2.0f - 1.0f) *
        intensity;

    return std::make_pair(x, y);
  }
};

} // namespace Effects
} // namespace CaptionEngine
