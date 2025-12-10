/**
 * @file transitions.cpp
 * @brief Video transition effects between clips
 */

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>


namespace CaptionEngine {
namespace Effects {

/// Helper clamp function
template <typename T> T clamp_value(T val, T min_val, T max_val) {
  if (val < min_val)
    return min_val;
  if (val > max_val)
    return max_val;
  return val;
}

/**
 * @brief Transition type enumeration
 */
enum class TransitionType {
  Fade,
  Dissolve,
  Wipe,
  Slide,
  Zoom,
  Spin,
  Flip,
  Glitch,
  Pixelate,
  Blur
};

/**
 * @brief Transition direction
 */
enum class TransitionDirection { Left, Right, Up, Down, InOut, OutIn };

/**
 * @brief Base transition configuration
 */
struct TransitionConfig {
  TransitionType type = TransitionType::Fade;
  TransitionDirection direction = TransitionDirection::Right;
  float duration = 0.5f;
  float easing = 0.5f;
  float softness = 0.1f;
  float scale = 2.0f;
  float rotation = 180.0f;
  int pixelate_max = 32;
};

/**
 * @brief Transition state calculator
 */
class TransitionCalculator {
public:
  explicit TransitionCalculator(const TransitionConfig &config)
      : config_(config) {}

  float get_progress(float time) const {
    float t = clamp_value(time / config_.duration, 0.0f, 1.0f);
    if (config_.easing > 0) {
      t = apply_easing(t, config_.easing);
    }
    return t;
  }

  float get_blend(float progress) const { return progress; }

  float get_wipe_mask(float u, float v, float progress) const {
    float edge;
    switch (config_.direction) {
    case TransitionDirection::Left:
      edge = 1.0f - u;
      break;
    case TransitionDirection::Right:
      edge = u;
      break;
    case TransitionDirection::Up:
      edge = 1.0f - v;
      break;
    case TransitionDirection::Down:
      edge = v;
      break;
    default:
      edge = u;
    }

    float threshold = progress;
    float soft = config_.softness;

    if (soft > 0) {
      return clamp_value((edge - threshold + soft) / (2 * soft), 0.0f, 1.0f);
    } else {
      return edge < threshold ? 1.0f : 0.0f;
    }
  }

  std::pair<float, float> get_slide_offset(float progress,
                                           bool is_incoming) const {
    float offset = is_incoming ? (1.0f - progress) : progress;
    switch (config_.direction) {
    case TransitionDirection::Left:
      return std::make_pair(is_incoming ? offset : -offset, 0.0f);
    case TransitionDirection::Right:
      return std::make_pair(is_incoming ? -offset : offset, 0.0f);
    case TransitionDirection::Up:
      return std::make_pair(0.0f, is_incoming ? offset : -offset);
    case TransitionDirection::Down:
      return std::make_pair(0.0f, is_incoming ? -offset : offset);
    default:
      return std::make_pair(0.0f, 0.0f);
    }
  }

  float get_zoom_scale(float progress, bool is_incoming) const {
    float max_scale = config_.scale;
    if (config_.direction == TransitionDirection::InOut) {
      if (is_incoming) {
        return 1.0f / (max_scale - (max_scale - 1) * progress);
      } else {
        return 1.0f + (max_scale - 1) * progress;
      }
    } else {
      if (is_incoming) {
        return max_scale - (max_scale - 1) * progress;
      } else {
        return 1.0f / (1.0f + (max_scale - 1) * progress);
      }
    }
  }

  int get_block_size(float progress) const {
    float peak = 1.0f - std::abs(progress - 0.5f) * 2;
    return 1 + static_cast<int>(peak * (config_.pixelate_max - 1));
  }

  float get_blur_strength(float progress) const {
    float peak = 1.0f - std::abs(progress - 0.5f) * 2;
    return peak * 0.1f;
  }

private:
  float apply_easing(float t, float strength) const {
    float cubic = t < 0.5f ? 4 * t * t * t : 1 - std::pow(-2 * t + 2, 3) / 2;
    return t + (cubic - t) * strength;
  }

  TransitionConfig config_;
};

/**
 * @brief Cross-fade transition
 */
struct FadeTransition {
  float operator()(float /*u*/, float /*v*/, float progress) const {
    return progress;
  }
};

/**
 * @brief Dissolve transition with noise
 */
struct DissolveTransition {
  uint64_t seed = 12345;
  float softness = 0.1f;

  float operator()(float u, float v, float progress) const {
    uint32_t x = static_cast<uint32_t>(u * 1000);
    uint32_t y = static_cast<uint32_t>(v * 1000);
    uint32_t hash = x * 374761393 + y * 668265263;
    hash = (hash ^ (hash >> 13)) * 1274126177;
    float noise = static_cast<float>(hash & 0xFFFF) / 65535.0f;

    float threshold = progress;
    return clamp_value((noise - threshold + softness) / (2 * softness), 0.0f,
                       1.0f);
  }
};

/**
 * @brief Circular wipe transition
 */
struct CircleWipe {
  float center_x = 0.5f;
  float center_y = 0.5f;
  float softness = 0.05f;
  bool invert = false;

  float operator()(float u, float v, float progress) const {
    float dx = u - center_x;
    float dy = v - center_y;
    float dist = std::sqrt(dx * dx + dy * dy);

    float max_dist = 0.72f;
    float threshold = progress * max_dist;

    float result;
    if (softness > 0) {
      result = clamp_value((dist - threshold + softness) / (2 * softness), 0.0f,
                           1.0f);
    } else {
      result = dist < threshold ? 1.0f : 0.0f;
    }

    return invert ? 1.0f - result : result;
  }
};

/**
 * @brief Diagonal wipe transition
 */
struct DiagonalWipe {
  float angle = 45.0f;
  float softness = 0.1f;

  float operator()(float u, float v, float progress) const {
    float rad = angle * 3.14159f / 180.0f;
    float nx = std::cos(rad);
    float ny = std::sin(rad);

    float proj = u * nx + v * ny;
    float max_proj = std::abs(nx) + std::abs(ny);

    float threshold = progress * max_proj;

    if (softness > 0) {
      return clamp_value((proj - threshold + softness) / (2 * softness), 0.0f,
                         1.0f);
    } else {
      return proj < threshold ? 1.0f : 0.0f;
    }
  }
};

/**
 * @brief Radial wipe (clock wipe)
 */
struct RadialWipe {
  float center_x = 0.5f;
  float center_y = 0.5f;
  float start_angle = -90.0f;
  bool clockwise = true;
  float softness = 0.02f;

  float operator()(float u, float v, float progress) const {
    float dx = u - center_x;
    float dy = v - center_y;

    float angle_val = std::atan2(dy, dx);

    float start_rad = start_angle * 3.14159f / 180.0f;
    float rel_angle = angle_val - start_rad;
    if (!clockwise)
      rel_angle = -rel_angle;

    float norm_angle = (rel_angle + 3.14159f) / (2 * 3.14159f);
    if (norm_angle < 0)
      norm_angle += 1;
    if (norm_angle > 1)
      norm_angle -= 1;

    float threshold = progress;

    if (softness > 0) {
      return clamp_value((norm_angle - threshold + softness) / (2 * softness),
                         0.0f, 1.0f);
    } else {
      return norm_angle < threshold ? 1.0f : 0.0f;
    }
  }
};

} // namespace Effects
} // namespace CaptionEngine
