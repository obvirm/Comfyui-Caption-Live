#pragma once
/**
 * @file math.hpp
 * @brief Header-only math utilities for compile-time optimization
 *
 * All functions are constexpr or inline to enable compiler optimizations
 * across translation units.
 */

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace CaptionEngine::Math {

// ============================================================================
// Basic Math Utilities
// ============================================================================

/// Constexpr lerp (linear interpolation)
template <typename T> [[nodiscard]] constexpr T lerp(T a, T b, T t) noexcept {
  return a + t * (b - a);
}

/// Constexpr clamp
template <typename T>
[[nodiscard]] constexpr T clamp(T val, T min_val, T max_val) noexcept {
  return val < min_val ? min_val : (val > max_val ? max_val : val);
}

/// Constexpr saturate (clamp to [0, 1])
template <typename T> [[nodiscard]] constexpr T saturate(T val) noexcept {
  return clamp(val, T{0}, T{1});
}

/// Constexpr abs
template <typename T> [[nodiscard]] constexpr T abs(T val) noexcept {
  return val < T{0} ? -val : val;
}

/// Constexpr sign (-1, 0, or 1)
template <typename T> [[nodiscard]] constexpr T sign(T val) noexcept {
  return val < T{0} ? T{-1} : (val > T{0} ? T{1} : T{0});
}

/// Constexpr min
template <typename T> [[nodiscard]] constexpr T min(T a, T b) noexcept {
  return a < b ? a : b;
}

/// Constexpr max
template <typename T> [[nodiscard]] constexpr T max(T a, T b) noexcept {
  return a > b ? a : b;
}

/// Map value from one range to another
template <typename T>
[[nodiscard]] constexpr T map_range(T val, T in_min, T in_max, T out_min,
                                    T out_max) noexcept {
  return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min);
}

// ============================================================================
// Smooth Interpolation Functions
// ============================================================================

/// Smooth step (cubic Hermite interpolation)
[[nodiscard]] constexpr float smoothstep(float edge0, float edge1,
                                         float x) noexcept {
  float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

/// Smoother step (quintic, C2 continuous)
[[nodiscard]] constexpr float smootherstep(float edge0, float edge1,
                                           float x) noexcept {
  float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// ============================================================================
// Easing Functions (for animations)
// ============================================================================

namespace Ease {

// Linear
[[nodiscard]] constexpr float linear(float t) noexcept { return t; }

// Quadratic
[[nodiscard]] constexpr float in_quad(float t) noexcept { return t * t; }

[[nodiscard]] constexpr float out_quad(float t) noexcept {
  return t * (2.0f - t);
}

[[nodiscard]] constexpr float in_out_quad(float t) noexcept {
  return t < 0.5f ? 2.0f * t * t
                  : 1.0f - (-2.0f * t + 2.0f) * (-2.0f * t + 2.0f) / 2.0f;
}

// Cubic
[[nodiscard]] constexpr float in_cubic(float t) noexcept { return t * t * t; }

[[nodiscard]] constexpr float out_cubic(float t) noexcept {
  float f = t - 1.0f;
  return f * f * f + 1.0f;
}

[[nodiscard]] constexpr float in_out_cubic(float t) noexcept {
  if (t < 0.5f) {
    return 4.0f * t * t * t;
  } else {
    float f = 2.0f * t - 2.0f;
    return 0.5f * f * f * f + 1.0f;
  }
}

// Quartic
[[nodiscard]] constexpr float in_quart(float t) noexcept {
  return t * t * t * t;
}

[[nodiscard]] constexpr float out_quart(float t) noexcept {
  float f = t - 1.0f;
  return 1.0f - f * f * f * f;
}

// Exponential
[[nodiscard]] inline float in_expo(float t) noexcept {
  return t == 0.0f ? 0.0f : std::pow(2.0f, 10.0f * (t - 1.0f));
}

[[nodiscard]] inline float out_expo(float t) noexcept {
  return t == 1.0f ? 1.0f : 1.0f - std::pow(2.0f, -10.0f * t);
}

// Elastic
[[nodiscard]] inline float out_elastic(float t) noexcept {
  constexpr float c4 = (2.0f * 3.14159265f) / 3.0f;
  if (t == 0.0f)
    return 0.0f;
  if (t == 1.0f)
    return 1.0f;
  return std::pow(2.0f, -10.0f * t) * std::sin((t * 10.0f - 0.75f) * c4) + 1.0f;
}

// Bounce
[[nodiscard]] inline float out_bounce(float t) noexcept {
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

[[nodiscard]] inline float in_bounce(float t) noexcept {
  return 1.0f - out_bounce(1.0f - t);
}

// Back (overshooting)
[[nodiscard]] constexpr float in_back(float t) noexcept {
  constexpr float c1 = 1.70158f;
  constexpr float c3 = c1 + 1.0f;
  return c3 * t * t * t - c1 * t * t;
}

[[nodiscard]] constexpr float out_back(float t) noexcept {
  constexpr float c1 = 1.70158f;
  constexpr float c3 = c1 + 1.0f;
  float f = t - 1.0f;
  return 1.0f + c3 * f * f * f + c1 * f * f;
}

} // namespace Ease

// ============================================================================
// Integer Math
// ============================================================================

/// Constexpr power of 2 check
[[nodiscard]] constexpr bool is_power_of_2(uint32_t n) noexcept {
  return n != 0 && (n & (n - 1)) == 0;
}

/// Next power of 2
[[nodiscard]] constexpr uint32_t next_power_of_2(uint32_t n) noexcept {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

/// Align value up to alignment
[[nodiscard]] constexpr uint32_t align_up(uint32_t value,
                                          uint32_t alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

/// Divide and round up
[[nodiscard]] constexpr uint32_t div_ceil(uint32_t a, uint32_t b) noexcept {
  return (a + b - 1) / b;
}

// ============================================================================
// 2D Vector (for simple operations)
// ============================================================================

template <typename T> struct Vec2 {
  T x, y;

  constexpr Vec2() noexcept : x{}, y{} {}
  constexpr Vec2(T x_, T y_) noexcept : x(x_), y(y_) {}

  [[nodiscard]] constexpr Vec2 operator+(Vec2 other) const noexcept {
    return {x + other.x, y + other.y};
  }

  [[nodiscard]] constexpr Vec2 operator-(Vec2 other) const noexcept {
    return {x - other.x, y - other.y};
  }

  [[nodiscard]] constexpr Vec2 operator*(T scalar) const noexcept {
    return {x * scalar, y * scalar};
  }

  [[nodiscard]] constexpr T dot(Vec2 other) const noexcept {
    return x * other.x + y * other.y;
  }

  [[nodiscard]] inline T length() const noexcept {
    return std::sqrt(x * x + y * y);
  }

  [[nodiscard]] inline Vec2 normalized() const noexcept {
    T len = length();
    return len > T{0} ? Vec2{x / len, y / len} : Vec2{};
  }
};

using Vec2f = Vec2<float>;
using Vec2i = Vec2<int32_t>;

} // namespace CaptionEngine::Math
