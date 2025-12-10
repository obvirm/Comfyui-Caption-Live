#pragma once

#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>


namespace CaptionEngine::Deterministic {

// Fixed-point arithmetic for cross-platform consistency (16.16)
class Fixed32 {
public:
  using value_type = int32_t;
  static constexpr int FRACTION_BITS = 16;
  static constexpr float SCALE = static_cast<float>(1 << FRACTION_BITS);

  constexpr Fixed32() noexcept : value_(0) {}
  constexpr explicit Fixed32(float f) noexcept
      : value_(static_cast<value_type>(f * SCALE)) {}
  constexpr explicit Fixed32(int i) noexcept : value_(i << FRACTION_BITS) {}

  // Raw construction
  static constexpr Fixed32 from_raw(value_type v) noexcept {
    Fixed32 result;
    result.value_ = v;
    return result;
  }

  constexpr float to_float() const noexcept {
    return static_cast<float>(value_) / SCALE;
  }

  constexpr Fixed32 operator+(Fixed32 other) const noexcept {
    return Fixed32::from_raw(value_ + other.value_);
  }

  constexpr Fixed32 operator-(Fixed32 other) const noexcept {
    return Fixed32::from_raw(value_ - other.value_);
  }

  constexpr Fixed32 operator*(Fixed32 other) const noexcept {
    int64_t product = static_cast<int64_t>(value_) * other.value_;
    return Fixed32::from_raw(static_cast<value_type>(product >> FRACTION_BITS));
  }

  // Comparison
  constexpr auto operator<=>(const Fixed32 &) const = default;

private:
  value_type value_;
};

// Deterministic random number generator (Xoshiro256**)
class DeterministicRNG {
public:
  explicit DeterministicRNG(uint64_t seed);

  // Next 64-bit integer
  uint64_t next() noexcept;

  // Deterministic float in [0, 1)
  float next_float() noexcept;

  // Deterministic Fixed32 in [0, 1)
  Fixed32 next_fixed() noexcept;

private:
  std::array<uint64_t, 4> state_;
};

// Frame consistency validation
struct FrameHash {
  uint64_t pixel_hash;
  uint64_t metadata_hash;
};

// Compute deterministic hash for frame data
uint64_t compute_pixel_hash(std::span<const uint8_t> pixels);

} // namespace CaptionEngine::Deterministic
