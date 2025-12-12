#pragma once
/**
 * @file deterministic.hpp
 * @brief Deterministic computing utilities for cross-platform consistency
 *
 * This header provides:
 * - Fixed-point arithmetic (Fixed32) for deterministic floating-point
 * - Xoshiro256** RNG for reproducible random numbers
 * - FNV-1a hashing for frame validation
 *
 * All functions are inline for maximum optimization.
 */

#include "gpu/backend.hpp"
#include <array>
#include <cstdint>

namespace CaptionEngine::Deterministic {

// ============================================================================
// Fixed-Point Arithmetic (16.16)
// ============================================================================

/**
 * @brief Fixed-point number with 16.16 format
 *
 * Provides deterministic arithmetic that produces identical results
 * across all platforms (no floating-point variance).
 */
class Fixed32 {
public:
  using value_type = int32_t;
  static constexpr int FRACTION_BITS = 16;
  static constexpr float SCALE = static_cast<float>(1 << FRACTION_BITS);

  constexpr Fixed32() noexcept : value_(0) {}

  constexpr explicit Fixed32(float f) noexcept
      : value_(static_cast<value_type>(f * SCALE)) {}

  constexpr explicit Fixed32(int i) noexcept : value_(i << FRACTION_BITS) {}

  /// Construct from raw fixed-point value
  [[nodiscard]] static constexpr Fixed32 from_raw(value_type v) noexcept {
    Fixed32 result;
    result.value_ = v;
    return result;
  }

  /// Convert to floating-point
  [[nodiscard]] constexpr float to_float() const noexcept {
    return static_cast<float>(value_) / SCALE;
  }

  /// Get raw value
  [[nodiscard]] constexpr value_type raw() const noexcept { return value_; }

  // Arithmetic operators
  [[nodiscard]] constexpr Fixed32 operator+(Fixed32 other) const noexcept {
    return Fixed32::from_raw(value_ + other.value_);
  }

  [[nodiscard]] constexpr Fixed32 operator-(Fixed32 other) const noexcept {
    return Fixed32::from_raw(value_ - other.value_);
  }

  [[nodiscard]] constexpr Fixed32 operator*(Fixed32 other) const noexcept {
    int64_t product = static_cast<int64_t>(value_) * other.value_;
    return Fixed32::from_raw(static_cast<value_type>(product >> FRACTION_BITS));
  }

  [[nodiscard]] constexpr Fixed32 operator/(Fixed32 other) const noexcept {
    if (other.value_ == 0)
      return Fixed32{};
    int64_t dividend = static_cast<int64_t>(value_) << FRACTION_BITS;
    return Fixed32::from_raw(static_cast<value_type>(dividend / other.value_));
  }

  // Compound assignment
  constexpr Fixed32 &operator+=(Fixed32 other) noexcept {
    value_ += other.value_;
    return *this;
  }

  constexpr Fixed32 &operator-=(Fixed32 other) noexcept {
    value_ -= other.value_;
    return *this;
  }

  // Comparison (C++17 compatible)
  [[nodiscard]] constexpr bool operator==(const Fixed32 &other) const noexcept {
    return value_ == other.value_;
  }
  [[nodiscard]] constexpr bool operator!=(const Fixed32 &other) const noexcept {
    return value_ != other.value_;
  }
  [[nodiscard]] constexpr bool operator<(const Fixed32 &other) const noexcept {
    return value_ < other.value_;
  }
  [[nodiscard]] constexpr bool operator>(const Fixed32 &other) const noexcept {
    return value_ > other.value_;
  }
  [[nodiscard]] constexpr bool operator<=(const Fixed32 &other) const noexcept {
    return value_ <= other.value_;
  }
  [[nodiscard]] constexpr bool operator>=(const Fixed32 &other) const noexcept {
    return value_ >= other.value_;
  }

private:
  value_type value_;
};

// ============================================================================
// Deterministic RNG (Xoshiro256**)
// ============================================================================

/**
 * @brief High-quality, fast PRNG with deterministic output
 *
 * Uses xoshiro256** algorithm which provides:
 * - Period of 2^256 - 1
 * - Excellent statistical quality
 * - Very fast generation
 * - Fully deterministic (same seed = same sequence)
 */
class DeterministicRNG {
public:
  /// Initialize with seed
  explicit DeterministicRNG(uint64_t seed) noexcept {
    // Use splitmix64 to expand single seed to full state
    state_[0] = splitmix64(seed);
    state_[1] = splitmix64(seed);
    state_[2] = splitmix64(seed);
    state_[3] = splitmix64(seed);
  }

  /// Generate next 64-bit random value
  [[nodiscard]] uint64_t next() noexcept {
    const uint64_t result = rotl(state_[1] * 5, 7) * 9;
    const uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);

    return result;
  }

  /// Generate random float in [0, 1)
  [[nodiscard]] float next_float() noexcept {
    // Use upper 53 bits for maximum precision
    return static_cast<float>(next() >> 11) * 0x1.0p-53f;
  }

  /// Generate random double in [0, 1)
  [[nodiscard]] double next_double() noexcept {
    return static_cast<double>(next() >> 11) * 0x1.0p-53;
  }

  /// Generate Fixed32 in [0, 1)
  [[nodiscard]] Fixed32 next_fixed() noexcept {
    // Generate 16-bit fraction (0x0000 to 0xFFFF maps to 0.0 to ~1.0)
    return Fixed32::from_raw(static_cast<int32_t>(next() & 0xFFFF));
  }

  /// Generate random int in [min, max] inclusive
  [[nodiscard]] int32_t next_int(int32_t min_val, int32_t max_val) noexcept {
    if (min_val >= max_val)
      return min_val;
    uint64_t range = static_cast<uint64_t>(max_val - min_val) + 1;
    return min_val + static_cast<int32_t>(next() % range);
  }

  /// Generate random float in [min, max]
  [[nodiscard]] float next_float(float min_val, float max_val) noexcept {
    return min_val + next_float() * (max_val - min_val);
  }

private:
  std::array<uint64_t, 4> state_;

  /// Rotate left helper
  [[nodiscard]] static constexpr uint64_t rotl(uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
  }

  /// Splitmix64 for seed expansion
  [[nodiscard]] static uint64_t splitmix64(uint64_t &x) noexcept {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

// ============================================================================
// Hashing
// ============================================================================

/// Frame hash result for validation
struct FrameHash {
  uint64_t pixel_hash;
  uint64_t metadata_hash;

  [[nodiscard]] constexpr bool
  operator==(const FrameHash &other) const noexcept {
    return pixel_hash == other.pixel_hash &&
           metadata_hash == other.metadata_hash;
  }
  [[nodiscard]] constexpr bool
  operator!=(const FrameHash &other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief FNV-1a 64-bit hash
 *
 * Fast, deterministic hash suitable for frame validation.
 */
[[nodiscard]] inline uint64_t
compute_pixel_hash(std::span<const uint8_t> pixels) noexcept {
  constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
  constexpr uint64_t FNV_PRIME = 1099511628211ULL;

  uint64_t hash = FNV_OFFSET;
  for (uint8_t byte : pixels) {
    hash ^= byte;
    hash *= FNV_PRIME;
  }
  return hash;
}

/**
 * @brief FNV-1a hash with seed
 */
[[nodiscard]] inline uint64_t hash_with_seed(std::span<const uint8_t> data,
                                             uint64_t seed) noexcept {
  uint64_t hash = seed;
  constexpr uint64_t FNV_PRIME = 1099511628211ULL;

  for (uint8_t byte : data) {
    hash ^= byte;
    hash *= FNV_PRIME;
  }
  return hash;
}

/**
 * @brief Combine multiple hashes
 */
[[nodiscard]] constexpr uint64_t combine_hashes(uint64_t h1,
                                                uint64_t h2) noexcept {
  // Use a simple mixing function
  return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

} // namespace CaptionEngine::Deterministic
