/**
 * @file deterministic.cpp
 * @brief Deterministic core implementation
 */

#include "core/deterministic.hpp"

namespace CaptionEngine::Deterministic {

// --- DeterministicRNG ---

// Helper splitmix64 to initialize state from a single seed
static uint64_t splitmix64(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

// Rotl helper
static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

DeterministicRNG::DeterministicRNG(uint64_t seed) {
  // Seeding xoshiro256** using splitmix64
  state_[0] = splitmix64(seed);
  state_[1] = splitmix64(seed);
  state_[2] = splitmix64(seed);
  state_[3] = splitmix64(seed);
}

uint64_t DeterministicRNG::next() noexcept {
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

float DeterministicRNG::next_float() noexcept {
  // Standard conversion to [0, 1) float
  return (next() >> 11) * 0x1.0p-53;
}

Fixed32 DeterministicRNG::next_fixed() noexcept {
  // Generate 16.16 fixed point directly?
  // Just map uint64 range to [0, 1) in fixed point
  // Fixed32 1.0 is 1 << 16.
  // Random 16 bits.
  return Fixed32::from_raw(next() & 0xFFFF);
}

// --- Hashing ---

uint64_t compute_pixel_hash(std::span<const uint8_t> pixels) {
  // FNV-1a 64-bit hash
  uint64_t hash = 14695981039346656037ULL;
  for (uint8_t byte : pixels) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

} // namespace CaptionEngine::Deterministic
