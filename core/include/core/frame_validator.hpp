/**
 * @file frame_validator.hpp
 * @brief Deterministic frame validation and checksum system
 */

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace CaptionEngine {
namespace Validation {

/**
 * @brief XXHash3-based fast hashing for frame data
 */
class FrameHasher {
public:
  /// Compute 64-bit hash of pixel data
  [[nodiscard]] static uint64_t
  hash_pixels(std::span<const uint8_t> pixels) noexcept;

  /// Compute hash with seed for deterministic variation
  [[nodiscard]] static uint64_t hash_with_seed(std::span<const uint8_t> data,
                                               uint64_t seed) noexcept;

  /// Fast CRC32 checksum
  [[nodiscard]] static uint32_t crc32(std::span<const uint8_t> data) noexcept;

  /// Combine multiple hashes
  [[nodiscard]] static uint64_t
  combine_hashes(std::initializer_list<uint64_t> hashes) noexcept;
};

/**
 * @brief Frame validation result
 */
struct ValidationResult {
  bool passed;
  uint64_t expected_hash;
  uint64_t actual_hash;
  std::vector<std::pair<uint32_t, uint32_t>>
      mismatch_regions; // (offset, length)

  explicit operator bool() const noexcept { return passed; }
};

/**
 * @brief Frame validator for deterministic rendering verification
 */
class FrameValidator {
public:
  struct FrameHash {
    uint64_t data_hash;
    uint64_t metadata_hash;
    uint32_t checksum;
    uint32_t width;
    uint32_t height;
    uint32_t channels;

    bool operator==(const FrameHash &) const = default;
  };

  /// Compute comprehensive frame hash
  [[nodiscard]] FrameHash compute_hash(std::span<const uint8_t> pixels,
                                       uint32_t width, uint32_t height,
                                       uint32_t channels,
                                       double time) const noexcept;

  /// Validate frame against expected hash
  [[nodiscard]] ValidationResult
  validate(std::span<const uint8_t> pixels,
           const FrameHash &expected) const noexcept;

  /// Compare two frames for exact match
  [[nodiscard]] ValidationResult
  compare_frames(std::span<const uint8_t> frame_a,
                 std::span<const uint8_t> frame_b, uint32_t width,
                 uint32_t height) const noexcept;

  /// Generate golden reference hash
  [[nodiscard]] std::vector<FrameHash>
  generate_golden_reference(std::span<const std::span<const uint8_t>> frames,
                            uint32_t width, uint32_t height, double fps) const;
};

/**
 * @brief Golden reference manager for CI/CD testing
 */
class GoldenReferenceManager {
public:
  /// Load golden reference from file
  bool load(const std::string &path);

  /// Save golden reference to file
  bool save(const std::string &path) const;

  /// Add frame hash to reference
  void add_frame(const FrameValidator::FrameHash &hash);

  /// Get frame hash at index
  [[nodiscard]] const FrameValidator::FrameHash *get_frame(size_t index) const;

  /// Get total frame count
  [[nodiscard]] size_t frame_count() const noexcept;

  /// Validate sequence against reference
  [[nodiscard]] std::vector<ValidationResult>
  validate_sequence(const FrameValidator &validator,
                    std::span<const std::span<const uint8_t>> frames) const;

private:
  std::vector<FrameValidator::FrameHash> hashes_;
  std::string template_hash_;
  double duration_ = 0.0;
  double fps_ = 60.0;
};

} // namespace Validation
} // namespace CaptionEngine
