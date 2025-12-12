#pragma once
/**
 * @file consistency_test.hpp
 * @brief WASM-Native consistency testing framework
 *
 * Provides infrastructure for verifying pixel-perfect consistency
 * between WebAssembly and Native builds of CaptionEngine.
 */

#include "gpu/backend.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>


namespace CaptionEngine {
namespace Testing {

// ============================================================================
// Test Case Definitions
// ============================================================================

/**
 * @brief Test case definition for consistency testing
 */
struct TestCase {
  std::string name;                              ///< Unique test name
  std::string template_json;                     ///< Template JSON to render
  std::vector<double> test_times{0.0, 0.5, 1.0}; ///< Times to test at
  uint32_t width = 1920;                         ///< Output width
  uint32_t height = 1080;                        ///< Output height

  /// Default constructor
  TestCase() = default;

  /// Named constructor
  TestCase(std::string name_, std::string json_)
      : name(std::move(name_)), template_json(std::move(json_)) {}
};

/**
 * @brief Result for a single frame comparison
 */
struct FrameResult {
  double time;
  uint64_t expected_hash;
  uint64_t actual_hash;
  bool passed;
  double similarity; ///< 0.0 = no match, 1.0 = perfect match
  std::string error;
};

/**
 * @brief Result for a full test case
 */
struct ConsistencyResult {
  std::string test_name;
  bool passed;
  size_t total_frames;
  size_t matched_frames;
  std::vector<FrameResult> frame_results;
  std::vector<std::string> failures;
  double min_similarity; ///< Worst case similarity

  /// Check if test passed
  explicit operator bool() const noexcept { return passed; }
};

// ============================================================================
// Golden Reference Data Structures
// ============================================================================

/**
 * @brief Frame hash data for golden reference
 */
struct GoldenFrame {
  double time;
  uint64_t pixel_hash;
  uint32_t checksum;
  uint32_t width;
  uint32_t height;
};

/**
 * @brief Test data in golden reference
 */
struct GoldenTest {
  std::string name;
  std::string template_hash; ///< Hash of template JSON for validation
  std::vector<GoldenFrame> frames;
};

/**
 * @brief Complete golden reference file
 */
struct GoldenReference {
  std::string version = "1.0";
  std::string generated_at;
  std::string platform;
  std::string engine_version;
  std::vector<GoldenTest> tests;

  /// Load from JSON file
  static std::optional<GoldenReference> load(const std::string &path);

  /// Save to JSON file
  bool save(const std::string &path) const;
};

// ============================================================================
// Test Runner
// ============================================================================

/**
 * @brief Callback for frame rendering
 *
 * @param template_json Template to render
 * @param time Time value for animation
 * @param width Output width
 * @param height Output height
 * @return RGBA pixel data or empty on failure
 */
using RenderCallback = std::function<std::vector<uint8_t>(
    const std::string &template_json, double time, uint32_t width,
    uint32_t height)>;

/**
 * @brief Progress callback during test execution
 */
using ProgressCallback =
    std::function<void(size_t current, size_t total, const std::string &test)>;

/**
 * @brief Main consistency test runner
 *
 * Manages test execution, golden reference generation, and validation.
 */
class ConsistencyTestRunner {
public:
  /// Constructor with render callback
  explicit ConsistencyTestRunner(RenderCallback render_fn);

  /// Default destructor
  ~ConsistencyTestRunner();

  // --- Test Management ---

  /// Add a test case
  void add_test(const TestCase &test);

  /// Add multiple test cases
  void add_tests(const std::vector<TestCase> &tests);

  /// Clear all test cases
  void clear_tests();

  /// Get all test cases
  [[nodiscard]] const std::vector<TestCase> &tests() const;

  // --- Golden Reference Generation ---

  /**
   * @brief Generate golden reference from current platform
   * @param output_path Path to save golden reference JSON
   * @return Results of generation (all should pass)
   */
  std::vector<ConsistencyResult>
  generate_golden(const std::string &output_path);

  // --- Validation ---

  /**
   * @brief Validate against golden reference
   * @param golden_path Path to golden reference JSON
   * @return Validation results for each test
   */
  std::vector<ConsistencyResult> validate(const std::string &golden_path);

  /**
   * @brief Compare two frame buffers
   * @param frame_a First frame (RGBA)
   * @param frame_b Second frame (RGBA)
   * @param width Frame width
   * @param height Frame height
   * @param tolerance Per-channel difference tolerance (0-255)
   * @return Similarity ratio (0.0 - 1.0)
   */
  static double compare_frames(std::span<const uint8_t> frame_a,
                               std::span<const uint8_t> frame_b, uint32_t width,
                               uint32_t height, uint8_t tolerance = 0);

  // --- Progress Reporting ---

  /// Set progress callback
  void set_progress_callback(ProgressCallback callback);

  // --- Built-in Test Cases ---

  /**
   * @brief Get standard built-in test cases
   * @return Vector of predefined test cases
   */
  static std::vector<TestCase> builtin_tests();

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute FNV-1a hash of pixel data
[[nodiscard]] uint64_t hash_pixels(std::span<const uint8_t> pixels);

/// Compute CRC32 checksum
[[nodiscard]] uint32_t checksum_crc32(std::span<const uint8_t> data);

/// Hash a string (for template validation)
[[nodiscard]] uint64_t hash_string(const std::string &str);

/// Get current timestamp as ISO 8601 string
[[nodiscard]] std::string current_timestamp();

/// Get platform identifier string
[[nodiscard]] std::string platform_string();

} // namespace Testing
} // namespace CaptionEngine
