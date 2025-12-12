/**
 * @file consistency_test.cpp
 * @brief WASM-Native consistency testing implementation
 */

#include "testing/consistency_test.hpp"
#include "core/deterministic.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

namespace CaptionEngine {
namespace Testing {

// ============================================================================
// Utility Functions Implementation
// ============================================================================

uint64_t hash_pixels(std::span<const uint8_t> pixels) {
  // FNV-1a 64-bit hash
  constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
  constexpr uint64_t FNV_PRIME = 1099511628211ULL;

  uint64_t hash = FNV_OFFSET;
  for (uint8_t byte : pixels) {
    hash ^= byte;
    hash *= FNV_PRIME;
  }
  return hash;
}

uint32_t checksum_crc32(std::span<const uint8_t> data) {
  // CRC32 lookup table
  static constexpr auto crc32_table = []() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t crc = i;
      for (int j = 0; j < 8; ++j) {
        crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
      }
      table[i] = crc;
    }
    return table;
  }();

  uint32_t crc = 0xFFFFFFFF;
  for (uint8_t byte : data) {
    crc = crc32_table[(crc ^ byte) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFF;
}

uint64_t hash_string(const std::string &str) {
  return hash_pixels(std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(str.data()), str.size()));
}

std::string current_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
#ifdef _WIN32
  localtime_s(&tm_buf, &time);
#else
  localtime_r(&time, &tm_buf);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}

std::string platform_string() {
#if defined(_WIN32)
  return "Windows/MSVC";
#elif defined(__APPLE__)
  return "macOS/Clang";
#elif defined(__linux__)
  return "Linux/GCC";
#elif defined(__EMSCRIPTEN__)
  return "WASM/Emscripten";
#else
  return "Unknown";
#endif
}

// ============================================================================
// GoldenReference Implementation
// ============================================================================

std::optional<GoldenReference> GoldenReference::load(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::nullopt;
  }

  try {
    json j = json::parse(file);

    GoldenReference ref;
    ref.version = j.value("version", "1.0");
    ref.generated_at = j.value("generated_at", "");
    ref.platform = j.value("platform", "");
    ref.engine_version = j.value("engine_version", "");

    for (const auto &test_json : j["tests"]) {
      GoldenTest test;
      test.name = test_json.value("name", "");
      test.template_hash = test_json.value("template_hash", "");

      for (const auto &frame_json : test_json["frames"]) {
        GoldenFrame frame;
        frame.time = frame_json.value("time", 0.0);
        // Parse hex string to uint64
        std::string hash_str = frame_json.value("pixel_hash", "0");
        frame.pixel_hash = std::stoull(hash_str, nullptr, 16);
        frame.checksum = frame_json.value("checksum", 0u);
        frame.width = frame_json.value("width", 1920u);
        frame.height = frame_json.value("height", 1080u);
        test.frames.push_back(frame);
      }
      ref.tests.push_back(std::move(test));
    }

    return ref;
  } catch (const std::exception &e) {
    std::cerr << "Error loading golden reference: " << e.what() << std::endl;
    return std::nullopt;
  }
}

bool GoldenReference::save(const std::string &path) const {
  try {
    json j;
    j["version"] = version;
    j["generated_at"] = generated_at;
    j["platform"] = platform;
    j["engine_version"] = engine_version;

    j["tests"] = json::array();
    for (const auto &test : tests) {
      json test_json;
      test_json["name"] = test.name;
      test_json["template_hash"] = test.template_hash;
      test_json["frames"] = json::array();

      for (const auto &frame : test.frames) {
        json frame_json;
        frame_json["time"] = frame.time;
        // Store hash as hex string
        std::ostringstream oss;
        oss << std::hex << frame.pixel_hash;
        frame_json["pixel_hash"] = oss.str();
        frame_json["checksum"] = frame.checksum;
        frame_json["width"] = frame.width;
        frame_json["height"] = frame.height;
        test_json["frames"].push_back(frame_json);
      }
      j["tests"].push_back(test_json);
    }

    std::ofstream file(path);
    if (!file.is_open()) {
      return false;
    }
    file << j.dump(2);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error saving golden reference: " << e.what() << std::endl;
    return false;
  }
}

// ============================================================================
// ConsistencyTestRunner Implementation
// ============================================================================

struct ConsistencyTestRunner::Impl {
  RenderCallback render_fn;
  ProgressCallback progress_fn;
  std::vector<TestCase> tests;
};

ConsistencyTestRunner::ConsistencyTestRunner(RenderCallback render_fn)
    : pimpl_(std::make_unique<Impl>()) {
  pimpl_->render_fn = std::move(render_fn);
}

ConsistencyTestRunner::~ConsistencyTestRunner() = default;

void ConsistencyTestRunner::add_test(const TestCase &test) {
  pimpl_->tests.push_back(test);
}

void ConsistencyTestRunner::add_tests(const std::vector<TestCase> &tests) {
  pimpl_->tests.insert(pimpl_->tests.end(), tests.begin(), tests.end());
}

void ConsistencyTestRunner::clear_tests() { pimpl_->tests.clear(); }

const std::vector<TestCase> &ConsistencyTestRunner::tests() const {
  return pimpl_->tests;
}

void ConsistencyTestRunner::set_progress_callback(ProgressCallback callback) {
  pimpl_->progress_fn = std::move(callback);
}

double ConsistencyTestRunner::compare_frames(std::span<const uint8_t> frame_a,
                                             std::span<const uint8_t> frame_b,
                                             uint32_t width, uint32_t height,
                                             uint8_t tolerance) {
  if (frame_a.size() != frame_b.size()) {
    return 0.0;
  }

  size_t total_pixels = width * height;
  size_t matching_pixels = 0;

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t offset = i * 4;
    bool matches = true;

    for (size_t c = 0; c < 4; ++c) {
      int diff = std::abs(static_cast<int>(frame_a[offset + c]) -
                          static_cast<int>(frame_b[offset + c]));
      if (diff > tolerance) {
        matches = false;
        break;
      }
    }

    if (matches) {
      ++matching_pixels;
    }
  }

  return static_cast<double>(matching_pixels) /
         static_cast<double>(total_pixels);
}

std::vector<ConsistencyResult>
ConsistencyTestRunner::generate_golden(const std::string &output_path) {
  std::vector<ConsistencyResult> results;
  GoldenReference golden;

  golden.generated_at = current_timestamp();
  golden.platform = platform_string();
  golden.engine_version = "2.0.0";

  size_t total_tests = pimpl_->tests.size();
  size_t current_test = 0;

  for (const auto &test : pimpl_->tests) {
    ++current_test;
    if (pimpl_->progress_fn) {
      pimpl_->progress_fn(current_test, total_tests, test.name);
    }

    ConsistencyResult result;
    result.test_name = test.name;
    result.total_frames = test.test_times.size();
    result.matched_frames = 0;
    result.passed = true;
    result.min_similarity = 1.0;

    GoldenTest golden_test;
    golden_test.name = test.name;

    // Hash template for validation
    std::ostringstream oss;
    oss << std::hex << hash_string(test.template_json);
    golden_test.template_hash = oss.str();

    for (double time : test.test_times) {
      FrameResult frame_result;
      frame_result.time = time;
      frame_result.passed = false;
      frame_result.similarity = 0.0;

      // Render frame
      auto pixels =
          pimpl_->render_fn(test.template_json, time, test.width, test.height);

      if (pixels.empty()) {
        frame_result.error = "Render failed";
        result.failures.push_back("Frame at t=" + std::to_string(time) +
                                  ": render failed");
        result.passed = false;
      } else {
        // Compute hash
        uint64_t pixel_hash = hash_pixels(pixels.data(), pixels.size());
        uint32_t checksum = checksum_crc32(pixels.data(), pixels.size());

        GoldenFrame golden_frame;
        golden_frame.time = time;
        golden_frame.pixel_hash = pixel_hash;
        golden_frame.checksum = checksum;
        golden_frame.width = test.width;
        golden_frame.height = test.height;
        golden_test.frames.push_back(golden_frame);

        frame_result.actual_hash = pixel_hash;
        frame_result.expected_hash = pixel_hash; // Same during generation
        frame_result.passed = true;
        frame_result.similarity = 1.0;
        ++result.matched_frames;
      }

      result.frame_results.push_back(frame_result);
    }

    golden.tests.push_back(std::move(golden_test));
    results.push_back(std::move(result));
  }

  // Save golden reference
  if (!golden.save(output_path)) {
    std::cerr << "Failed to save golden reference to: " << output_path
              << std::endl;
  }

  return results;
}

std::vector<ConsistencyResult>
ConsistencyTestRunner::validate(const std::string &golden_path) {
  std::vector<ConsistencyResult> results;

  auto golden_opt = GoldenReference::load(golden_path);
  if (!golden_opt) {
    ConsistencyResult err;
    err.test_name = "LOAD_ERROR";
    err.passed = false;
    err.failures.push_back("Failed to load golden reference: " + golden_path);
    results.push_back(err);
    return results;
  }

  const auto &golden = *golden_opt;

  std::cout << "Validating against golden reference:" << std::endl;
  std::cout << "  Generated: " << golden.generated_at << std::endl;
  std::cout << "  Platform:  " << golden.platform << std::endl;
  std::cout << "  Tests:     " << golden.tests.size() << std::endl;

  size_t total_tests = golden.tests.size();
  size_t current_test = 0;

  for (const auto &golden_test : golden.tests) {
    ++current_test;
    if (pimpl_->progress_fn) {
      pimpl_->progress_fn(current_test, total_tests, golden_test.name);
    }

    // Find matching test case
    const TestCase *test_case = nullptr;
    for (const auto &t : pimpl_->tests) {
      if (t.name == golden_test.name) {
        test_case = &t;
        break;
      }
    }

    ConsistencyResult result;
    result.test_name = golden_test.name;
    result.total_frames = golden_test.frames.size();
    result.matched_frames = 0;
    result.passed = true;
    result.min_similarity = 1.0;

    if (!test_case) {
      result.passed = false;
      result.failures.push_back("Test case not found in runner");
      results.push_back(std::move(result));
      continue;
    }

    // Validate template hash
    std::ostringstream oss;
    oss << std::hex << hash_string(test_case->template_json);
    if (oss.str() != golden_test.template_hash) {
      result.failures.push_back(
          "Template hash mismatch - template may have changed");
    }

    // Validate each frame
    for (const auto &golden_frame : golden_test.frames) {
      FrameResult frame_result;
      frame_result.time = golden_frame.time;
      frame_result.expected_hash = golden_frame.pixel_hash;
      frame_result.passed = false;
      frame_result.similarity = 0.0;

      // Render frame
      auto pixels =
          pimpl_->render_fn(test_case->template_json, golden_frame.time,
                            golden_frame.width, golden_frame.height);

      if (pixels.empty()) {
        frame_result.error = "Render failed";
        result.failures.push_back(
            "Frame at t=" + std::to_string(golden_frame.time) +
            ": render failed");
        result.passed = false;
      } else {
        uint64_t actual_hash = hash_pixels(pixels.data(), pixels.size());
        frame_result.actual_hash = actual_hash;

        if (actual_hash == golden_frame.pixel_hash) {
          frame_result.passed = true;
          frame_result.similarity = 1.0;
          ++result.matched_frames;
        } else {
          // Hash mismatch - compute similarity for debugging
          // Note: We don't have the reference pixels, only the hash
          // So we report 0 similarity for hash mismatch
          frame_result.passed = false;
          frame_result.similarity = 0.0;
          result.passed = false;

          std::ostringstream err;
          err << "Frame at t=" << golden_frame.time << ": hash mismatch "
              << "(expected: " << std::hex << golden_frame.pixel_hash
              << ", got: " << actual_hash << ")";
          result.failures.push_back(err.str());
        }
      }

      result.min_similarity =
          std::min(result.min_similarity, frame_result.similarity);
      result.frame_results.push_back(frame_result);
    }

    results.push_back(std::move(result));
  }

  return results;
}

// ============================================================================
// Built-in Test Cases
// ============================================================================

std::vector<TestCase> ConsistencyTestRunner::builtin_tests() {
  std::vector<TestCase> tests;

  // Test 1: Simple text
  {
    TestCase test;
    test.name = "simple_text";
    test.template_json = R"({
      "canvas": {"width": 1920, "height": 1080},
      "duration": 2.0,
      "fps": 60.0,
      "layers": [
        {
          "type": "text",
          "content": "Hello World",
          "style": {
            "font_size": 72,
            "color": "#FFFFFF",
            "font_family": "Inter"
          },
          "position": {"x": 0.5, "y": 0.5}
        }
      ]
    })";
    test.test_times = {0.0, 1.0, 2.0};
    test.width = 1920;
    test.height = 1080;
    tests.push_back(test);
  }

  // Test 2: Box highlight animation
  {
    TestCase test;
    test.name = "box_highlight";
    test.template_json = R"({
      "canvas": {"width": 1920, "height": 1080},
      "duration": 3.0,
      "fps": 60.0,
      "layers": [
        {
          "type": "text",
          "content": "Word One Word Two Word Three",
          "style": {
            "font_size": 64,
            "color": "#FFFFFF"
          },
          "position": {"x": 0.5, "y": 0.5},
          "animation": {
            "type": "box_highlight",
            "segments": [
              {"text": "Word One", "start": 0.0, "end": 1.0},
              {"text": "Word Two", "start": 1.0, "end": 2.0},
              {"text": "Word Three", "start": 2.0, "end": 3.0}
            ],
            "box_color": "#39E55F"
          }
        }
      ]
    })";
    test.test_times = {0.5, 1.5, 2.5};
    tests.push_back(test);
  }

  // Test 3: Multi-layer compositing
  {
    TestCase test;
    test.name = "multi_layer";
    test.template_json = R"({
      "canvas": {"width": 1920, "height": 1080},
      "duration": 1.0,
      "fps": 60.0,
      "layers": [
        {
          "type": "text",
          "content": "Background Layer",
          "style": {"font_size": 48, "color": "#888888"},
          "position": {"x": 0.5, "y": 0.3}
        },
        {
          "type": "text",
          "content": "Middle Layer",
          "style": {"font_size": 64, "color": "#AAAAAA"},
          "position": {"x": 0.5, "y": 0.5}
        },
        {
          "type": "text",
          "content": "Foreground Layer",
          "style": {"font_size": 80, "color": "#FFFFFF"},
          "position": {"x": 0.5, "y": 0.7}
        }
      ]
    })";
    test.test_times = {0.0, 0.5, 1.0};
    tests.push_back(test);
  }

  // Test 4: Text with stroke (tests effect rendering)
  {
    TestCase test;
    test.name = "text_stroke";
    test.template_json = R"({
      "canvas": {"width": 1920, "height": 1080},
      "duration": 1.0,
      "fps": 60.0,
      "layers": [
        {
          "type": "text",
          "content": "Stroke Test",
          "style": {
            "font_size": 96,
            "color": "#FFFFFF",
            "stroke_color": "#000000",
            "stroke_width": 4
          },
          "position": {"x": 0.5, "y": 0.5}
        }
      ]
    })";
    test.test_times = {0.0};
    tests.push_back(test);
  }

  // Test 5: Deterministic RNG test (particles)
  {
    TestCase test;
    test.name = "deterministic_rng";
    test.template_json = R"({
      "canvas": {"width": 512, "height": 512},
      "duration": 2.0,
      "fps": 30.0,
      "layers": [
        {
          "type": "text",
          "content": "RNG Test",
          "style": {"font_size": 48, "color": "#FFFFFF"},
          "position": {"x": 0.5, "y": 0.5}
        }
      ]
    })";
    // Multiple frames to test RNG consistency
    test.test_times = {0.0, 0.033, 0.066, 0.1, 0.5, 1.0, 1.5, 2.0};
    test.width = 512;
    test.height = 512;
    tests.push_back(test);
  }

  return tests;
}

} // namespace Testing
} // namespace CaptionEngine
