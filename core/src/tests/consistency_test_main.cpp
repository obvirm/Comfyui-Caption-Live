/**
 * @file consistency_test_main.cpp
 * @brief CLI entry point for consistency testing
 *
 * Usage:
 *   consistency_test --generate [output_path]
 *   consistency_test --validate [golden_path]
 *   consistency_test --list
 *   consistency_test --help
 */

#include "engine.hpp"
#include "testing/consistency_test.hpp"


#include <cstring>
#include <iomanip>
#include <iostream>

using namespace CaptionEngine::Testing;

// ANSI color codes for output
#ifdef _WIN32
#include <windows.h>
static void enable_ansi_colors() {
  HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD dwMode = 0;
  GetConsoleMode(hOut, &dwMode);
  SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#else
static void enable_ansi_colors() {}
#endif

constexpr const char *COLOR_RESET = "\033[0m";
constexpr const char *COLOR_GREEN = "\033[32m";
constexpr const char *COLOR_RED = "\033[31m";
constexpr const char *COLOR_YELLOW = "\033[33m";
constexpr const char *COLOR_CYAN = "\033[36m";
constexpr const char *COLOR_BOLD = "\033[1m";

void print_header() {
  std::cout << COLOR_BOLD << COLOR_CYAN;
  std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║           CaptionEngine Consistency Test Runner               ║
║              WASM ⟷ Native Validation System                  ║
╚═══════════════════════════════════════════════════════════════╝
)" << COLOR_RESET
            << std::endl;
}

void print_usage() {
  std::cout << COLOR_BOLD << "Usage:" << COLOR_RESET << std::endl;
  std::cout << "  consistency_test --generate [output.json]" << std::endl;
  std::cout << "      Generate golden reference from current platform\n"
            << std::endl;
  std::cout << "  consistency_test --validate <golden.json>" << std::endl;
  std::cout << "      Validate current build against golden reference\n"
            << std::endl;
  std::cout << "  consistency_test --list" << std::endl;
  std::cout << "      List all built-in test cases\n" << std::endl;
  std::cout << "  consistency_test --help" << std::endl;
  std::cout << "      Show this help message\n" << std::endl;
}

void print_test_list() {
  auto tests = ConsistencyTestRunner::builtin_tests();

  std::cout << COLOR_BOLD << "Built-in Test Cases:" << COLOR_RESET << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  for (const auto &test : tests) {
    std::cout << COLOR_CYAN << "  • " << test.name << COLOR_RESET << std::endl;
    std::cout << "    Resolution: " << test.width << "x" << test.height
              << std::endl;
    std::cout << "    Frames: " << test.test_times.size() << " (at ";
    for (size_t i = 0; i < test.test_times.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << test.test_times[i] << "s";
    }
    std::cout << ")" << std::endl;
    std::cout << std::endl;
  }
}

void print_result(const ConsistencyResult &result) {
  if (result.passed) {
    std::cout << COLOR_GREEN << "  ✓ " << COLOR_RESET;
  } else {
    std::cout << COLOR_RED << "  ✗ " << COLOR_RESET;
  }

  std::cout << COLOR_BOLD << result.test_name << COLOR_RESET;
  std::cout << " [" << result.matched_frames << "/" << result.total_frames
            << " frames]";

  if (result.passed) {
    std::cout << COLOR_GREEN << " PASSED" << COLOR_RESET;
  } else {
    std::cout << COLOR_RED << " FAILED" << COLOR_RESET;
  }
  std::cout << std::endl;

  // Print failures
  for (const auto &failure : result.failures) {
    std::cout << COLOR_YELLOW << "      → " << failure << COLOR_RESET
              << std::endl;
  }
}

void print_summary(const std::vector<ConsistencyResult> &results) {
  size_t passed = 0;
  size_t total = results.size();
  size_t total_frames = 0;
  size_t matched_frames = 0;

  for (const auto &r : results) {
    if (r.passed)
      ++passed;
    total_frames += r.total_frames;
    matched_frames += r.matched_frames;
  }

  std::cout << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  std::cout << COLOR_BOLD << "Summary:" << COLOR_RESET << std::endl;
  std::cout << "  Tests:  " << passed << "/" << total;
  if (passed == total) {
    std::cout << COLOR_GREEN << " (100%)" << COLOR_RESET;
  } else {
    std::cout << COLOR_RED << " (" << (passed * 100 / total) << "%)"
              << COLOR_RESET;
  }
  std::cout << std::endl;

  std::cout << "  Frames: " << matched_frames << "/" << total_frames;
  if (matched_frames == total_frames) {
    std::cout << COLOR_GREEN << " (100%)" << COLOR_RESET;
  } else {
    std::cout << COLOR_RED << " (" << (matched_frames * 100 / total_frames)
              << "%)" << COLOR_RESET;
  }
  std::cout << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  if (passed == total) {
    std::cout << COLOR_GREEN << COLOR_BOLD
              << "✓ All tests passed! Consistency verified." << COLOR_RESET
              << std::endl;
  } else {
    std::cout << COLOR_RED << COLOR_BOLD
              << "✗ Some tests failed. See failures above." << COLOR_RESET
              << std::endl;
  }
}

int main(int argc, char *argv[]) {
  enable_ansi_colors();
  print_header();

  if (argc < 2) {
    print_usage();
    return 1;
  }

  std::string mode = argv[1];

  // --help
  if (mode == "--help" || mode == "-h") {
    print_usage();
    return 0;
  }

  // --list
  if (mode == "--list" || mode == "-l") {
    print_test_list();
    return 0;
  }

  // Create render callback using Engine
  auto render_fn = [](const std::string &template_json, double time,
                      uint32_t width, uint32_t height) -> std::vector<uint8_t> {
    try {
      // Create engine for this render
      // Note: In production, reuse engine instance
      CaptionEngine::Engine engine(width, height);

      // Load template
      if (!engine.load_template(template_json)) {
        std::cerr << "Failed to load template" << std::endl;
        return {};
      }

      // Render frame
      auto result = engine.process_frame(time, nullptr);
      if (!result) {
        std::cerr << "Failed to render frame" << std::endl;
        return {};
      }

      return *result;
    } catch (const std::exception &e) {
      std::cerr << "Render error: " << e.what() << std::endl;
      return {};
    }
  };

  ConsistencyTestRunner runner(render_fn);
  runner.add_tests(ConsistencyTestRunner::builtin_tests());

  // Progress callback
  runner.set_progress_callback(
      [](size_t current, size_t total, const std::string &test) {
        std::cout << "\r" << COLOR_CYAN << "[" << current << "/" << total << "]"
                  << COLOR_RESET << " Testing: " << test << "...          "
                  << std::flush;
      });

  // --generate
  if (mode == "--generate" || mode == "-g") {
    std::string output_path = "golden_reference.json";
    if (argc >= 3) {
      output_path = argv[2];
    }

    std::cout << "Generating golden reference..." << std::endl;
    std::cout << "Platform: " << platform_string() << std::endl;
    std::cout << "Output:   " << output_path << std::endl;
    std::cout << std::endl;

    auto results = runner.generate_golden(output_path);

    std::cout << "\r" << std::string(60, ' ') << "\r"; // Clear progress line

    std::cout << COLOR_BOLD << "Results:" << COLOR_RESET << std::endl;
    for (const auto &result : results) {
      print_result(result);
    }

    print_summary(results);

    bool all_passed = true;
    for (const auto &r : results) {
      if (!r.passed)
        all_passed = false;
    }

    std::cout << std::endl;
    std::cout << "Golden reference saved to: " << COLOR_CYAN << output_path
              << COLOR_RESET << std::endl;

    return all_passed ? 0 : 1;
  }

  // --validate
  if (mode == "--validate" || mode == "-v") {
    if (argc < 3) {
      std::cerr << COLOR_RED
                << "Error: --validate requires golden reference path"
                << COLOR_RESET << std::endl;
      print_usage();
      return 1;
    }

    std::string golden_path = argv[2];

    std::cout << "Validating against golden reference..." << std::endl;
    std::cout << "Platform: " << platform_string() << std::endl;
    std::cout << "Golden:   " << golden_path << std::endl;
    std::cout << std::endl;

    auto results = runner.validate(golden_path);

    std::cout << "\r" << std::string(60, ' ') << "\r"; // Clear progress line

    std::cout << COLOR_BOLD << "Results:" << COLOR_RESET << std::endl;
    for (const auto &result : results) {
      print_result(result);
    }

    print_summary(results);

    bool all_passed = true;
    for (const auto &r : results) {
      if (!r.passed)
        all_passed = false;
    }

    return all_passed ? 0 : 1;
  }

  std::cerr << COLOR_RED << "Unknown mode: " << mode << COLOR_RESET
            << std::endl;
  print_usage();
  return 1;
}
