#pragma once
/**
 * @file emscripten.hpp
 * @brief Emscripten/WASM platform bindings
 */

#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace CaptionEngine::Platform {

/**
 * @brief Emscripten async callback wrapper
 */
template <typename T> class AsyncPromise {
public:
  using Callback = std::function<void(T)>;

  void then(Callback cb) { callback_ = std::move(cb); }
  void resolve(T value) {
    if (callback_)
      callback_(std::move(value));
  }

private:
  Callback callback_;
};

/**
 * @brief JavaScript interop utilities
 */
class JSInterop {
public:
  /// Call JavaScript function with arguments
  template <typename... Args>
  static emscripten::val call_js(const std::string &func_name, Args &&...args);

  /// Get JavaScript global object
  static emscripten::val global();

  /// Create Uint8Array from C++ buffer
  static emscripten::val to_uint8_array(const std::vector<uint8_t> &buffer);

  /// Convert Uint8Array to C++ buffer
  static std::vector<uint8_t> from_uint8_array(const emscripten::val &array);

  /// Create ImageData from pixel buffer
  static emscripten::val to_image_data(const uint8_t *pixels, uint32_t width,
                                       uint32_t height);
};

/**
 * @brief File system access for Emscripten
 */
class WasmFS {
public:
  /// Mount MEMFS at path
  static void mount_memfs(const std::string &path);

  /// Preload file from URL to path
  static void preload_file(const std::string &url, const std::string &path);

  /// Read file contents
  static std::optional<std::vector<uint8_t>> read_file(const std::string &path);

  /// Write file contents
  static bool write_file(const std::string &path,
                         std::span<const uint8_t> data);
};

/// Initialize WASM module (called automatically)
void wasm_init();

/// Get memory usage info
struct MemoryInfo {
  size_t used_bytes;
  size_t total_bytes;
  size_t heap_size;
};
MemoryInfo wasm_memory_info();

} // namespace CaptionEngine::Platform

// Macro for exposing functions to JavaScript
#define CAPTION_EXPORT_FUNCTION(name) EMSCRIPTEN_KEEPALIVE extern "C"

#else

// Stub macros when not building for WASM
#define CAPTION_EXPORT_FUNCTION(name)

#endif // __EMSCRIPTEN__
