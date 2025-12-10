/**
 * @file emscripten.cpp
 * @brief WebAssembly exports for browser
 */

#ifdef __EMSCRIPTEN__

#include "platform/emscripten.hpp"
#include "engine/engine.hpp"
#include <emscripten/bind.h>

using namespace emscripten;
using namespace CaptionEngine;

// Global engine instance
static std::unique_ptr<Engine> g_engine;

// Initialize engine
void init_engine(int width, int height) {
  EngineConfig config;
  config.width = width;
  config.height = height;
  config.preferred_backend = BackendType::WebGPU;
  g_engine = std::make_unique<Engine>(config);
}

// Render frame from JSON template
val render_frame_js(const std::string &template_json, double time) {
  if (!g_engine) {
    return val::null();
  }

  FrameData frame = g_engine->render_frame(template_json, time);

  // Create JavaScript object with frame data
  val result = val::object();
  result.set("width", frame.width);
  result.set("height", frame.height);
  result.set("timestamp", frame.timestamp);

  // Create Uint8Array for pixel data
  val pixels = val::global("Uint8Array").new_(frame.pixels.size());
  val memory_view = val::module_property("HEAPU8");

  // Copy pixels to Uint8Array
  for (size_t i = 0; i < frame.pixels.size(); ++i) {
    pixels.set(i, frame.pixels[i]);
  }

  result.set("pixels", pixels);

  return result;
}

// Get frame as ImageData
val render_to_image_data(const std::string &template_json, double time) {
  if (!g_engine) {
    return val::null();
  }

  FrameData frame = g_engine->render_frame(template_json, time);

  // Create ImageData via JavaScript
  val image_data = Platform::JSInterop::to_image_data(
      frame.pixels.data(), frame.width, frame.height);

  return image_data;
}

// Compute frame hash for validation
uint64_t compute_hash(const std::string &template_json, double time) {
  if (!g_engine) {
    return 0;
  }

  FrameData frame = g_engine->render_frame(template_json, time);
  return g_engine->compute_frame_hash(frame);
}

// Get current backend name
std::string get_backend() {
  if (!g_engine) {
    return "none";
  }
  return g_engine->current_backend();
}

// Check if WebGPU is available
bool is_webgpu_available() {
  return val::global("navigator")["gpu"].as<bool>();
}

// Export to JavaScript via embind
EMSCRIPTEN_BINDINGS(caption_engine) {
  function("initEngine", &init_engine);
  function("renderFrame", &render_frame_js);
  function("renderToImageData", &render_to_image_data);
  function("computeHash", &compute_hash);
  function("getBackend", &get_backend);
  function("isWebGPUAvailable", &is_webgpu_available);

  // Export quality enum
  enum_<Quality>("Quality")
      .value("Draft", Quality::Draft)
      .value("Preview", Quality::Preview)
      .value("Final", Quality::Final);
}

namespace CaptionEngine::Platform {

val JSInterop::global() { return val::global(); }

val JSInterop::to_uint8_array(const std::vector<uint8_t> &buffer) {
  val array = val::global("Uint8Array").new_(buffer.size());
  for (size_t i = 0; i < buffer.size(); ++i) {
    array.set(i, buffer[i]);
  }
  return array;
}

std::vector<uint8_t> JSInterop::from_uint8_array(const val &array) {
  size_t length = array["length"].as<size_t>();
  std::vector<uint8_t> result(length);
  for (size_t i = 0; i < length; ++i) {
    result[i] = array[i].as<uint8_t>();
  }
  return result;
}

val JSInterop::to_image_data(const uint8_t *pixels, uint32_t width,
                             uint32_t height) {
  // Create Uint8ClampedArray
  size_t size = width * height * 4;
  val data = val::global("Uint8ClampedArray").new_(size);

  for (size_t i = 0; i < size; ++i) {
    data.set(i, pixels[i]);
  }

  // Create ImageData
  return val::global("ImageData").new_(data, width, height);
}

void wasm_init() {
  // Initialize any WASM-specific setup
}

MemoryInfo wasm_memory_info() {
  MemoryInfo info;
  val memory = val::module_property("HEAP8");
  info.heap_size = memory["length"].as<size_t>();
  // Note: actual used memory tracking requires additional instrumentation
  info.used_bytes = 0;
  info.total_bytes = info.heap_size;
  return info;
}

} // namespace CaptionEngine::Platform

#endif // __EMSCRIPTEN__
