/**
 * @file wasm_bindings.cpp
 * @brief WebAssembly bindings using Emscripten
 */

#ifdef __EMSCRIPTEN__

#include "engine/engine.hpp"
#include "gpu/backend.hpp" // Fix for libc++ rebind_pointer_t
#include <algorithm>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <iostream>
#include <memory>


using namespace emscripten;

// Global persistent engine instance
// This ensures backends (like WebGPU) stay alive across frames
static std::unique_ptr<CaptionEngine::Engine> g_engine;

// Lazy initialization helper
CaptionEngine::Engine &get_engine() {
  if (!g_engine) {
    // Initialize with default config (which includes Auto backend -> WebGPU
    // preference on WASM)
    CaptionEngine::EngineConfig config;
    config.backend =
        CaptionEngine::BackendType::WebGPU; // Force WebGPU preference
                                            // explicitly for now
    g_engine = std::make_unique<CaptionEngine::Engine>(config);
    std::cout << "✅ C++ Engine Initialized (WASM)" << std::endl;
  }
  return *g_engine;
}

// Wrapper to return Uint8Array to JavaScript
val render_frame_js(const std::string &template_json, double time) {
  auto &engine = get_engine();
  auto frame = engine.render_frame(template_json, time);

  // Create Uint8Array from pixel data
  return val(typed_memory_view(frame.pixels.size(), frame.pixels.data()));
}

val render_frame_rgba_js(const std::string &template_json, double time) {
  auto &engine = get_engine();
  auto frame = engine.render_frame(template_json, time);

  // Return dimensions along with data
  val result = val::object();
  result.set("width", frame.width);
  result.set("height", frame.height);
  result.set("data",
             val(typed_memory_view(frame.pixels.size(), frame.pixels.data())));

  return result;
}

/**
 * @brief Process frame with input image (matches Python process_frame API)
 * @param template_json JSON scene description
 * @param time Current timestamp
 * @param input_data Uint8ClampedArray from canvas ImageData (RGBA, 0-255)
 * @param width Image width
 * @param height Image height
 * @return Object with composited image data
 */
val process_frame_js(const std::string &template_json, double time,
                     val input_data, int width, int height) {
  auto &engine = get_engine();

  // Convert JS Uint8ClampedArray to C++ FrameData
  CaptionEngine::FrameData input_frame;
  input_frame.width = width;
  input_frame.height = height;
  input_frame.pixels.resize(width * height * 4);

  // Copy data from JS array
  auto length = input_data["length"].as<size_t>();
  for (size_t i = 0; i < length && i < input_frame.pixels.size(); ++i) {
    input_frame.pixels[i] = input_data[i].as<uint8_t>();
  }

  // Render and composite
  auto result_frame =
      engine.render_frame_composite(template_json, time, input_frame);

  // Return result object
  val result = val::object();
  result.set("width", result_frame.width);
  result.set("height", result_frame.height);
  result.set("data", val(typed_memory_view(result_frame.pixels.size(),
                                           result_frame.pixels.data())));

  return result;
}

/**
 * @brief Process frame with Float32Array input (matches Python numpy format)
 * @param template_json JSON scene description
 * @param time Current timestamp
 * @param input_data Float32Array (RGBA, 0.0-1.0)
 * @param width Image width
 * @param height Image height
 * @return Object with composited Float32Array data
 */
val process_frame_float_js(const std::string &template_json, double time,
                           val input_data, int width, int height) {
  auto &engine = get_engine();

  // Convert JS Float32Array to C++ FrameData
  CaptionEngine::FrameData input_frame;
  input_frame.width = width;
  input_frame.height = height;
  input_frame.pixels.resize(width * height * 4);

  // Copy and convert float to uint8
  size_t pixel_count = width * height * 4;
  for (size_t i = 0; i < pixel_count; ++i) {
    float val_f = input_data[i].as<float>();
    input_frame.pixels[i] =
        static_cast<uint8_t>(std::clamp(val_f * 255.0f, 0.0f, 255.0f));
  }

  // Render and composite
  auto result_frame =
      engine.render_frame_composite(template_json, time, input_frame);

  // Convert result back to float32 array
  std::vector<float> result_float(result_frame.pixels.size());
  for (size_t i = 0; i < result_frame.pixels.size(); ++i) {
    result_float[i] = result_frame.pixels[i] / 255.0f;
  }

  val result = val::object();
  result.set("width", result_frame.width);
  result.set("height", result_frame.height);
  result.set("data",
             val(typed_memory_view(result_float.size(), result_float.data())));

  return result;
}

// Test compute shader execution
bool test_compute_js() {
  auto &engine = get_engine();
  return engine.test_gpu_compute();
}

// Check if engine is ready (backend initialized)
bool is_engine_ready_js() {
  // In future, check backend readiness status
  if (!g_engine)
    return false;
  // We initiate init in constructor.
  return true;
}

EMSCRIPTEN_BINDINGS(caption_engine) {
  // Original functions
  function("render_frame", &render_frame_js);
  function("render_frame_rgba", &render_frame_rgba_js);
  function("test_compute", &test_compute_js);
  function("is_engine_ready", &is_engine_ready_js);

  // NEW: Unified process_frame API (matches Python)
  function("process_frame", &process_frame_js);
  function("process_frame_float", &process_frame_float_js);

  // Expose FrameData if needed
  value_object<CaptionEngine::FrameData>("FrameData")
      .field("width", &CaptionEngine::FrameData::width)
      .field("height", &CaptionEngine::FrameData::height)
      .field("timestamp", &CaptionEngine::FrameData::timestamp);
}

#else

// Stub when not building for Emscripten
namespace {
int wasm_dummy = 0;
}

#endif
