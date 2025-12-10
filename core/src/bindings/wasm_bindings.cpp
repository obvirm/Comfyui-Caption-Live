/**
 * @file wasm_bindings.cpp
 * @brief WebAssembly bindings using Emscripten
 */

#ifdef __EMSCRIPTEN__

#include "engine/engine.hpp"
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
        CaptionEngine::BackendType::WebGPU; // Force WebGPU preference explicitly
                                           // for now
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
  function("render_frame", &render_frame_js);
  function("render_frame_rgba", &render_frame_rgba_js);
  function("test_compute", &test_compute_js);
  function("is_engine_ready", &is_engine_ready_js);

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
