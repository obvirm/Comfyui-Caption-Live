/**
 * @file engine.cpp
 * @brief Core engine implementation
 */

#include "engine/engine.hpp"
#include "compute/compiler.hpp"
#include "core/deterministic.hpp"
#include "engine/renderer.hpp"
#include "engine/template.hpp"
#include "graphics/backend.hpp"

#include <stdexcept>

namespace CaptionEngine {

// Engine implementation
struct Engine::Impl {
  EngineConfig config;
  std::unique_ptr<Renderer> renderer;
  std::unique_ptr<ComputeBackend> backend;

  Impl(const EngineConfig &cfg) : config(cfg) {
    // Initialize renderer
    renderer = std::make_unique<Renderer>();

    // Initialize compute backend
    switch (config.backend) {
    case BackendType::Auto:
      backend = ComputeBackend::create_best();
      break;
    case BackendType::CPU:
      backend = std::make_unique<CPUBackend>();
      break;
#if defined(__EMSCRIPTEN__)
    case BackendType::WebGPU:
      backend = std::make_unique<WebGPUBackend>();
      break;
#endif
    default:
      backend = ComputeBackend::create_best();
      break;
    }
  }
};

Engine::Engine() : Engine(EngineConfig{}) {}

Engine::Engine(const EngineConfig &config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Engine::~Engine() = default;

Engine::Engine(Engine &&) noexcept = default;
Engine &Engine::operator=(Engine &&) noexcept = default;

FrameData Engine::render_frame(const std::string &template_json, double time) {
  // Parse template
  Template tmpl = Template::from_json(template_json);

  // Render frame
  ImageBuffer buf = pimpl_->renderer->render_frame(tmpl, time);

  // Convert to FrameData
  FrameData frame;
  frame.pixels = std::move(buf.data);
  frame.width = buf.width;
  frame.height = buf.height;
  frame.timestamp = time;

  return frame;
}

FrameData Engine::render_frame_composite(const std::string &template_json,
                                         double time,
                                         const FrameData &input_image) {
  // Parse template
  Template tmpl = Template::from_json(template_json);

  // Create input buffer
  ImageBuffer input;
  input.data = input_image.pixels; // Copy
  input.width = input_image.width;
  input.height = input_image.height;

  // Render with compositing
  ImageBuffer buf = pimpl_->renderer->render_frame(tmpl, time, input);

  // Convert to FrameData
  FrameData frame;
  frame.pixels = std::move(buf.data);
  frame.width = buf.width;
  frame.height = buf.height;
  frame.timestamp = time;

  return frame;
}

std::vector<uint8_t> Engine::export_png(const FrameData &frame) {
  // Use stb_image_write for PNG encoding
  // Implementation in separate file
  return {}; // TODO: Implement
}

BackendType Engine::current_backend() const {
  // Map backend name to type
  const auto name = pimpl_->backend->name();
  if (name == "CPU")
    return BackendType::CPU;
  if (name == "WebGPU")
    return BackendType::WebGPU;
  if (name == "Vulkan")
    return BackendType::Vulkan;
  if (name == "CUDA")
    return BackendType::CUDA;
  if (name == "Metal")
    return BackendType::Metal;
  return BackendType::Auto;
}

bool Engine::validate_frame(const FrameData &frame,
                            uint64_t expected_hash) const {
  return compute_frame_hash(frame) == expected_hash;
}

uint64_t Engine::compute_frame_hash(const FrameData &frame) const {
  // Simple FNV-1a hash for deterministic validation
  uint64_t hash = 14695981039346656037ULL;
  for (uint8_t byte : frame.pixels) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

// Embedded Rain Shader (for consistent test without file I/O)
const char *RAIN_WGSL = R"(
struct FrameUniforms {
    width: u32,
    height: u32,
    time: f32,
    seed: u32,
};
@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>;

fn hash(p: u32) -> u32 {
    var p_ = p;
    p_ = (p_ + 0x7ed55d16u) + (p_ << 12u);
    p_ = (p_ ^ 0xc761c23cu) ^ (p_ >> 19u);
    p_ = (p_ + 0x165667b1u) + (p_ << 5u);
    p_ = (p_ + 0xd3a2646cu) ^ (p_ << 9u);
    p_ = (p_ + 0xfd7046c5u) + (p_ << 3u);
    p_ = (p_ ^ 0xb55a4f09u) ^ (p_ >> 16u);
    return p_;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= uniforms.width || y >= uniforms.height) { return; }
    
    // Deterministic visual test: Gradient + Noise
    let idx = y * uniforms.width + x;
    let h = hash(idx + uniforms.seed);
    
    let u = f32(x) / f32(uniforms.width);
    let v = f32(y) / f32(uniforms.height);
    
    // Blue gradient background
    let r = 0u;
    let g = u32(v * 100.0);
    let b = u32(v * 255.0);
    
    // Noise overlay (White)
    if ((h % 100u) < 2u) {
       outputBuffer[idx] = 0xFFFFFFFFu;
    } else {
       outputBuffer[idx] = 0xFF000000u | (b << 16) | (g << 8) | r;
    }
}
)";

// Test structure matching shader
struct TestUniforms {
  uint32_t width;
  uint32_t height;
  float time;
  uint32_t seed;
};

// Run isolated test if available
bool run_rain_test_impl(ComputeBackend *backend);

bool Engine::test_gpu_compute() {
  if (!pimpl_->backend)
    return false;
  return run_rain_test_impl(pimpl_->backend.get());
}

} // namespace CaptionEngine
