#include "compute/compiler.hpp"
#include "core/deterministic.hpp"
#include "graphics/backend.hpp"
#include <iostream>
#include <span>
#include <vector>


// Standalone test function to avoid Engine header conflicts
namespace CaptionEngine {

// Embedded Rain Shader
const char *RAIN_WGSL_TEST = R"(
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
    
    let idx = y * uniforms.width + x;
    let h = hash(idx + uniforms.seed);
    
    let v = f32(y) / f32(uniforms.height);
    
    // Blue gradient background
    let r = 0u;
    let g = u32(v * 100.0);
    let b = u32(v * 255.0);
    
    // Noise overlay (White raindrops)
    if ((h % 100u) < 2u) {
       outputBuffer[idx] = 0xFFFFFFFFu;
    } else {
       outputBuffer[idx] = 0xFF000000u | (b << 16) | (g << 8) | r;
    }
}
)";

struct TestUniforms {
  uint32_t width;
  uint32_t height;
  float time;
  uint32_t seed;
};

// Externally exposed function
bool run_rain_test_impl(ComputeBackend *backend) {
  if (!backend)
    return false;

  std::cout << "🌧️ Starting Isolated Rain Test..." << std::endl;

  uint32_t width = 256;
  uint32_t height = 256;
  size_t pixel_count = width * height;
  size_t buffer_size = pixel_count * 4;

  // Compile
  auto [kernel_opt, error] = KernelCompiler::compile_kernel(
      "rain_test_iso", RAIN_WGSL_TEST, KernelCompiler::Target::WGSL);

  if (!kernel_opt) {
    std::cerr << "❌ Kernel Compilation Failed!" << std::endl;
    return false;
  }

  // Register
  if (!backend->register_kernel(*kernel_opt)) {
    std::cerr << "❌ Kernel Registration Failed!" << std::endl;
    return false;
  }

  // Create Buffers
  BufferHandle uniformBuf =
      backend->create_buffer(sizeof(TestUniforms), MemoryType::HostVisible);
  BufferHandle storageBuf =
      backend->create_buffer(buffer_size, MemoryType::DeviceLocal);

  if (!uniformBuf || !storageBuf) {
    std::cerr << "❌ Buffer Creation Failed!" << std::endl;
    return false;
  }

  // RNG
  CaptionEngine::Deterministic::DeterministicRNG rng(12345);
  uint32_t seed_val = static_cast<uint32_t>(rng.next());

  // Upload
  TestUniforms uniforms = {width, height, 1.0f, seed_val};
  backend->upload_buffer(
      uniformBuf,
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&uniforms),
                               sizeof(uniforms)));

  // Dispatch
  std::vector<BufferHandle> buffers = {uniformBuf, storageBuf};
  // 256/16 = 16 groups
  backend->dispatch_compute("rain_test_iso", buffers, {16, 16, 1});

  std::cout << "✅ Rain Test Dispatched Successfully! (Seed: " << seed_val
            << ")" << std::endl;

  // Cleanup
  backend->destroy_buffer(uniformBuf);
  backend->destroy_buffer(storageBuf);

  return true;
}

} // namespace CaptionEngine
