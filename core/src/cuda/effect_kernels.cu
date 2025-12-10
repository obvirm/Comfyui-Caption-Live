/**
 * @file effect_kernels.cu
 * @brief CUDA kernels for caption effects
 */

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace CaptionEngine {
namespace CUDA {

// ============================================================
// Constants
// ============================================================

constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

// ============================================================
// Utility Device Functions
// ============================================================

__device__ __forceinline__ float clamp_f(float val, float min_val,
                                         float max_val) {
  return fminf(fmaxf(val, min_val), max_val);
}

__device__ __forceinline__ uint32_t pack_rgba(float r, float g, float b,
                                              float a) {
  uint8_t ri = static_cast<uint8_t>(clamp_f(r, 0.0f, 1.0f) * 255.0f);
  uint8_t gi = static_cast<uint8_t>(clamp_f(g, 0.0f, 1.0f) * 255.0f);
  uint8_t bi = static_cast<uint8_t>(clamp_f(b, 0.0f, 1.0f) * 255.0f);
  uint8_t ai = static_cast<uint8_t>(clamp_f(a, 0.0f, 1.0f) * 255.0f);
  return (ai << 24) | (bi << 16) | (gi << 8) | ri;
}

__device__ __forceinline__ void unpack_rgba(uint32_t color, float &r, float &g,
                                            float &b, float &a) {
  r = static_cast<float>(color & 0xFF) / 255.0f;
  g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
  b = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
  a = static_cast<float>((color >> 24) & 0xFF) / 255.0f;
}

// Simple hash for deterministic random
__device__ __forceinline__ uint32_t hash_uint(uint32_t x) {
  x ^= x >> 17;
  x *= 0xed5ad4bbU;
  x ^= x >> 11;
  x *= 0xac4c1b51U;
  x ^= x >> 15;
  x *= 0x31848babU;
  x ^= x >> 14;
  return x;
}

__device__ __forceinline__ float hash_to_float(uint32_t hash) {
  return static_cast<float>(hash & 0xFFFFFF) / 16777216.0f;
}

// ============================================================
// Blend Modes
// ============================================================

__device__ __forceinline__ float blend_normal(float base, float overlay,
                                              float alpha) {
  return base * (1.0f - alpha) + overlay * alpha;
}

__device__ __forceinline__ float blend_multiply(float base, float overlay) {
  return base * overlay;
}

__device__ __forceinline__ float blend_screen(float base, float overlay) {
  return 1.0f - (1.0f - base) * (1.0f - overlay);
}

__device__ __forceinline__ float blend_overlay(float base, float overlay) {
  if (base < 0.5f) {
    return 2.0f * base * overlay;
  } else {
    return 1.0f - 2.0f * (1.0f - base) * (1.0f - overlay);
  }
}

// ============================================================
// Effect Kernels
// ============================================================

/**
 * @brief Text glow effect kernel
 */
__global__ void kernel_text_glow(uint32_t *output, const uint32_t *input,
                                 int width, int height, float glow_radius,
                                 uint32_t glow_color, float glow_intensity) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  uint32_t pixel = input[idx];

  float r, g, b, a;
  unpack_rgba(pixel, r, g, b, a);

  // Sample surrounding pixels for glow
  float glow_acc = 0.0f;
  int radius = static_cast<int>(glow_radius);

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      int sx = clamp_f(static_cast<float>(x + dx), 0.0f,
                       static_cast<float>(width - 1));
      int sy = clamp_f(static_cast<float>(y + dy), 0.0f,
                       static_cast<float>(height - 1));

      uint32_t sample = input[sy * width + sx];
      float sample_a = static_cast<float>((sample >> 24) & 0xFF) / 255.0f;

      float dist = sqrtf(static_cast<float>(dx * dx + dy * dy));
      float weight = fmaxf(0.0f, 1.0f - dist / glow_radius);
      glow_acc += sample_a * weight;
    }
  }

  // Apply glow color
  float gr, gg, gb, ga;
  unpack_rgba(glow_color, gr, gg, gb, ga);

  glow_acc = clamp_f(glow_acc * glow_intensity, 0.0f, 1.0f);

  // Combine glow with original
  float final_r = blend_screen(gr * glow_acc, r);
  float final_g = blend_screen(gg * glow_acc, g);
  float final_b = blend_screen(gb * glow_acc, b);
  float final_a = fmaxf(a, glow_acc * ga);

  output[idx] = pack_rgba(final_r, final_g, final_b, final_a);
}

/**
 * @brief Glitch effect kernel
 */
__global__ void kernel_glitch(uint32_t *output, const uint32_t *input,
                              int width, int height, float intensity,
                              float time, uint32_t seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  // Deterministic random based on position and time
  uint32_t frame_hash = hash_uint(seed + static_cast<uint32_t>(time * 60.0f));

  // Only affect certain rows
  uint32_t row_hash = hash_uint(y + frame_hash);
  float row_prob = hash_to_float(row_hash);

  if (row_prob > intensity * 0.2f) {
    output[idx] = input[idx];
    return;
  }

  // Calculate offset
  uint32_t offset_hash = hash_uint(row_hash + 1);
  int x_offset = static_cast<int>((hash_to_float(offset_hash) * 2.0f - 1.0f) *
                                  50.0f * intensity);

  // RGB split
  uint32_t split_hash = hash_uint(row_hash + 2);
  int r_offset = static_cast<int>((hash_to_float(split_hash) * 2.0f - 1.0f) *
                                  5.0f * intensity);

  // Sample with offsets
  int r_x = clamp_f(static_cast<float>(x + x_offset + r_offset), 0.0f,
                    static_cast<float>(width - 1));
  int g_x = clamp_f(static_cast<float>(x + x_offset), 0.0f,
                    static_cast<float>(width - 1));
  int b_x = clamp_f(static_cast<float>(x + x_offset - r_offset), 0.0f,
                    static_cast<float>(width - 1));

  uint32_t r_pixel = input[y * width + r_x];
  uint32_t g_pixel = input[y * width + g_x];
  uint32_t b_pixel = input[y * width + b_x];

  uint8_t r = r_pixel & 0xFF;
  uint8_t g = (g_pixel >> 8) & 0xFF;
  uint8_t b = (b_pixel >> 16) & 0xFF;
  uint8_t a = (g_pixel >> 24) & 0xFF;

  output[idx] = (a << 24) | (b << 16) | (g << 8) | r;
}

/**
 * @brief Wave distortion kernel
 */
__global__ void kernel_wave_distortion(uint32_t *output, const uint32_t *input,
                                       int width, int height, float amplitude_x,
                                       float amplitude_y, float frequency,
                                       float time) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  constexpr float PI = 3.14159265358979f;

  // Calculate wave offset
  float phase = time * 2.0f;
  float dx = sinf(y * frequency * PI + phase) * amplitude_x;
  float dy = sinf(x * frequency * PI + phase) * amplitude_y;

  // Sample from offset position
  int src_x =
      clamp_f(static_cast<float>(x) + dx, 0.0f, static_cast<float>(width - 1));
  int src_y =
      clamp_f(static_cast<float>(y) + dy, 0.0f, static_cast<float>(height - 1));

  output[y * width + x] = input[src_y * width + src_x];
}

/**
 * @brief Chromatic aberration kernel
 */
__global__ void kernel_chromatic_aberration(uint32_t *output,
                                            const uint32_t *input, int width,
                                            int height, float strength,
                                            float center_x, float center_y) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float u = static_cast<float>(x) / width;
  float v = static_cast<float>(y) / height;

  float dx = u - center_x;
  float dy = v - center_y;
  float dist = sqrtf(dx * dx + dy * dy);

  if (dist < 0.001f) {
    output[y * width + x] = input[y * width + x];
    return;
  }

  float norm_x = dx / dist;
  float norm_y = dy / dist;
  float offset = dist * strength * width;

  // Sample R, G, B from different positions
  int r_x = clamp_f(x - norm_x * offset, 0.0f, static_cast<float>(width - 1));
  int r_y = clamp_f(y - norm_y * offset, 0.0f, static_cast<float>(height - 1));
  int b_x = clamp_f(x + norm_x * offset, 0.0f, static_cast<float>(width - 1));
  int b_y = clamp_f(y + norm_y * offset, 0.0f, static_cast<float>(height - 1));

  uint32_t r_pixel = input[r_y * width + r_x];
  uint32_t g_pixel = input[y * width + x];
  uint32_t b_pixel = input[b_y * width + b_x];

  uint8_t r = r_pixel & 0xFF;
  uint8_t g = (g_pixel >> 8) & 0xFF;
  uint8_t b = (b_pixel >> 16) & 0xFF;
  uint8_t a = (g_pixel >> 24) & 0xFF;

  output[y * width + x] = (a << 24) | (b << 16) | (g << 8) | r;
}

/**
 * @brief Gaussian blur kernel (horizontal pass)
 */
__global__ void kernel_blur_horizontal(uint32_t *output, const uint32_t *input,
                                       int width, int height, float sigma) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int radius = static_cast<int>(sigma * 3.0f);

  float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f, a_acc = 0.0f;
  float weight_sum = 0.0f;

  for (int dx = -radius; dx <= radius; ++dx) {
    int sx = clamp_f(static_cast<float>(x + dx), 0.0f,
                     static_cast<float>(width - 1));

    float weight = expf(-(dx * dx) / (2.0f * sigma * sigma));
    weight_sum += weight;

    uint32_t pixel = input[y * width + sx];
    float pr, pg, pb, pa;
    unpack_rgba(pixel, pr, pg, pb, pa);

    r_acc += pr * weight;
    g_acc += pg * weight;
    b_acc += pb * weight;
    a_acc += pa * weight;
  }

  output[y * width + x] = pack_rgba(r_acc / weight_sum, g_acc / weight_sum,
                                    b_acc / weight_sum, a_acc / weight_sum);
}

/**
 * @brief Gaussian blur kernel (vertical pass)
 */
__global__ void kernel_blur_vertical(uint32_t *output, const uint32_t *input,
                                     int width, int height, float sigma) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int radius = static_cast<int>(sigma * 3.0f);

  float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f, a_acc = 0.0f;
  float weight_sum = 0.0f;

  for (int dy = -radius; dy <= radius; ++dy) {
    int sy = clamp_f(static_cast<float>(y + dy), 0.0f,
                     static_cast<float>(height - 1));

    float weight = expf(-(dy * dy) / (2.0f * sigma * sigma));
    weight_sum += weight;

    uint32_t pixel = input[sy * width + x];
    float pr, pg, pb, pa;
    unpack_rgba(pixel, pr, pg, pb, pa);

    r_acc += pr * weight;
    g_acc += pg * weight;
    b_acc += pb * weight;
    a_acc += pa * weight;
  }

  output[y * width + x] = pack_rgba(r_acc / weight_sum, g_acc / weight_sum,
                                    b_acc / weight_sum, a_acc / weight_sum);
}

/**
 * @brief Alpha composite kernel
 */
__global__ void kernel_composite(uint32_t *output, const uint32_t *background,
                                 const uint32_t *foreground, int width,
                                 int height, float opacity) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  float br, bg, bb, ba;
  float fr, fg, fb, fa;

  unpack_rgba(background[idx], br, bg, bb, ba);
  unpack_rgba(foreground[idx], fr, fg, fb, fa);

  fa *= opacity;

  float out_a = fa + ba * (1.0f - fa);
  float out_r = (fr * fa + br * ba * (1.0f - fa)) / fmaxf(out_a, 0.001f);
  float out_g = (fg * fa + bg * ba * (1.0f - fa)) / fmaxf(out_a, 0.001f);
  float out_b = (fb * fa + bb * ba * (1.0f - fa)) / fmaxf(out_a, 0.001f);

  output[idx] = pack_rgba(out_r, out_g, out_b, out_a);
}

// ============================================================
// Host Launch Functions
// ============================================================

extern "C" {

void launch_text_glow(uint32_t *output, const uint32_t *input, int width,
                      int height, float glow_radius, uint32_t glow_color,
                      float glow_intensity, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel_text_glow<<<grid, block, 0, stream>>>(
      output, input, width, height, glow_radius, glow_color, glow_intensity);
}

void launch_glitch(uint32_t *output, const uint32_t *input, int width,
                   int height, float intensity, float time, uint32_t seed,
                   cudaStream_t stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel_glitch<<<grid, block, 0, stream>>>(output, input, width, height,
                                            intensity, time, seed);
}

void launch_wave_distortion(uint32_t *output, const uint32_t *input, int width,
                            int height, float amplitude_x, float amplitude_y,
                            float frequency, float time, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel_wave_distortion<<<grid, block, 0, stream>>>(
      output, input, width, height, amplitude_x, amplitude_y, frequency, time);
}

void launch_chromatic_aberration(uint32_t *output, const uint32_t *input,
                                 int width, int height, float strength,
                                 float center_x, float center_y,
                                 cudaStream_t stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel_chromatic_aberration<<<grid, block, 0, stream>>>(
      output, input, width, height, strength, center_x, center_y);
}

void launch_blur(uint32_t *output, const uint32_t *input, uint32_t *temp_buffer,
                 int width, int height, float sigma, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // Two-pass separable Gaussian blur
  kernel_blur_horizontal<<<grid, block, 0, stream>>>(temp_buffer, input, width,
                                                     height, sigma);
  kernel_blur_vertical<<<grid, block, 0, stream>>>(output, temp_buffer, width,
                                                   height, sigma);
}

void launch_composite(uint32_t *output, const uint32_t *background,
                      const uint32_t *foreground, int width, int height,
                      float opacity, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  kernel_composite<<<grid, block, 0, stream>>>(output, background, foreground,
                                               width, height, opacity);
}

} // extern "C"

} // namespace CUDA
} // namespace CaptionEngine
