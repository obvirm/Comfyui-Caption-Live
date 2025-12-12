/**
 * @file cuda_kernels.hpp
 * @brief Host-side declarations for CUDA effect kernels
 */

#pragma once

#include <cstdint>

#if defined(CAPTION_HAS_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#else
typedef void *cudaStream_t;
#endif

namespace CaptionEngine {
namespace CUDA {

// ============================================================
// Kernel Launch Functions
// ============================================================

/**
 * @brief Launch text glow effect kernel
 * @param output Output pixel buffer (RGBA32)
 * @param input Input pixel buffer (RGBA32)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param glow_radius Glow radius in pixels
 * @param glow_color Glow color (packed RGBA)
 * @param glow_intensity Glow intensity (0.0 - 1.0+)
 * @param stream CUDA stream
 */
extern "C" void launch_text_glow(uint32_t *output, const uint32_t *input,
                                 int width, int height, float glow_radius,
                                 uint32_t glow_color, float glow_intensity,
                                 cudaStream_t stream);

/**
 * @brief Launch glitch effect kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param intensity Glitch intensity (0.0 - 1.0)
 * @param time Current time in seconds
 * @param seed Random seed for determinism
 * @param stream CUDA stream
 */
extern "C" void launch_glitch(uint32_t *output, const uint32_t *input,
                              int width, int height, float intensity,
                              float time, uint32_t seed, cudaStream_t stream);

/**
 * @brief Launch wave distortion kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param amplitude_x Horizontal wave amplitude in pixels
 * @param amplitude_y Vertical wave amplitude in pixels
 * @param frequency Wave frequency
 * @param time Current time in seconds
 * @param stream CUDA stream
 */
extern "C" void launch_wave_distortion(uint32_t *output, const uint32_t *input,
                                       int width, int height, float amplitude_x,
                                       float amplitude_y, float frequency,
                                       float time, cudaStream_t stream);

/**
 * @brief Launch chromatic aberration kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param strength Effect strength
 * @param center_x Center X (0.0 - 1.0, default 0.5)
 * @param center_y Center Y (0.0 - 1.0, default 0.5)
 * @param stream CUDA stream
 */
extern "C" void launch_chromatic_aberration(uint32_t *output,
                                            const uint32_t *input, int width,
                                            int height, float strength,
                                            float center_x, float center_y,
                                            cudaStream_t stream);

/**
 * @brief Launch separable Gaussian blur
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param temp_buffer Temporary buffer (same size as input/output)
 * @param width Image width
 * @param height Image height
 * @param sigma Blur sigma
 * @param stream CUDA stream
 */
extern "C" void launch_blur(uint32_t *output, const uint32_t *input,
                            uint32_t *temp_buffer, int width, int height,
                            float sigma, cudaStream_t stream);

/**
 * @brief Launch alpha composite kernel
 * @param output Output pixel buffer
 * @param background Background layer
 * @param foreground Foreground layer
 * @param width Image width
 * @param height Image height
 * @param opacity Foreground opacity multiplier
 * @param stream CUDA stream
 */
extern "C" void launch_composite(uint32_t *output, const uint32_t *background,
                                 const uint32_t *foreground, int width,
                                 int height, float opacity,
                                 cudaStream_t stream);

/**
 * @brief Launch text stroke/outline kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param stroke_width Stroke width in pixels
 * @param stroke_color Stroke color (packed RGBA)
 * @param stream CUDA stream
 */
extern "C" void launch_text_stroke(uint32_t *output, const uint32_t *input,
                                   int width, int height, float stroke_width,
                                   uint32_t stroke_color, cudaStream_t stream);

/**
 * @brief Launch drop shadow kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param offset_x Shadow offset X in pixels
 * @param offset_y Shadow offset Y in pixels
 * @param blur_radius Shadow blur radius
 * @param shadow_color Shadow color (packed RGBA)
 * @param opacity Shadow opacity (0.0-1.0)
 * @param stream CUDA stream
 */
extern "C" void launch_drop_shadow(uint32_t *output, const uint32_t *input,
                                   int width, int height, float offset_x,
                                   float offset_y, float blur_radius,
                                   uint32_t shadow_color, float opacity,
                                   cudaStream_t stream);

/**
 * @brief Launch typewriter reveal animation kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param progress Animation progress (0.0-1.0)
 * @param direction Reveal direction (0=LTR, 1=RTL, 2=TTB, 3=BTT)
 * @param edge_softness Edge softness for smooth reveal
 * @param stream CUDA stream
 */
extern "C" void launch_typewriter_reveal(uint32_t *output,
                                         const uint32_t *input, int width,
                                         int height, float progress,
                                         int direction, float edge_softness,
                                         cudaStream_t stream);

/**
 * @brief Launch particle system kernel
 * @param output Output pixel buffer
 * @param input Input pixel buffer
 * @param width Image width
 * @param height Image height
 * @param time Current time in seconds
 * @param seed Random seed for deterministic particles
 * @param particle_count Number of particles to render
 * @param particle_color Particle color (packed RGBA)
 * @param stream CUDA stream
 */
extern "C" void launch_particles(uint32_t *output, const uint32_t *input,
                                 int width, int height, float time,
                                 uint32_t seed, int particle_count,
                                 uint32_t particle_color, cudaStream_t stream);

} // namespace CUDA
} // namespace CaptionEngine
