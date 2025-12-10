/**
 * @file distortions.cpp
 * @brief Visual distortion effects (glitch, wave, chromatic aberration)
 */

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace CaptionEngine {
namespace Effects {

/// Simple inline RNG for deterministic effects
class DistortionRNG {
public:
  explicit DistortionRNG(uint64_t seed) : state_(seed ? seed : 0xDEADBEEF) {}

  uint64_t next() noexcept {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1DULL;
  }

  float next_float() noexcept {
    return static_cast<float>(next() & 0xFFFFFF) / 16777216.0f;
  }

private:
  uint64_t state_;
};

/**
 * @brief Glitch effect with horizontal displacement
 */
struct GlitchEffect {
  float intensity = 0.1f;
  float slice_min_height = 5;
  float slice_max_height = 50;
  float offset_range = 50;
  float probability = 0.1f;
  float rgb_split = 5.0f;
  uint64_t seed = 12345;

  struct GlitchSlice {
    int y_start;
    int height;
    int x_offset;
    int r_offset;
    int g_offset;
    int b_offset;
  };

  std::vector<GlitchSlice> generate_slices(int image_height, float time) const {
    std::vector<GlitchSlice> slices;

    uint64_t frame_seed = seed + static_cast<uint64_t>(time * 60);
    DistortionRNG rng(frame_seed);

    if (rng.next_float() > probability * intensity) {
      return slices;
    }

    int num_slices = 3 + static_cast<int>(rng.next() % 8);
    for (int i = 0; i < num_slices; ++i) {
      GlitchSlice s;
      s.y_start =
          static_cast<int>(rng.next() % static_cast<uint64_t>(image_height));
      s.height = static_cast<int>(slice_min_height) +
                 static_cast<int>(rng.next() %
                                  static_cast<uint64_t>(slice_max_height -
                                                        slice_min_height));
      s.x_offset = static_cast<int>((rng.next_float() * 2.0f - 1.0f) *
                                    offset_range * intensity);
      s.r_offset = static_cast<int>((rng.next_float() * 2.0f - 1.0f) *
                                    rgb_split * intensity);
      s.g_offset = 0;
      s.b_offset = static_cast<int>((rng.next_float() * 2.0f - 1.0f) *
                                    rgb_split * intensity);

      slices.push_back(s);
    }

    return slices;
  }
};

/**
 * @brief Wave distortion effect
 */
struct WaveDistortion {
  float amplitude_x = 10.0f;
  float amplitude_y = 10.0f;
  float frequency_x = 0.02f;
  float frequency_y = 0.02f;
  float speed = 2.0f;
  float phase_offset = 0.0f;

  std::pair<float, float> get_displacement(int x, int y, float time) const {
    float phase = time * speed + phase_offset;
    float dx =
        std::sin(static_cast<float>(y) * frequency_y + phase) * amplitude_x;
    float dy =
        std::sin(static_cast<float>(x) * frequency_x + phase) * amplitude_y;
    return std::make_pair(dx, dy);
  }

  std::pair<float, float> get_source_uv(float u, float v, float time) const {
    constexpr float pi2 = 6.28318f;
    float phase = time * speed + phase_offset;
    float du = std::sin(v * frequency_y * pi2 + phase) * amplitude_x / 1920.0f;
    float dv = std::sin(u * frequency_x * pi2 + phase) * amplitude_y / 1080.0f;
    return std::make_pair(u + du, v + dv);
  }
};

/**
 * @brief Chromatic aberration effect
 */
struct ChromaticAberration {
  float strength = 0.01f;
  float center_x = 0.5f;
  float center_y = 0.5f;
  bool radial = true;
  float angle = 0.0f;

  struct RGBOffset {
    float r_u, r_v;
    float g_u, g_v;
    float b_u, b_v;
  };

  RGBOffset get_offsets(float u, float v) const {
    RGBOffset off = {};
    off.g_u = u;
    off.g_v = v;

    if (radial) {
      float dx = u - center_x;
      float dy = v - center_y;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist > 0.001f) {
        float norm_x = dx / dist;
        float norm_y = dy / dist;
        float offset = dist * strength;

        off.r_u = u - norm_x * offset;
        off.r_v = v - norm_y * offset;
        off.b_u = u + norm_x * offset;
        off.b_v = v + norm_y * offset;
      } else {
        off.r_u = off.b_u = u;
        off.r_v = off.b_v = v;
      }
    } else {
      float offset_x = std::cos(angle) * strength;
      float offset_y = std::sin(angle) * strength;

      off.r_u = u - offset_x;
      off.r_v = v - offset_y;
      off.b_u = u + offset_x;
      off.b_v = v + offset_y;
    }

    return off;
  }
};

/**
 * @brief VHS/retro effect
 */
struct VHSEffect {
  float scan_line_strength = 0.2f;
  float noise_strength = 0.05f;
  float color_bleed = 2.0f;
  float vertical_jitter = 0;
  float horizontal_glitch_prob = 0.01f;
  uint64_t seed = 54321;

  float get_scanline(int y) const {
    return 1.0f - scan_line_strength * static_cast<float>(y % 2);
  }

  float get_noise(int x, int y, float time) const {
    uint64_t pixel_seed = seed + static_cast<uint64_t>(x) +
                          static_cast<uint64_t>(y) * 1920ULL +
                          static_cast<uint64_t>(time * 60) * 1920ULL * 1080ULL;
    DistortionRNG rng(pixel_seed);
    return (rng.next_float() * 2.0f - 1.0f) * noise_strength;
  }
};

/**
 * @brief Zoom blur effect
 */
struct ZoomBlur {
  float strength = 0.1f;
  float center_x = 0.5f;
  float center_y = 0.5f;
  int samples = 10;

  std::vector<std::pair<float, float>> get_sample_uvs(float u, float v) const {
    std::vector<std::pair<float, float>> samples_out;
    samples_out.reserve(static_cast<size_t>(samples));

    float dx = u - center_x;
    float dy = v - center_y;

    for (int i = 0; i < samples; ++i) {
      float t =
          static_cast<float>(i) / static_cast<float>(samples - 1) * strength;
      samples_out.push_back(std::make_pair(u - dx * t, v - dy * t));
    }

    return samples_out;
  }
};

/**
 * @brief Motion blur effect
 */
struct MotionBlur {
  float angle = 0.0f;
  float strength = 0.05f;
  int samples = 8;

  std::vector<std::pair<float, float>> get_sample_uvs(float u, float v) const {
    std::vector<std::pair<float, float>> samples_out;
    samples_out.reserve(static_cast<size_t>(samples));

    float dx = std::cos(angle) * strength;
    float dy = std::sin(angle) * strength;

    for (int i = 0; i < samples; ++i) {
      float t = static_cast<float>(i) / static_cast<float>(samples - 1) - 0.5f;
      samples_out.push_back(std::make_pair(u + dx * t, v + dy * t));
    }

    return samples_out;
  }
};

/**
 * @brief Pixelation effect
 */
struct Pixelation {
  int block_size = 8;

  std::pair<float, float> get_source_uv(float u, float v, int width,
                                        int height) const {
    int px = static_cast<int>(u * static_cast<float>(width));
    int py = static_cast<int>(v * static_cast<float>(height));

    px = (px / block_size) * block_size + block_size / 2;
    py = (py / block_size) * block_size + block_size / 2;

    return std::make_pair(static_cast<float>(px) / static_cast<float>(width),
                          static_cast<float>(py) / static_cast<float>(height));
  }
};

} // namespace Effects
} // namespace CaptionEngine
