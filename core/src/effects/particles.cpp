/**
 * @file particles.cpp
 * @brief Particle system for decorative effects
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace CaptionEngine {
namespace Effects {

/// Particle blending mode
enum class ParticleBlend { Alpha, Additive, Multiply };

/// Particle shape
enum class ParticleShape { Circle, Square, Star, Heart, Custom };

/// Simple deterministic RNG for particles
class ParticleRNG {
public:
  explicit ParticleRNG(uint64_t seed) {
    state_ = seed;
    if (state_ == 0)
      state_ = 0xDEADBEEF;
  }

  uint64_t next() noexcept {
    // xorshift64
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

/// Single particle state
struct Particle {
  float x = 0, y = 0;
  float vx = 0, vy = 0;
  float ax = 0, ay = 0;
  float rotation = 0;
  float angular_vel = 0;
  float scale = 1.0f;
  float life = 1.0f;
  float max_life = 1.0f;
  uint32_t color = 0xFFFFFFFF;
  float alpha = 1.0f;

  bool is_alive() const { return life > 0; }

  void update(float dt) {
    vx += ax * dt;
    vy += ay * dt;
    x += vx * dt;
    y += vy * dt;
    rotation += angular_vel * dt;
    life -= dt / max_life;

    alpha = life * 2.0f;
    if (alpha < 0.0f)
      alpha = 0.0f;
    if (alpha > 1.0f)
      alpha = 1.0f;
  }
};

/// Emission shape
struct EmitterShape {
  enum Type { Point, Line, Circle, Rectangle };
  Type type = Point;
  float x = 0, y = 0;
  float width = 0, height = 0;
  float radius = 0;
  float angle_min = 0, angle_max = 6.28318f;

  std::pair<float, float> sample(ParticleRNG &rng) const {
    switch (type) {
    case Point:
      return std::make_pair(x, y);
    case Line:
      return std::make_pair(x + rng.next_float() * width, y);
    case Circle: {
      float a = rng.next_float() * 6.28318f;
      float r = std::sqrt(rng.next_float()) * radius;
      return std::make_pair(x + r * std::cos(a), y + r * std::sin(a));
    }
    case Rectangle:
      return std::make_pair(x + rng.next_float() * width,
                            y + rng.next_float() * height);
    default:
      return std::make_pair(x, y);
    }
  }
};

/// Particle emitter configuration
struct EmitterConfig {
  EmitterShape shape;

  float rate = 100;
  int burst_count = 0;

  float speed_min = 50, speed_max = 100;
  float direction = -1.57f;
  float spread = 0.5f;

  float life_min = 0.5f, life_max = 2.0f;
  float scale_min = 0.5f, scale_max = 1.5f;
  float rotation_min = 0, rotation_max = 6.28f;
  float angular_vel_min = 0, angular_vel_max = 3.14f;

  float gravity = 100;
  float drag = 0.1f;

  ParticleShape particle_shape = ParticleShape::Circle;
  ParticleBlend blend_mode = ParticleBlend::Additive;
  std::vector<uint32_t> colors = {0xFFFFFFFF};

  float scale_start = 1.0f;
  float scale_end = 0.0f;
};

/**
 * @brief Particle emitter/system
 */
class ParticleEmitter {
public:
  explicit ParticleEmitter(const EmitterConfig &config, uint64_t seed = 42)
      : config_(config), rng_(seed) {
    particles_.reserve(1000);
  }

  void update(float dt) {
    emit_accumulator_ += dt;
    while (emit_accumulator_ >= 1.0f / config_.rate &&
           particles_.size() < max_particles_) {
      emit_accumulator_ -= 1.0f / config_.rate;
      spawn_particle();
    }

    for (auto &p : particles_) {
      p.ay = config_.gravity;
      p.vx *= 1.0f - config_.drag * dt;
      p.vy *= 1.0f - config_.drag * dt;
      p.update(dt);

      float life_t = 1.0f - p.life;
      p.scale = config_.scale_start +
                (config_.scale_end - config_.scale_start) * life_t;
    }

    particles_.erase(
        std::remove_if(particles_.begin(), particles_.end(),
                       [](const Particle &p) { return !p.is_alive(); }),
        particles_.end());
  }

  void burst(int count) {
    for (int i = 0; i < count && particles_.size() < max_particles_; ++i) {
      spawn_particle();
    }
  }

  const std::vector<Particle> &particles() const { return particles_; }
  size_t count() const { return particles_.size(); }
  void clear() { particles_.clear(); }
  void set_max_particles(size_t max) {
    max_particles_ = max;
    particles_.reserve(max);
  }

private:
  void spawn_particle() {
    Particle p;

    auto pos = config_.shape.sample(rng_);
    p.x = pos.first;
    p.y = pos.second;

    float speed = config_.speed_min +
                  rng_.next_float() * (config_.speed_max - config_.speed_min);
    float angle =
        config_.direction + (rng_.next_float() * 2.0f - 1.0f) * config_.spread;
    p.vx = std::cos(angle) * speed;
    p.vy = std::sin(angle) * speed;

    p.max_life = config_.life_min +
                 rng_.next_float() * (config_.life_max - config_.life_min);
    p.life = 1.0f;

    p.scale = config_.scale_min +
              rng_.next_float() * (config_.scale_max - config_.scale_min);

    p.rotation =
        config_.rotation_min +
        rng_.next_float() * (config_.rotation_max - config_.rotation_min);
    p.angular_vel =
        config_.angular_vel_min +
        rng_.next_float() * (config_.angular_vel_max - config_.angular_vel_min);

    if (!config_.colors.empty()) {
      size_t idx = static_cast<size_t>(rng_.next()) % config_.colors.size();
      p.color = config_.colors[idx];
    }

    particles_.push_back(p);
  }

  EmitterConfig config_;
  ParticleRNG rng_;
  std::vector<Particle> particles_;
  float emit_accumulator_ = 0;
  size_t max_particles_ = 10000;
};

/// Preset particle effects
namespace Presets {

inline EmitterConfig confetti() {
  EmitterConfig cfg;
  cfg.shape.type = EmitterShape::Line;
  cfg.shape.y = -50;
  cfg.shape.width = 1920;
  cfg.rate = 50;
  cfg.speed_min = 200;
  cfg.speed_max = 400;
  cfg.direction = 1.57f;
  cfg.spread = 0.3f;
  cfg.life_min = 3;
  cfg.life_max = 5;
  cfg.gravity = 200;
  cfg.angular_vel_max = 10;
  cfg.colors = {0xFFFF0000, 0xFF00FF00, 0xFF0000FF,
                0xFFFFFF00, 0xFFFF00FF, 0xFF00FFFF};
  cfg.particle_shape = ParticleShape::Square;
  return cfg;
}

inline EmitterConfig sparkle() {
  EmitterConfig cfg;
  cfg.shape.type = EmitterShape::Rectangle;
  cfg.shape.width = 1920;
  cfg.shape.height = 1080;
  cfg.rate = 20;
  cfg.speed_min = 0;
  cfg.speed_max = 20;
  cfg.life_min = 0.3f;
  cfg.life_max = 0.8f;
  cfg.scale_min = 0.2f;
  cfg.scale_max = 0.5f;
  cfg.gravity = 0;
  cfg.colors = {0xFFFFFFFF};
  cfg.blend_mode = ParticleBlend::Additive;
  cfg.particle_shape = ParticleShape::Star;
  return cfg;
}

inline EmitterConfig rain() {
  EmitterConfig cfg;
  cfg.shape.type = EmitterShape::Line;
  cfg.shape.y = -50;
  cfg.shape.width = 2200;
  cfg.shape.x = -100;
  cfg.rate = 500;
  cfg.speed_min = 800;
  cfg.speed_max = 1200;
  cfg.direction = 1.8f;
  cfg.spread = 0.05f;
  cfg.life_min = 1;
  cfg.life_max = 2;
  cfg.gravity = 0;
  cfg.scale_min = 0.1f;
  cfg.scale_max = 0.3f;
  cfg.colors = {0x80FFFFFF};
  return cfg;
}

inline EmitterConfig fire() {
  EmitterConfig cfg;
  cfg.shape.type = EmitterShape::Circle;
  cfg.shape.radius = 30;
  cfg.rate = 100;
  cfg.speed_min = 50;
  cfg.speed_max = 150;
  cfg.direction = -1.57f;
  cfg.spread = 0.5f;
  cfg.life_min = 0.5f;
  cfg.life_max = 1.5f;
  cfg.gravity = -100;
  cfg.colors = {0xFF0000FF, 0xFF0066FF, 0xFF00CCFF};
  cfg.blend_mode = ParticleBlend::Additive;
  cfg.scale_start = 1.0f;
  cfg.scale_end = 0.0f;
  return cfg;
}

} // namespace Presets

} // namespace Effects
} // namespace CaptionEngine
