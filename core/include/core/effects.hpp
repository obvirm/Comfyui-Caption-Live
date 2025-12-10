#pragma once
/**
 * @file effects.hpp
 * @brief Effect registry and factory
 */

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace CaptionEngine {

/// Effect type enumeration
enum class EffectType {
  // Text Effects
  Bounce,
  Typewriter,
  ColoredWord,
  BoxHighlight,
  Karaoke,
  Wave,
  Shake,

  // Particle Effects
  Confetti,
  Sparkle,
  Rain,
  Fire,
  Snow,

  // Distortion Effects
  Glitch,
  ChromaticAberration,
  VHS,
  Wave_Distort,
  ZoomBlur,
  MotionBlur,
  Pixelate,

  // Transitions
  Fade,
  Dissolve,
  Wipe,
  CircleWipe,
  DiagonalWipe,
  RadialWipe,
  Slide,
  Zoom,
  Spin
};

/// Effect parameter types
enum class ParamType {
  Float,
  Int,
  Bool,
  Color, // RGBA uint32
  String,
  Vec2,
  Vec3,
  Vec4
};

/// Parameter descriptor
struct EffectParam {
  std::string name;
  ParamType type;
  float default_float = 0;
  int default_int = 0;
  bool default_bool = false;
  uint32_t default_color = 0xFFFFFFFF;
  float min_val = 0;
  float max_val = 1;
};

/// Effect descriptor
struct EffectDescriptor {
  std::string name;
  std::string category;
  EffectType type;
  std::vector<EffectParam> params;
  bool supports_gpu = true;
  bool deterministic = true;
};

/// Effect instance runtime state
struct EffectState {
  EffectType type;
  std::unordered_map<std::string, float> float_params;
  std::unordered_map<std::string, int> int_params;
  std::unordered_map<std::string, bool> bool_params;
  std::unordered_map<std::string, uint32_t> color_params;

  float get_float(const std::string &name, float default_val = 0) const {
    auto it = float_params.find(name);
    return it != float_params.end() ? it->second : default_val;
  }

  int get_int(const std::string &name, int default_val = 0) const {
    auto it = int_params.find(name);
    return it != int_params.end() ? it->second : default_val;
  }

  bool get_bool(const std::string &name, bool default_val = false) const {
    auto it = bool_params.find(name);
    return it != bool_params.end() ? it->second : default_val;
  }

  uint32_t get_color(const std::string &name,
                     uint32_t default_val = 0xFFFFFFFF) const {
    auto it = color_params.find(name);
    return it != color_params.end() ? it->second : default_val;
  }
};

/**
 * @brief Effect registry for all available effects
 */
class EffectRegistry {
public:
  /// Get singleton instance
  static EffectRegistry &instance();

  /// Register an effect descriptor
  void register_effect(const EffectDescriptor &desc);

  /// Get effect by type
  [[nodiscard]] const EffectDescriptor *get(EffectType type) const;

  /// Get effect by name
  [[nodiscard]] const EffectDescriptor *get(const std::string &name) const;

  /// List all registered effects
  [[nodiscard]] std::vector<EffectDescriptor> list_all() const;

  /// List effects by category
  [[nodiscard]] std::vector<EffectDescriptor>
  list_by_category(const std::string &category) const;

  /// Create default state for effect
  [[nodiscard]] EffectState create_default_state(EffectType type) const;

private:
  EffectRegistry();
  void register_builtins();

  std::unordered_map<EffectType, EffectDescriptor> effects_;
  std::unordered_map<std::string, EffectType> name_map_;
};

/// Initialize all built-in effects
void init_effects();

/// Effect string conversion
[[nodiscard]] std::string effect_type_to_string(EffectType type);
[[nodiscard]] EffectType string_to_effect_type(const std::string &name);

} // namespace CaptionEngine
