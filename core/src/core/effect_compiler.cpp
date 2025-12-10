/**
 * @file effect_compiler.cpp
 * @brief Effect configuration compiler and validator
 */

#include "core/effects.hpp"
#include <algorithm>
#include <stdexcept>

namespace CaptionEngine {

// Singleton instance
EffectRegistry &EffectRegistry::instance() {
  static EffectRegistry instance;
  return instance;
}

EffectRegistry::EffectRegistry() { register_builtins(); }

void EffectRegistry::register_effect(const EffectDescriptor &desc) {
  effects_[desc.type] = desc;
  name_map_[desc.name] = desc.type;
}

const EffectDescriptor *EffectRegistry::get(EffectType type) const {
  auto it = effects_.find(type);
  return it != effects_.end() ? &it->second : nullptr;
}

const EffectDescriptor *EffectRegistry::get(const std::string &name) const {
  auto it = name_map_.find(name);
  if (it == name_map_.end())
    return nullptr;
  return get(it->second);
}

std::vector<EffectDescriptor> EffectRegistry::list_all() const {
  std::vector<EffectDescriptor> result;
  result.reserve(effects_.size());
  for (const auto &[type, desc] : effects_) {
    result.push_back(desc);
  }
  return result;
}

std::vector<EffectDescriptor>
EffectRegistry::list_by_category(const std::string &category) const {
  std::vector<EffectDescriptor> result;
  for (const auto &[type, desc] : effects_) {
    if (desc.category == category) {
      result.push_back(desc);
    }
  }
  return result;
}

EffectState EffectRegistry::create_default_state(EffectType type) const {
  EffectState state;
  state.type = type;

  const auto *desc = get(type);
  if (!desc)
    return state;

  for (const auto &param : desc->params) {
    switch (param.type) {
    case ParamType::Float:
      state.float_params[param.name] = param.default_float;
      break;
    case ParamType::Int:
      state.int_params[param.name] = param.default_int;
      break;
    case ParamType::Bool:
      state.bool_params[param.name] = param.default_bool;
      break;
    case ParamType::Color:
      state.color_params[param.name] = param.default_color;
      break;
    default:
      break;
    }
  }

  return state;
}

void EffectRegistry::register_builtins() {
  // Text Effects
  register_effect(
      {"bounce",
       "Text",
       EffectType::Bounce,
       {{"scale", ParamType::Float, 1.25f, 0, false, 0, 1.0f, 2.0f},
        {"duration", ParamType::Float, 0.15f, 0, false, 0, 0.05f, 1.0f}},
       true,
       true});

  register_effect(
      {"typewriter",
       "Text",
       EffectType::Typewriter,
       {{"speed", ParamType::Float, 20.0f, 0, false, 0, 1.0f, 100.0f},
        {"show_cursor", ParamType::Bool, 0, 0, true, 0, 0, 0}},
       true,
       true});

  register_effect(
      {"colored_word",
       "Text",
       EffectType::ColoredWord,
       {{"active_color", ParamType::Color, 0, 0, false, 0xFF5FE539, 0, 0},
        {"inactive_color", ParamType::Color, 0, 0, false, 0xFFFFFFFF, 0, 0}},
       true,
       true});

  register_effect(
      {"box_highlight",
       "Text",
       EffectType::BoxHighlight,
       {{"box_color", ParamType::Color, 0, 0, false, 0xFF5FE539, 0, 0},
        {"radius", ParamType::Float, 8.0f, 0, false, 0, 0, 50.0f},
        {"padding", ParamType::Float, 8.0f, 0, false, 0, 0, 50.0f}},
       true,
       true});

  register_effect(
      {"karaoke",
       "Text",
       EffectType::Karaoke,
       {{"sung_color", ParamType::Color, 0, 0, false, 0xFF00FFFF, 0, 0},
        {"unsung_color", ParamType::Color, 0, 0, false, 0xFFFFFFFF, 0, 0}},
       true,
       true});

  register_effect(
      {"wave",
       "Text",
       EffectType::Wave,
       {{"amplitude", ParamType::Float, 5.0f, 0, false, 0, 0, 50.0f},
        {"frequency", ParamType::Float, 4.0f, 0, false, 0, 0.1f, 20.0f},
        {"speed", ParamType::Float, 2.0f, 0, false, 0, 0.1f, 10.0f}},
       true,
       true});

  register_effect(
      {"shake",
       "Text",
       EffectType::Shake,
       {{"intensity", ParamType::Float, 3.0f, 0, false, 0, 0, 20.0f},
        {"frequency", ParamType::Float, 30.0f, 0, false, 0, 1.0f, 60.0f}},
       true,
       true});

  // Particle Effects
  register_effect(
      {"confetti",
       "Particles",
       EffectType::Confetti,
       {{"rate", ParamType::Float, 50.0f, 0, false, 0, 1.0f, 200.0f},
        {"gravity", ParamType::Float, 200.0f, 0, false, 0, 0, 1000.0f}},
       true,
       true});

  register_effect(
      {"sparkle",
       "Particles",
       EffectType::Sparkle,
       {{"rate", ParamType::Float, 20.0f, 0, false, 0, 1.0f, 100.0f},
        {"size", ParamType::Float, 0.5f, 0, false, 0, 0.1f, 2.0f}},
       true,
       true});

  register_effect(
      {"rain",
       "Particles",
       EffectType::Rain,
       {{"rate", ParamType::Float, 500.0f, 0, false, 0, 10.0f, 2000.0f},
        {"speed", ParamType::Float, 1000.0f, 0, false, 0, 100.0f, 2000.0f}},
       true,
       true});

  register_effect(
      {"fire",
       "Particles",
       EffectType::Fire,
       {{"rate", ParamType::Float, 100.0f, 0, false, 0, 10.0f, 500.0f},
        {"size", ParamType::Float, 1.0f, 0, false, 0, 0.1f, 3.0f}},
       true,
       true});

  // Distortion Effects
  register_effect(
      {"glitch",
       "Distortion",
       EffectType::Glitch,
       {{"intensity", ParamType::Float, 0.1f, 0, false, 0, 0, 1.0f},
        {"probability", ParamType::Float, 0.1f, 0, false, 0, 0, 1.0f},
        {"rgb_split", ParamType::Float, 5.0f, 0, false, 0, 0, 50.0f}},
       true,
       true});

  register_effect({"chromatic_aberration",
                   "Distortion",
                   EffectType::ChromaticAberration,
                   {{"strength", ParamType::Float, 0.01f, 0, false, 0, 0, 0.1f},
                    {"radial", ParamType::Bool, 0, 0, true, 0, 0, 0}},
                   true,
                   true});

  register_effect(
      {"vhs",
       "Distortion",
       EffectType::VHS,
       {{"scanline_strength", ParamType::Float, 0.2f, 0, false, 0, 0, 1.0f},
        {"noise_strength", ParamType::Float, 0.05f, 0, false, 0, 0, 0.5f}},
       true,
       true});

  // Transitions
  register_effect(
      {"fade",
       "Transition",
       EffectType::Fade,
       {{"duration", ParamType::Float, 0.5f, 0, false, 0, 0.1f, 5.0f}},
       true,
       true});

  register_effect(
      {"dissolve",
       "Transition",
       EffectType::Dissolve,
       {{"duration", ParamType::Float, 0.5f, 0, false, 0, 0.1f, 5.0f},
        {"softness", ParamType::Float, 0.1f, 0, false, 0, 0, 0.5f}},
       true,
       true});

  register_effect(
      {"wipe",
       "Transition",
       EffectType::Wipe,
       {
           {"duration", ParamType::Float, 0.5f, 0, false, 0, 0.1f, 5.0f},
           {"softness", ParamType::Float, 0.1f, 0, false, 0, 0, 0.5f},
           {"direction", ParamType::Int, 0, 0, false, 0, 0, 3}
           // 0=left, 1=right, 2=up, 3=down
       },
       true,
       true});
}

// String conversion utilities
std::string effect_type_to_string(EffectType type) {
  const auto *desc = EffectRegistry::instance().get(type);
  return desc ? desc->name : "unknown";
}

EffectType string_to_effect_type(const std::string &name) {
  const auto *desc = EffectRegistry::instance().get(name);
  if (!desc) {
    throw std::runtime_error("Unknown effect: " + name);
  }
  return desc->type;
}

void init_effects() {
  // Force singleton initialization
  EffectRegistry::instance();
}

} // namespace CaptionEngine
