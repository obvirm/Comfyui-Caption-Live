/**
 * @file python.cpp
 * @brief Python bindings using pybind11
 */

#ifndef __EMSCRIPTEN__

#include "core/effects.hpp"
#include "engine/engine.hpp"
#include "engine/template.hpp"
#include "platform/pybind.hpp"


namespace py = pybind11;
using namespace CaptionEngine;

// NumPy utilities implementation
namespace CaptionEngine::Platform {

py::array_t<uint8_t> NumPyUtils::to_numpy_rgba(const uint8_t *data,
                                               uint32_t width,
                                               uint32_t height) {

  // Create array with shape (H, W, 4)
  py::array_t<uint8_t> result({height, width, 4u});
  auto buf = result.mutable_data();

  std::memcpy(buf, data, width * height * 4);

  return result;
}

std::vector<uint8_t>
NumPyUtils::from_numpy_rgba(const py::array_t<uint8_t> &arr) {
  auto buf = arr.unchecked<3>();
  size_t height = buf.shape(0);
  size_t width = buf.shape(1);
  size_t channels = buf.shape(2);

  std::vector<uint8_t> result(height * width * 4);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      size_t idx = (y * width + x) * 4;
      for (size_t c = 0; c < std::min(channels, size_t(4)); ++c) {
        result[idx + c] = buf(y, x, c);
      }
      // Fill alpha if missing
      if (channels < 4) {
        result[idx + 3] = 255;
      }
    }
  }

  return result;
}

py::array_t<float> NumPyUtils::to_numpy_float(const float *data, size_t size) {
  py::array_t<float> result(size);
  std::memcpy(result.mutable_data(), data, size * sizeof(float));
  return result;
}

bool NumPyUtils::is_contiguous(const py::array &arr) {
  return arr.flags() & py::array::c_style;
}

// GIL utilities
GILUtils::ScopedRelease::ScopedRelease() : release_() {}
GILUtils::ScopedRelease::~ScopedRelease() = default;

GILUtils::ScopedAcquire::ScopedAcquire() : acquire_() {}
GILUtils::ScopedAcquire::~ScopedAcquire() = default;

// Module registration
void register_engine_module(py::module_ &m) {
  // FrameData class
  py::class_<FrameData>(m, "FrameData")
      .def(py::init<>())
      .def_readwrite("pixels", &FrameData::pixels)
      .def_readwrite("width", &FrameData::width)
      .def_readwrite("height", &FrameData::height)
      .def_readwrite("timestamp", &FrameData::timestamp)
      .def("to_numpy", [](const FrameData &fd) {
        return NumPyUtils::to_numpy_rgba(fd.pixels.data(), fd.width, fd.height);
      });

  // BackendType enum
  py::enum_<BackendType>(m, "BackendType")
      .value("Auto", BackendType::Auto)
      .value("CPU", BackendType::CPU)
      .value("WebGPU", BackendType::WebGPU)
      .value("Vulkan", BackendType::Vulkan)
      .value("CUDA", BackendType::CUDA)
      .value("Metal", BackendType::Metal)
      .export_values();

  // Quality enum
  py::enum_<Quality>(m, "Quality")
      .value("Draft", Quality::Draft)
      .value("Preview", Quality::Preview)
      .value("Final", Quality::Final)
      .export_values();

  // EngineConfig class
  py::class_<EngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("width", &EngineConfig::width)
      .def_readwrite("height", &EngineConfig::height)
      .def_readwrite("preferred_backend", &EngineConfig::preferred_backend)
      .def_readwrite("quality", &EngineConfig::quality);

  // Engine class
  py::class_<Engine>(m, "Engine")
      .def(py::init<const EngineConfig &>())
      .def("render_frame", &Engine::render_frame, py::arg("template_json"),
           py::arg("time"), py::call_guard<py::gil_scoped_release>())
      .def("export_png", &Engine::export_png)
      .def("current_backend", &Engine::current_backend)
      .def("compute_frame_hash", &Engine::compute_frame_hash);
}

void register_template_module(py::module_ &m) {
  py::class_<Position>(m, "Position")
      .def(py::init<>())
      .def_readwrite("x", &Position::x)
      .def_readwrite("y", &Position::y)
      .def_readwrite("anchor_x", &Position::anchor_x)
      .def_readwrite("anchor_y", &Position::anchor_y);

  py::class_<TextStyle>(m, "TextStyle")
      .def(py::init<>())
      .def_readwrite("font_family", &TextStyle::font_family)
      .def_readwrite("font_size", &TextStyle::font_size)
      .def_readwrite("color", &TextStyle::color)
      .def_readwrite("stroke_color", &TextStyle::stroke_color)
      .def_readwrite("stroke_width", &TextStyle::stroke_width);

  py::class_<Segment>(m, "Segment")
      .def(py::init<>())
      .def_readwrite("text", &Segment::text)
      .def_readwrite("start_time", &Segment::start_time)
      .def_readwrite("end_time", &Segment::end_time);

  py::class_<Template>(m, "Template")
      .def(py::init<>())
      .def_static("from_json", &Template::from_json)
      .def("to_json", &Template::to_json)
      .def("hash", &Template::hash);
}

void register_effects_module(py::module_ &m) {
  py::enum_<EffectType>(m, "EffectType")
      .value("Bounce", EffectType::Bounce)
      .value("Typewriter", EffectType::Typewriter)
      .value("ColoredWord", EffectType::ColoredWord)
      .value("BoxHighlight", EffectType::BoxHighlight)
      .value("Karaoke", EffectType::Karaoke)
      .value("Wave", EffectType::Wave)
      .value("Shake", EffectType::Shake)
      .value("Confetti", EffectType::Confetti)
      .value("Sparkle", EffectType::Sparkle)
      .value("Rain", EffectType::Rain)
      .value("Fire", EffectType::Fire)
      .value("Glitch", EffectType::Glitch)
      .value("ChromaticAberration", EffectType::ChromaticAberration)
      .value("VHS", EffectType::VHS)
      .value("Fade", EffectType::Fade)
      .value("Dissolve", EffectType::Dissolve)
      .export_values();

  m.def("init_effects", &init_effects);
  m.def("effect_type_to_string", &effect_type_to_string);
  m.def("string_to_effect_type", &string_to_effect_type);
}

} // namespace CaptionEngine::Platform

// Main pybind11 module definition
PYBIND11_MODULE(caption_engine_py, m) {
  m.doc() = "Caption Engine - C++ Backend for ComfyUI";

  CaptionEngine::Platform::register_engine_module(m);
  CaptionEngine::Platform::register_template_module(m);
  CaptionEngine::Platform::register_effects_module(m);
}

#endif // !__EMSCRIPTEN__
