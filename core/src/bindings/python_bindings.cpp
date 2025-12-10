/**
 * @file python_bindings.cpp
 * @brief Python bindings using pybind11
 */

#include "engine/engine.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(caption_engine_py, m) {
  m.doc() = "Caption Engine - C++ Backend for ComfyUI";

  // FrameData
  py::class_<CaptionEngine::FrameData>(m, "FrameData")
      .def(py::init<>())
      .def_readwrite("pixels", &CaptionEngine::FrameData::pixels)
      .def_readwrite("width", &CaptionEngine::FrameData::width)
      .def_readwrite("height", &CaptionEngine::FrameData::height)
      .def_readwrite("timestamp", &CaptionEngine::FrameData::timestamp);

  // BackendType enum
  py::enum_<CaptionEngine::BackendType>(m, "BackendType")
      .value("Auto", CaptionEngine::BackendType::Auto)
      .value("CPU", CaptionEngine::BackendType::CPU)
      .value("WebGPU", CaptionEngine::BackendType::WebGPU)
      .value("Vulkan", CaptionEngine::BackendType::Vulkan)
      .value("CUDA", CaptionEngine::BackendType::CUDA)
      .value("Metal", CaptionEngine::BackendType::Metal);

  // Quality enum
  py::enum_<CaptionEngine::Quality>(m, "Quality")
      .value("Low", CaptionEngine::Quality::Low)
      .value("Medium", CaptionEngine::Quality::Medium)
      .value("High", CaptionEngine::Quality::High)
      .value("Ultra", CaptionEngine::Quality::Ultra);

  // EngineConfig
  py::class_<CaptionEngine::EngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("backend", &CaptionEngine::EngineConfig::backend)
      .def_readwrite("quality", &CaptionEngine::EngineConfig::quality)
      .def_readwrite("deterministic",
                     &CaptionEngine::EngineConfig::deterministic);

  // Engine
  py::class_<CaptionEngine::Engine>(m, "Engine")
      .def(py::init<>())
      .def(py::init<const CaptionEngine::EngineConfig &>())
      .def("render_frame", &CaptionEngine::Engine::render_frame)
      .def("export_png", &CaptionEngine::Engine::export_png)
      .def("current_backend", &CaptionEngine::Engine::current_backend)
      .def("compute_frame_hash", &CaptionEngine::Engine::compute_frame_hash);

  // Convenience function for ComfyUI
  m.def(
      "render_caption_frame",
      [](const std::string &template_json, double time) {
        CaptionEngine::Engine engine;
        auto frame = engine.render_frame(template_json, time);
        return engine.export_png(frame);
      },
      "Render a single caption frame");
}
