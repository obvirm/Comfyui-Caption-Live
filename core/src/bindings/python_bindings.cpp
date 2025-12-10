/**
 * @file python_bindings.cpp
 * @brief Python bindings using pybind11
 */

#include "engine/engine.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Global engine instance for efficiency
static std::unique_ptr<CaptionEngine::Engine> g_engine;

CaptionEngine::Engine &get_engine() {
  if (!g_engine) {
    g_engine = std::make_unique<CaptionEngine::Engine>();
  }
  return *g_engine;
}

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

  // =========================================================================
  // UNIFIED PROCESS_FRAME: Accepts input numpy array, returns composited array
  // =========================================================================
  m.def(
      "process_frame",
      [](const std::string &template_json, double time,
         py::array_t<float, py::array::c_style | py::array::forcecast>
             input_image) {
        // Get buffer info
        py::buffer_info buf = input_image.request();

        if (buf.ndim != 3) {
          throw std::runtime_error("Input must be 3D array (H, W, C)");
        }

        int height = static_cast<int>(buf.shape[0]);
        int width = static_cast<int>(buf.shape[1]);
        int channels = static_cast<int>(buf.shape[2]);

        if (channels < 3 || channels > 4) {
          throw std::runtime_error(
              "Input must have 3 or 4 channels (RGB/RGBA)");
        }

        float *ptr = static_cast<float *>(buf.ptr);

        // Convert float [0-1] input to uint8 FrameData
        CaptionEngine::FrameData input_frame;
        input_frame.width = width;
        input_frame.height = height;
        input_frame.pixels.resize(width * height * 4);

        for (int i = 0; i < height * width; ++i) {
          int src_idx = i * channels;
          int dst_idx = i * 4;

          input_frame.pixels[dst_idx + 0] = static_cast<uint8_t>(
              std::clamp(ptr[src_idx + 0] * 255.0f, 0.0f, 255.0f));
          input_frame.pixels[dst_idx + 1] = static_cast<uint8_t>(
              std::clamp(ptr[src_idx + 1] * 255.0f, 0.0f, 255.0f));
          input_frame.pixels[dst_idx + 2] = static_cast<uint8_t>(
              std::clamp(ptr[src_idx + 2] * 255.0f, 0.0f, 255.0f));
          input_frame.pixels[dst_idx + 3] =
              (channels == 4) ? static_cast<uint8_t>(std::clamp(
                                    ptr[src_idx + 3] * 255.0f, 0.0f, 255.0f))
                              : 255;
        }

        // Render caption overlay and composite
        auto &engine = get_engine();
        auto result =
            engine.render_frame_composite(template_json, time, input_frame);

        // Convert uint8 result back to float numpy array
        auto output = py::array_t<float>({height, width, 4});
        py::buffer_info out_buf = output.request();
        float *out_ptr = static_cast<float *>(out_buf.ptr);

        for (size_t i = 0; i < result.pixels.size(); ++i) {
          out_ptr[i] = result.pixels[i] / 255.0f;
        }

        return output;
      },
      py::arg("template_json"), py::arg("time"), py::arg("input_image"),
      "Process a single frame: render caption and composite over input image.\n"
      "Args:\n"
      "  template_json: JSON scene description\n"
      "  time: Current timestamp in seconds\n"
      "  input_image: Input numpy array (H, W, C) float32 [0-1]\n"
      "Returns:\n"
      "  Composited numpy array (H, W, 4) float32 [0-1]");

  // Legacy render function
  m.def(
      "render_caption_frame",
      [](const std::string &template_json, double time) {
        auto &engine = get_engine();
        auto frame = engine.render_frame(template_json, time);
        return engine.export_png(frame);
      },
      "Render a single caption frame as PNG bytes");
}
