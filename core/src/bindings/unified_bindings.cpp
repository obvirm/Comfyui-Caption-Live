/**
 * @file unified_bindings.cpp
 * @brief Python bindings for Unified Render Pipeline
 *
 * Exposes the RenderPipeline to Python with numpy array support.
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "engine/render_pipeline.hpp"

namespace py = pybind11;

namespace CaptionEngine {

PYBIND11_MODULE(caption_engine_unified, m) {
  m.doc() = "Caption Engine - Unified GPU Rendering Pipeline";

  // Backend type enum
  py::enum_<GPU::BackendType>(m, "BackendType")
      .value("Auto", GPU::BackendType::Auto)
      .value("Vulkan", GPU::BackendType::Vulkan)
      .value("WebGPU", GPU::BackendType::WebGPU)
      .value("CUDA", GPU::BackendType::CUDA)
      .export_values();

  // Render target
  py::class_<RenderTarget>(m, "RenderTarget")
      .def(py::init<>())
      .def_readwrite("width", &RenderTarget::width)
      .def_readwrite("height", &RenderTarget::height)
      .def_readwrite("fps", &RenderTarget::fps);

  // Frame timing
  py::class_<FrameTiming>(m, "FrameTiming")
      .def(py::init<>())
      .def_readwrite("current_time", &FrameTiming::currentTime)
      .def_readwrite("duration", &FrameTiming::duration)
      .def_readwrite("frame_index", &FrameTiming::frameIndex)
      .def_readwrite("delta_time", &FrameTiming::deltaTime);

  // Text style
  py::class_<Text::TextStyle>(m, "TextStyle")
      .def(py::init<>())
      .def_readwrite("font_size", &Text::TextStyle::fontSize)
      .def_readwrite("outline_width", &Text::TextStyle::outlineWidth);

  // Caption layer
  py::class_<CaptionLayer>(m, "CaptionLayer")
      .def(py::init<>())
      .def_readwrite("text", &CaptionLayer::text)
      .def_readwrite("text_style", &CaptionLayer::textStyle);

  // Scene template
  py::class_<SceneTemplate>(m, "SceneTemplate")
      .def(py::init<>())
      .def_readwrite("target", &SceneTemplate::target)
      .def_readwrite("duration", &SceneTemplate::duration)
      .def_readwrite("layers", &SceneTemplate::layers)
      .def_static("from_json", &SceneTemplate::fromJSON)
      .def("to_json", &SceneTemplate::toJSON);

  // Frame output
  py::class_<FrameOutput>(m, "FrameOutput")
      .def(py::init<>())
      .def_readonly("width", &FrameOutput::width)
      .def_readonly("height", &FrameOutput::height)
      .def_readonly("timestamp", &FrameOutput::timestamp)
      .def_readonly("checksum", &FrameOutput::checksum)
      .def("to_numpy", [](const FrameOutput &self) {
        return py::array_t<uint8_t>({self.height, self.width, 4u},
                                    {self.width * 4, 4, 1}, self.pixels.data());
      });

  // Render Pipeline
  py::class_<RenderPipeline>(m, "RenderPipeline")
      .def(py::init<>())
      .def(py::init<GPU::BackendType>())
      .def("initialize", &RenderPipeline::initialize)
      .def("load_font",
           [](RenderPipeline &self, py::bytes data) {
             std::string str = data;
             return self.loadFont(reinterpret_cast<const uint8_t *>(str.data()),
                                  str.size());
           })
      .def("load_font_file", &RenderPipeline::loadFontFile)
      .def("render_frame", &RenderPipeline::renderFrame)
      .def("render_frame_composite",
           [](RenderPipeline &self, const SceneTemplate &scene,
              const FrameTiming &timing, py::array_t<uint8_t> input) {
             auto buf = input.request();
             if (buf.ndim != 3) {
               throw std::runtime_error("Input must be HxWx4 array");
             }
             uint32_t h = buf.shape[0];
             uint32_t w = buf.shape[1];
             return self.renderFrameComposite(
                 scene, timing, static_cast<const uint8_t *>(buf.ptr), w, h);
           })
      .def("render_sequence",
           [](RenderPipeline &self, const SceneTemplate &scene,
              py::function progress) {
             return self.renderSequence(
                 scene, [&progress](uint32_t frame, uint32_t total) {
                   progress(frame, total);
                 });
           })
      .def("active_backend", &RenderPipeline::activeBackend)
      .def("backend_name", &RenderPipeline::backendName)
      .def("has_cuda", &RenderPipeline::hasCUDAAcceleration)
      .def("is_ready", &RenderPipeline::isReady)
      .def_static("calculate_checksum", &RenderPipeline::calculateChecksum)
      .def_static("frames_match", &RenderPipeline::framesMatch);

  // Global pipeline accessor
  m.def("get_pipeline", &getGlobalPipeline, py::return_value_policy::reference,
        "Get the global render pipeline instance");

  // Convenience function matching old API
  m.def(
      "process_frame",
      [](const std::string &template_json, double time,
         py::array_t<uint8_t> input_image) -> py::array_t<uint8_t> {
        auto &pipeline = getGlobalPipeline();

        // Parse scene
        SceneTemplate scene = SceneTemplate::fromJSON(template_json);

        // Setup timing
        FrameTiming timing;
        timing.currentTime = time;
        timing.duration = scene.duration;

        // Get input dimensions
        auto buf = input_image.request();
        uint32_t h = buf.shape[0];
        uint32_t w = buf.shape[1];

        // Initialize if needed
        if (!pipeline.isReady()) {
          RenderTarget target;
          target.width = w;
          target.height = h;
          pipeline.initialize(target);
        }

        // Render
        FrameOutput output = pipeline.renderFrameComposite(
            scene, timing, static_cast<const uint8_t *>(buf.ptr), w, h);

        // Return as numpy array
        return py::array_t<uint8_t>({output.height, output.width, 4u},
                                    {output.width * 4, 4, 1},
                                    output.pixels.data());
      },
      "Process frame with scene template (compatibility API)",
      py::arg("template_json"), py::arg("time"), py::arg("input_image"));
}

} // namespace CaptionEngine
