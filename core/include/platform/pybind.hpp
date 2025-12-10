#pragma once
/**
 * @file pybind.hpp
 * @brief Python bindings configuration and utilities
 */

#ifndef __EMSCRIPTEN__

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace CaptionEngine::Platform {

namespace py = pybind11;

/**
 * @brief NumPy array utilities
 */
class NumPyUtils {
public:
  /// Convert RGBA pixel buffer to NumPy array (H, W, 4)
  static py::array_t<uint8_t> to_numpy_rgba(const uint8_t *data, uint32_t width,
                                            uint32_t height);

  /// Convert NumPy array to RGBA pixel buffer
  static std::vector<uint8_t> from_numpy_rgba(const py::array_t<uint8_t> &arr);

  /// Convert float buffer to NumPy array
  static py::array_t<float> to_numpy_float(const float *data, size_t size);

  /// Check if array is contiguous
  static bool is_contiguous(const py::array &arr);

  /// Get array shape as tuple
  template <typename T>
  static std::vector<size_t> shape(const py::array_t<T> &arr);
};

/**
 * @brief PyTorch tensor interop
 */
class TorchInterop {
public:
  /// Check if PyTorch is available
  static bool available();

  /// Convert pixel buffer to PyTorch tensor (requires torch)
  static py::object to_torch_tensor(const uint8_t *data, uint32_t width,
                                    uint32_t height);

  /// Convert PyTorch tensor to pixel buffer
  static std::vector<uint8_t> from_torch_tensor(const py::object &tensor);

  /// Get tensor device (cpu, cuda:0, etc.)
  static std::string tensor_device(const py::object &tensor);
};

/**
 * @brief GIL (Global Interpreter Lock) utilities
 */
class GILUtils {
public:
  /// Release GIL for long-running operations
  class ScopedRelease {
  public:
    ScopedRelease();
    ~ScopedRelease();

  private:
    py::gil_scoped_release release_;
  };

  /// Acquire GIL for Python calls
  class ScopedAcquire {
  public:
    ScopedAcquire();
    ~ScopedAcquire();

  private:
    py::gil_scoped_acquire acquire_;
  };
};

/// Python module initialization helper
void register_engine_module(py::module_ &m);
void register_template_module(py::module_ &m);
void register_effects_module(py::module_ &m);

} // namespace CaptionEngine::Platform

#endif // !__EMSCRIPTEN__
