#pragma once
/**
 * @file kernels.hpp
 * @brief Pre-built compute kernels for common effects
 */

#include "compute/types.hpp"
#include <functional>
#include <string>
#include <unordered_map>

namespace CaptionEngine::Kernels {

/// Kernel factory function type
using KernelFactory = std::function<ComputeKernel(ComputeKernel::Format)>;

/**
 * @brief Registry for pre-built compute kernels
 */
class KernelRegistry {
public:
  /// Get singleton instance
  static KernelRegistry &instance();

  /// Register a kernel factory
  void register_kernel(const std::string &name, KernelFactory factory);

  /// Get kernel for target format
  [[nodiscard]] std::optional<ComputeKernel>
  get(const std::string &name, ComputeKernel::Format format) const;

  /// List all registered kernel names
  [[nodiscard]] std::vector<std::string> list_kernels() const;

private:
  KernelRegistry() = default;
  std::unordered_map<std::string, KernelFactory> factories_;
};

/// Built-in kernel: Box blur
[[nodiscard]] ComputeKernel blur_box(ComputeKernel::Format format);

/// Built-in kernel: Gaussian blur (separable)
[[nodiscard]] ComputeKernel blur_gaussian(ComputeKernel::Format format);

/// Built-in kernel: Chromatic aberration
[[nodiscard]] ComputeKernel chromatic_aberration(ComputeKernel::Format format);

/// Built-in kernel: Glitch effect
[[nodiscard]] ComputeKernel glitch(ComputeKernel::Format format);

/// Built-in kernel: Color grading
[[nodiscard]] ComputeKernel color_grading(ComputeKernel::Format format);

/// Built-in kernel: Particle simulation step
[[nodiscard]] ComputeKernel particle_simulate(ComputeKernel::Format format);

/// Built-in kernel: Text SDF rendering
[[nodiscard]] ComputeKernel text_sdf(ComputeKernel::Format format);

/// Built-in kernel: Image compositing (alpha blend)
[[nodiscard]] ComputeKernel composite_alpha(ComputeKernel::Format format);

/// Initialize kernel registry with all built-in kernels
void register_builtins();

} // namespace CaptionEngine::Kernels
