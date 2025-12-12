#pragma once
/**
 * @file types.hpp
 * @brief Shared types for text rendering
 *
 * Common types used across render_pipeline and text rendering.
 */

#include <glm/glm.hpp>
#include <string>

namespace CaptionEngine {

/// Text alignment
enum class TextAlign { Left, Center, Right };

/// Text style for captions and rendering
struct TextStyle {
  glm::vec4 color = glm::vec4(1.0f);        ///< RGBA
  glm::vec4 outlineColor = glm::vec4(0.0f); ///< Outline RGBA
  float outlineWidth = 0.0f;                ///< Outline size
  float fontSize = 48.0f;                   ///< Display size
  TextAlign align = TextAlign::Center;
};

} // namespace CaptionEngine
