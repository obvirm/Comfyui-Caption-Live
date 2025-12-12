/**
 * @file deterministic.cpp
 * @brief Deterministic core implementation (minimal - most is header-only)
 *
 * This file is kept minimal as most functionality has been moved to
 * the header for compile-time optimization. It exists primarily for
 * backward compatibility and any future non-inline implementations.
 */

#include "core/deterministic.hpp"

// All implementation is now in deterministic.hpp for inlining optimization.
// This file is retained in case we need to add non-inline implementations
// in the future (e.g., for very large functions that shouldn't be inlined).

namespace CaptionEngine::Deterministic {

// Currently empty - all functions are header-only

} // namespace CaptionEngine::Deterministic
