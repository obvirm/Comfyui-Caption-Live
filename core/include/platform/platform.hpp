#pragma once
/**
 * @file platform.hpp
 * @brief Unified Platform Detection and Configuration
 *
 * This header provides centralized platform detection macros and
 * configuration for the CaptionEngine. Include this header to get
 * consistent platform detection across the entire codebase.
 *
 * Usage:
 *   #include "platform/platform.hpp"
 *
 *   #if CE_PLATFORM_WASM
 *       // WebAssembly-specific code
 *   #elif CE_PLATFORM_WINDOWS
 *       // Windows-specific code
 *   #endif
 */

#include <cstdint>
#include <string_view>

namespace CaptionEngine {

// ============================================================================
// PLATFORM DETECTION
// ============================================================================

// Operating System Detection
#if defined(__EMSCRIPTEN__)
#define CE_PLATFORM_WASM 1
#define CE_PLATFORM_NAME "WebAssembly"
#elif defined(_WIN32) || defined(_WIN64)
#define CE_PLATFORM_WINDOWS 1
#define CE_PLATFORM_NAME "Windows"
#ifdef _WIN64
#define CE_PLATFORM_64BIT 1
#else
#define CE_PLATFORM_32BIT 1
#endif
#elif defined(__linux__)
#define CE_PLATFORM_LINUX 1
#define CE_PLATFORM_NAME "Linux"
#define CE_PLATFORM_UNIX 1
#elif defined(__APPLE__) && defined(__MACH__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
#define CE_PLATFORM_IOS 1
#define CE_PLATFORM_NAME "iOS"
#else
#define CE_PLATFORM_MACOS 1
#define CE_PLATFORM_NAME "macOS"
#endif
#define CE_PLATFORM_APPLE 1
#define CE_PLATFORM_UNIX 1
#elif defined(__ANDROID__)
#define CE_PLATFORM_ANDROID 1
#define CE_PLATFORM_NAME "Android"
#define CE_PLATFORM_UNIX 1
#else
#define CE_PLATFORM_UNKNOWN 1
#define CE_PLATFORM_NAME "Unknown"
#endif

// Default to 0 for undefined platforms
#ifndef CE_PLATFORM_WASM
#define CE_PLATFORM_WASM 0
#endif
#ifndef CE_PLATFORM_WINDOWS
#define CE_PLATFORM_WINDOWS 0
#endif
#ifndef CE_PLATFORM_LINUX
#define CE_PLATFORM_LINUX 0
#endif
#ifndef CE_PLATFORM_MACOS
#define CE_PLATFORM_MACOS 0
#endif
#ifndef CE_PLATFORM_IOS
#define CE_PLATFORM_IOS 0
#endif
#ifndef CE_PLATFORM_ANDROID
#define CE_PLATFORM_ANDROID 0
#endif
#ifndef CE_PLATFORM_APPLE
#define CE_PLATFORM_APPLE 0
#endif
#ifndef CE_PLATFORM_UNIX
#define CE_PLATFORM_UNIX 0
#endif

// Desktop vs Mobile
#define CE_PLATFORM_DESKTOP                                                    \
  (CE_PLATFORM_WINDOWS || CE_PLATFORM_LINUX || CE_PLATFORM_MACOS)
#define CE_PLATFORM_MOBILE (CE_PLATFORM_IOS || CE_PLATFORM_ANDROID)

// ============================================================================
// COMPILER DETECTION
// ============================================================================

#if defined(__clang__)
#define CE_COMPILER_CLANG 1
#define CE_COMPILER_NAME "Clang"
#define CE_COMPILER_VERSION __clang_major__
#elif defined(__GNUC__) || defined(__GNUG__)
#define CE_COMPILER_GCC 1
#define CE_COMPILER_NAME "GCC"
#define CE_COMPILER_VERSION __GNUC__
#elif defined(_MSC_VER)
#define CE_COMPILER_MSVC 1
#define CE_COMPILER_NAME "MSVC"
#define CE_COMPILER_VERSION _MSC_VER
#elif defined(__NVCC__)
#define CE_COMPILER_NVCC 1
#define CE_COMPILER_NAME "NVCC"
#else
#define CE_COMPILER_UNKNOWN 1
#define CE_COMPILER_NAME "Unknown"
#endif

// ============================================================================
// GPU BACKEND DETECTION
// ============================================================================

// CUDA (NVIDIA)
#if defined(CAPTION_HAS_CUDA) || defined(HAS_CUDA) || defined(__CUDACC__)
#define CE_HAS_CUDA 1
#else
#define CE_HAS_CUDA 0
#endif

// Vulkan
#if defined(CAPTION_HAS_VULKAN) || defined(HAS_VULKAN)
#define CE_HAS_VULKAN 1
#else
#define CE_HAS_VULKAN 0
#endif

// Metal (Apple only)
#if CE_PLATFORM_APPLE && (defined(CAPTION_HAS_METAL) || defined(HAS_METAL))
#define CE_HAS_METAL 1
#else
#define CE_HAS_METAL 0
#endif

// DirectX 12 (Windows only)
#if CE_PLATFORM_WINDOWS && (defined(CAPTION_HAS_DX12) || defined(HAS_DX12))
#define CE_HAS_DX12 1
#else
#define CE_HAS_DX12 0
#endif

// WebGPU (WASM via Dawn, or native via wgpu-native)
#if CE_PLATFORM_WASM || defined(CAPTION_HAS_WEBGPU) || defined(HAS_WEBGPU)
#define CE_HAS_WEBGPU 1
#else
#define CE_HAS_WEBGPU 0
#endif

// ============================================================================
// DEFAULT BACKEND SELECTION
// ============================================================================

namespace Platform {

/// Backend preference order for each platform
enum class BackendPreference {
  Auto,
  CUDA,
  Vulkan,
  Metal,
  DirectX12,
  WebGPU,
  CPU
};

/// Get default backend for current platform
constexpr BackendPreference get_default_backend() {
#if CE_PLATFORM_WASM
  return BackendPreference::WebGPU;
#elif CE_PLATFORM_WINDOWS
#if CE_HAS_CUDA
  return BackendPreference::CUDA;
#elif CE_HAS_VULKAN
  return BackendPreference::Vulkan;
#elif CE_HAS_DX12
  return BackendPreference::DirectX12;
#else
  return BackendPreference::CPU;
#endif
#elif CE_PLATFORM_LINUX
#if CE_HAS_CUDA
  return BackendPreference::CUDA;
#elif CE_HAS_VULKAN
  return BackendPreference::Vulkan;
#else
  return BackendPreference::CPU;
#endif
#elif CE_PLATFORM_MACOS
#if CE_HAS_METAL
  return BackendPreference::Metal;
#elif CE_HAS_VULKAN
  return BackendPreference::Vulkan;
#else
  return BackendPreference::CPU;
#endif
#else
  return BackendPreference::CPU;
#endif
}

/// Get platform name as string
constexpr std::string_view get_platform_name() { return CE_PLATFORM_NAME; }

/// Check if running in browser (WebAssembly)
constexpr bool is_browser() { return CE_PLATFORM_WASM; }

/// Check if running on desktop
constexpr bool is_desktop() { return CE_PLATFORM_DESKTOP; }

/// Check if running on mobile
constexpr bool is_mobile() { return CE_PLATFORM_MOBILE; }

/// Check if GPU compute is available
constexpr bool has_gpu_compute() {
  return CE_HAS_CUDA || CE_HAS_VULKAN || CE_HAS_METAL || CE_HAS_WEBGPU;
}

} // namespace Platform

// ============================================================================
// FEATURE FLAGS
// ============================================================================

// SIMD Support
#if defined(__SSE4_2__) || defined(__AVX__) || defined(__AVX2__)
#define CE_HAS_SIMD_X86 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define CE_HAS_SIMD_NEON 1
#endif
#if CE_PLATFORM_WASM && defined(__wasm_simd128__)
#define CE_HAS_SIMD_WASM 1
#endif

#define CE_HAS_SIMD                                                            \
  (defined(CE_HAS_SIMD_X86) || defined(CE_HAS_SIMD_NEON) ||                    \
   defined(CE_HAS_SIMD_WASM))

// Threading Support
#if !CE_PLATFORM_WASM
#define CE_HAS_THREADS 1
#else
  // WASM may have threads via SharedArrayBuffer
#ifdef __EMSCRIPTEN_PTHREADS__
#define CE_HAS_THREADS 1
#else
#define CE_HAS_THREADS 0
#endif
#endif

// Filesystem Support
#if CE_PLATFORM_WASM
#define CE_HAS_FILESYSTEM 0 // Use Emscripten FS API
#else
#define CE_HAS_FILESYSTEM 1
#endif

// ============================================================================
// EXPORT/IMPORT MACROS
// ============================================================================

// DLL Export/Import for Windows
#if CE_PLATFORM_WINDOWS
#ifdef CAPTION_ENGINE_EXPORTS
#define CE_API __declspec(dllexport)
#else
#define CE_API __declspec(dllimport)
#endif
#else
#define CE_API __attribute__((visibility("default")))
#endif

// Force inline
#if CE_COMPILER_MSVC
#define CE_FORCEINLINE __forceinline
#elif CE_COMPILER_GCC || CE_COMPILER_CLANG
#define CE_FORCEINLINE __attribute__((always_inline)) inline
#else
#define CE_FORCEINLINE inline
#endif

// Likely/Unlikely branch hints
#if CE_COMPILER_GCC || CE_COMPILER_CLANG
#define CE_LIKELY(x) __builtin_expect(!!(x), 1)
#define CE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define CE_LIKELY(x) (x)
#define CE_UNLIKELY(x) (x)
#endif

// ============================================================================
// DEBUGGING AND LOGGING
// ============================================================================

#ifdef NDEBUG
#define CE_DEBUG 0
#else
#define CE_DEBUG 1
#endif

// Debug assertion
#if CE_DEBUG
#include <cassert>
#define CE_ASSERT(cond, msg) assert((cond) && (msg))
#else
#define CE_ASSERT(cond, msg) ((void)0)
#endif

} // namespace CaptionEngine

// ============================================================================
// PLATFORM-SPECIFIC INCLUDES
// ============================================================================

// Include platform-specific headers
#if CE_PLATFORM_WASM
#include "platform/emscripten.hpp"
#endif

// ============================================================================
// SUMMARY MACROS (for easy checking)
// ============================================================================

/**
 * Quick reference:
 *
 * Platform:
 *   CE_PLATFORM_WASM      - WebAssembly (Emscripten)
 *   CE_PLATFORM_WINDOWS   - Windows (32/64-bit)
 *   CE_PLATFORM_LINUX     - Linux
 *   CE_PLATFORM_MACOS     - macOS
 *   CE_PLATFORM_IOS       - iOS
 *   CE_PLATFORM_ANDROID   - Android
 *   CE_PLATFORM_DESKTOP   - Desktop (Windows/Linux/macOS)
 *   CE_PLATFORM_MOBILE    - Mobile (iOS/Android)
 *
 * GPU Backends:
 *   CE_HAS_CUDA           - NVIDIA CUDA
 *   CE_HAS_VULKAN         - Vulkan
 *   CE_HAS_METAL          - Apple Metal
 *   CE_HAS_DX12           - DirectX 12
 *   CE_HAS_WEBGPU         - WebGPU
 *
 * Features:
 *   CE_HAS_SIMD           - SIMD support available
 *   CE_HAS_THREADS        - Threading support
 *   CE_HAS_FILESYSTEM     - Native filesystem access
 *   CE_DEBUG              - Debug build
 */
