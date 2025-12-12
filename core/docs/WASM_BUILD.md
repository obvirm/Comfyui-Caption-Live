# Building Caption Engine for WebAssembly (WASM)

This document describes how to build the C++ Caption Engine for WebAssembly using Emscripten.

## Prerequisites

1.  **Emscripten SDK (EMSCRIPTEN)**
    *   Must be installed and activated in your current shell.
    *   Windows (via Scoop): `scoop install emscripten`
    *   Then activate: `emsdk_env.bat` (or usually auto-handled by Scoop's shim)

2.  **CMake** (3.15+)
3.  **Ninja** or **Make**
4.  **PowerShell** (for build scripts)

## Quick Build (Windows PowerShell)

We provide a helper script `build_wasm.ps1` in the `core` directory:

```powershell
cd core
.\build_wasm.ps1
```

This script will:
1.  Create `build_wasm` directory.
2.  Configure CMake with Emscripten toolchain.
3.  Build the project.
4.  Copy artifacts (`.js`, `.wasm`, `.data`) to `../web/wasm/`.

## Manual Build Steps

If you want to run CMake manually:

```bash
mkdir build_wasm
cd build_wasm

# Configure (Windows)
emcmake cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCE_PLATFORM_WASM=ON

# Build
cmake --build . --config Release
```

## Troubleshooting & "Gotchas"

### 1. `CaptionEngine::std` Namespace Pollution
**Symptom:** Errors like `no member named 'move' in namespace 'CaptionEngine::std'`.
**Cause:** Including standard headers (or headers that include them, like `emscripten/bind.h`) *inside* a `namespace CaptionEngine { ... }` block.
**Solution:** Ensure all platform and standard includes in `platform.hpp` are **outside** the namespace block.

```cpp
// Correct structure in platform.hpp
namespace CaptionEngine {
    // ... declarations ...
} // Close namespace FIRST

// Then include platform headers
#if CE_PLATFORM_WASM
#include "platform/emscripten.hpp"
#endif
```

### 2. Linker Errors (`undefined symbol`)
**Symptom:** `wasm-ld: error: ... undefined symbol: CaptionEngine::WebGPUBackend`.
**Cause:** The engine was trying to instantiate `ComputeBackend` (aliased to `WebGPUBackend`) which is not fully implemented/linked.
**Solution:** The project has moved to a `UnifiedBackend`. Ensure `engine.cpp` uses `Compute::UnifiedBackend` instead of the legacy `ComputeBackend`.

```cpp
// src/engine.cpp
#include "compute/unified_backend.hpp"
// ...
backend = Compute::UnifiedBackend::create_best();
```

### 3. `rebind_pointer_t` Errors
**Symptom:** `no template named '__rebind_pointer_t' in namespace 'std'`.
**Cause:** Complex interaction between `nlohmann/json`, C++20, and Emscripten's `libc++`.
**Solution:**
*   Use C++17 (`set(CMAKE_CXX_STANDARD 17)`).
*   Ensure `pointer_traits` specialization for `std::unique_ptr` is in the **global** `std` namespace, not nested.

### 4. Browser Caching
**Symptom:** Changes to C++ code aren't reflected in the browser.
**Solution:** The browser caches `.wasm` files aggressively. Always **Disable Cache** in DevTools (Network tab) when developing.

## Deployment

The build produces three key files in `bin/`:
1.  `caption_engine_wasm.js`: Glue code.
2.  `caption_engine_wasm.wasm`: The compiled binary.
3.  `caption_engine_wasm.data`: Virtual filesystem (assets/fonts).

These must be deployed to `web/wasm/` so the frontend can load them. The `build_wasm.ps1` script processes this automatically.
