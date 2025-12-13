cmake_minimum_required(VERSION 3.20)
project(CaptionEngine 
    VERSION 2.0.0 
    LANGUAGES CXX
)

# Modern C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_WASM "Build for WebAssembly" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(ENABLE_METAL "Enable Metal backend (macOS)" OFF)
option(ENABLE_UNIFIED_PIPELINE "Enable unified render pipeline" ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Third-party dependencies
add_subdirectory(third_party)

# =============================================================================
# Core Engine Library
# =============================================================================
set(CORE_SOURCES
    src/engine.cpp
    src/template.cpp
    src/renderer.cpp
    src/text_renderer.cpp
    src/core/deterministic.cpp
    src/core/template_parser.cpp
    src/core/effect_compiler.cpp
    src/core/frame_validator.cpp
    src/compute/compiler.cpp
    src/compute/unified_backend.cpp
    src/backends/cpu_backend.cpp
    src/backends/gpu_stubs.cpp
    src/text/sdf_text_renderer.cpp
)

set(EFFECTS_SOURCES
    src/effects/text.cpp
    src/effects/particles.cpp
    src/effects/distortions.cpp
    src/effects/transitions.cpp
)

set(GPU_SOURCES
    # GPU Backend Abstraction - Added for unified pipeline
    src/gpu/webgpu_backend.cpp
)

set(UNIFIED_PIPELINE_SOURCES
    # Unified Render Pipeline
    src/render_pipeline.cpp
    src/text/sdf_generator.cpp
)

set(TEST_SOURCES
    src/tests/rain_test.cpp
)

add_library(caption_engine STATIC
    ${CORE_SOURCES}
    ${EFFECTS_SOURCES}
    ${TEST_SOURCES}
)

# Add unified pipeline sources if enabled
set(UNIFIED_PIPELINE_AVAILABLE FALSE CACHE INTERNAL "Unified pipeline deps found")

if(ENABLE_UNIFIED_PIPELINE)
    # Check for required dependencies
    find_package(Freetype QUIET)
    find_package(glm QUIET)
    
    if(Freetype_FOUND AND glm_FOUND)
        set(UNIFIED_PIPELINE_AVAILABLE TRUE CACHE INTERNAL "Unified pipeline deps found")
        target_sources(caption_engine PRIVATE ${UNIFIED_PIPELINE_SOURCES})
        target_compile_definitions(caption_engine PUBLIC 
            CAPTION_UNIFIED_PIPELINE=1
            CE_HAS_FREETYPE=1
        )
        target_link_libraries(caption_engine PRIVATE Freetype::Freetype glm::glm)
        message(STATUS "Unified Render Pipeline enabled with FreeType + GLM")
    else()
        message(STATUS "Unified Render Pipeline DISABLED - missing dependencies:")
        if(NOT Freetype_FOUND)
            message(STATUS "  - FreeType not found")
        endif()
        if(NOT glm_FOUND)
            message(STATUS "  - GLM not found")
        endif()
        message(STATUS "  Install with: vcpkg install freetype glm")
    endif()
endif()

# =============================================================================
# Skia Graphics Engine (Optional - for high-quality text rendering)
# =============================================================================
option(ENABLE_SKIA "Enable Skia for high-quality text rendering" ON)
set(SKIA_ROOT "E:/skia" CACHE PATH "Path to Skia source root")
set(SKIA_OUT_DIR "${SKIA_ROOT}/out/Release" CACHE PATH "Path to Skia build output")

if(ENABLE_SKIA AND NOT EMSCRIPTEN)
    # Check if manually built Skia exists
    if(EXISTS "${SKIA_OUT_DIR}/skia.lib")
        target_sources(caption_engine PRIVATE
            src/text/skia_text_renderer.cpp
        )
        
        # Link Skia libraries (manually built)
        target_link_libraries(caption_engine PRIVATE
            ${SKIA_OUT_DIR}/skia.lib
            ${SKIA_OUT_DIR}/harfbuzz.lib
            ${SKIA_OUT_DIR}/skcms.lib
        )
        
        # Skia include directories
        target_include_directories(caption_engine PRIVATE
            ${SKIA_ROOT}
            ${SKIA_ROOT}/include
        )
        
        target_compile_definitions(caption_engine PUBLIC 
            CE_HAS_SKIA=1
            CAPTION_HAS_SKIA=1
            SK_BUILD_FOR_WIN=1
            NOMINMAX
        )
        
        # Windows-specific libs needed by Skia
        if(WIN32)
            target_link_libraries(caption_engine PRIVATE
                opengl32
                user32
                gdi32
            )
        endif()
        
        message(STATUS "╔═══════════════════════════════════════════════════╗")
        message(STATUS "║  Skia Graphics Engine ENABLED (Manual Build)      ║")
        message(STATUS "║  High-quality anti-aliased text rendering         ║")
        message(STATUS "╚═══════════════════════════════════════════════════╝")
    else()
        message(STATUS "Skia not found at ${SKIA_OUT_DIR} - using stb_truetype fallback")
        message(STATUS "  Build Skia: cd ${SKIA_ROOT} && gn gen out/Release && ninja -C out/Release")
    endif()
endif()

target_include_directories(caption_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core libraries (always linked)
target_link_libraries(caption_engine PRIVATE
    nlohmann_json
    stb
    glm::glm
)

# Compile definitions for feature detection
target_compile_definitions(caption_engine PUBLIC
    $<$<BOOL:${BUILD_WASM}>:BUILD_WASM=1>
)

# =============================================================================
# WebAssembly Target
# =============================================================================
if(BUILD_WASM OR EMSCRIPTEN)
    message(STATUS "Building WebAssembly target")
    
    # Add WebGPU backend for WASM
    target_sources(caption_engine PRIVATE ${GPU_SOURCES})
    target_compile_definitions(caption_engine PUBLIC HAS_WEBGPU=1)
    
    add_executable(caption_engine_wasm
        src/bindings/wasm_bindings.cpp
    )
    
    target_link_libraries(caption_engine_wasm PRIVATE caption_engine)
    
    target_compile_options(caption_engine PRIVATE "--use-port=emdawnwebgpu")
    target_compile_options(caption_engine_wasm PRIVATE "--use-port=emdawnwebgpu")
    
    # Modern optimizations per Master Plan
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -msimd128")

    set_target_properties(caption_engine_wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -s WASM=1 \
            -s MODULARIZE=1 \
            -s EXPORT_ES6=1 \
            -s EXPORT_NAME='CaptionEngine' \
            -s ALLOW_MEMORY_GROWTH=1 \
            --use-port=emdawnwebgpu \
            -s EXPORTED_RUNTIME_METHODS=['UTF8ToString','stringToUTF8'] \
            -s ASSERTIONS=0 \
            -s DISABLE_EXCEPTION_CATCHING=0 \
            -s ENVIRONMENT=web,worker \
            --bind \
            --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@/assets \
        "
    )
endif()

# =============================================================================
# Vulkan Backend (Native Desktop)
# =============================================================================
if(ENABLE_VULKAN AND NOT EMSCRIPTEN)
    find_package(Vulkan QUIET)
    
    if(Vulkan_FOUND)
        target_sources(caption_engine PRIVATE
            src/gpu/vulkan_backend.cpp
        )
        target_include_directories(caption_engine PRIVATE ${Vulkan_INCLUDE_DIRS})
        target_link_libraries(caption_engine PRIVATE Vulkan::Vulkan)
        target_compile_definitions(caption_engine PUBLIC HAS_VULKAN=1)
        message(STATUS "Vulkan backend enabled - ${Vulkan_LIBRARY}")
    else()
        message(WARNING "Vulkan not found, Vulkan backend disabled. Install Vulkan SDK.")
    endif()
endif()

# =============================================================================
# CUDA Backend (NVIDIA GPU Compute)
# =============================================================================
if(ENABLE_CUDA AND NOT EMSCRIPTEN)
    find_package(CUDAToolkit QUIET)
    
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        
        # CUDA source files
        add_library(caption_cuda STATIC
            src/gpu/cuda_backend.cu
            src/cuda/effect_kernels.cu
        )
        
        target_include_directories(caption_cuda PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        
        set_target_properties(caption_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "60;70;75;80;86;89;90"  # Pascal to Ada Lovelace
            POSITION_INDEPENDENT_CODE ON
        )
        
        # Define HAS_CUDA and CAPTION_HAS_CUDA for detection
        target_compile_definitions(caption_cuda PUBLIC 
            HAS_CUDA=1 
            CAPTION_HAS_CUDA=1
        )
        target_link_libraries(caption_cuda PRIVATE CUDA::cudart)
        
        # Add CUDA compute adapter (C++ file that bridges to ComputeBackend)
        target_sources(caption_engine PRIVATE
            src/backends/cuda_compute_adapter.cpp
        )
        
        # Link CUDA to main engine
        target_link_libraries(caption_engine PUBLIC caption_cuda)
        target_compile_definitions(caption_engine PUBLIC 
            HAS_CUDA=1
            CAPTION_HAS_CUDA=1
        )
        
        message(STATUS "CUDA backend enabled - Architectures: 60-90")
        message(STATUS "  Effect kernels: glow, glitch, wave, chromatic, blur, composite")
        message(STATUS "  New kernels: stroke, shadow, typewriter, particles")
    else()
        message(WARNING "CUDA Toolkit not found, CUDA backend disabled")
    endif()
endif()

# =============================================================================
# Metal Backend (macOS/iOS)
# =============================================================================
if(ENABLE_METAL AND APPLE AND NOT EMSCRIPTEN)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(FOUNDATION_LIBRARY Foundation)
    
    if(METAL_LIBRARY AND METALKIT_LIBRARY)
        target_sources(caption_engine PRIVATE
            src/gpu/metal_backend.mm
        )
        target_link_libraries(caption_engine PRIVATE 
            ${METAL_LIBRARY} 
            ${METALKIT_LIBRARY}
            ${FOUNDATION_LIBRARY}
        )
        target_compile_definitions(caption_engine PUBLIC HAS_METAL=1)
        message(STATUS "Metal backend enabled")
    else()
        message(WARNING "Metal framework not found")
    endif()
endif()

# =============================================================================
# DirectX 12 Backend (Windows Native)
# =============================================================================
option(ENABLE_DX12 "Enable DirectX 12 backend (Windows only)" OFF)

if(ENABLE_DX12 AND WIN32 AND NOT EMSCRIPTEN)
    # Check for Windows SDK
    if(CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "10.0")
        target_sources(caption_engine PRIVATE
            src/gpu/dx12_backend.cpp
        )
        
        # Link DirectX 12 libraries (included in Windows SDK)
        target_link_libraries(caption_engine PRIVATE
            d3d12
            dxgi
            dxguid
            d3dcompiler
        )
        
        # Define macros for DX12 detection
        target_compile_definitions(caption_engine PUBLIC 
            HAS_DX12=1
            CAPTION_HAS_DX12=1
        )
        
        message(STATUS "╔═══════════════════════════════════════════════════╗")
        message(STATUS "║  DirectX 12 Backend ENABLED                       ║")
        message(STATUS "║  Requires: Windows 10+ with Windows SDK 10.0+     ║")
        message(STATUS "╚═══════════════════════════════════════════════════╝")
    else()
        message(WARNING "DirectX 12 requires Windows 10 or later (SDK 10.0+)")
    endif()
endif()


# =============================================================================
# Python Bindings
# =============================================================================
if(BUILD_PYTHON AND NOT EMSCRIPTEN)
    find_package(Python3 COMPONENTS Interpreter Development NumPy QUIET)
    
    if(Python3_FOUND)
        # Find pybind11 via pip installation
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_RESULT
        )
        
        if(PYBIND11_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 CONFIG REQUIRED)
            
            # Legacy Python bindings (caption_engine_py)
            pybind11_add_module(caption_engine_py src/bindings/python_bindings.cpp)
            target_link_libraries(caption_engine_py PRIVATE caption_engine)
            target_include_directories(caption_engine_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
            
            # Copy to parent directory for easy import
            add_custom_command(TARGET caption_engine_py POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:caption_engine_py> ${CMAKE_CURRENT_SOURCE_DIR}/../caption_engine_py.pyd
            )
            
            # Unified Pipeline Python bindings (caption_engine_unified)
            if(UNIFIED_PIPELINE_AVAILABLE)
                pybind11_add_module(caption_engine_unified src/bindings/unified_bindings.cpp)
                target_link_libraries(caption_engine_unified PRIVATE caption_engine glm::glm)
                target_include_directories(caption_engine_unified PRIVATE 
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${Python3_NumPy_INCLUDE_DIRS}
                )
                
                # Copy unified module to parent
                add_custom_command(TARGET caption_engine_unified POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:caption_engine_unified> ${CMAKE_CURRENT_SOURCE_DIR}/../caption_engine_unified.pyd
                )
                
                message(STATUS "Unified Pipeline Python bindings enabled")
            endif()
            
            message(STATUS "Python bindings enabled - Python ${Python3_VERSION}")
        else()
            message(WARNING "pybind11 not found via pip, Python bindings disabled")
        endif()
    else()
        message(WARNING "Python3 not found, Python bindings disabled")
    endif()
endif()

# =============================================================================
# Tests
# =============================================================================
if(BUILD_TESTS AND NOT EMSCRIPTEN)
    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# Install Rules
# =============================================================================
install(TARGETS caption_engine
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# =============================================================================
# Build Summary
# =============================================================================
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════╗")
message(STATUS "║      Caption Engine Build Configuration           ║")
message(STATUS "╠═══════════════════════════════════════════════════╣")
message(STATUS "║  Version:        ${PROJECT_VERSION}                            ║")
message(STATUS "║  Build Type:     ${CMAKE_BUILD_TYPE}                          ║")
message(STATUS "╠═══════════════════════════════════════════════════╣")
message(STATUS "║  Targets:                                         ║")
message(STATUS "║    WASM:         ${BUILD_WASM}                              ║")
message(STATUS "║    Python:       ${BUILD_PYTHON}                               ║")
message(STATUS "║    Tests:        ${BUILD_TESTS}                               ║")
message(STATUS "╠═══════════════════════════════════════════════════╣")
message(STATUS "║  GPU Backends:                                    ║")
message(STATUS "║    Unified:      ${ENABLE_UNIFIED_PIPELINE}                               ║")
message(STATUS "║    CUDA:         ${ENABLE_CUDA}                              ║")
message(STATUS "║    Vulkan:       ${ENABLE_VULKAN}                              ║")
message(STATUS "║    Metal:        ${ENABLE_METAL}                              ║")
message(STATUS "╚═══════════════════════════════════════════════════╝")
message(STATUS "")
