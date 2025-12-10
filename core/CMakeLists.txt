cmake_minimum_required(VERSION 3.20)
project(CaptionEngine 
    VERSION 1.0.0 
    LANGUAGES CXX
)

# Modern C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_WASM "Build for WebAssembly" OFF)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(ENABLE_METAL "Enable Metal backend (macOS)" OFF)

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
)

set(EFFECTS_SOURCES
    src/effects/text.cpp
    src/effects/particles.cpp
    src/effects/distortions.cpp
    src/effects/transitions.cpp
)

set(PLATFORM_SOURCES
    src/platform/emscripten.cpp
    src/platform/python.cpp
)

set(TEST_SOURCES
    src/tests/rain_test.cpp
)

add_library(caption_engine STATIC
    ${CORE_SOURCES}
    ${EFFECTS_SOURCES}
    ${TEST_SOURCES}
)

# WebGPU backend for WASM
if(BUILD_WASM OR EMSCRIPTEN)
    target_sources(caption_engine PRIVATE src/backends/webgpu_backend.cpp)
endif()

target_include_directories(caption_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(caption_engine PRIVATE
    nlohmann_json
    stb
    freetype
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
# CUDA Backend (Optional)
# =============================================================================
if(ENABLE_CUDA AND NOT EMSCRIPTEN)
    enable_language(CUDA)
    
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "CUDA enabled - Building GPU acceleration backend")
    
    # CUDA sources
    set(CUDA_SOURCES
        src/compute/cuda_backend.cpp
        src/cuda/effect_kernels.cu
    )
    
    # Create CUDA library
    add_library(caption_cuda STATIC ${CUDA_SOURCES})
    
    target_include_directories(caption_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    # CUDA compile options
    set_target_properties(caption_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_compile_definitions(caption_cuda PUBLIC
        CAPTION_HAS_CUDA=1
    )
    
    # Link CUDA Runtime
    target_link_libraries(caption_cuda PRIVATE
        CUDA::cudart
    )
    
    # Link caption_cuda to main engine if CUDA enabled
    target_link_libraries(caption_engine PRIVATE caption_cuda)
    target_compile_definitions(caption_engine PUBLIC CAPTION_HAS_CUDA=1)
    
endif()

# =============================================================================
# Python Bindings
# =============================================================================
if(BUILD_PYTHON AND NOT EMSCRIPTEN)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
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
        
        pybind11_add_module(caption_engine_py src/bindings/python_bindings.cpp)
        target_link_libraries(caption_engine_py PRIVATE caption_engine)
        target_include_directories(caption_engine_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        
        # Copy to parent directory for easy import
        add_custom_command(TARGET caption_engine_py POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:caption_engine_py> ${CMAKE_CURRENT_SOURCE_DIR}/../caption_engine_py.pyd
        )
        message(STATUS "pybind11 found, Python bindings enabled")
    else()
        message(WARNING "pybind11 not found via pip, Python bindings disabled")
    endif()
endif()

# =============================================================================
# CUDA Backend
# =============================================================================
if(ENABLE_CUDA AND NOT EMSCRIPTEN)
    find_package(CUDAToolkit QUIET)
    
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        
        add_library(caption_cuda STATIC
            src/backends/cuda_backend.cu
        )
        
        set_target_properties(caption_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "60;70;75;80;86;89;90"  # Updated for RTX 40 series
        )
        
        target_link_libraries(caption_engine PUBLIC caption_cuda)
        target_link_libraries(caption_cuda PRIVATE CUDA::cudart)
        target_compile_definitions(caption_engine PUBLIC HAS_CUDA=1)
        message(STATUS "CUDA backend enabled")
    else()
        message(WARNING "CUDA Toolkit not found, CUDA backend disabled")
    endif()
endif()

# =============================================================================
# Vulkan Backend
# =============================================================================
if(ENABLE_VULKAN AND NOT EMSCRIPTEN)
    find_package(Vulkan QUIET)
    
    if(Vulkan_FOUND)
        target_sources(caption_engine PRIVATE
            src/backends/vulkan_backend.cpp
        )
        target_link_libraries(caption_engine PRIVATE Vulkan::Vulkan)
        target_compile_definitions(caption_engine PUBLIC HAS_VULKAN=1)
        message(STATUS "Vulkan backend enabled")
    else()
        message(WARNING "Vulkan not found, Vulkan backend disabled")
    endif()
endif()

# =============================================================================
# Metal Backend (macOS)
# =============================================================================
if(ENABLE_METAL AND APPLE AND NOT EMSCRIPTEN)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    
    if(METAL_LIBRARY AND METALKIT_LIBRARY)
        target_sources(caption_engine PRIVATE
            src/backends/metal_backend.mm
        )
        target_link_libraries(caption_engine PRIVATE 
            ${METAL_LIBRARY} 
            ${METALKIT_LIBRARY}
        )
        target_compile_definitions(caption_engine PUBLIC HAS_METAL=1)
        message(STATUS "Metal backend enabled")
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
message(STATUS "=== Caption Engine Build Configuration ===")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  WASM:            ${BUILD_WASM}")
message(STATUS "  Python:          ${BUILD_PYTHON}")
message(STATUS "  CUDA:            ${ENABLE_CUDA}")
message(STATUS "  Vulkan:          ${ENABLE_VULKAN}")
message(STATUS "  Metal:           ${ENABLE_METAL}")
message(STATUS "  Tests:           ${BUILD_TESTS}")
message(STATUS "================================================")

