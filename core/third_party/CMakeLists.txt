# Third-party dependencies for Caption Engine

# nlohmann/json (header-only)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# stb libraries (header-only)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# FreeType stub - using stb_truetype instead for simplicity
add_library(freetype INTERFACE)
# No actual library needed - stb_truetype is header-only

# GLM (header-only math library)
# Skip if vcpkg already provides it (to avoid target conflicts)
find_package(glm CONFIG QUIET)
if(NOT glm_FOUND)
    message(STATUS "GLM not found via vcpkg, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1
    )
    FetchContent_MakeAvailable(glm)
else()
    message(STATUS "Using GLM from vcpkg")
endif()

