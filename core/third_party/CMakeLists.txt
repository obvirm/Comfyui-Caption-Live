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
