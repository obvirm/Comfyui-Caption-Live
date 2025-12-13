# =============================================================================
# CaptionEngine Test Suite
# =============================================================================

# Consistency Test - WASM/Native validation
# Requires unified pipeline for full functionality
if(UNIFIED_PIPELINE_AVAILABLE)
    add_executable(consistency_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tests/consistency_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tests/consistency_test_main.cpp
    )

    target_include_directories(consistency_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(consistency_test PRIVATE
        caption_engine
        nlohmann_json
    )

    # CTest integration
    add_test(NAME consistency_generate 
             COMMAND consistency_test --generate ${CMAKE_CURRENT_SOURCE_DIR}/golden/reference.json
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    
    message(STATUS "  - consistency_test (WASM/Native validation)")
else()
    message(STATUS "  - consistency_test SKIPPED (requires unified pipeline)")
endif()

# Copy web test files to build directory
add_custom_target(copy_web_tests ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/web
        ${CMAKE_BINARY_DIR}/tests/web
    COMMENT "Copying web test harness"
)

# =============================================================================
# Rain Test (Existing) - requires compute backends
# =============================================================================
if(UNIFIED_PIPELINE_AVAILABLE)
    add_executable(rain_test
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tests/rain_test.cpp
    )

    target_include_directories(rain_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(rain_test PRIVATE
        caption_engine
    )

    add_test(NAME rain_test COMMAND rain_test)
    message(STATUS "  - rain_test (Compute shader test)")
else()
    message(STATUS "  - rain_test SKIPPED (requires unified pipeline)")
endif()

message(STATUS "Tests enabled:")
