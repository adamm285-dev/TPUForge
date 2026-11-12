cmake_minimum_required(VERSION 3.20)
project(TPUForge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core engine library
add_library(tpuforge_core STATIC
    src/core/device.cpp
    src/core/tensor.cpp
    src/core/kv_cache.cpp
    src/speculative/verifier.cpp
    src/runtime/pjrt_driver.cpp
)

target_include_directories(tpuforge_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Command-line benchmark / runner
add_executable(tpuforge_bench
    src/main.cpp
)

target_link_libraries(tpuforge_bench PRIVATE tpuforge_core)
