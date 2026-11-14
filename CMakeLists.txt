cmake_minimum_required(VERSION 3.20)
project(TPUForge VERSION 0.2.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TPUFORGE_NATIVE_ARCH "Enable -march=native optimization" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Warning and optional target-arch flags (build optimization comes from CMAKE_BUILD_TYPE)
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra)
    if(TPUFORGE_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core engine library
add_library(tpuforge_core STATIC
    src/core/device.cpp
    src/core/kv_cache.cpp
    src/core/static_arena.cpp
    src/speculative/verifier.cpp
    src/speculative/tree_speculator.cpp
    src/speculative/openai_server.cpp
    src/runtime/pjrt_driver.cpp
)

set_target_properties(tpuforge_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(tpuforge_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(tpuforge_core PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)
elseif(WIN32)
    target_link_libraries(tpuforge_core PUBLIC ws2_32)
endif()

# Command-line benchmark / runner
add_executable(tpuforge_bench
    src/main.cpp
)

target_link_libraries(tpuforge_bench PRIVATE tpuforge_core)

if(MINGW)
    target_link_options(tpuforge_bench PRIVATE -static)
endif()

# Test PJRT C ABI plugin shared library (optional when packaging minimal bundles)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/pjrt_test_plugin.c")
    add_library(pjrt_test_plugin SHARED
        tests/fixtures/pjrt_test_plugin.c
    )
    set_target_properties(pjrt_test_plugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    if(MINGW)
        target_link_options(pjrt_test_plugin PRIVATE -static)
    endif()
endif()

# Optional nanobind Python extension
if(NOT DEFINED Python_EXECUTABLE)
    if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
        set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif()
endif()

find_package(Python COMPONENTS Interpreter Development.Module QUIET)

if(Python_Interpreter_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE nanobind_cmake_res
    )

    if(nanobind_cmake_res EQUAL 0 AND EXISTS "${nanobind_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
        find_package(nanobind CONFIG REQUIRED)

        nanobind_add_module(_tpuforge_ext NB_STATIC
            src/speculative/py_bindings.cpp
        )
        target_link_libraries(_tpuforge_ext PRIVATE tpuforge_core)
        set_target_properties(_tpuforge_ext PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
        if(MINGW)
            target_link_options(_tpuforge_ext PRIVATE -static)
        endif()
    else()
        message(WARNING "nanobind not found via ${Python_EXECUTABLE}; skipping _tpuforge_ext target")
    endif()
else()
    message(WARNING "Python Development.Module not found; skipping _tpuforge_ext target")
endif()
