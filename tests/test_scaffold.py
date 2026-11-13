from pathlib import Path


def test_core_engine_headers_and_sources_present():
    root = Path(__file__).resolve().parent.parent
    required = [
        "CMakeLists.txt",
        "include/tpuforge/pjrt_c_api.h",
        "include/tpuforge/device.h",
        "include/tpuforge/kv_cache.h",
        "include/tpuforge/speculative.h",
        "src/core/device.cpp",
        "src/core/kv_cache.cpp",
        "src/speculative/verifier.cpp",
        "src/runtime/pjrt_driver.cpp",
        "src/main.cpp",
    ]
    for rel in required:
        assert (root / rel).exists(), f"Missing required engine file: {rel}"
