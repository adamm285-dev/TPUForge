import os
import subprocess
import sys
from pathlib import Path


def test_core_engine_headers_and_sources_present():
    root = Path(__file__).resolve().parent.parent
    required = [
        "CMakeLists.txt",
        "include/tpuforge/pjrt_c_api.h",
        "include/tpuforge/pjrt_driver.h",
        "include/tpuforge/device.h",
        "include/tpuforge/kv_cache.h",
        "include/tpuforge/static_arena.h",
        "include/tpuforge/speculative.h",
        "include/tpuforge/tree_speculator.h",
        "src/core/device.cpp",
        "src/core/kv_cache.cpp",
        "src/core/static_arena.cpp",
        "src/speculative/verifier.cpp",
        "src/speculative/tree_speculator.cpp",
        "src/speculative/py_bindings.cpp",
        "src/runtime/pjrt_driver.cpp",
        "src/main.cpp",
    ]
    for rel in required:
        assert (root / rel).exists(), f"Missing required engine file: {rel}"


def test_native_clang_build_and_bench_execution():
    import shutil

    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    ucrt_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(ucrt_bin) and ucrt_bin not in env.get("PATH", ""):
        env["PATH"] = f"{ucrt_bin};{env.get('PATH', '')}"

    cmake_bin = shutil.which("cmake", path=env.get("PATH")) or r"C:\msys64\ucrt64\bin\cmake.exe"
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    py_exe = str(venv_py if venv_py.exists() else Path(sys.executable))

    build_dir = root / "build"
    if not (build_dir / "build.ninja").exists():
        cfg = subprocess.run(
            [
                cmake_bin,
                "-B",
                "build",
                "-S",
                ".",
                "-G",
                "Ninja",
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={py_exe}",
            ],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
        )
        assert cfg.returncode == 0, f"CMake configure failed:\n{cfg.stderr}"

    bld = subprocess.run(
        [cmake_bin, "--build", "build"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert bld.returncode == 0, f"CMake build failed:\n{bld.stderr}"

    exe_name = "tpuforge_bench.exe" if os.name == "nt" else "tpuforge_bench"
    exe_path = build_dir / exe_name
    assert exe_path.exists(), f"Expected compiled binary at {exe_path}"

    run = subprocess.run(
        [str(exe_path)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"Binary execution failed:\n{run.stderr}"
    assert "MXU systolic layout: 128x128" in run.stdout
    assert "Tokens Accepted: 6" in run.stdout
    assert "Effective Speedup: 7x" in run.stdout
    assert "Engine initialized and verified successfully." in run.stdout
