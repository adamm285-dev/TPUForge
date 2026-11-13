"""Pytest configuration ensuring native _tpuforge_ext and .venv packages load cleanly."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"

# Ensure MSYS2 UCRT64 runtime DLLs are discoverable when Python loads _tpuforge_ext.pyd
UCRT_BIN = Path(r"C:\msys64\ucrt64\bin")
if os.name == "nt" and UCRT_BIN.is_dir():
    os.add_dll_directory(str(UCRT_BIN))
    if str(UCRT_BIN) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{UCRT_BIN};{os.environ.get('PATH', '')}"

# Ensure native extension in build/ is importable regardless of test runner invocation
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

# Bridge .venv site-packages (jax, jaxlib, numpy, nanobind) when invoked from global Python
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.is_dir() and str(VENV_SITE) not in sys.path:
    sys.path.append(str(VENV_SITE))


def find_pjrt_test_plugin() -> Path:
    """Locate the compiled C11 PJRT test plugin shared library."""
    candidates = [
        BUILD_DIR / "libpjrt_test_plugin.dll",
        BUILD_DIR / "pjrt_test_plugin.dll",
        BUILD_DIR / "libpjrt_test_plugin.so",
        BUILD_DIR / "libpjrt_test_plugin.dylib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"PJRT test plugin library not found in {BUILD_DIR}: checked {[p.name for p in candidates]}"
    )


def ensure_native_build() -> None:
    """Ensure CMake targets (tpuforge_bench, _tpuforge_ext, pjrt_test_plugin) are up to date."""
    if not (BUILD_DIR / "build.ninja").exists():
        cmake_bin = shutil.which("cmake") or r"C:\msys64\ucrt64\bin\cmake.exe"
        venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
        py_exe = str(venv_py if venv_py.exists() else Path(sys.executable))
        subprocess.run(
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
            cwd=str(ROOT),
            check=True,
        )
    cmake_bin = shutil.which("cmake") or r"C:\msys64\ucrt64\bin\cmake.exe"
    subprocess.run([cmake_bin, "--build", "build"], cwd=str(ROOT), check=True)


ensure_native_build()
