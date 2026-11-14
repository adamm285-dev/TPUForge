"""Package clean TPUForge C++ and Python sources into a self-contained Kaggle/Cloud TPU zip."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

DEFAULT_INCLUDE_PATHS = (
    "CMakeLists.txt",
    "pyproject.toml",
    "README.md",
    "include",
    "src",
    "benchmarks/qwen_jax.py",
    "benchmarks/quantization_jax.py",
    "benchmarks/block_diffusion_jax.py",
    "benchmarks/ring_attention_jax.py",
    "benchmarks/run_qwen_bench.py",
    "benchmarks/package_kaggle_bundle.py",
    "benchmarks/__init__.py",
    "tests/fixtures/pjrt_test_plugin.c",
    "colab/TPUForge_Benchmark.ipynb",
)

_SKIP_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", "build", "models", "results"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".zip", ".o", ".obj", ".a"}


def _iter_bundle_files(root_dir: Path) -> list[tuple[Path, str]]:
    """Collect sorted (abs_path, arcname) pairs excluding build artifacts and weights."""
    files: list[tuple[Path, str]] = []
    seen_arcnames: set[str] = set()

    for rel in DEFAULT_INCLUDE_PATHS:
        target = root_dir / rel
        if not target.exists():
            continue
        if target.is_file():
            arc = target.relative_to(root_dir).as_posix()
            if arc not in seen_arcnames:
                seen_arcnames.add(arc)
                files.append((target, arc))
        elif target.is_dir():
            for child in sorted(target.rglob("*")):
                if not child.is_file():
                    continue
                if any(part in _SKIP_DIR_NAMES for part in child.parts):
                    continue
                if child.suffix.lower() in _SKIP_SUFFIXES:
                    continue
                arc = child.relative_to(root_dir).as_posix()
                if arc not in seen_arcnames:
                    seen_arcnames.add(arc)
                    files.append((child, arc))
    return sorted(files, key=lambda pair: pair[1])


def build_kaggle_bundle(root_dir: Path, output_zip: Path) -> Path:
    """Create a reproducible zip bundle ready for upload as a Kaggle Dataset or Colab archive."""
    root = root_dir.resolve()
    entries = _iter_bundle_files(root)
    if not entries:
        raise FileNotFoundError(f"No packable TPUForge sources found under {root}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arcname in entries:
            zf.write(abs_path, arcname=arcname)
    return output_zip


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build standalone TPUForge source bundle zip for Kaggle/Cloud TPU."
    )
    parser.add_argument(
        "--root",
        default=str(repo_root),
        help="TPUForge repository root directory.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "benchmarks" / "results" / "tpuforge_kaggle_bundle.zip"),
        help="Destination .zip path.",
    )
    args = parser.parse_args()

    out = build_kaggle_bundle(Path(args.root), Path(args.output))
    with zipfile.ZipFile(out, "r") as zf:
        count = len(zf.namelist())
    size_kb = out.stat().st_size / 1024.0
    print(f"Created Kaggle bundle: {out} ({count} files, {size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
