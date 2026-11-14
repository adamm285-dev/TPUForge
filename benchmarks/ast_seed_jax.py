"""Out-of-band repository AST, declaration, and syntax idiom seed extractor for SuffixIndex."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

_MAX_SNIPPET_CHARS = 240
_SKIP_DIRS = frozenset({".git", ".venv", "venv", "build", "__pycache__", ".pytest_cache", ".ruff_cache"})
_PY_EXTENSIONS = frozenset({".py"})
_CPP_EXTENSIONS = frozenset({".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx"})

DEFAULT_CODE_IDIOMS: tuple[str, ...] = (
    "from __future__ import annotations",
    'if __name__ == "__main__":',
    "@dataclass(frozen=True)",
    "isinstance(",
    "raise ValueError(",
    "def __init__(self, ",
    "return ",
    " -> None:",
    "std::vector<int32_t>",
    "static_cast<size_t>(",
)

_CPP_DECL_RE = re.compile(
    r"^(?:#include\b|using\b|typedef\b|(?:template\s*<[^>]+>\s*)?(?:class|struct)\s+\w+"
    r"|(?:static\s+|inline\s+|virtual\s+|explicit\s+|constexpr\s+)*"
    r"[A-Za-z_][\w:<>,\s*&]+\s+[A-Za-z_~]\w*(?:::\w+)*\s*\([^;{}]*\))"
)


def _append_unique(out: list[str], seen: set[str], raw: str | None) -> None:
    if not raw:
        return
    cleaned = " ".join(raw.strip().splitlines()).strip()
    if len(cleaned) < 2 or len(cleaned) > _MAX_SNIPPET_CHARS:
        return
    if cleaned not in seen:
        seen.add(cleaned)
        out.append(cleaned)


def _extract_python_signature_line(source_lines: list[str], lineno: int) -> str:
    idx = max(0, lineno - 1)
    parts: list[str] = []
    for line in source_lines[idx : idx + 6]:
        stripped = line.strip()
        if not stripped:
            continue
        parts.append(stripped)
        if stripped.endswith(":"):
            break
    return " ".join(parts)


def extract_python_ast_snippets(source_code: str) -> list[str]:
    """Extract deduplicated Python AST signatures, class annotations, imports, and idioms."""
    if not source_code or not source_code.strip():
        return []

    out: list[str] = []
    seen: set[str] = set()
    lines = source_code.splitlines()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "class ", "def ", "async def ", "return ")):
                _append_unique(out, seen, stripped)
        return out

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(source_code, node) or ast.unparse(node)
            _append_unique(out, seen, seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _extract_python_signature_line(lines, node.lineno)
            _append_unique(out, seen, sig)
        elif isinstance(node, ast.ClassDef):
            hdr = _extract_python_signature_line(lines, node.lineno)
            _append_unique(out, seen, hdr)
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):
                    seg = ast.get_source_segment(source_code, stmt) or ast.unparse(stmt)
                    _append_unique(out, seen, seg)
        elif isinstance(node, (ast.Return, ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(source_code, node) or ast.unparse(node)
            _append_unique(out, seen, seg)

    return out


def extract_cpp_header_snippets(source_code: str) -> list[str]:
    """Extract deduplicated C/C++ #include, type declarations, and function signatures."""
    if not source_code or not source_code.strip():
        return []

    out: list[str] = []
    seen: set[str] = set()
    for raw_line in source_code.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("//", "/*")):
            continue
        if _CPP_DECL_RE.match(line):
            cleaned = line.rstrip("{").strip()
            _append_unique(out, seen, cleaned)
    return out


def _iter_source_files(paths: Iterable[Path | str], max_files: int) -> list[Path]:
    collected: list[Path] = []
    seen: set[Path] = set()
    allowed_exts = _PY_EXTENSIONS | _CPP_EXTENSIONS

    for raw_path in paths:
        if len(collected) >= max_files:
            break
        p = Path(raw_path)
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in allowed_exts and p not in seen:
                seen.add(p)
                collected.append(p)
            continue

        candidates = sorted(p.rglob("*"))
        for child in candidates:
            if len(collected) >= max_files:
                break
            if any(part in _SKIP_DIRS for part in child.parts):
                continue
            if child.is_file() and child.suffix.lower() in allowed_exts and child not in seen:
                seen.add(child)
                collected.append(child)

    return collected


def extract_repo_ast_snippets(
    paths: Iterable[Path | str],
    include_default_idioms: bool = True,
    max_files: int = 128,
) -> list[str]:
    """Walk source files/directories and extract deduplicated AST/declaration seed snippets."""
    out: list[str] = []
    seen: set[str] = set()

    for path in _iter_source_files(paths, max_files=max(0, int(max_files))):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        suffix = path.suffix.lower()
        if suffix in _PY_EXTENSIONS:
            snippets = extract_python_ast_snippets(text)
        else:
            snippets = extract_cpp_header_snippets(text)
        for snip in snippets:
            _append_unique(out, seen, snip)

    if include_default_idioms:
        for idiom in DEFAULT_CODE_IDIOMS:
            _append_unique(out, seen, idiom)

    return out


def encode_seed_sequences(
    snippets: Iterable[str],
    encode_fn: Callable[[str], Sequence[int]],
) -> list[list[int]]:
    """Encode snippets into token sequences suitable for SuffixIndex::seed_sequences."""
    sequences: list[list[int]] = []
    for raw in snippets:
        if not raw:
            continue
        ids = [int(tok) for tok in encode_fn(raw)]
        if len(ids) >= 2:
            sequences.append(ids)
    return sequences


def encode_with_tokenizer(snippets: Iterable[str], tokenizer: Any) -> list[list[int]]:
    """Encode AST/idiom snippets with HF tokenizers, transformers, or ByteTokenizer."""

    def _encode_one(text: str) -> list[int]:
        try:
            enc = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            enc = tokenizer.encode(text)
        if hasattr(enc, "ids"):
            return [int(t) for t in enc.ids]
        return [int(t) for t in enc]

    return encode_seed_sequences(snippets, _encode_one)
