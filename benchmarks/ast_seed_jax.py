"""Out-of-band repository AST, declaration, and syntax idiom seed extractor for SuffixIndex."""

from __future__ import annotations

import ast
import io
import os
import re
import tokenize
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

_MAX_SNIPPET_CHARS = 240
_MAX_SOURCE_FILE_BYTES = 512 * 1024
_SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "build", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
)
_PY_EXTENSIONS = frozenset({".py", ".pyi"})
_CPP_EXTENSIONS = frozenset({".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx"})
_PY_STMT_START_KEYWORDS = frozenset({"import", "from", "class", "def", "async", "return"})
_CPP_CONTROL_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "return", "catch", "sizeof", "alignof", "static_assert"}
)

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
    r"^(?:#include\b|using\b|typedef\b|(?:template\s*<[^;{}]+>\s*)?(?:class|struct)\s+\w+"
    r"|(?:template\s*<[^;{}]+>\s*)?(?:static\s+|inline\s+|virtual\s+|explicit\s+|constexpr\s+)*"
    r"[A-Za-z_][\w:<>,\s*&]+\s+[A-Za-z_~]\w*(?:::\w+)*\s*\([^;{}]*\))"
)
_CPP_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _append_unique(out: list[str], seen: set[str], raw: str | None) -> None:
    if not raw:
        return
    cleaned = " ".join(part.strip() for part in raw.strip().splitlines() if part.strip())
    if len(cleaned) < 2 or len(cleaned) > _MAX_SNIPPET_CHARS:
        return
    if cleaned not in seen:
        seen.add(cleaned)
        out.append(cleaned)


def _slice_node_lines(lines: list[str], node: ast.AST) -> str:
    """Slice AST node source lines in O(1) without rescanning the full buffer."""
    start = getattr(node, "lineno", 0)
    if start <= 0:
        return ""
    end = getattr(node, "end_lineno", start) or start
    return "\n".join(lines[start - 1 : min(len(lines), end)])


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


def _fallback_python_token_lines(source_code: str, lines: list[str]) -> list[str]:
    """Token-aware fallback statement extractor when a Python file fails ast.parse."""
    matched: list[str] = []
    at_stmt_start = True
    try:
        for tok_type, tok_str, start, _end, _line in tokenize.generate_tokens(
            io.StringIO(source_code).readline
        ):
            if tok_type in (tokenize.COMMENT, tokenize.ENCODING):
                continue
            if tok_type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                at_stmt_start = True
                continue
            if tok_type == tokenize.OP and tok_str == ";":
                at_stmt_start = True
                continue
            if at_stmt_start:
                if tok_type == tokenize.NAME and tok_str in _PY_STMT_START_KEYWORDS:
                    row = start[0] - 1
                    if 0 <= row < len(lines):
                        matched.append(lines[row].strip())
                at_stmt_start = False
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return matched


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
        for stmt_line in _fallback_python_token_lines(source_code, lines):
            _append_unique(out, seen, stmt_line)
        return out

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = _slice_node_lines(lines, node) or ast.unparse(node)
            _append_unique(out, seen, seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _extract_python_signature_line(lines, node.lineno)
            _append_unique(out, seen, sig)
        elif isinstance(node, ast.ClassDef):
            hdr = _extract_python_signature_line(lines, node.lineno)
            _append_unique(out, seen, hdr)
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):
                    seg = _slice_node_lines(lines, stmt) or ast.unparse(stmt)
                    _append_unique(out, seen, seg)
        elif isinstance(node, (ast.Return, ast.Assign, ast.AnnAssign)):
            seg = _slice_node_lines(lines, node) or ast.unparse(node)
            _append_unique(out, seen, seg)

    return out


def _strip_cpp_comments(source_code: str) -> list[str]:
    """Remove C/C++ block and line comments before declaration scanning."""
    no_blocks = _CPP_BLOCK_COMMENT_RE.sub("", source_code)
    cleaned_lines: list[str] = []
    for raw_line in no_blocks.splitlines():
        in_str: str | None = None
        escaped = False
        cut_idx = len(raw_line)
        for idx, ch in enumerate(raw_line):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_str is not None:
                escaped = True
                continue
            if ch in ('"', "'"):
                if in_str is None:
                    in_str = ch
                elif in_str == ch:
                    in_str = None
                continue
            if in_str is None and ch == "/" and idx + 1 < len(raw_line) and raw_line[idx + 1] == "/":
                cut_idx = idx
                break
        stripped = raw_line[:cut_idx].strip()
        if stripped:
            cleaned_lines.append(stripped)
    return cleaned_lines


def _starts_with_cpp_control(text: str) -> bool:
    head = text.lstrip()
    for kw in _CPP_CONTROL_KEYWORDS:
        if re.match(rf"^{kw}\b", head):
            return True
    return False


def extract_cpp_ast_snippets(source_code: str) -> list[str]:
    """Extract deduplicated C/C++ #include, type declarations, and single/multi-line signatures."""
    if not source_code or not source_code.strip():
        return []

    out: list[str] = []
    seen: set[str] = set()
    lines = _strip_cpp_comments(source_code)
    n_lines = len(lines)
    i = 0

    while i < n_lines:
        line = lines[i]
        if _CPP_DECL_RE.match(line) and not _starts_with_cpp_control(line):
            _append_unique(out, seen, line.rstrip("{").strip())
            i += 1
            continue

        # Join multi-line template prefixes or multi-line C++ function/method signatures.
        is_template = line.startswith("template") and i + 1 < n_lines
        has_open_paren = "(" in line and ")" not in line and not _starts_with_cpp_control(line)
        if is_template or has_open_paren:
            parts: list[str] = [line]
            j = i + 1
            while j < min(n_lines, i + 6):
                nxt = lines[j]
                parts.append(nxt)
                joined = " ".join(parts)
                if nxt.endswith((";", "{")) or (")" in joined and (";" in nxt or "{" in nxt)):
                    break
                j += 1
            candidate = " ".join(parts)
            normalized = candidate.rstrip("{").strip()
            if _CPP_DECL_RE.match(normalized) and not _starts_with_cpp_control(normalized):
                _append_unique(out, seen, normalized)
                i = j + 1
                continue

        i += 1

    return out


extract_cpp_header_snippets = extract_cpp_ast_snippets


def _iter_source_files(
    paths: str | Path | Iterable[Path | str],
    max_files: int,
) -> list[Path]:
    if max_files <= 0:
        return []

    raw_items: Iterable[Path | str] = (paths,) if isinstance(paths, (str, Path)) else paths
    collected: list[Path] = []
    seen: set[Path] = set()
    allowed_exts = _PY_EXTENSIONS | _CPP_EXTENSIONS

    for raw_path in raw_items:
        if len(collected) >= max_files:
            break
        p = Path(raw_path)
        if not p.exists():
            continue
        if p.is_file():
            try:
                resolved = p.resolve()
                size = p.stat().st_size
            except OSError:
                continue
            if p.suffix.lower() in allowed_exts and size <= _MAX_SOURCE_FILE_BYTES and resolved not in seen:
                seen.add(resolved)
                collected.append(p)
            continue

        for root, dirnames, filenames in os.walk(p):
            if len(collected) >= max_files:
                break
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            root_path = Path(root)
            for fname in sorted(filenames):
                if len(collected) >= max_files:
                    break
                child = root_path / fname
                if child.suffix.lower() not in allowed_exts:
                    continue
                try:
                    resolved = child.resolve()
                    size = child.stat().st_size
                except OSError:
                    continue
                if size <= _MAX_SOURCE_FILE_BYTES and resolved not in seen:
                    seen.add(resolved)
                    collected.append(child)

    return collected


def extract_repo_ast_snippets(
    paths: str | Path | Iterable[Path | str],
    include_default_idioms: bool = True,
    max_files: int = 128,
    max_snippets: int = 512,
) -> list[str]:
    """Walk source files/directories and extract deduplicated AST/declaration seed snippets."""
    if max_files <= 0 or max_snippets <= 0:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for path in _iter_source_files(paths, max_files=int(max_files)):
        if len(out) >= max_snippets:
            break
        try:
            with path.open("r", encoding="utf-8") as fh:
                text = fh.read(_MAX_SOURCE_FILE_BYTES + 1)
        except (OSError, UnicodeDecodeError):
            continue
        if len(text.encode("utf-8")) > _MAX_SOURCE_FILE_BYTES:
            continue
        suffix = path.suffix.lower()
        if suffix in _PY_EXTENSIONS:
            snippets = extract_python_ast_snippets(text)
        else:
            snippets = extract_cpp_ast_snippets(text)
        for snip in snippets:
            _append_unique(out, seen, snip)
            if len(out) >= max_snippets:
                break

    if include_default_idioms and len(out) < max_snippets:
        for idiom in DEFAULT_CODE_IDIOMS:
            _append_unique(out, seen, idiom)
            if len(out) >= max_snippets:
                break

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
