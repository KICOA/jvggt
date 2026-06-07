#!/usr/bin/env python3
"""Static checks for jvggt PyTorch->Jittor port (no jittor import required)."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "jvggt"

FORWARD_OK = {"forward_features", "forward_features_list", "forward_nested"}

BANNED_PATTERNS = [
    (r"\bimport torch\b", "import torch"),
    (r"\bfrom torch\b", "from torch"),
    (r"requires_grad_\(", "requires_grad_()"),
    (r"register_buffer\(", "register_buffer()"),
    (r"PyTorchModelHubMixin", "PyTorchModelHubMixin"),
    (r"indexing\s*=\s*[\"']xy[\"']", "meshgrid indexing='xy'"),
    (r"cartesian_prod\(", "cartesian_prod()"),
    (r"\.device\b", ".device (check manually)"),
]

ALLOW_DEVICE_IN = {
    "heads/utils.py",  # docstring only after fix
}


def _check_annotations() -> list[str]:
    bad: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            ann = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ann = node.returns
            elif isinstance(node, ast.arg):
                ann = node.annotation
            if ann is None:
                continue
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id == "Tensor":
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} bare Tensor annotation")
    return bad


def _check_forward_methods() -> list[str]:
    bad: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "forward" or (
                    node.name.startswith("forward") and node.name not in FORWARD_OK
                ):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} def {node.name} (use execute)")
    return bad


def _check_banned_regex() -> list[str]:
    bad: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = str(path.relative_to(PKG)).replace("\\", "/")
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if ".device" in line and rel in ALLOW_DEVICE_IN and "device (" in line:
                continue
            if ".device" in line and 'device (jt.device' in line:
                continue  # docstring
            for pat, label in BANNED_PATTERNS:
                if re.search(pat, line):
                    # skip comments-only lines for .device in docstrings
                    stripped = line.strip()
                    if label.startswith(".device") and (stripped.startswith("#") or 'optional' in line):
                        continue
                    bad.append(f"{path.relative_to(ROOT)}:{line_no} {label}: {stripped[:80]}")
    return bad


def _check_f_import() -> list[str]:
    """Files using F.xxx must import F from jvggt.ops."""
    bad: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        has_f_use = False
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if re.search(r"(?<![\w.])F\.\w+", code):
                has_f_use = True
                break
        if has_f_use and "from jvggt.ops import F" not in text:
            bad.append(f"{path.relative_to(ROOT)} uses F.* without importing F from jvggt.ops")
    return bad


def _check_syntax() -> None:
    for path in sorted(PKG.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> None:
    errors: list[str] = []
    _check_syntax()
    errors.extend(_check_annotations())
    errors.extend(_check_forward_methods())
    errors.extend(_check_banned_regex())
    errors.extend(_check_f_import())

    if errors:
        print("FAILED static checks:")
        for e in errors:
            print(" ", e)
        sys.exit(1)

    n = len(list(PKG.rglob("*.py")))
    print(f"OK: {n} files passed static jvggt checks")


if __name__ == "__main__":
    main()
