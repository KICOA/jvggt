#!/usr/bin/env python3
"""Verify jvggt modules import without torch (catches undefined Tensor etc.)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure compiler before jittor (same as demo).
from jvggt.jittor_env import configure_jittor_compiler

configure_jittor_compiler()


def _check_no_bare_tensor_in_annotations() -> None:
    """AST check: return/arg annotations must not reference bare name Tensor."""
    bad: list[str] = []
    for path in sorted((ROOT / "jvggt").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            ann = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns:
                    ann = node.returns
            elif isinstance(node, ast.arg):
                ann = node.annotation
            if ann is None:
                continue
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id == "Tensor":
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    if bad:
        raise SystemExit("Bare 'Tensor' in annotations (use jt.Var):\n  " + "\n  ".join(bad))


def main() -> None:
    static = ROOT / "tools" / "check_jittor_static.py"
    if static.is_file():
        import subprocess

        r = subprocess.run([sys.executable, str(static)], cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(r.returncode)

    _check_no_bare_tensor_in_annotations()
    try:
        import jittor as jt  # noqa: F401
    except ModuleNotFoundError:
        print("static checks OK; jittor not installed — skip runtime import test")
        return

    from jvggt.models.vggt import VGGT
    from jvggt.inference import create_vggt_model, run_vggt_forward

    print("import ok:", VGGT, run_vggt_forward)


if __name__ == "__main__":
    main()
