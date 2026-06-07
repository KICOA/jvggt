#!/usr/bin/env python3
"""Convert model.pt to model.npz for faster reload (no torch, low peak RAM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jvggt.pt_loader import save_state_dict_to_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch model.pt to numpy .npz")
    parser.add_argument("--input", type=str, default=str(ROOT / "model.pt"))
    parser.add_argument("--output", type=str, default=str(ROOT / "model.npz"))
    args = parser.parse_args()

    print(f"Converting {args.input} -> {args.output}")
    print("Streaming tensors to disk (mmap); peak RAM stays low, but needs free disk ~5GB+ ...")
    n = save_state_dict_to_npz(args.input, args.output)
    print(f"Done. Wrote {n} tensors to {args.output}")


if __name__ == "__main__":
    main()
