# Load VGGT checkpoints into Jittor models (no torch required).

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import os
import re

import jittor as jt
import numpy as np

from jvggt.pt_loader import load_state_dict_from_pt

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_MODEL_PT = _REPO_ROOT / "model.pt"
_LOCAL_MODEL_NPZ = _REPO_ROOT / "model.npz"
_HF_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def resolve_weights_path(weights: Optional[str] = None) -> str:
    """
    Resolve checkpoint path.

    Priority:
      1. Explicit ``weights`` (file, .pt / .npz, or http URL)
      2. ``<repo>/model.npz`` then ``<repo>/model.pt``
      3. HuggingFace URL for ``model.pt``
    """
    if weights:
        if weights.startswith(("http://", "https://")):
            return weights
        path = Path(weights)
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"Weights file not found: {weights}")

    if _LOCAL_MODEL_NPZ.is_file():
        return str(_LOCAL_MODEL_NPZ)
    if _LOCAL_MODEL_PT.is_file():
        return str(_LOCAL_MODEL_PT)

    return _HF_MODEL_URL


def load_state_dict(weights_path: str) -> Dict[str, np.ndarray]:
    """Load a state dict from ``.npz``, ``.pt`` (no torch), or download URL."""
    if weights_path.startswith(("http://", "https://")):
        return _load_state_dict_from_url(weights_path)

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(weights_path)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}
    if suffix == ".pt":
        return load_state_dict_from_pt(str(path))

    raise ValueError(f"Unsupported weights format: {path} (use .pt or .npz)")


def _load_state_dict_from_url(url: str, cache_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    from urllib.request import urlretrieve

    cache_dir = cache_dir or str(_REPO_ROOT)
    filename = os.path.join(cache_dir, os.path.basename(url.split("?")[0]))
    if not os.path.isfile(filename):
        print(f"Downloading weights from {url} ...")
        urlretrieve(url, filename)
    return load_state_dict(filename)


def jittor_key_to_checkpoint_key(name: str) -> str:
    """Jittor ``nn.Sequential`` uses ``.layers.N.``; PyTorch checkpoints use ``.N.``."""
    return re.sub(r"\.layers\.(\d+)\.", r".\1.", name)


def _lookup_checkpoint(state_dict: Dict[str, np.ndarray], param_name: str) -> tuple[str, np.ndarray] | None:
    for key in (param_name, jittor_key_to_checkpoint_key(param_name)):
        if key in state_dict:
            return key, state_dict[key]
    return None


def assign_state_dict(model: jt.nn.Module, state_dict: Dict[str, np.ndarray]) -> None:
    """Assign numpy weights into a Jittor model."""
    jt_state = model.state_dict()
    missing = []
    mismatched = []
    assigned = 0
    for name, param in jt_state.items():
        found = _lookup_checkpoint(state_dict, name)
        if found is None:
            missing.append(name)
            continue
        ckpt_key, arr = found
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        param_shape = tuple(int(s) for s in param.shape)
        if arr.shape != param_shape:
            mismatched.append((name, param_shape, arr.shape))
            continue
        if hasattr(param, "assign"):
            param.assign(arr)
        else:
            jt_state[name] = jt.array(arr)
        assigned += 1
    if missing:
        print(f"Warning: {len(missing)} keys missing in checkpoint (showing first 5): {missing[:5]}")
    if mismatched:
        print(f"Warning: {len(mismatched)} keys shape-mismatch (showing first 5):")
        for name, want, got in mismatched[:5]:
            print(f"  {name}: model{want} vs ckpt{got}")
    print(f"Weights assigned: {assigned}/{len(jt_state)} parameters")


def load_vggt_pretrained(model: jt.nn.Module, weights: Optional[str] = None) -> jt.nn.Module:
    """Load pretrained VGGT weights into a Jittor model (no torch)."""
    path = resolve_weights_path(weights)
    print(f"Loading weights from: {path}")
    state_dict = load_state_dict(path)
    assign_state_dict(model, state_dict)
    model.eval()
    return model
