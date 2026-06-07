#!/usr/bin/env python3
"""Quick check: jvggt state_dict vs model.pt keys and point_head weight stats."""
from jvggt.jittor_env import configure_jittor_compiler

configure_jittor_compiler()

import numpy as np

from jvggt.models.vggt import VGGT
from jvggt.pt_loader import load_state_dict_from_pt
from jvggt.weight_loader import assign_state_dict, resolve_weights_path


def main():
    model = VGGT()
    jt_keys = set(model.state_dict().keys())
    path = resolve_weights_path(None)
    sd = load_state_dict_from_pt(path)
    pt_keys = set(sd.keys())
    missing = sorted(jt_keys - pt_keys)
    print(f"checkpoint: {path}")
    print(f"jt params: {len(jt_keys)}, pt keys: {len(pt_keys)}, missing: {len(missing)}")
    for k in missing[:15]:
        print(f"  missing: {k}")
    ph_missing = [k for k in missing if k.startswith("point_head")]
    print(f"point_head missing: {len(ph_missing)}")
    assign_state_dict(model, sd)
    # after assign, check a param on model
    w = model.state_dict()["point_head.scratch.output_conv2.1.weight"]
    arr = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
    print(f"assigned point_head output conv weight: shape={arr.shape} mean={arr.mean():.6f} std={arr.std():.6f}")


if __name__ == "__main__":
    main()
