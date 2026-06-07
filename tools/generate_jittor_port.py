#!/usr/bin/env python3
"""Generate jvggt inference package from PyTorch vggt sources."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "vggt"
DST = ROOT / "jvggt"

FILE_MAP = [
    ("layers/mlp.py", "layers/mlp.py"),
    ("layers/layer_scale.py", "layers/layer_scale.py"),
    ("layers/drop_path.py", "layers/drop_path.py"),
    ("layers/patch_embed.py", "layers/patch_embed.py"),
    ("layers/swiglu_ffn.py", "layers/swiglu_ffn.py"),
    ("layers/attention.py", "layers/attention.py"),
    ("layers/block.py", "layers/block.py"),
    ("layers/rope.py", "layers/rope.py"),
    ("layers/vision_transformer.py", "layers/vision_transformer.py"),
    ("layers/__init__.py", "layers/__init__.py"),
    ("models/aggregator.py", "models/aggregator.py"),
    # models/vggt.py is hand-maintained in jvggt/models/vggt.py (run_depth / VRAM opts)
    # heads/head_act.py is hand-maintained (sign/expm1 helpers)
    ("heads/utils.py", "heads/utils.py"),
    ("heads/camera_head.py", "heads/camera_head.py"),
    ("heads/dpt_head.py", "heads/dpt_head.py"),
    ("utils/pose_enc.py", "utils/pose_enc.py"),
    ("utils/rotation.py", "utils/rotation.py"),
    ("utils/load_fn.py", "utils/load_fn.py"),
]


def convert_imports(text: str) -> str:
    # Normalize torch imports before generic replacement to avoid duplicate jittor imports.
    text = re.sub(
        r"^import torch\.nn as nn\s*\nimport torch\s*$",
        "import jittor as jt\nfrom jittor import nn",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^import torch\s*\nimport torch\.nn as nn\s*$",
        "import jittor as jt\nfrom jittor import nn",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^import torch\.nn\.functional as F\s*$",
        "import jittor as jt\nfrom jittor import nn\nfrom jvggt.ops import F",
        text,
        flags=re.M,
    )
    text = re.sub(r"^import torch\.nn as nn\s*$", "import jittor as jt\nfrom jittor import nn", text, flags=re.M)
    text = re.sub(r"^import torch\s*$", "import jittor as jt", text, flags=re.M)
    text = re.sub(r"^from torch import nn\s*$", "import jittor as jt\nfrom jittor import nn", text, flags=re.M)
    text = re.sub(r"^from torch import Tensor, nn\s*$", "import jittor as jt\nfrom jittor import nn", text, flags=re.M)
    text = re.sub(r"^from torch import Tensor\s*$", "import jittor as jt", text, flags=re.M)
    text = re.sub(r"^from torch\.utils\.checkpoint import checkpoint\s*$", "", text, flags=re.M)
    text = re.sub(r"^from torch\.nn\.init import trunc_normal_\s*$", "from jvggt.ops import trunc_normal_", text, flags=re.M)
    text = re.sub(r"^from huggingface_hub import PyTorchModelHubMixin.*\n", "", text, flags=re.M)
    text = text.replace("from vggt.", "from jvggt.")
    text = text.replace("requires_grad_(False)", "stop_grad()")
    text = text.replace(".requires_grad_(False)", ".stop_grad()")
    text = re.sub(r"\bdef forward\b", "def execute", text)
    text = text.replace("super().forward", "super().execute")
    text = text.replace("torch.", "jt.")
    text = text.replace("nn.functional.", "F.")
    text = text.replace("jt.Tensor", "jt.Var")
    # Remove accidental duplicate jittor import lines.
    text = re.sub(r"^from torch import nn, Tensor\s*$", "import jittor as jt\nfrom jittor import nn", text, flags=re.M)
    text = re.sub(r"^from torch import Tensor, nn\s*$", "import jittor as jt\nfrom jittor import nn", text, flags=re.M)
    text = text.replace(": Tensor", ": jt.Var")
    text = text.replace("Union[float, Tensor]", "Union[float, jt.Var]")
    text = text.replace("List[Tensor]", "List[jt.Var]")
    text = text.replace("isinstance(x_or_x_list, Tensor)", "isinstance(x_or_x_list, jt.Var)")
    text = text.replace(") -> Tensor:", ") -> jt.Var:")
    text = text.replace(") -> List[Tensor]:", ") -> List[jt.Var]:")
    text = text.replace("q, k, v = qkv.unbind(0)", "q, k, v = jt.unbind(qkv, 0)")
    text = text.replace("attn.softmax(dim=-1)", "jt.nn.softmax(attn, dim=-1)")
    text = text.replace(".to(previous_dtype)", ".astype(previous_dtype)")
    text = text.replace("self.mask_token.to(x.dtype)", "self.mask_token.astype(x.dtype)")
    text = text.replace("residual.to(dtype=x.dtype)", "residual.astype(x.dtype)")
    text = text.replace("jt.randperm(b, device=x.device)", "jt.randperm(b)")
    text = text.replace("nn.init.normal_(", "normal_(")
    text = text.replace("return x.mul_(self.gamma)", "return x * self.gamma")
    text = text.replace("x.size(", "x.shape[")
    text = re.sub(r"(^import jittor as jt\n)+", "import jittor as jt\n", text, flags=re.M)
    return text


def patch_file(rel_src: str, text: str) -> str:
    text = convert_imports(text)

    if rel_src == "models/vggt.py":
        text = text.replace("class VGGT(nn.Module, PyTorchModelHubMixin):", "class VGGT(nn.Module):")
        text = text.replace("from jvggt.heads.track_head import TrackHead\n", "")
        text = text.replace(
            "        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None\n",
            "",
        )
        text = text.replace(
            "enable_camera=True, enable_point=True, enable_depth=True, enable_track=True",
            "enable_camera=True, enable_point=True, enable_depth=True, enable_track=False",
        )
        text = text.replace("with jt.cuda.amp.autocast(enabled=False):", "with jt.no_grad():")
        text = text.replace(
            "if not self.training:\n            predictions[\"images\"] = images",
            'predictions["images"] = images',
        )

    if rel_src == "models/aggregator.py":
        text = text.replace("nn.init.normal_(", "normal_(")
        if "from jvggt.ops import normal_" not in text:
            text = text.replace(
                "from jvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2",
                "from jvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2\nfrom jvggt.ops import normal_",
            )
        text = text.replace(
            "self.patch_embed.mask_token.requires_grad_(False)",
            "self.patch_embed.mask_token.stop_grad()",
        )
        text = text.replace("from jittor.utils.checkpoint import checkpoint\n", "")
        text = text.replace(
            "if self.training:\n                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)\n            else:\n                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)",
            "tokens = self.frame_blocks[frame_idx](tokens, pos=pos)",
        )
        text = text.replace(
            "if self.training:\n                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)\n            else:\n                tokens = self.global_blocks[global_idx](tokens, pos=pos)",
            "tokens = self.global_blocks[global_idx](tokens, pos=pos)",
        )
        text = text.replace(
            'self.register_buffer(name, jt.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)',
            "setattr(self, name, jt.array(value, dtype='float32').reshape(1, 1, 3, 1, 1))",
        )
        text = text.replace(
            'self.register_buffer(name, jt.float32(value).view(1, 1, 3, 1, 1), persistent=False)',
            "setattr(self, name, jt.array(value, dtype='float32').reshape(1, 1, 3, 1, 1))",
        )
        text = text.replace(
            "device=images.device",
            "",
        )
        text = text.replace(
            ".to(images.device).to(pos.dtype)",
            ".astype(pos.dtype)",
        )

    if rel_src == "layers/vision_transformer.py":
        if "from jvggt.ops import trunc_normal_, normal_, zeros_" not in text:
            text = text.replace(
                "from jvggt.ops import trunc_normal_",
                "from jvggt.ops import trunc_normal_, normal_, zeros_",
            )
        # nn.functional.* -> F.* but original file has no F import
        if "F." in text and "from jvggt.ops import F" not in text:
            text = text.replace(
                "from jvggt.ops import trunc_normal_, normal_, zeros_",
                "from jvggt.ops import F, trunc_normal_, normal_, zeros_",
            )
        text = text.replace("jt.init.gauss(", "normal_(")
        text = text.replace("nn.init.zeros_(", "zeros_(")
        text = text.replace(
            "if self.training:\n                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)\n            else:\n                x = blk(x)",
            "x = blk(x)",
        )
        text = text.replace(
            "if self.training:\n                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)\n            else:\n                x = blk(x)",
            "x = blk(x)",
        )

    if rel_src == "heads/head_act.py":
        text = text.replace("from jvggt.ops import F", "from jvggt.ops import F, expm1, sign")
        text = text.replace("d.expm1()", "expm1(d)")
        text = text.replace(
            "return y.sign() * y.abs().expm1()",
            "return sign(y) * expm1(y.abs())",
        )

    if rel_src == "heads/dpt_head.py":
        text = text.replace("self.skip_add = nn.quantized.FloatFunctional()", "self.skip_add = None")
        text = text.replace("return self.skip_add.add(out, x)", "return out + x")
        text = text.replace("output = self.skip_add.add(output, res)", "output = output + res")
        text = text.replace("from jvggt.ops import F", "from jvggt.ops import F, ReLU")
        text = text.replace("nn.ReLU(inplace=True)", "ReLU(inplace=True)")

    if rel_src == "utils/rotation.py":
        text = text.replace(
            "out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))",
            "idx = q_abs.argmax(dim=-1)\n    out = jt.gather(quat_candidates, dim=-2, index=idx.unsqueeze(-1).unsqueeze(-1).expand(batch_dim + (1, 4))).squeeze(-2)",
        )
        text = text.replace(
            "flr = jt.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)",
            "flr = jt.array(0.1).astype(q_abs.dtype)",
        )
        text = text.replace(
            "q_abs[..., None].max(flr)",
            "jt.maximum(q_abs[..., None], flr)",
        )
        text = text.replace("matrix.size(-1)", "matrix.shape[-1]")
        text = text.replace("matrix.size(-2)", "matrix.shape[-2]")

    if rel_src == "utils/load_fn.py":
        text = text.replace("from torchvision import transforms as TF", "from PIL import Image as PILImage")
        text = text.replace("to_tensor = TF.ToTensor()", "")
        text = text.replace(
            "img_tensor = to_tensor(square_img)",
            "img_tensor = jt.array(np.array(square_img).astype('float32') / 255.0).permute(2, 0, 1)",
        )
        text = text.replace(
            "img = to_tensor(img)  # Convert to tensor (0, 1)",
            "img = jt.array(np.array(img).astype('float32') / 255.0).permute(2, 0, 1)",
        )
        text = text.replace(
            "original_coords = jt.from_numpy(np.array(original_coords)).float()",
            "original_coords = jt.array(np.array(original_coords), dtype='float32')",
        )
        if "list_images_from_folder" not in text:
            text = text.replace(
                "import jittor as jt\nfrom PIL import Image",
                "import glob\nimport os\n\nimport jittor as jt\nfrom PIL import Image",
            )
            text = text.replace(
                "import numpy as np\n\n\n",
                "import numpy as np\n\n"
                "IMAGE_EXTENSIONS = {\".jpg\", \".jpeg\", \".png\", \".bmp\", \".webp\", \".tif\", \".tiff\"}\n\n\n"
                "def list_images_from_folder(image_folder: str) -> list[str]:\n"
                "    image_folder = os.path.normpath(image_folder)\n"
                "    paths: list[str] = []\n"
                "    for path in sorted(glob.glob(os.path.join(image_folder, \"*\"))):\n"
                "        if not os.path.isfile(path):\n"
                "            continue\n"
                "        if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS:\n"
                "            paths.append(path)\n"
                "    return paths\n\n\n"
                "def select_image_paths(image_paths: list[str], max_images: int) -> list[str]:\n"
                "    if max_images <= 0 or len(image_paths) <= max_images:\n"
                "        return image_paths\n"
                "    return image_paths[:max_images]\n\n\n",
            )

    if rel_src == "heads/utils.py":
        text = text.replace(
            "uu, vv = torch.meshgrid(x_coords, y_coords, indexing=\"xy\")",
            "uu, vv = jt.meshgrid(x_coords, y_coords)\n"
            "    uu, vv = uu.transpose(), vv.transpose()",
        )
        text = text.replace(
            "x_coords = jt.linspace(left_x, right_x, steps=width, dtype=dtype)",
            "x_coords = jt.linspace(left_x, right_x, steps=width)",
        )
        text = text.replace(
            "y_coords = jt.linspace(top_y, bottom_y, steps=height, dtype=dtype)",
            "y_coords = jt.linspace(top_y, bottom_y, steps=height)",
        )
        if "if dtype is not None:" not in text and "x_coords = jt.linspace(left_x, right_x, steps=width)" in text:
            text = text.replace(
                "    y_coords = jt.linspace(top_y, bottom_y, steps=height)\n\n    uu, vv = jt.meshgrid",
                "    y_coords = jt.linspace(top_y, bottom_y, steps=height)\n"
                "    if dtype is not None:\n"
                "        x_coords = x_coords.astype(dtype)\n"
                "        y_coords = y_coords.astype(dtype)\n\n"
                "    uu, vv = jt.meshgrid",
            )

    if rel_src == "layers/rope.py":
        text = text.replace(
            "self.position_cache: Dict[Tuple[int, int], jt.Tensor] = {}",
            "self.position_cache: Dict[Tuple[int, int], object] = {}",
        )
        text = text.replace(
            "self.frequency_cache: Dict[Tuple, Tuple[jt.Tensor, jt.Tensor]] = {}",
            "self.frequency_cache: Dict[Tuple, Tuple[object, object]] = {}",
        )

    if rel_src == "utils/pose_enc.py":
        text = text.replace(
            'intrinsics = jt.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)',
            'intrinsics = jt.zeros(pose_encoding.shape[:2] + (3, 3), dtype="float32")',
        )
        if "import math" not in text:
            text = text.replace("import jittor as jt", "import math\n\nimport jittor as jt")
        if "fov_h = jt.clamp" not in text:
            text = text.replace(
                "        fov_h = pose_encoding[..., 7]\n        fov_w = pose_encoding[..., 8]\n\n        R = quat_to_mat(quat)",
                "        fov_h = pose_encoding[..., 7]\n        fov_w = pose_encoding[..., 8]\n"
                "        fov_h = jt.clamp(fov_h, 1e-4, math.pi - 1e-4)\n"
                "        fov_w = jt.clamp(fov_w, 1e-4, math.pi - 1e-4)\n\n        R = quat_to_mat(quat)",
            )

    return text


def main() -> None:
    for rel_src, rel_dst in FILE_MAP:
        src_path = SRC / rel_src
        dst_path = DST / rel_dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        text = src_path.read_text(encoding="utf-8")
        text = patch_file(rel_src, text)
        dst_path.write_text(text, encoding="utf-8")
        print(f"Generated {rel_dst}")


if __name__ == "__main__":
    main()
