# VGGT Jittor 推理改造说明（change.md）

## 1. 改造目标

将 VGGT 的**推理路径**从 PyTorch 迁移到 [Jittor](https://github.com/Jittor/jittor)，便于在 Jittor 生态下部署与加速。

**明确不做的事：**

- 不修改 `vggt/` 下原有 PyTorch 实现
- 不修改 `training/` 训练代码与配置
- 不移植 `track_head` 及 `vggt/dependency/` 中的 VGGSfM 跟踪链路（推理 demo 默认不依赖）

---

## 2. 总体方案

| 项目 | 说明 |
|------|------|
| 新包目录 | `jvggt/`（Jittor 推理专用） |
| 源码来源 | 从 `vggt/` 对应文件自动转换生成 |
| 生成工具 | `tools/generate_jittor_port.py` |
| 权重格式 | 仍使用官方 PyTorch `model.pt`，经 numpy 写入 Jittor 参数 |
| 原 PyTorch 包 | `vggt/` 保持不动，训练与对照仍可用 |

---

## 3. 新增 / 变更文件清单

### 3.1 手写新增（不随生成脚本覆盖）

| 路径 | 作用 |
|------|------|
| `jvggt/__init__.py` | 包标识 |
| `jvggt/ops.py` | PyTorch `functional` 的 Jittor 兼容层 |
| `jvggt/weight_loader.py` | 下载并加载 HuggingFace `model.pt` |
| `jvggt/inference.py` | 推理封装：`create_vggt_model`、`run_vggt_forward` 等 |
| `jvggt/models/__init__.py` | 子包占位 |
| `jvggt/heads/__init__.py` | 子包占位 |
| `jvggt/utils/__init__.py` | 子包占位 |
| `tools/generate_jittor_port.py` | 批量生成 `jvggt/` 的转换脚本 |
| `change.md` | 本说明文档 |

### 3.2 由 `generate_jittor_port.py` 从 `vggt/` 生成

| 源 (`vggt/`) | 目标 (`jvggt/`) |
|--------------|-----------------|
| `layers/mlp.py` | `layers/mlp.py` |
| `layers/layer_scale.py` | `layers/layer_scale.py` |
| `layers/drop_path.py` | `layers/drop_path.py` |
| `layers/patch_embed.py` | `layers/patch_embed.py` |
| `layers/swiglu_ffn.py` | `layers/swiglu_ffn.py` |
| `layers/attention.py` | `layers/attention.py` |
| `layers/block.py` | `layers/block.py` |
| `layers/rope.py` | `layers/rope.py` |
| `layers/vision_transformer.py` | `layers/vision_transformer.py` |
| `layers/__init__.py` | `layers/__init__.py` |
| `models/aggregator.py` | `models/aggregator.py` |
| `models/vggt.py` | `models/vggt.py` |
| `heads/head_act.py` | `heads/head_act.py` |
| `heads/utils.py` | `heads/utils.py` |
| `heads/camera_head.py` | `heads/camera_head.py` |
| `heads/dpt_head.py` | `heads/dpt_head.py` |
| `utils/pose_enc.py` | `utils/pose_enc.py` |
| `utils/rotation.py` | `utils/rotation.py` |
| `utils/load_fn.py` | `utils/load_fn.py` |

**未生成（推理 demo 暂不需要）：**

- `heads/track_head.py` 及 `heads/track_modules/*`
- `vggt/dependency/*`（COLMAP BA、LightGlue 等仍依赖 PyTorch 侧逻辑）

### 3.3 已删除

| 路径 | 说明 |
|------|------|
| `vggt_jittor/` | 早期临时目录，已统一为 `jvggt/` |

---

## 4. 目录结构

```
jvggt/
├── __init__.py
├── ops.py                 # 兼容算子
├── weight_loader.py       # 权重加载
├── inference.py           # 高层推理 API
├── layers/                # DINO ViT、Attention、RoPE、Block 等
├── models/                # Aggregator、VGGT
├── heads/                 # CameraHead、DPTHead、激活与 UV 编码
└── utils/                 # load_fn、pose_enc、rotation
```

---

## 5. 推理数据流

```
输入图像路径
    ↓
jvggt.utils.load_fn.load_and_preprocess_images  →  jt.Var [N,3,H,W]
    ↓
jvggt.models.VGGT.forward
    ├── Aggregator（DINO patch embed + 交替 frame/global attention）
    ├── CameraHead  → pose_enc
    ├── DPTHead (depth)  → depth, depth_conf
    └── DPTHead (point)  → world_points, world_points_conf
    ↓
jvggt.utils.pose_enc.pose_encoding_to_extri_intri  → extrinsic, intrinsic
    ↓
.numpy()  → 供 viser / COLMAP / 可视化使用
```

**COLMAP 风格子路径**（对应原 `demo_colmap.run_VGGT`）见 `jvggt.inference.run_vggt_colmap`：仅调用 `aggregator` + `camera_head` + `depth_head`。

---

## 6. 代码级改动明细

### 6.1 全局 import 替换（`convert_imports`）

| PyTorch | Jittor |
|---------|--------|
| `import torch` | `import jittor as jt` |
| `import torch.nn as nn` | `from jittor import nn` |
| `import torch.nn.functional as F` | `from jvggt.ops import F` |
| `from torch.nn.init import trunc_normal_` | `from jvggt.ops import trunc_normal_` |
| `from vggt.xxx` | `from jvggt.xxx` |
| `torch.*` | `jt.*` |
| `from huggingface_hub import PyTorchModelHubMixin` | 删除 |
| `from torch.utils.checkpoint import checkpoint` | 删除 |

### 6.2 `jvggt/ops.py` 手写兼容 API

| API | 实现说明 |
|-----|----------|
| `F.scaled_dot_product_attention` | `QK^T / sqrt(d)` + softmax + `@V` |
| `F.silu` | `x * sigmoid(x)` |
| `F.interpolate` | `nn.interpolate`；`antialias=True` 的 bicubic 退化为普通 bicubic |
| `F.embedding` | `nn.embedding` |
| `F.pad` | `nn.pad` |
| `F.one_hot` | `scatter_` 实现 |
| `trunc_normal_` | numpy 截断正态初始化 |

### 6.3 按文件的特殊 patch

| 文件 | 改动 |
|------|------|
| `models/vggt.py` | 去掉 `PyTorchModelHubMixin`；`autocast` → `jt.no_grad()`；默认 `enable_track=False`；移除 `TrackHead` 引用与 track 前向分支 |
| `models/aggregator.py` | 去掉 training 时 `checkpoint`；`register_buffer` → `setattr` + `jt.array` |
| `layers/vision_transformer.py` | 去掉 block 级 `checkpoint`，推理恒为 `x = blk(x)` |
| `heads/dpt_head.py` | `nn.quantized.FloatFunctional` → `out + x` |
| `utils/rotation.py` | `mat_to_quat` 中 `one_hot` 高级索引 → `jt.gather` |
| `utils/load_fn.py` | 去掉 `torchvision`；PIL → `jt.array(...)/255` |
| `layers/rope.py` | 类型注解中 `jt.Tensor` → `object`（避免静态检查误报） |

### 6.4 权重加载（`jvggt/weight_loader.py`）

1. 优先 **`model.npz`**，其次仓库根目录 **`model.pt`**，否则从 HuggingFace 下载
2. 使用 **`jvggt/pt_loader.py`**（zipfile + pickle + numpy）读取 `.pt`，**不依赖 torch**
3. 各参数转为 `numpy` 后 `assign` 到 Jittor 模型

可选：一次性转换以加快下次加载：

```bash
python tools/convert_pt_to_npz.py --input model.pt --output model.npz
```

转换时把 zip 内 storage **流式写到临时目录并用 mmap 读取**，避免一次性 `zf.read()` 占满内存；写入 `model.npz` 也是**逐 tensor 追加**，不再 `np.savez(**整个字典)`。仍需约 5GB **磁盘**临时空间 + 最终 `model.npz` 体积。

---

## 7. 未修改内容

| 路径 | 说明 |
|------|------|
| `vggt/**` | 全部 PyTorch 源码 |
| `training/**` | 训练、数据、loss、分布式等 |
| `demo_viser.py` | 仍使用 `torch` + `vggt`（PyTorch 原版） |
| `demo_gradio.py` | 仍使用 `torch` + `vggt` |
| `demo_colmap.py` | 仍使用 `torch` + `vggt` |
| **`demo_jvggt.py`** | **新增**：`jvggt` 推理 + 复用 viser 可视化 |
| `requirements.txt` / `requirements_demo.txt` | 未新增 `jittor` 条目（需自行安装） |
| `visual_util.py`、`vggt/utils/geometry.py` 等 | 仍以 NumPy 为主，demo 可直接复用 |

---

## 8. 使用方式

### 8.1 安装依赖

```bash
pip install jittor
# 推理与读 model.pt 均不需要 torch
# 其余 demo 依赖见 requirements_demo.txt
```

### 8.2 完整前向（对应 `VGGT.forward`）

```python
import jittor as jt
from jvggt.inference import create_vggt_model, run_vggt_forward, predictions_to_numpy
from jvggt.utils.load_fn import load_and_preprocess_images

jt.flags.use_cuda = 1

model = create_vggt_model()
image_paths = ["examples/kitchen/images/00.png", "examples/kitchen/images/01.png"]
images = load_and_preprocess_images(image_paths)

predictions = run_vggt_forward(model, images)
result = predictions_to_numpy(predictions, image_hw=images.shape[-2:])

# result 含: pose_enc, depth, depth_conf, world_points, world_points_conf,
#           extrinsic, intrinsic, images 等（numpy）
```

### 8.3 COLMAP 子路径

```python
from jvggt.inference import create_vggt_model, run_vggt_colmap
from jvggt.utils.load_fn import load_and_preprocess_images

model = create_vggt_model()
images = load_and_preprocess_images(image_paths)  # [N,3,H,W]
extrinsic, intrinsic, depth_map, depth_conf = run_vggt_colmap(model, images)
```

### 8.4 运行 Jittor demo（`demo_jvggt.py`）

```bash
# 默认 examples/kitchen/images/；权重优先用项目根目录 model.pt
python demo_jvggt.py

# 显式指定你的 model.pt（与放在根目录效果相同）
python demo_jvggt.py --weights model.pt

# 只跑推理、打印输出 shape（不启动 viser）
python demo_jvggt.py --inference_only

# 指定样例目录
python demo_jvggt.py --image_folder examples/kitchen/images/

# 无 GPU
python demo_jvggt.py --cpu --inference_only
```

---

## 9. Windows 上 Jittor 编译失败（mspdbcore / cl.exe）

Jittor 首次 `import` 会编译 C++ 扩展。自带 `msvc.zip` 在部分机器上会报 `mspdbcore.dll` 或 `cl.exe` 失败。

**推荐做法（任选其一）：**

1. 安装 [Visual Studio 2022 生成工具](https://visualstudio.microsoft.com/zh-hans/downloads/) → 勾选 **“使用 C++ 的桌面开发”**  
2. 在项目根目录用脚本启动（会自动加载 VS 环境并设置 `cc_path`）：

```powershell
.\run_demo_jvggt.ps1 --inference_only
```

3. 或在 **“x64 Native Tools Command Prompt for VS 2022”** 里：

```powershell
conda activate jittor
cd F:\y26\vggt-main
python demo_jvggt.py --inference_only
```

`demo_jvggt.py` 会在导入 Jittor 前调用 `jvggt/jittor_env.py`，尝试自动查找系统 `cl.exe` 并设置环境变量 `cc_path`。

---

## 10. 与 PyTorch 推理的差异与限制

| 项目 | 说明 |
|------|------|
| 混合精度 | 原 demo 使用 `torch.cuda.amp.autocast(bfloat16/float16)`；Jittor 版默认 float32，可按需自行加 `jt.flags.auto_mixed_precision` |
| Track 头 | 未移植；`enable_track` 默认 `False` |
| xFormers | 原工程已禁用；Jittor 版使用标准 attention |
| DINO 位置编码 | `bicubic + antialias` 在 `ops.interpolate` 中未实现 antialias，与 PyTorch 可能有微小数值差 |
| 权重依赖 | 使用 `jvggt/pt_loader.py` 读 `.pt`，无需 `torch`；可选 `model.npz` 加快二次加载 |
| 数值一致性 | 未做逐层对齐测试；若需严格一致，建议对同一输入做 PyTorch / Jittor 输出 diff |

---

## 10. 重新生成 `jvggt/` 源码

当 `vggt/` 中推理相关文件更新后，在项目根目录执行：

```bash
python tools/generate_jittor_port.py
```

**注意：** 该命令会**覆盖** `jvggt/layers`、`jvggt/models`、`jvggt/heads`、`jvggt/utils` 下由脚本生成的文件，**不会**覆盖手写的 `ops.py`、`weight_loader.py`、`inference.py`。

若改过 `models/vggt.py` 中 track 相关手工 patch，重新生成后需确认 `generate_jittor_port.py` 内 `patch_file` 逻辑仍生效。

---

## 11. 后续可做事项（未实现）

- [x] 新增 `demo_jvggt.py`（jvggt 推理 + examples 样例 + viser）
- [ ] 修改 `demo_gradio.py` / `demo_colmap.py` 可选走 `jvggt`
- [ ] 增加 `requirements_jittor.txt`（`jittor` 等）
- [x] 使用 `jvggt/pt_loader.py` 读 `.pt`，运行时无需 `torch`；可选 `tools/convert_pt_to_npz.py` 生成 `model.npz`
- [ ] 移植 `track_head`（若需要点跟踪推理）
- [ ] PyTorch vs Jittor 输出对齐测试脚本

---

## 12. 变更摘要

| 类别 | 数量 / 说明 |
|------|-------------|
| 新增目录 | `jvggt/`（约 23 个 Python 文件） |
| 新增工具 | `tools/generate_jittor_port.py` |
| 修改原仓库核心代码 | **0**（`vggt/`、`training/` 未动） |
| 修改 demo | **0**（待接入） |
| 训练 | **无影响** |
