# Load PyTorch .pt (zip + pickle) checkpoints without importing torch.
# Large checkpoints stream storage blobs to disk and mmap them to avoid MemoryError.

from __future__ import annotations

import io
import os
import pickle
import shutil
import tempfile
import zipfile
from collections import OrderedDict
from typing import Callable, Dict, Iterator, Tuple

import numpy as np

_STORAGE_TO_DTYPE = {
    "FloatStorage": np.dtype(np.float32),
    "HalfStorage": np.dtype(np.float16),
    "DoubleStorage": np.dtype(np.float64),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
}


class _MmappedStorage:
    """Storage backed by a file on disk (memory-mapped on read)."""

    __slots__ = ("dtype", "_path", "_numel", "_mmap")

    def __init__(self, dtype: np.dtype, path: str, numel: int):
        self.dtype = dtype
        self._path = path
        self._numel = numel
        self._mmap = None

    @property
    def data(self) -> np.ndarray:
        if self._mmap is None:
            self._mmap = np.memmap(self._path, dtype=self.dtype, mode="r", shape=(self._numel,))
        return self._mmap


def _find_archive_prefix(z: zipfile.ZipFile) -> str:
    for name in z.namelist():
        if name.endswith("/data.pkl"):
            return name[: -len("data.pkl")]
    raise ValueError("Not a PyTorch zip checkpoint: missing */data.pkl")


def _storage_name_to_dtype(storage_cls) -> np.dtype:
    name = getattr(storage_cls, "__name__", str(storage_cls))
    if name in _STORAGE_TO_DTYPE:
        return _STORAGE_TO_DTYPE[name]
    if "BFloat16" in name:
        return np.dtype(np.uint16)
    raise NotImplementedError(f"Unsupported storage type: {name}")


class _PtCheckpointReader:
    """Low-memory reader for torch.save zip checkpoints."""

    def __init__(self, pt_path: str):
        self.pt_path = pt_path
        self._temp_dir: str | None = None
        self._zf: zipfile.ZipFile | None = None
        self._prefix: str | None = None

    def __enter__(self) -> _PtCheckpointReader:
        self._temp_dir = tempfile.mkdtemp(prefix="jvggt_ckpt_")
        self._zf = zipfile.ZipFile(self.pt_path, "r")
        self._prefix = _find_archive_prefix(self._zf)
        return self

    def __exit__(self, *args) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None

    def _persistent_load(self, saved_id: Tuple) -> _MmappedStorage:
        assert self._zf is not None and self._prefix is not None and self._temp_dir is not None
        typename, storage_cls, key, _location, numel = saved_id
        if typename != "storage":
            raise pickle.UnpicklingError(f"Unknown persistent id type: {typename}")
        dtype = _storage_name_to_dtype(storage_cls)
        out_path = os.path.join(self._temp_dir, f"storage_{key}.bin")
        if not os.path.isfile(out_path):
            zpath = f"{self._prefix}data/{key}"
            with self._zf.open(zpath, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return _MmappedStorage(dtype, out_path, int(numel))

    def _rebuild_tensor_v2(
        self,
        storage: _MmappedStorage,
        storage_offset: int,
        size: Tuple[int, ...],
        stride: Tuple[int, ...],
        requires_grad: bool,
        backward_hooks,
    ) -> np.ndarray:
        del requires_grad, backward_hooks
        numel = int(np.prod(size))
        flat = storage.data[storage_offset : storage_offset + numel]
        if stride == size or stride == tuple(s for s in size):
            return flat.reshape(size)
        base = storage.data[storage_offset:]
        item = storage.dtype.itemsize
        return np.lib.stride_tricks.as_strided(
            base,
            shape=size,
            strides=tuple(s * item for s in stride),
        )

    def _make_unpickler(self) -> pickle.Unpickler:
        assert self._zf is not None and self._prefix is not None
        reader = self

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return reader._rebuild_tensor_v2
                if module == "torch" and name.endswith("Storage"):
                    return type(name, (), {"__name__": name})
                return super().find_class(module, name)

            def persistent_load(self, saved_id):
                return reader._persistent_load(saved_id)

        return _Unpickler(io.BytesIO(self._zf.read(f"{self._prefix}data.pkl")))

    def load_raw(self) -> OrderedDict:
        raw = self._make_unpickler().load()
        if not isinstance(raw, (dict, OrderedDict)):
            raise TypeError(f"Expected state_dict dict, got {type(raw)}")
        return raw

    def iter_params(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield (name, contiguous ndarray) one at a time to limit peak RAM."""
        raw = self.load_raw()
        names = list(raw.keys())
        for name in names:
            tensor = raw.pop(name)
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Parameter {name!r} has type {type(tensor)}")
            yield name, np.ascontiguousarray(tensor)
        raw.clear()


def load_state_dict_from_pt(pt_path: str) -> Dict[str, np.ndarray]:
    """Load full state_dict into memory (needs ~5 GB RAM for VGGT-1B)."""
    with _PtCheckpointReader(pt_path) as reader:
        return dict(reader.iter_params())


def iter_state_dict_from_pt(pt_path: str) -> Iterator[Tuple[str, np.ndarray]]:
    """Stream parameters from a .pt file (lower peak RAM than load_state_dict_from_pt)."""
    with _PtCheckpointReader(pt_path) as reader:
        yield from reader.iter_params()


def save_state_dict_to_npz(
    pt_path: str,
    npz_path: str,
    progress_every: int = 50,
) -> int:
    """
    Convert .pt -> .npz by writing one tensor at a time (low peak RAM).

    Returns the number of tensors written.
    """
    import numpy.lib.format as nfmt

    count = 0
    with _PtCheckpointReader(pt_path) as reader, zipfile.ZipFile(
        npz_path, "w", compression=zipfile.ZIP_STORED
    ) as zf:
        for name, arr in reader.iter_params():
            buf = io.BytesIO()
            nfmt.write_array(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"  written {count} tensors ...")
    return count
