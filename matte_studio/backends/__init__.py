"""后端注册表与进程内缓存。

对外只暴露三个函数：

    detect(dir)                 → "onnx" | "torch" | None   按目录内容判断
    load(name, dir)             → Backend                    加载并缓存
    device_label(name, dir)     → str                        给界面看的设备描述
"""

from __future__ import annotations

from pathlib import Path

from .base import INFER_SIZE, Backend, attach_alpha, normalize_to_blob

__all__ = ["Backend", "INFER_SIZE", "attach_alpha", "normalize_to_blob",
           "detect", "load", "unload", "device_label", "onnx_file"]

_CACHE: dict[tuple[str, str], Backend] = {}

# 目录里出现这些文件就认作对应后端
ONNX_FILENAMES = ["model.onnx", "model_fp16.onnx", "model_int8.onnx",
                  "model_quantized.onnx", "model_uint8.onnx"]


def onnx_file(path: Path | None) -> Path | None:
    """在一个目录里找到可用的 .onnx 文件（优先 model.onnx）。"""
    if not path:
        return None
    path = Path(path)
    if path.is_file():
        return path if path.suffix.lower() == ".onnx" else None
    if not path.is_dir():
        return None
    for name in ONNX_FILENAMES:
        if (path / name).is_file():
            return path / name
    found = sorted(path.glob("*.onnx"))
    return found[0] if found else None


def detect(path: Path | str | None) -> str | None:
    """按目录内容判断该用哪个后端。ONNX 优先——依赖更轻。"""
    if not path:
        return None
    path = Path(path)
    if onnx_file(path):
        return "onnx"
    if path.is_dir() and (path / "config.json").is_file():
        if any((path / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")):
            return "torch"
    return None


def _create(name: str, model_dir: Path | str) -> Backend:
    if name == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend(onnx_file(Path(model_dir)))
    if name == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(model_dir)
    raise ValueError(f"未知后端：{name}（可选 onnx / torch）")


def load(name: str, model_dir: Path | str) -> Backend:
    """加载后端，同一个 (类型, 目录) 只加载一次。"""
    model_dir = Path(model_dir)
    resolved = name if name in ("onnx", "torch") else detect(model_dir)
    if not resolved:
        raise FileNotFoundError(
            f"目录里既没有 .onnx 也没有 torch 权重：{model_dir}")
    key = (resolved, str(model_dir.resolve() if model_dir.exists() else model_dir))
    if key not in _CACHE:
        _CACHE[key] = _create(resolved, model_dir)
    return _CACHE[key]


def unload() -> None:
    for be in _CACHE.values():
        try:
            be.close()
        except Exception:
            pass
    _CACHE.clear()


def device_label(name: str = "auto", model_dir: Path | str | None = None) -> str:
    """不加载模型也能给出设备信息，供界面顶栏显示。"""
    for (backend_name, _), be in _CACHE.items():
        try:
            return be.device            # 后端名由调用方单独展示，这里不重复
        except Exception:
            break

    # 还没加载过模型，只能探测运行时能力
    if name in ("auto", "torch"):
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                return f"{p.name} · {p.total_memory / 1024 ** 3:.0f} GB · 待命"
            return "CPU · 待命"
        except ImportError:
            if name == "torch":
                return "未安装 torch"

    try:
        import os

        import onnxruntime as ort
        # ONNX 后端默认只用 CPU，这里要跟实际选择保持一致，
        # 否则会显示成「CUDA」但实际在跑 CPU，反而误导。
        pref = (os.environ.get("MATTE_ORT_PROVIDER") or "cpu").lower()
        if pref == "cpu":
            return "CPU · 待命"
        avail = ort.get_available_providers()
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if p in avail:
                return f"{p.replace('ExecutionProvider', '')} · 待命"
        return "CPU · 待命"
    except ImportError:
        return "未安装推理依赖"
