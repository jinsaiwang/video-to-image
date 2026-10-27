"""推理后端的统一接口。

流水线（抽帧后的批量抠图）只认 `Backend` 这一个抽象：

    backend = create("onnx", model_dir)     # 已加载好
    mask = backend.infer_mask(pil_rgb)      # 与原图同尺寸的 L 蒙版

这样把 torch 换成 onnxruntime、以后再加别的模型，都不用改 matte.py 里的
断点续跑、重复帧复用、显存清理那些逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image

INFER_SIZE = 1024          # BiRefNet 系列的固定输入尺寸
MASK_RESAMPLE = Image.LANCZOS


class Backend(ABC):
    """所有推理后端的基类。"""

    name = "base"

    @abstractmethod
    def infer_mask(self, image: Image.Image) -> Image.Image:
        """输入任意尺寸的 RGB 图，返回与原图同尺寸的 L 模式蒙版。"""

    @property
    def device(self) -> str:
        """人类可读的设备描述，用于界面和日志。"""
        return "CPU"

    @property
    def runtime(self) -> str:
        """运行时与版本，例如 onnxruntime 1.29 / torch 2.13。"""
        return ""

    def describe(self) -> str:
        parts = [self.name, self.runtime, self.device]
        return " · ".join(p for p in parts if p)

    def close(self) -> None:
        """释放显存 / 会话。默认什么都不做。"""

    def tick(self, index: int) -> None:
        """每处理完一帧回调一次，后端可在此做周期性清理。

        之所以做成钩子而不是在流水线里写 `torch.cuda.empty_cache()`：
        ONNX 后端用不到它，而且每帧都清会强制同步、明显拖慢速度。
        """

    # ---------- 各后端共用的后处理 ----------
    @staticmethod
    def postprocess(logits: np.ndarray, size: tuple[int, int],
                    sigmoid: str = "auto") -> Image.Image:
        """把网络输出的蒙版张量变成原图尺寸的 L 图。

        `sigmoid` 决定要不要做 0~1 归一：

        - ``auto``（默认）按取值范围判断。RMBG-2.0 的 ONNX 导出已经把
          sigmoid 写进图里（实测输出恰好落在 0.0000~1.0000），此时不能再加一次。
        - ``always`` / ``never`` 供其它导出格式手动指定，可用环境变量
          MATTE_ONNX_SIGMOID 覆盖。
        """
        arr = np.asarray(logits, dtype=np.float32).squeeze()
        if arr.ndim != 2:
            raise ValueError(f"蒙版维度异常：{arr.shape}")

        if sigmoid == "auto":
            lo, hi = float(arr.min()), float(arr.max())
            do_sigmoid = lo < 0.0 or hi > 1.0
        else:
            do_sigmoid = sigmoid == "always"

        if do_sigmoid:
            arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -60, 60)))

        mask = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "L")
        if mask.size != tuple(size):
            mask = mask.resize(tuple(size), MASK_RESAMPLE)
        return mask


def normalize_to_blob(image: Image.Image, size: int = INFER_SIZE) -> np.ndarray:
    """RGB 图 → NCHW float32 张量，均值方差与官方 preprocessor 一致。

    与 torch 后端保持同一套预处理，两条推理路径的结果才可比。
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def attach_alpha(image: Image.Image, mask: Image.Image) -> Image.Image:
    """把蒙版贴成 alpha 通道，得到 RGBA 图。"""
    rgba = image.convert("RGB").convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def find_backend_error(exc: Exception, backend: str) -> Optional[str]:
    """把缺依赖的 ImportError 翻译成能直接照抄的安装命令。"""
    if not isinstance(exc, ImportError):
        return None
    if backend == "onnx":
        return ('缺少 ONNX 推理依赖，请安装：\n'
                '    pip install "matte-studio[onnx]"\n'
                '（CPU 版即可用；有 NVIDIA 显卡可改装 onnxruntime-gpu 提速）')
    return ('缺少 torch 推理依赖，请安装：\n'
            '    pip install "matte-studio[torch]"\n'
            '（没有 NVIDIA 显卡也能用，torch 会以 CPU 模式运行，只是慢一些）')
