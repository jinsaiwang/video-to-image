"""torch 后端：用 transformers 加载 RMBG-2.0（BiRefNet）原始权重。

重量级依赖全部延迟导入，所以 `import matte_studio` 在没有 torch 的环境里
依然可用（单元测试只依赖 pytest）。
"""

from __future__ import annotations

import warnings
from pathlib import Path

from PIL import Image

from .base import INFER_SIZE, Backend, attach_alpha


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, model_dir: str | Path):
        import torch                                        # noqa: PLC0415
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        # BiRefNet 内部还在用 timm 的旧导入路径，会刷一堆 FutureWarning，
        # 对使用者没有意义，只屏蔽这一类。
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")

        self._torch = torch
        self._preprocess = transforms.Compose([
            transforms.Resize((INFER_SIZE, INFER_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._to_pil = transforms.ToPILImage()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BiRefNet 对 FP16 支持不完善（实测报 expected scalar type Float but
        # found Half），因此固定 FP32，不对外暴露半精度开关。
        self._model = AutoModelForImageSegmentation.from_pretrained(
            str(model_dir), trust_remote_code=True
        ).eval().to(self._device)

    # ---------- 推理 ----------
    def infer_mask(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            preds = self._model(tensor)[-1].sigmoid().cpu()

        mask = self._to_pil(preds[0].squeeze())
        return mask.resize(image.size, Image.LANCZOS)

    def apply(self, image: Image.Image) -> Image.Image:
        return attach_alpha(image, self.infer_mask(image))

    # ---------- 信息 ----------
    @property
    def device(self) -> str:
        if self._device.type != "cuda":
            return "CPU"
        p = self._torch.cuda.get_device_properties(0)
        return f"{p.name} · {p.total_memory / 1024 ** 3:.0f} GB"

    @property
    def runtime(self) -> str:
        return f"torch {self._torch.__version__.split('+')[0]}"

    def tick(self, index: int) -> None:
        # 每帧都清会强制同步、明显拖慢速度，改成一分钟量级的周期清理。
        if index % 60 == 0 and self._device.type == "cuda":
            self._torch.cuda.empty_cache()

    def close(self) -> None:
        self._model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
