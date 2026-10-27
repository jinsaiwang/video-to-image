"""抠图推理。

模型后端目前只有一个：RMBG-2.0（BiRefNet 架构）。torch / transformers 全部
延迟导入，这样 `import matte_studio.matte` 在没有安装 torch 的环境里也不会炸，
单元测试才能只装 pytest 就跑。

后续接入 ONNX / rembg 时，只要实现同样的 `matte_one` + `matte_frames` 接口即可，
上层代码不需要改动。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

INFER_SIZE = 1024
EMPTY_CACHE_EVERY = 60      # 每 N 帧清一次显存；每帧都清会明显拖慢速度


class ModelHub:
    """进程内模型单例，避免每个任务重复加载数百 MB 权重。"""

    _model = None
    _device = None
    _dir = None

    @classmethod
    def get(cls, model_dir):
        import warnings

        # BiRefNet 内部还在用 timm 的旧导入路径，会刷一堆 FutureWarning，
        # 对使用者没有意义，只屏蔽这一类。
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")

        try:
            import torch                                    # noqa: PLC0415 延迟导入
            from transformers import AutoModelForImageSegmentation
        except ImportError as exc:
            raise ImportError(
                "缺少推理依赖，请先安装：\n"
                "    pip install \"matte-studio[gpu]\"\n"
                "（没有 NVIDIA 显卡也能用，torch 会以 CPU 模式运行，只是慢一些）"
            ) from exc

        if cls._model is None or str(cls._dir) != str(model_dir):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForImageSegmentation.from_pretrained(
                str(model_dir), trust_remote_code=True
            ).eval()
            model.to(device)
            # BiRefNet 对 FP16 支持不完善（实测报 expected scalar type Float
            # but found Half），因此固定 FP32，不对外暴露半精度开关。
            cls._model, cls._device, cls._dir = model, device, model_dir
        return cls._model, cls._device

    @classmethod
    def device_name(cls) -> str:
        try:
            import torch
        except ImportError:
            return "未安装 torch"

        if cls._device is not None:
            if cls._device.type == "cuda":
                p = torch.cuda.get_device_properties(0)
                return f"{p.name} · {p.total_memory / 1024 ** 3:.0f} GB"
            return "CPU"

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} · {p.total_memory / 1024 ** 3:.0f} GB · 待命"
        return "CPU"


def build_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((INFER_SIZE, INFER_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def matte_one(model, device, image: Image.Image, tf) -> Image.Image:
    """单帧推理，返回带 alpha 的 RGBA 图。

    流程与 RMBG-2.0 官方示例一致：缩放推理 → 蒙版还原到原图尺寸 → 贴 alpha。
    """
    import torch
    from torchvision import transforms

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = tf(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        preds = model(tensor)[-1].sigmoid().cpu()

    mask = transforms.ToPILImage()(preds[0].squeeze())
    mask = mask.resize(image.size, Image.LANCZOS)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def matte_frames(
    frame_dir: Path,
    alpha_dir: Path,
    model,
    device,
    on_progress: Optional[Callable[[int, int, float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """批量抠图。已存在的帧自动跳过（断点续跑）。"""
    frame_dir, alpha_dir = Path(frame_dir), Path(alpha_dir)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in frame_dir.glob("*.png") if p.is_file())
    total = len(files)
    tf = build_transform()

    done = skipped = failed = dup = 0
    started = time.time()
    prev_hash: Optional[str] = None
    prev_out: Optional[Path] = None

    for i, src in enumerate(files, 1):
        if should_stop and should_stop():
            break

        out = alpha_dir / src.name
        if out.exists() and out.stat().st_size > 0:
            skipped += 1
            if on_progress:
                on_progress(i, total, 0.0, src.name)
            continue

        # 慢放会让同一源帧重复出现，内容相同就不必再跑一次推理
        digest = hashlib.md5(src.read_bytes()).hexdigest()
        if digest == prev_hash and prev_out is not None and prev_out.exists():
            out.write_bytes(prev_out.read_bytes())
            dup += 1
            if on_progress:
                on_progress(i, total, 0.0, f"{src.name}（重复帧，复用上一帧结果）")
            continue

        t0 = time.time()
        try:
            rgba = matte_one(model, device, Image.open(src), tf)
            rgba.save(out, "PNG")
            done += 1
            prev_hash, prev_out = digest, out
        except Exception as exc:                       # 单帧失败不影响整批
            failed += 1
            if on_progress:
                on_progress(i, total, -1.0, f"{src.name} 失败：{exc}")
            continue

        if i % EMPTY_CACHE_EVERY == 0 and getattr(device, "type", "") == "cuda":
            import torch
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(i, total, (time.time() - t0) * 1000, src.name)

    if getattr(device, "type", "") == "cuda":
        import torch
        torch.cuda.empty_cache()

    return {"total": total, "done": done, "skipped": skipped,
            "failed": failed, "duplicated": dup,
            "elapsed": round(time.time() - started, 2)}
