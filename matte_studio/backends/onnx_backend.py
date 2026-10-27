"""ONNX Runtime 后端。

相比 torch 后端，它把安装体积从 3GB+ 降到 100MB 级，且 CPU 上就能跑出
可用的速度——这是让别人愿意「克隆下来试一下」的关键。

模型文件是 BiRefNet（RMBG-2.0 的网络结构）的 ONNX 导出，输入固定
1×3×1024×1024，输出是若干张侧向蒙版，取最后一张。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .base import INFER_SIZE, Backend, attach_alpha

# 默认只走 CPU。原因不是保守，是实测：
#   RTX 5060 8GB 上 BiRefNet fp32 的 ONNX 图会在可变形卷积节点申请 800MB 以上的
#   单块显存，CUDA arena 直接 OOM；放开上限虽然能跑，但每帧 42 秒，比 CPU 还慢。
#   显存更大（≥16GB）或想自己调的人可以设 MATTE_ORT_PROVIDER=cuda 打开。
PROVIDER_PRIORITY = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

TORCH_DLL_HINT = """\
onnxruntime 的 CUDA 加速器需要 CUDA 13 的 cublasLt 与 cuDNN 9。
如果本机装过 torch（pip 版自带这些 DLL），可以把 torch/lib 加进搜索路径，
不必另外安装 CUDA Toolkit。"""


def _inject_torch_libs() -> str | None:
    """把 torch 自带的 CUDA DLL 目录加进搜索路径，返回该目录。

    不 import torch：一是不想为了借两个 DLL 付 3 秒的启动代价，
    二是避免 torch 先建好自己的 CUDA 上下文后与 onnxruntime 抢显存。
    """
    if not hasattr(os, "add_dll_directory"):         # 非 Windows 不需要
        return None
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return None
        lib = Path(spec.origin).parent / "lib"
        if not lib.is_dir():
            return None
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        return str(lib)
    except Exception:
        return None


def resolve_providers(preferred: str | None = None) -> list[str]:
    """按 MATTE_ORT_PROVIDER 决定用哪些执行器。

    cpu（默认） 只用 CPU，不加载 CUDA 相关 DLL，启动干净也不会 OOM
    cuda        优先 CUDA，缺依赖时自动回退 CPU
    auto        有什么用什么
    """
    choice = (preferred or os.environ.get("MATTE_ORT_PROVIDER") or "cpu").lower()
    import onnxruntime as ort
    available = set(ort.get_available_providers())

    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        _inject_torch_libs()
        if "CUDAExecutionProvider" not in available:
            print("[提示] 当前 onnxruntime 不带 CUDA 支持，回退 CPU。"
                  "需要 GPU 请安装 onnxruntime-gpu。", file=sys.stderr)
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if choice == "auto":
        return [p for p in PROVIDER_PRIORITY if p in available] or ["CPUExecutionProvider"]
    return [choice] if choice in available else ["CPUExecutionProvider"]


def _pick_output(session) -> str:
    """挑出最终蒙版那一路输出。

    BiRefNet 导出后往往带多个 4 维输出（不同解码层），官方示例取最后一个；
    这里同样取最后一个 4 维输出，并对维度数量做防御。
    """
    candidates = [o for o in session.get_outputs() if len(o.shape) == 4]
    if not candidates:
        if not session.get_outputs():
            raise RuntimeError("ONNX 模型没有任何输出")
        return session.get_outputs()[-1].name
    return candidates[-1].name


class OnnxBackend(Backend):
    name = "onnx"

    def __init__(self, model_path: str | Path, providers=None):
        import onnxruntime as ort                        # noqa: PLC0415

        self._ort = ort
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 ONNX 模型：{model_path}")

        providers = providers or resolve_providers()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3                      # 只报错误，屏蔽 warning

        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers)
        self._providers = list(self._session.get_providers())
        self._input = self._session.get_inputs()[0].name
        self._output = _pick_output(self._session)
        # RMBG-2.0 的官方 ONNX 导出自带 sigmoid，默认 auto 会自动识别；
        # 换用别人的导出（输出是 logits）时可用环境变量强制 always。
        self._sigmoid = (os.environ.get("MATTE_ONNX_SIGMOID") or "auto").lower()
        self.model_path = model_path

    # ---------- 推理 ----------
    def infer_mask(self, image: Image.Image) -> Image.Image:
        from .base import normalize_to_blob

        blob = normalize_to_blob(image, INFER_SIZE)
        raw = self._session.run([self._output], {self._input: blob})[0]
        return self.postprocess(np.asarray(raw)[0], image.size, sigmoid=self._sigmoid)

    def apply(self, image: Image.Image) -> Image.Image:
        return attach_alpha(image, self.infer_mask(image))

    # ---------- 信息 ----------
    @property
    def device(self) -> str:
        if "CUDAExecutionProvider" in self._providers:
            return self._cuda_name() or "CUDA"
        if "DmlExecutionProvider" in self._providers:
            return "DirectML"
        if "CoreMLExecutionProvider" in self._providers:
            return "CoreML"
        return "CPU"

    @staticmethod
    def _cuda_name() -> str | None:
        """借 torch 拿显卡型号；没装 torch 就只显示 CUDA。"""
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} · {p.total_memory / 1024 ** 3:.0f} GB"

    @property
    def runtime(self) -> str:
        return f"onnxruntime {self._ort.__version__} ({self._providers[0].replace('ExecutionProvider', '')})"

    def close(self) -> None:
        self._session = None
        if "CUDAExecutionProvider" in self._providers:
            try:                                          # 有 torch 才能清显存
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
