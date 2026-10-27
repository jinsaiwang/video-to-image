"""配置解析：所有路径与端口都从这里取，代码里不再出现硬编码路径。

优先级（从高到低）：
    1. 命令行参数
    2. 环境变量（MATTE_MODEL_DIR / MATTE_JOBS_DIR / MATTE_HOST / MATTE_PORT）
    3. 当前目录下的 ./RMBG-2.0-onnx、./RMBG-2.0 与 ./jobs（便于「克隆后直接跑」）
    4. 用户目录 ~/.matte-studio/{models, jobs}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .backends import detect

APP_DIR_NAME = ".matte-studio"
# 查找顺序：先找 torch 权重目录（有 GPU 时它最快），再找 ONNX 目录（装了就能跑）。
# 两者都在时可以用 --backend onnx 强制走轻量路线。
MODEL_DIR_NAMES = ("RMBG-2.0", "RMBG-2.0-onnx")
# `matte download` 默认下载到这里
DOWNLOAD_DIR_NAME = "RMBG-2.0-onnx"


def user_data_dir() -> Path:
    """跨平台的用户数据目录，可用 MATTE_HOME 覆盖（便于测试）。"""
    env = os.environ.get("MATTE_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / APP_DIR_NAME


def _as_path(value) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def model_candidates(explicit: str | Path | None = None) -> list[Path]:
    """按优先级列出所有可能的模型目录。"""
    out: list[Path] = []
    for p in (_as_path(explicit), _as_path(os.environ.get("MATTE_MODEL_DIR"))):
        if p:
            out.append(p)
    for name in MODEL_DIR_NAMES:
        out.append(Path.cwd() / name)
    for name in MODEL_DIR_NAMES:
        out.append(user_data_dir() / "models" / name)
    return out


def resolve_model_dir(explicit: str | Path | None = None) -> Path | None:
    for p in model_candidates(explicit):
        if detect(p):
            return p
    # 都没找到时返回「期望位置」，用于提示用户放哪里 / 下载到哪里
    return Path.cwd() / DOWNLOAD_DIR_NAME


def resolve_jobs_dir(explicit: str | Path | None = None) -> Path:
    for p in (_as_path(explicit),
              _as_path(os.environ.get("MATTE_JOBS_DIR")),
              Path.cwd() / "jobs"):
        if p:
            return p
    return user_data_dir() / "jobs"


@dataclass
class Config:
    model_dir: Path | None = None
    jobs_dir: Path = field(default_factory=resolve_jobs_dir)
    backend: str = "auto"          # auto | onnx | torch
    host: str = "127.0.0.1"
    port: int = 7860
    open_browser: bool = True

    @classmethod
    def load(cls, model_dir=None, jobs_dir=None, host=None,
             port=None, open_browser=None, backend=None) -> "Config":
        cfg = cls(
            model_dir=resolve_model_dir(model_dir),
            jobs_dir=resolve_jobs_dir(jobs_dir),
            backend=backend or os.environ.get("MATTE_BACKEND") or "auto",
            host=host or os.environ.get("MATTE_HOST") or "127.0.0.1",
            port=int(port or os.environ.get("MATTE_PORT") or 7860),
            open_browser=True if open_browser is None else bool(open_browser),
        )
        cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    # ---------- 校验 ----------
    @property
    def detected_backend(self) -> str | None:
        return detect(self.model_dir)

    @property
    def active_backend(self) -> str | None:
        """实际会用的后端：显式指定优先，否则按目录内容判断。"""
        return None if self.backend == "auto" else self.backend

    @property
    def model_ready(self) -> bool:
        return self.detected_backend is not None

    @property
    def effective_backend(self) -> str:
        """最终会用的后端名，用于显示设备与日志（永不返回 auto）。"""
        return self.active_backend or self.detected_backend or "auto"

    def require_model(self) -> Path:
        if not self.model_ready:
            raise FileNotFoundError(
                f"找不到可用的抠图模型：{self.model_dir}\n"
                f"一行命令下载（国内加 --mirror cn）：\n"
                f"    matte download\n"
                f"或用 --model-dir / MATTE_MODEL_DIR 指向已有权重目录。"
            )
        return self.model_dir

    def describe(self) -> str:
        backend = self.active_backend or self.detected_backend or "未就绪"
        return "\n".join([
            f"模型目录：{self.model_dir}"
            + ("" if self.model_ready else "   [未就绪，可运行 matte download]"),
            f"推理后端：{backend}",
            f"输出目录：{self.jobs_dir}",
            f"监听地址：{self.host}:{self.port}",
        ])
