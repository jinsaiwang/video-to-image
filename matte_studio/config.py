"""配置解析：所有路径与端口都从这里取，代码里不再出现硬编码路径。

优先级（从高到低）：
    1. 命令行参数
    2. 环境变量（MATTE_MODEL_DIR / MATTE_JOBS_DIR / MATTE_HOST / MATTE_PORT）
    3. 当前目录下的 ./RMBG-2.0 与 ./jobs（便于「克隆后直接跑」）
    4. 用户目录 ~/.matte-studio/{models, jobs}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR_NAME = ".matte-studio"
MODEL_DIR_NAME = "RMBG-2.0"


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


def _is_model_dir(path: Path | None) -> bool:
    if not path or not path.is_dir():
        return False
    # 有 config.json 且至少有一个权重文件，才认为可用
    has_cfg = (path / "config.json").is_file()
    has_weight = any((path / n).is_file() for n in
                     ("model.safetensors", "pytorch_model.bin"))
    return has_cfg and has_weight


def resolve_model_dir(explicit: str | Path | None = None) -> Path | None:
    candidates = [
        _as_path(explicit),
        _as_path(os.environ.get("MATTE_MODEL_DIR")),
        Path.cwd() / MODEL_DIR_NAME,
        user_data_dir() / "models" / MODEL_DIR_NAME,
    ]
    for p in candidates:
        if _is_model_dir(p):
            return p
    # 都没找到时返回「期望位置」，用于提示用户放哪里
    return candidates[2]


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
    host: str = "127.0.0.1"
    port: int = 7860
    open_browser: bool = True

    @classmethod
    def load(cls, model_dir=None, jobs_dir=None, host=None,
             port=None, open_browser=None) -> "Config":
        cfg = cls(
            model_dir=resolve_model_dir(model_dir),
            jobs_dir=resolve_jobs_dir(jobs_dir),
            host=host or os.environ.get("MATTE_HOST") or "127.0.0.1",
            port=int(port or os.environ.get("MATTE_PORT") or 7860),
            open_browser=True if open_browser is None else bool(open_browser),
        )
        cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    # ---------- 校验 ----------
    @property
    def model_ready(self) -> bool:
        return _is_model_dir(self.model_dir)

    def require_model(self) -> Path:
        if not self.model_ready:
            raise FileNotFoundError(
                f"找不到抠图模型：{self.model_dir}\n"
                f"请把 RMBG-2.0 放进去，或用 --model-dir / MATTE_MODEL_DIR 指定。\n"
                f"下载地址：https://huggingface.co/briaai/RMBG-2.0"
            )
        return self.model_dir

    def describe(self) -> str:
        return "\n".join([
            f"模型目录：{self.model_dir}"
            + ("" if self.model_ready else "   [未就绪]"),
            f"输出目录：{self.jobs_dir}",
            f"监听地址：{self.host}:{self.port}",
        ])
