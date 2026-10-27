"""模型获取：定位本地权重，或从 HuggingFace / 镜像下载。

开箱即用的前提是「不需要手动找 900MB 权重」，所以这里内置了下载器。

网络说明：国内直连 huggingface.co 通常不通，本模块所有下载都走
`HF_ENDPOINT` 环境变量，默认 https://huggingface.co，可换成
https://hf-mirror.com。`matte download --mirror cn` 就是干这个的。
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# RMBG-2.0 的 ONNX 导出（BiRefNet 结构，CC BY-NC 4.0，与原始权重同一许可）
ONNX_REPO = "yamura4/RMBG-2.0-ONNX"
TORCH_REPO = "briaai/RMBG-2.0"

# 变体 → (仓库内路径, 说明)
ONNX_VARIANTS = {
    "fp32": ("onnx/model.onnx", "全精度，兼容性最好（默认，约 976 MB）"),
    "fp16": ("onnx/model_fp16.onnx", "半精度，体积减半，GPU 上更快（约 489 MB）"),
    "int8": ("onnx/model_int8.onnx", "8bit 量化，CPU 最快（约 348 MB）"),
}

MIRRORS = {
    "hf": "https://huggingface.co",
    "cn": "https://hf-mirror.com",
}


def default_endpoint() -> str:
    """优先用环境变量指定的镜像。"""
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def resolve_endpoint(mirror: str | None = None) -> str:
    if not mirror:
        return default_endpoint()
    return MIRRORS.get(mirror, mirror.rstrip("/"))


def file_url(repo: str, remote_path: str, endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/{repo}/resolve/main/{remote_path.lstrip('/')}"


def _fmt(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def download_file(url: str, dest: Path, on_progress=None, timeout: int = 60) -> Path:
    """流式下载到 dest（先写 .part，成功后再改名，避免半截文件被当成可用模型）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "matte-studio"})

    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done, t0, last = 0, time.time(), 0.0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            now = time.time()
            if on_progress and (now - last > 0.2 or done == total):
                last = now
                on_progress(done, total, done / max(now - t0, 1e-6))
    part.replace(dest)
    return dest


_last_step = [-1]


def _console_progress(done: int, total: int, speed: float):
    """终端里原地刷新；被重定向到文件时改为每 10% 打一行，避免刷屏。"""
    if total:
        pct = done / total * 100
        msg = (f"  下载中 {done / 1048576:.1f} / {total / 1048576:.1f} MB "
               f"({pct:.0f}%)  {speed / 1048576:.1f} MB/s")
        step = int(pct // 10)
    else:
        msg = f"  下载中 {_fmt(done)}  {speed / 1048576:.1f} MB/s"
        step = int(done // (64 * 1048576))

    if sys.stderr.isatty():
        print("\r\033[K" + msg, end="", file=sys.stderr, flush=True)
    elif step != _last_step[0]:
        print(msg, file=sys.stderr, flush=True)
    _last_step[0] = step


def ensure_onnx_model(
    dest_dir: Path | str,
    variant: str = "fp32",
    mirror: str | None = None,
    on_progress=None,
    force: bool = False,
) -> Path:
    """把指定变体下载到 dest_dir/model.onnx，已存在则直接返回。"""
    if variant not in ONNX_VARIANTS:
        raise ValueError(f"未知变体 {variant}，可选：{', '.join(ONNX_VARIANTS)}")

    remote, _desc = ONNX_VARIANTS[variant]
    dest_dir = Path(dest_dir)
    target = dest_dir / ("model.onnx" if variant == "fp32" else f"model_{variant}.onnx")
    if target.is_file() and target.stat().st_size > 0 and not force:
        return target

    endpoint = resolve_endpoint(mirror)
    url = file_url(ONNX_REPO, remote, endpoint)
    progress = on_progress or _console_progress
    print(f"从 {endpoint} 下载 {remote}（{variant}）", file=sys.stderr)
    try:
        download_file(url, target, on_progress=progress)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"下载失败 HTTP {exc.code}：{url}\n"
            f"若在国内网络，试试：matte download --mirror cn"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 {endpoint}（{exc.reason}）\n"
            f"若在国内网络，试试：matte download --mirror cn"
        ) from exc
    finally:
        print("", file=sys.stderr)
    return target
