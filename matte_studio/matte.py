"""批量抠图：断点续跑、重复帧复用、单帧失败隔离。

推理本身交给 backends 里的具体实现，这里只负责「把一整个目录的帧可靠地
处理完」，所以换后端不影响这部分逻辑。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .backends.base import Backend


def matte_frames(
    frame_dir: Path,
    alpha_dir: Path,
    backend: Backend,
    on_progress: Optional[Callable[[int, int, float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """批量抠图。已存在的帧自动跳过（断点续跑）。"""
    frame_dir, alpha_dir = Path(frame_dir), Path(alpha_dir)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in frame_dir.glob("*.png") if p.is_file())
    total = len(files)

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
            with Image.open(src) as im:
                rgba = backend.apply(im)
            rgba.save(out, "PNG")
            done += 1
            prev_hash, prev_out = digest, out
        except Exception as exc:                       # 单帧失败不影响整批
            failed += 1
            if on_progress:
                on_progress(i, total, -1.0, f"{src.name} 失败：{exc}")
            continue

        backend.tick(i)

        if on_progress:
            on_progress(i, total, (time.time() - t0) * 1000, src.name)

    return {"total": total, "done": done, "skipped": skipped,
            "failed": failed, "duplicated": dup,
            "elapsed": round(time.time() - started, 2),
            "backend": backend.name, "device": backend.device}
