"""视频探测与抽帧。

依赖 OpenCV。调速逻辑不在这里，而是由 planner 先算好「输出第 i 帧取源第几帧」，
本模块只负责按计划把画面写盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .naming import frame_name
from .planner import Plan, build_plan


def probe_video(video_path: str | Path) -> dict:
    """读取视频基础信息。cv2 对中文路径可直接打开（5.x 实测）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    info["duration"] = round(info["total_frames"] / fps, 3) if fps else 0.0
    return info


def write_png(path: Path, image: np.ndarray) -> bool:
    """保存 PNG。用 imencode + write_bytes 而不是 cv2.imwrite，后者在中文路径下会静默失败。"""
    ok, buf = cv2.imencode(".png", image)
    if ok:
        Path(path).write_bytes(buf.tobytes())
    return bool(ok)


def extract_frames(
    video_path: str | Path,
    frame_dir: Path,
    mode: str = "all",
    sample_count: int = 60,
    speed: float = 1.0,
    prefix: str = "frame_",
    start: int = 0,
    digits: int = 6,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """按计划导出帧到 frame_dir。

    返回 dict：saved / fps / source_total / mode / sample_count / speed / plan
    """
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0

    # 帧数未知时无法按比例重采样，退化为顺序导出全部帧
    if total <= 0:
        saved = 0
        while True:
            if should_stop and should_stop():
                break
            ok, frame = cap.read()
            if not ok:
                break
            write_png(frame_dir / frame_name(saved, prefix, start, digits), frame)
            saved += 1
            if on_progress:
                on_progress(saved, 0)
        cap.release()
        return {"saved": saved, "fps": fps, "source_total": 0,
                "mode": mode, "sample_count": 0, "speed": 1.0, "plan": None}

    plan: Plan = build_plan(total, speed=speed, mode=mode, sample_count=sample_count)
    wanted = plan.duplicates()

    saved = 0
    cursor = 0
    while True:
        if should_stop and should_stop():
            break
        ok, frame = cap.read()
        if not ok:
            break
        outs = wanted.get(cursor)
        if outs:
            first_path = frame_dir / frame_name(outs[0], prefix, start, digits)
            write_png(first_path, frame)
            # 慢放时同一源帧要出现多次，复制文件比重新编码 PNG 快得多
            for dup in outs[1:]:
                target = frame_dir / frame_name(dup, prefix, start, digits)
                target.write_bytes(first_path.read_bytes())
            saved += len(outs)
            if on_progress:
                on_progress(saved, plan.out_count)
        cursor += 1

    cap.release()
    return {"saved": saved, "fps": fps, "source_total": total,
            "mode": mode, "sample_count": int(sample_count) if mode == "sample" else 0,
            "speed": plan.speed, "plan": plan}
