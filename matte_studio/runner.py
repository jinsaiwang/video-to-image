"""串联整条流水线：抽帧 → 抠图 → 落盘结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from . import frames, matte
from .naming import frame_name


def run_job(
    job,
    model_dir: Path,
    on_progress: Optional[Callable[[dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """跑完一个任务，返回结果 dict，同时写入 result.json。"""
    def emit(**kwargs):
        if on_progress:
            on_progress(kwargs)

    cfg = job.config
    mode = cfg.get("mode", "all")
    speed = float(cfg.get("speed", 1.0))
    prefix = cfg.get("prefix") or "frame_"
    start = int(cfg.get("start", 0) or 0)
    digits = int(cfg.get("digits", 6) or 6)
    sample_count = int(cfg.get("sample_count", 60) or 60)

    job_dir = Path(job.dir)
    frame_dir, alpha_dir = job_dir / "frames", job_dir / "alpha"

    info = frames.probe_video(job.video_path)
    job.info = info

    # ---- 阶段 1：抽帧 ----
    emit(stage="extract", done=0, total=info["total_frames"], msg="开始抽帧")
    ex = frames.extract_frames(
        job.video_path, frame_dir, mode=mode, sample_count=sample_count,
        speed=speed, prefix=prefix, start=start, digits=digits,
        on_progress=lambda done, total: emit(
            stage="extract", done=done, total=total, msg=f"抽帧 {done}/{total or '?'}"),
        should_stop=should_stop,
    )
    emit(stage="extract", done=ex["saved"], total=ex["saved"],
         msg=f"抽帧完成，共 {ex['saved']} 帧")

    # ---- 阶段 2：抠图 ----
    emit(stage="load_model", done=0, total=1, msg="加载抠图模型（首次约 7 秒）")
    model, device = matte.ModelHub.get(model_dir)
    emit(stage="load_model", done=1, total=1, msg=f"模型就绪 · {matte.ModelHub.device_name()}")

    def _progress(i, total, ms, name):
        emit(stage="matte", done=i, total=total, ms=round(ms), msg=name, index=i - 1)

    mt = matte.matte_frames(frame_dir, alpha_dir, model, device,
                            on_progress=_progress, should_stop=should_stop)
    emit(stage="matte", done=mt["total"] - mt["skipped"], total=mt["total"],
         msg=(f"抠图完成：新增 {mt['done']}，复用重复帧 {mt['duplicated']}，"
              f"跳过 {mt['skipped']}，失败 {mt['failed']}"))

    produced = sorted(p for p in alpha_dir.glob("*.png") if p.is_file())
    result = {
        "video": str(job.video_path),
        "fps": ex["fps"],
        "source_total": ex["source_total"],
        "mode": ex["mode"],
        "sample_count": ex["sample_count"],
        "speed": ex["speed"],
        "duration": round(ex["source_total"] / ex["fps"] / ex["speed"], 3) if ex["fps"] else 0.0,
        "frames": len(produced),
        "frames_expected": mt["total"],
        "naming": {
            "prefix": prefix, "start": start, "digits": digits,
            "sample": frame_name(0, prefix, start, digits),
            "first": produced[0].name if produced else None,
            "last": produced[-1].name if produced else None,
        },
        "extract": {k: v for k, v in ex.items() if k != "plan"},
        "matte": mt,
        "frame_dir": str(frame_dir),
        "alpha_dir": str(alpha_dir),
        "device": matte.ModelHub.device_name(),
    }
    (job_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    job.result = result
    job.invalidate()
    emit(stage="done", done=mt["total"], total=mt["total"], msg="全部完成", result=result)
    return result
