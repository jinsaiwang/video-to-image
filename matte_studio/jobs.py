"""任务对象与任务注册表。

任务 ID 直接使用输出目录名：重启后从磁盘恢复时 ID 不变，浏览器记住的
「上次任务」才能在服务重启后依然打开（随机 ID 一重启就失效）。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .naming import safe_name


class Job:
    """一个视频 → 透明帧的任务。"""

    def __init__(self, jobs_dir: Path, name: str, config: dict,
                 folder_name: str | None = None):
        temp_id = uuid.uuid4().hex[:8]
        self.name = name
        self.config = config

        folder = safe_name(folder_name or name, fallback=temp_id)
        self.dir = Path(jobs_dir) / folder
        if self.dir.exists():                    # 重名加后缀，不覆盖旧结果
            self.dir = Path(jobs_dir) / f"{folder}_{temp_id[:4]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.id = self.dir.name

        self.status = "queued"                   # queued|ready|running|done|error|cancelled
        self.events: list[dict] = []
        self.subscribers: list = []
        self.stop_flag = threading.Event()
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.info: dict = {}
        self.video_path: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._file_cache: dict[str, list[Path]] = {}
        self._lock = threading.Lock()

    # ---------- 进度事件 ----------
    def emit(self, **event):
        event["t"] = time.time()
        event["status"] = self.status
        with self._lock:
            self.events.append(event)
            if len(self.events) > 200:
                self.events = self.events[-200:]
            subscribers = list(self.subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    # ---------- 文件索引 ----------
    def files(self, kind: str = "alpha") -> list[Path]:
        """按序号返回帧文件列表。文件名可由用户自定义，不能靠固定格式拼路径。"""
        sub = "alpha" if kind == "alpha" else "frames"
        cache = self._file_cache.get(kind)
        if cache is None or self.status == "running":
            cache = sorted(p for p in (self.dir / sub).glob("*.png") if p.is_file())
            self._file_cache[kind] = cache
        return cache

    def invalidate(self):
        self._file_cache.clear()

    # ---------- 序列化 ----------
    def snapshot(self) -> dict:
        return {
            "id": self.id, "name": self.name, "config": self.config,
            "status": self.status, "result": self.result, "error": self.error,
            "info": self.info, "started_at": self.started_at,
            "finished_at": self.finished_at, "dir": str(self.dir),
        }

    @classmethod
    def from_disk(cls, path: Path) -> Optional["Job"]:
        """从磁盘上的 result.json 恢复一个已完成任务。"""
        result_file = Path(path) / "result.json"
        if not result_file.is_file():
            return None
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            return None

        obj = cls.__new__(cls)
        obj.id = path.name
        obj.name = path.name
        naming = result.get("naming") or {}
        obj.config = {
            "mode": result.get("mode"),
            "speed": result.get("speed", 1.0),
            "prefix": naming.get("prefix"),
            "start": naming.get("start"),
            "digits": naming.get("digits"),
        }
        obj.dir = Path(path)
        obj.status = "done"
        obj.events = []
        obj.subscribers = []
        obj.stop_flag = threading.Event()
        obj.result = result
        obj.error = None
        obj.info = {"total_frames": result.get("source_total"),
                    "fps": result.get("fps"),
                    "duration": result.get("duration")}
        obj.video_path = None
        obj.started_at = None
        obj.finished_at = result_file.stat().st_mtime
        obj._file_cache = {}
        obj._lock = threading.Lock()
        return obj


class JobStore:
    """内存中的任务表，启动时可从磁盘批量恢复。"""

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> Job:
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def rekey(self, old_id: str, job: Job):
        self._jobs.pop(old_id, None)
        self._jobs[job.id] = job

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def restore(self, jobs_dir: Path) -> int:
        """扫描 jobs 目录，恢复已完成的任务。"""
        jobs_dir = Path(jobs_dir)
        if not jobs_dir.is_dir():
            return 0
        count = 0
        for sub in sorted(jobs_dir.iterdir()):
            if not sub.is_dir():
                continue
            job = Job.from_disk(sub)
            if job:
                self._jobs[job.id] = job
                count += 1
        return count

    def recent(self, limit: int = 20) -> list[dict]:
        items = [{
            "id": j.id, "name": j.name, "status": j.status,
            "frames": (j.result or {}).get("frames"),
            "speed": (j.result or {}).get("speed", 1.0),
            "duration": (j.result or {}).get("duration"),
            "finished_at": j.finished_at,
        } for j in self._jobs.values()]
        items.sort(key=lambda x: x.get("finished_at") or 0, reverse=True)
        return items[:limit]
