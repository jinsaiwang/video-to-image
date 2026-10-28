"""任务队列：先进先出、位置准确、排队中可取消。

这里不碰真实推理：用替身把「跑一个任务」替换成「卡住等放行」，这样才能
稳定地制造出「有任务正在跑、后面排着几个」的状态。
"""

from __future__ import annotations

import threading
import time

from matte_studio import server
from matte_studio.config import Config
from matte_studio.jobs import Job


class FakeRun:
    """记录执行顺序，并在 release 之前一直占着队列。"""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.order: list[str] = []

    def __call__(self, job, model_dir, **kwargs):
        self.order.append(job.id)
        self.started.set()
        self.release.wait(5)
        return {}


def wait_until(cond, timeout=3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def make_app(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(server.runner, "run_job", fake)
    monkeypatch.setattr(Config, "require_model", lambda self: self.model_dir)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return server.App(Config(jobs_dir=jobs_dir, model_dir=tmp_path / "model"))


def make_job(app, tmp_path, name):
    job = Job(tmp_path / "jobs", name + ".mp4", {})
    job.video_path = str(tmp_path / f"{name}.mp4")
    job.status = "ready"
    app.store.add(job)
    return job


def test_runs_one_at_a_time_in_submit_order(tmp_path, monkeypatch):
    fake = FakeRun()
    app = make_app(tmp_path, monkeypatch, fake)
    a, b = (make_job(app, tmp_path, n) for n in ("a", "b"))

    assert app.submit(a) == 0              # 队列是空的，它直接开始
    assert fake.started.wait(2)
    assert wait_until(lambda: a.status == "running")

    assert app.submit(b) == 1
    assert b.status == "queued"
    assert b.queue_ahead == 1

    snap = app.queue_snapshot()
    assert snap["running"]["id"] == a.id
    assert [w["id"] for w in snap["waiting"]] == [b.id]
    assert snap["count"] == 1

    fake.release.set()
    assert wait_until(lambda: b.status == "done", 4)
    assert fake.order == [a.id, b.id]      # 严格按提交顺序


def test_queue_positions_shift_after_cancel(tmp_path, monkeypatch):
    fake = FakeRun()
    app = make_app(tmp_path, monkeypatch, fake)
    a, b, c = (make_job(app, tmp_path, n) for n in ("a", "b", "c"))

    app.submit(a)
    assert fake.started.wait(2)
    assert wait_until(lambda: a.status == "running")
    app.submit(b)
    app.submit(c)
    assert c.queue_ahead == 2              # 一个在跑 + b 在前面

    got = app.cancel(b)
    assert got["dequeued"] is True         # 排队中的任务直接出队
    assert b.status == "cancelled"
    assert c.queue_ahead == 1              # 后面的人往前补一位

    fake.release.set()
    assert wait_until(lambda: c.status == "done", 4)
    assert fake.order == [a.id, c.id]      # 被取消的 b 从未执行


def test_cancel_running_job_only_asks_to_stop(tmp_path, monkeypatch):
    fake = FakeRun()
    app = make_app(tmp_path, monkeypatch, fake)
    a = make_job(app, tmp_path, "a")

    app.submit(a)
    assert fake.started.wait(2)
    assert wait_until(lambda: a.status == "running")

    got = app.cancel(a)
    assert got["dequeued"] is False        # 正在跑的只能请求停止
    assert a.stop_flag.is_set()
    fake.release.set()
    assert wait_until(lambda: a.status == "cancelled", 4)


def test_queue_wakes_up_after_going_idle(tmp_path, monkeypatch):
    """队列跑空后工作线程会停在 wait() 上，新任务必须能把它叫醒。

    曾经的 bug：submit 只往列表里塞任务、不 notify，于是「跑完一批再提交」
    会永远卡在 queued。第一批能跑只是因为提交时线程刚好还没进入等待。
    """
    fake = FakeRun()
    fake.release.set()                     # 不阻塞，提交后立刻跑完
    app = make_app(tmp_path, monkeypatch, fake)

    a = make_job(app, tmp_path, "a")
    app.submit(a)
    assert wait_until(lambda: a.status == "done")
    time.sleep(0.2)                        # 让工作线程回到 wait()

    b = make_job(app, tmp_path, "b")
    app.submit(b)
    assert wait_until(lambda: b.status == "done")
    assert fake.order == [a.id, b.id]


def test_progress_only_tracks_extract_and_matte(tmp_path, monkeypatch):
    fake = FakeRun()
    app = make_app(tmp_path, monkeypatch, fake)
    job = make_job(app, tmp_path, "p")
    job.status = "running"

    job.emit(stage="load_model", done=1, total=1, msg="模型就绪")
    assert job.progress == {}              # 加载模型不算处理进度

    job.emit(stage="matte", done=5, total=241, msg="m")
    assert job.progress == {"stage": "matte", "done": 5, "total": 241}
    assert job.snapshot()["progress"]["total"] == 241
