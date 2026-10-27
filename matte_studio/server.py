"""本地 HTTP 服务：上传、调速、启动、进度推送、帧图片。

只用标准库 http.server + SSE，不引入 Web 框架。这样安装依赖更少，
也不会和用户已有的 Web 服务端口/版本冲突。
"""

from __future__ import annotations

import io
import json
import os
import queue
import socket
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__, runner
from .config import Config
from .jobs import Job, JobStore
from .backends import device_label
from .naming import clean_prefix, safe_name

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 2 * 1024 ** 3          # 2GB


class App:
    """把配置、任务表、GPU 锁集中在一处，方便测试时替换。"""

    def __init__(self, config: Config):
        self.config = config
        self.store = JobStore()
        self.gpu_lock = threading.Semaphore(1)
        self.restored = self.store.restore(config.jobs_dir)

    # ---------- 任务执行 ----------
    def start_job(self, job: Job):
        job.status = "running"
        job.started_at = time.time()
        job.emit(stage="start", msg=f"任务开始：{job.name}")
        with self.gpu_lock:
            try:
                runner.run_job(job, self.config.require_model(),
                               backend=self.config.active_backend or "auto",
                               on_progress=lambda ev: job.emit(**ev),
                               should_stop=job.stop_flag.is_set)
                job.status = "cancelled" if job.stop_flag.is_set() else "done"
                job.emit(stage=job.status,
                         msg="已取消" if job.status == "cancelled" else "全部完成")
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.emit(stage="error", msg=job.error,
                         traceback=traceback.format_exc()[-1500:])
        job.finished_at = time.time()
        job.invalidate()


def frame_bytes(job: Job, index: int, kind: str, width: int):
    """取第 index 帧，可按宽度生成缩略图。"""
    from PIL import Image

    files = job.files(kind)
    if not files or index < 0 or index >= len(files):
        return None
    path = files[index]
    if width <= 0:
        return path.read_bytes(), "image/png"

    image = Image.open(path)
    w, h = image.size
    width = min(width, w)
    image = image.resize((width, max(1, round(h * width / w))), Image.LANCZOS)
    buf = io.BytesIO()
    if kind == "alpha":                      # 透明帧必须保留 alpha
        image.save(buf, "PNG", optimize=True)
        return buf.getvalue(), "image/png"
    image.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue(), "image/jpeg"


class Handler(BaseHTTPRequestHandler):
    server_version = f"MatteStudio/{__version__}"

    # ---------- 工具 ----------
    @property
    def app(self) -> App:
        return self.server.app

    def log_message(self, fmt, *args):       # 静音访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, data: bytes, ctype: str, cache="no-store"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    # ---------- GET ----------
    def do_GET(self):
        url = urlparse(self.path)
        path, qs = unquote(url.path), parse_qs(url.query)   # 任务 ID 可能是中文

        if path in ("/", "/index.html"):
            return self._raw((WEB_DIR / "index.html").read_bytes(),
                             "text/html; charset=utf-8")

        if path == "/api/info":
            cfg = self.app.config
            versions = {}
            for mod, label in (("onnxruntime", "onnxruntime"), ("torch", "torch")):
                try:
                    versions[label] = __import__(mod).__version__
                except ImportError:
                    versions[label] = None
            return self._json({
                "version": __version__,
                "model_dir": str(cfg.model_dir),
                "model_exists": cfg.model_ready,
                "backend": cfg.effective_backend,
                **versions,
                "device": device_label(cfg.effective_backend, cfg.model_dir),
                "jobs_dir": str(cfg.jobs_dir),
            })

        if path == "/api/jobs":
            return self._json({"jobs": self.app.store.recent()})

        if path.startswith("/api/jobs/") and path.endswith("/events"):
            return self._sse(path.split("/")[3])

        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            job = self.app.store.get(parts[2]) if len(parts) > 2 else None
            if not job:
                return self._json({"error": "任务不存在"}, 404)

            if len(parts) == 3:
                return self._json(job.snapshot())

            if parts[3] == "frame":
                try:
                    index = int(qs.get("i", ["0"])[0])
                    width = int(qs.get("w", ["0"])[0])
                except ValueError:
                    return self._json({"error": "参数不合法"}, 400)
                kind = qs.get("kind", ["alpha"])[0]
                got = frame_bytes(job, index, kind, width)
                if not got:
                    return self._json({"error": "帧不存在"}, 404)
                data, ctype = got
                return self._raw(data, ctype, cache="public, max-age=3600")

        return self._json({"error": "not found"}, 404)

    # ---------- POST ----------
    def do_POST(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        path = unquote(url.path)

        if path == "/api/jobs":
            return self._create_job(qs)
        if path.startswith("/api/jobs/") and path.endswith("/start"):
            return self._start_job(qs, path.strip("/").split("/")[2])
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            return self._simple(path, lambda job: (job.stop_flag.set(),
                                                   job.emit(stage="cancel_request",
                                                            msg="正在停止，处理完当前帧后退出"),
                                                   {"ok": True})[-1])
        if path.startswith("/api/jobs/") and path.endswith("/open"):
            return self._simple(path, self._open_folder)

        return self._json({"error": "not found"}, 404)

    # ---------- 具体动作 ----------
    def _create_job(self, qs):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json({"error": "空文件"}, 400)
        if length > MAX_UPLOAD:
            return self._json({"error": "文件过大"}, 413)

        name = unquote(qs.get("name", ["video.mp4"])[0])
        job = Job(self.app.config.jobs_dir, name, {"model_dir": str(self.app.config.model_dir)})
        video_path = job.dir / ("source" + (Path(name).suffix or ".mp4"))

        remaining = length
        with open(video_path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            job.status, job.error = "error", "上传中断"
            return self._json({"error": "上传中断", "job": job.id}, 400)

        job.video_path = str(video_path)
        try:
            from .frames import probe_video
            job.info = probe_video(video_path)
        except Exception as exc:
            job.info = {}
            job.error = f"无法读取视频信息：{exc}"

        job.status = "ready"
        self.app.store.add(job)
        # 只上传不启动，等用户预览调速、确认参数后再调 /start
        return self._json({"job": job.id, "name": job.name,
                           "dir": job.dir.name, "info": job.info})

    def _start_job(self, qs, job_id):
        job = self.app.store.get(job_id)
        if not job:
            return self._json({"error": "任务不存在"}, 404)
        if job.status != "ready":
            return self._json({"error": f"任务状态为 {job.status}，无法启动"}, 409)

        try:
            speed = float(qs.get("speed", ["1"])[0])
            start = max(0, int(qs.get("start", ["0"])[0]))
            digits = min(8, max(1, int(qs.get("digits", ["6"])[0])))
            count = int(qs.get("count", ["60"])[0])
        except ValueError:
            return self._json({"error": "参数不合法"}, 400)

        job.config.update({
            "mode": qs.get("mode", ["all"])[0],
            "sample_count": count,
            "speed": speed,
            "prefix": clean_prefix(unquote(qs.get("prefix", ["frame_"])[0])),
            "start": start,
            "digits": digits,
        })

        # 任务名可能在选完视频后才改：此时目录里只有源视频，直接改名即可
        want = unquote(qs.get("jobname", [""])[0]).strip()
        if want:
            self._rename_job(job, want)

        threading.Thread(target=self.app.start_job, args=(job,), daemon=True).start()
        return self._json({"ok": True, "job": job.id, "dir": job.dir.name,
                           "config": job.config})

    def _rename_job(self, job: Job, want: str):
        target = job.dir.parent / safe_name(want, fallback=job.id)
        if target.resolve() == job.dir.resolve():
            return
        if target.exists():
            target = job.dir.parent / f"{safe_name(want, fallback=job.id)}_{job.id[:4]}"
        try:
            old_id = job.id
            job.dir.rename(target)
            job.dir = target
            job.id = target.name
            self.app.store.rekey(old_id, job)
            if job.video_path:      # 目录改名后源视频路径也要跟着变
                job.video_path = str(target / Path(job.video_path).name)
        except OSError:
            pass

    def _simple(self, path, action):
        job = self.app.store.get(path.strip("/").split("/")[2])
        if not job:
            return self._json({"error": "任务不存在"}, 404)
        return self._json(action(job))

    def _open_folder(self, job: Job):
        target = job.dir / "alpha" if (job.dir / "alpha").exists() else job.dir
        try:
            if os.name == "nt":
                os.startfile(str(target))          # noqa: S606
            elif sys_platform() == "darwin":
                os.system(f'open "{target}"')
            else:
                os.system(f'xdg-open "{target}"')
            return {"ok": True, "path": str(target)}
        except Exception as exc:
            return {"error": str(exc)}

    # ---------- SSE ----------
    def _sse(self, job_id: str):
        job = self.app.store.get(job_id)
        if not job:
            return self._json({"error": "任务不存在"}, 404)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=1000)
        with job._lock:
            backlog = list(job.events[-60:])
            job.subscribers.append(q)

        def send(payload: dict):
            self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            send({"stage": "hello", "job": job.snapshot()})
            for event in backlog:
                send(event)

            last_ping = time.time()
            while True:
                try:
                    send(q.get(timeout=2))
                except queue.Empty:
                    if time.time() - last_ping > 10:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
                    if (job.status in ("done", "error", "cancelled")
                            and time.time() - (job.finished_at or time.time()) > 3):
                        break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with job._lock:
                if q in job.subscribers:
                    job.subscribers.remove(q)


def sys_platform() -> str:
    import sys
    return sys.platform


def port_in_use(host: str, port: int) -> bool:
    """端口是否已有服务在监听。

    Windows 上 SO_REUSEADDR 允许两个进程绑定同一端口，只用 bind 试探会误判
    为空闲，导致新进程「启动成功」但请求全被旧进程接走。所以先尝试连接。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def create_server(config: Config, tries: int = 20):
    """在 config.port 起向上找一个真正空闲的端口。"""
    app = App(config)
    last_error = None
    for port in range(config.port, config.port + tries):
        if port_in_use(config.host, port):
            print(f"  [端口] {port} 已被占用，尝试下一个")
            continue
        try:
            httpd = ThreadingHTTPServer((config.host, port), Handler)
        except OSError as exc:
            last_error = exc
            continue
        httpd.daemon_threads = True
        httpd.app = app
        config.port = port
        return httpd, app
    raise RuntimeError(f"找不到空闲端口（最后错误：{last_error}）")


def serve(config: Config):
    httpd, app = create_server(config)
    url = f"http://{config.host}:{config.port}"

    print("=" * 58)
    print(f"  Matte Studio v{__version__}  ·  视频转透明连续帧")
    print("=" * 58)
    print(f"  地址：{url}")
    print(f"  模型：{config.model_dir}" + ("" if config.model_ready else "   [未就绪]"))
    print(f"  输出：{config.jobs_dir}")
    if app.restored:
        print(f"  历史：已恢复 {app.restored} 个任务，可在页面底部打开")
    print("  关闭本窗口即停止服务")
    print("=" * 58)

    if not config.model_ready:
        print("  [提醒] 没有找到模型权重，界面会提示；也可以用 --model-dir 指定")

    if config.open_browser:
        threading.Timer(1.2, lambda: _open_browser(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


def _open_browser(url: str):
    try:
        if webbrowser.open(url, new=1):
            return
    except Exception:
        pass
    try:
        if os.name == "nt":
            os.startfile(url)                  # noqa: S606
    except Exception:
        pass
