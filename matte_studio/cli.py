"""命令行入口。

    matte                 # 等同于 matte serve
    matte serve           # 启动本地可视化服务
    matte run in.mp4 …    # 无界面直接跑完整流水线
    matte info            # 打印当前配置与设备信息
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .jobs import Job


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--model-dir", help="RMBG-2.0 模型目录（默认自动查找）")
    p.add_argument("--jobs-dir", help="任务输出根目录")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matte",
        description="Matte Studio · 视频转透明连续帧（抽帧 + 调速 + 抠图）",
    )
    parser.add_argument("-V", "--version", action="version", version=f"matte {__version__}")
    sub = parser.add_subparsers(dest="command")

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="启动本地可视化服务")
    _add_common(p_serve)
    p_serve.add_argument("--host", default=None, help="监听地址，默认 127.0.0.1")
    p_serve.add_argument("--port", type=int, default=None, help="端口，默认 7860（占用则自动顺延）")
    p_serve.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")

    # ---- run ----
    p_run = sub.add_parser("run", help="无界面跑完整个流程")
    p_run.add_argument("video", help="输入视频路径")
    p_run.add_argument("-o", "--output", help="输出目录（默认 <jobs-dir>/<视频名>）")
    _add_common(p_run)
    p_run.add_argument("--speed", type=float, default=1.0, help="播放速度，0.25~4，默认 1")
    p_run.add_argument("--mode", choices=["all", "sample"], default="all",
                       help="all=逐帧全部导出（默认），sample=均匀抽样")
    p_run.add_argument("--count", type=int, default=60, help="sample 模式下的帧数")
    p_run.add_argument("--prefix", default="frame_", help="帧文件名前缀，默认 frame_")
    p_run.add_argument("--start", type=int, default=0, help="起始编号，默认 0")
    p_run.add_argument("--digits", type=int, default=6, help="编号位数，默认 6")

    # ---- info ----
    p_info = sub.add_parser("info", help="查看配置与设备信息")
    _add_common(p_info)

    return parser


def cmd_info(config: Config) -> int:
    from .matte import ModelHub
    print(f"版本：{__version__}")
    print(config.describe())
    print(f"推理设备：{ModelHub.device_name()}")
    if not config.model_ready:
        print("\n[提示] 模型未就绪。下载后放到上面的模型目录，或用 --model-dir 指定：")
        print("       https://huggingface.co/briaai/RMBG-2.0")
    return 0


def cmd_serve(config: Config) -> int:
    from .server import serve
    serve(config)
    return 0


def cmd_run(config: Config, args) -> int:
    video = Path(args.video).expanduser()
    if not video.is_file():
        print(f"[错误] 找不到视频：{video}", file=sys.stderr)
        return 2

    config.require_model()          # 先校验，避免抽完帧才发现没模型

    if args.output:
        out = Path(args.output).expanduser()
        jobs_dir, folder = (out.parent or Path(".")), out.name
    else:
        jobs_dir, folder = config.jobs_dir, video.stem

    job = Job(jobs_dir, video.name, {
        "mode": args.mode,
        "sample_count": args.count,
        "speed": args.speed,
        "prefix": args.prefix,
        "start": args.start,
        "digits": args.digits,
        "model_dir": str(config.model_dir),
    }, folder_name=folder)
    job.video_path = str(video)
    job.status = "ready"

    from . import runner

    last_stage = [""]
    tty = sys.stderr.isatty()

    def on_progress(event: dict):
        stage = event.get("stage", "")
        done, total = event.get("done", 0), event.get("total", 0)
        if stage == "matte" and event.get("ms", 0):
            msg = f"抠图 {done}/{total}  {event['ms']:.0f} ms/帧"
        else:
            msg = event.get("msg", "")
        if msg:
            if tty:                       # 终端里原地刷新，重定向时逐行输出
                print(f"\r\033[K{msg}", end="", file=sys.stderr, flush=True)
            else:
                print(msg, file=sys.stderr, flush=True)
        last_stage[0] = stage

    try:
        result = runner.run_job(job, config.model_dir, on_progress=on_progress)
    except KeyboardInterrupt:
        print("\n已中断，已完成的帧保留在输出目录", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n[失败] {exc}", file=sys.stderr)
        return 1
    finally:
        if tty:
            print("", file=sys.stderr)

    print(f"完成：{result['frames']} 帧 · {result['duration']} 秒 · "
          f"速度 {result['speed']}× · {result['device']}")
    print(f"输出：{job.dir / 'alpha'}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"

    config = Config.load(
        model_dir=getattr(args, "model_dir", None),
        jobs_dir=getattr(args, "jobs_dir", None),
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        open_browser=(False if getattr(args, "no_browser", False) else None),
    )

    if command == "info":
        return cmd_info(config)
    if command == "run":
        return cmd_run(config, args)
    return cmd_serve(config)


if __name__ == "__main__":
    raise SystemExit(main())
