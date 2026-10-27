# Matte Studio · 视频转透明连续帧

把一段视频变成**带 alpha 通道的连续帧序列**：抽帧 → 调速 → 抠图，全程可视化，也可以纯命令行批量跑。

> **English**: Matte Studio turns a video into a sequence of transparent PNG frames. It extracts frames, applies playback-speed changes (0.25×–4×) and removes the background frame by frame, then lets you preview the result in a local web UI. Built for animation workflows that need frame-exact output (After Effects / Spine / Unity).

## 它解决什么

把视频抠成透明帧，工具链通常要拼三四个命令行工具：`ffmpeg` 抽帧 → 手动调速度 → 循环跑抠图 → 自己重命名。这个项目把它们串成一条流水线，并且补上了实际使用中最容易出错的几件事：

| 常见做法 | 本项目 |
|---|---|
| 均匀抽 60 帧 | **默认逐帧全部导出**，保住原始动画节奏 |
| 调速要先用 ffmpeg 重新编码一遍 | 抽帧时做**时间重采样**，不重编码、无画质损失 |
| 崩了从头再跑 | **断点续跑**，已完成的帧自动跳过 |
| 文件名固定 `frame_000001.png` | 前缀 / 起始编号 / 位数都可自定义，直接进 AE、Spine |
| 只能单张图抠 | 面向**视频序列**：整批推理，模型只加载一次 |

## 特性

- **可视化预览调速**：改速度时浏览器里直接按新倍速播放，确认后才开跑
- **两种推理后端**：ONNX Runtime（装起来轻）与 torch（显卡上快），可切换
- **一行下载模型**：`matte download` 自动取权重，支持国内镜像
- **零前端依赖**：后端只用 Python 标准库 `http.server` + SSE，不需要 Node 或 Web 框架
- **断点续跑 / 重复帧复用**：中断后跳过已完成的帧，慢放的重复画面不重复推理
- **结果可回看**：任务落盘为档案，重启服务后仍能在页面里打开历史结果
- **CLI 可用**：`matte run in.mp4 --speed 2` 一行跑完，适合批处理

## 安装

先选后端，两者产出的结果几乎一致（实测 alpha 蒙版 IoU 0.9999），差别在装起来多重、跑起来多快：

| | ONNX 后端 | torch 后端 |
|---|---|---|
| 安装体积 | **约 100 MB** | 约 3 GB |
| 适合 | 没有显卡、只想试一下 | 有 NVIDIA 显卡、要跑长视频 |
| 本机实测 | 15 s/帧（CPU） | **0.71 s/帧**（RTX 5060） |

```bash
git clone https://github.com/jinsaiwang/video-to-image.git
cd video-to-image

# 方案 A：轻量（推荐先试这个）
pip install -e ".[onnx]"

# 方案 B：有显卡，追求速度
pip install -e ".[torch]"

# 有显卡且想要 ONNX 的 GPU 加速，把 onnxruntime 换成：
#   pip install onnxruntime-gpu
```

> 基础依赖（`numpy` / `opencv-python` / `Pillow`）很轻，装完就能启动界面、抽帧、调速；
> 推理依赖拆成可选，不装也能先把流程跑通，只是不能抠图。

## 获取模型

**不用手动找模型**，一条命令下载到当前目录的 `RMBG-2.0-onnx/`：

```bash
matte download --mirror cn     # 国内网络用 cn（走 hf-mirror.com）
matte download                 # 海外直连 huggingface.co
```

可选变体：`--variant fp32`（默认，976 MB）/ `fp16`（489 MB）/ `int8`（348 MB）。
下载器支持断点续传之外的重复检测：文件已在就跳过。

也可以手动准备 torch 权重，放到下面任一位置：

```
./RMBG-2.0/                     ← HuggingFace 仓库文件原样下载
~/.matte-studio/models/RMBG-2.0/
```

或用参数指定：`matte serve --model-dir /path/to/RMBG-2.0`，或设 `MATTE_MODEL_DIR`。

> ⚠️ **许可证提醒**：RMBG-2.0 模型由 BRIA AI 发布，**仅限非商业使用**。本仓库代码是 MIT，
> 但模型不受 MIT 约束。商用请先阅读 [模型条款](https://huggingface.co/briaai/RMBG-2.0)。

## 性能实测

同一台机器（RTX 5060 Laptop 8 GB + 笔记本 CPU），同一段 10 秒 / 241 帧视频，单帧耗时：

| 后端 | 设备 | 单帧 | 241 帧预计 | 说明 |
|---|---|---|---|---|
| torch | RTX 5060 | **0.71 s** | 约 3 分钟 | 推荐配置 |
| onnx | CPU | 15 s | 约 60 分钟 | 能跑，长视频不划算 |
| onnx | CPU（int8 模型） | 45 s | — | **更慢，别选** |
| onnx | RTX 5060（CUDA） | OOM / 42 s | — | 8 GB 显存不够，见下 |

**为什么 ONNX 不用 GPU**：BiRefNet 的可变形卷积在 ONNX 图上会申请 800 MB 以上的单块显存，
8 GB 卡直接 OOM；放开显存上限能跑，但每帧 42 秒，比 CPU 还慢。所以 ONNX 后端**默认只用 CPU**，
需要时设 `MATTE_ORT_PROVIDER=cuda` 自行打开（16 GB 以上的卡可能表现不同，欢迎反馈实测数据）。

**质量是否一致**：用同一帧分别跑两条后端，最终 alpha 蒙版 **IoU 0.9999、平均差 0.24/255**，
透明像素占比 69.6% vs 69.7%——差异只在边缘一两个像素，肉眼不可见。

## 快速开始

### 图形界面

```bash
matte serve          # 或双击 启动.bat
```

浏览器会自动打开 <http://127.0.0.1:7860>。流程：

1. 拖入视频（此时只上传，不会开始处理）
2. 调整播放速度，左侧实时按新倍速预览
3. 设置抽帧方式与输出命名
4. 点「开始处理」
5. 跑完在结果页用棋盘格背景播放，逐帧检查

### 命令行

```bash
matte run input.mp4 --speed 2 -o out/pet
```

输出结构：

```
out/pet/
├── frames/          原始帧（不透明）
├── alpha/           ★ 透明帧，带 alpha 通道
│   ├── frame_000000.png
│   └── ...
└── result.json      参数、帧数、耗时记录
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--speed 2` | 2 倍速；`0.5` 为慢放。范围 0.25–4 |
| `--mode sample --count 24` | 均匀抽样 24 帧（默认 `all` 逐帧全部导出） |
| `--prefix pet_ --start 1 --digits 4` | 输出 `pet_0001.png` 起 |
| `--model-dir PATH` | 指定模型目录 |
| `--backend onnx\|torch` | 强制后端，默认按模型目录自动判断 |

## 调速是怎么做的

不做二次编码，直接在抽帧时做时间重采样：

```
输出第 i 帧  ←  源视频第 int(i × speed) 帧
```

以 241 帧（24fps，10.04 秒）为例：

| 速度 | 输出帧数 | 时长 | 行为 |
|---|---|---|---|
| 4× | 60 | 2.50 s | 每 4 帧取 1 帧 |
| 2× | 120 | 5.00 s | 每 2 帧取 1 帧 |
| 1× | 241 | 10.04 s | 原样 |
| 0.5× | 482 | 20.08 s | 每帧重复 2 次 |

**注意**：慢放是重复帧，不是补帧。动作不会变细腻，只是每张画面停留更久；
需要光流插值补帧（`minterpolate`）属于另一套方案，尚未实现。

## 项目结构

```
matte_studio/
├── naming.py         帧文件名、目录名清洗（纯标准库，可单测）
├── planner.py        调速与抽帧计划（纯函数，可单测）
├── config.py         配置解析，无硬编码路径
├── models.py         模型下载（HuggingFace / 镜像）
├── frames.py         视频探测与抽帧（OpenCV）
├── matte.py          批量抠图：断点续跑、重复帧复用、失败隔离
├── backends/
│   ├── base.py       后端接口与公用预处理 / 后处理
│   ├── torch_backend.py   transformers + BiRefNet 原始权重
│   └── onnx_backend.py    onnxruntime 会话
├── jobs.py           任务对象与任务表，支持从磁盘恢复
├── runner.py         串联抽帧 → 抠图 → 落盘
├── server.py         本地 HTTP 服务 + SSE 进度推送
├── cli.py            命令行入口
└── web/index.html    单文件界面
```

抠图逻辑与推理运行时是分开的：`matte.py` 只管「把一整个目录的帧可靠地处理完」，
具体推理由 `backends/` 里实现。加新模型（比如更轻的 u2net、可商用的 rembg）只需要再写一个后端。

## 开发

测试不依赖模型、不依赖 GPU，几十秒跑完：

```bash
pip install -e .
pip install numpy Pillow pytest
pytest
```

CI 会在 Python 3.10 / 3.11 / 3.12 上跑同一套测试（不装 torch、不装 OpenCV）。

## 配置项

所有配置都可用环境变量覆盖，代码里没有硬编码路径：

| 变量 | 说明 | 默认 |
|---|---|---|
| `MATTE_MODEL_DIR` | 模型目录 | 自动查找 `./RMBG-2.0`、`./RMBG-2.0-onnx`、`~/.matte-studio/models/` |
| `MATTE_BACKEND` | 强制后端 `auto` / `onnx` / `torch` | `auto` |
| `MATTE_ORT_PROVIDER` | ONNX 执行器 `cpu` / `cuda` / `auto` | `cpu` |
| `MATTE_ONNX_SIGMOID` | 输出是否已含 sigmoid：`auto` / `always` / `never` | `auto` |
| `MATTE_JOBS_DIR` | 任务输出根目录 | `./jobs` |
| `MATTE_HOST` | 监听地址 | `127.0.0.1` |
| `MATTE_PORT` | 端口 | `7860`（占用则自动顺延） |
| `MATTE_HOME` | 用户数据目录 | `~/.matte-studio` |
| `HF_ENDPOINT` | 模型下载源 | `https://huggingface.co` |

## 路线图

- [x] 抽帧 + 调速 + 抠图全链路
- [x] 可视化预览、断点续跑、自定义命名
- [x] 任务持久化，重启后可回看
- [x] ONNX Runtime 后端（免 torch，安装体积降到 100MB 级）
- [x] 模型自动下载 + 国内镜像
- [ ] 可商用替代模型：`rembg`（u2net / isnet）+ 轻量 CPU 快速档
- [ ] 导出格式：WebP、APNG、带 alpha 的 WebM、精灵图集
- [ ] 多文件批量队列
- [ ] 时序一致性优化（减少逐帧独立的边缘抖动）

## 许可

代码采用 [MIT](LICENSE)。模型权重有独立许可，详见上文与 LICENSE 末尾说明。
