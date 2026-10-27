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
- **零前端依赖**：后端只用 Python 标准库 `http.server` + SSE，不需要 Node 或 Web 框架
- **断点续跑**：中断后重新启动会跳过已完成的帧
- **重复帧复用**：慢放产生的重复画面不会重复跑模型，0.5× 的耗时与 1× 基本一致
- **结果可回看**：任务落盘为档案，重启服务后仍能在页面里打开历史结果
- **CLI 可用**：`matte run in.mp4 --speed 2` 一行跑完，适合批处理

## 安装

```bash
git clone https://github.com/jinsaiwang/matte-studio.git
cd matte-studio
pip install -e ".[gpu]"      # 有 NVIDIA 显卡
# 或者
pip install -e ".[gpu]"      # 无显卡也能用，torch 会退回 CPU，只是慢
```

> 基础依赖（`numpy` / `opencv-python` / `Pillow`）很轻；`torch` 那套有 2GB+，
> 所以拆成可选依赖。不装 `[gpu]` 也能启动界面、抽帧、调速，只是无法抠图。
> 计划在后续版本接入 ONNX Runtime，把安装体积降到 100MB 左右。

## 准备模型

本项目**不包含模型权重**（体积大，且许可证不同）。请自行下载 RMBG-2.0 放到以下任一位置：

```
./RMBG-2.0/                     ← 推荐，放仓库根目录
~/.matte-studio/models/RMBG-2.0/
```

或用参数指定：

```bash
matte serve --model-dir /path/to/RMBG-2.0
# 或
export MATTE_MODEL_DIR=/path/to/RMBG-2.0
```

模型目录里应包含 `config.json`、`model.safetensors`、`birefnet.py`、`BiRefNet_config.py`
（即 HuggingFace 仓库文件原样下载）。

> ⚠️ **许可证提醒**：RMBG-2.0 模型由 BRIA AI 发布，**仅限非商业使用**。本仓库代码是 MIT，
> 但模型不受 MIT 约束。商用请先阅读 [模型条款](https://huggingface.co/briaai/RMBG-2.0)，
> 或改用可商用的替代后端。

## 快速开始

### 图形界面

```bash
matte serve
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
├── naming.py     帧文件名、目录名清洗（纯标准库，可单测）
├── planner.py    调速与抽帧计划（纯函数，可单测）
├── config.py     配置解析，无硬编码路径
├── frames.py     视频探测与抽帧（OpenCV）
├── matte.py      抠图推理（torch 延迟导入）
├── jobs.py       任务对象与任务表，支持从磁盘恢复
├── runner.py     串联抽帧 → 抠图 → 落盘
├── server.py     本地 HTTP 服务 + SSE 进度推送
├── cli.py        命令行入口
└── web/index.html  单文件界面
```

## 开发

测试只覆盖不依赖模型的部分，因此**不需要显卡、也不需要下载模型**：

```bash
pip install -e .
pip install pytest
pytest
```

## 配置项

所有配置都可用环境变量覆盖，代码里没有硬编码路径：

| 变量 | 说明 | 默认 |
|---|---|---|
| `MATTE_MODEL_DIR` | 模型目录 | 自动查找 `./RMBG-2.0`、`~/.matte-studio/models/` |
| `MATTE_JOBS_DIR` | 任务输出根目录 | `./jobs` |
| `MATTE_HOST` | 监听地址 | `127.0.0.1` |
| `MATTE_PORT` | 端口 | `7860`（占用则自动顺延） |
| `MATTE_HOME` | 用户数据目录 | `~/.matte-studio` |

## 路线图

- [x] 抽帧 + 调速 + 抠图全链路
- [x] 可视化预览、断点续跑、自定义命名
- [x] 任务持久化，重启后可回看
- [ ] ONNX Runtime 后端（免 torch，安装体积降到 100MB 级）
- [ ] 可插拔模型后端：`rembg`（可商用）、`BiRefNet`、自定义
- [ ] 导出格式：WebP、APNG、带 alpha 的 WebM、精灵图集
- [ ] 多文件批量队列
- [ ] 时序一致性优化（减少逐帧独立的边缘抖动）

## 许可

代码采用 [MIT](LICENSE)。模型权重有独立许可，详见上文与 LICENSE 末尾说明。
