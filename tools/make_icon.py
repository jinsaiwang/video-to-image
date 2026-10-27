"""生成 Matte Studio 的应用图标。

设计：靛蓝渐变方块 + 白色画框（内含棋盘格＝透明）+ 橙色播放徽标。
含义直白——「视频 → 透明帧」。

所有尺寸都在 4 倍画布上绘制再降采样，保证 16×16 也不糊。
用法：python tools/make_icon.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets"
SIZES = [16, 24, 32, 48, 64, 128, 256]
SUPERSAMPLE = 4

TILE_TOP = (74, 86, 232)        # 靛蓝
TILE_BOTTOM = (128, 76, 255)    # 紫罗兰
CHECK_A = (255, 255, 255)
CHECK_B = (212, 219, 240)
ACCENT = (255, 122, 69)         # 橙


def _gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    """左上到右下的对角渐变。"""
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    t = np.clip(x * 0.35 + y * 0.65, 0.0, 1.0)[..., None]
    c1 = np.array(top, dtype=np.float32)
    c2 = np.array(bottom, dtype=np.float32)
    arr = c1 + (c2 - c1) * t
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def _checkerboard(size: int, box: tuple[int, int, int, int], cells: int) -> Image.Image:
    """在整幅画布上画一块棋盘格，代表「透明」。"""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cw, ch = w / cells, h / cells
    for row in range(cells):
        for col in range(cells):
            fill = CHECK_A if (row + col) % 2 == 0 else CHECK_B
            d.rectangle([x0 + col * cw, y0 + row * ch,
                         x0 + (col + 1) * cw, y0 + (row + 1) * ch], fill=fill)
    return layer


def _cells_for(size: int) -> int:
    """小图标用大格子：16px 上 4×4 的棋盘会糊成一片灰。"""
    if size <= 20:
        return 2
    if size <= 40:
        return 3
    return 4


def render(size: int) -> Image.Image:
    """按目标尺寸渲染一张 RGBA 图标。"""
    s = size * SUPERSAMPLE
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # ---- 底板：圆角方块 + 渐变 ----
    radius = int(s * 0.235)
    tile_mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(tile_mask).rounded_rectangle(
        [0, 0, s - 1, s - 1], radius=radius, fill=255)
    canvas.paste(_gradient(s, TILE_TOP, TILE_BOTTOM), (0, 0), tile_mask)

    # ---- 白色画框（面板）----
    pad = int(s * 0.155)
    panel = (pad, pad, s - pad, s - pad)
    panel_radius = int(s * 0.070)

    frame = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    fd = ImageDraw.Draw(frame)
    fd.rounded_rectangle(panel, radius=panel_radius, fill=(255, 255, 255, 255))

    # 面板内缩一圈再贴棋盘格，形成白色描边的观感
    inset = max(1, int(s * 0.035))
    inner = (panel[0] + inset, panel[1] + inset,
             panel[2] - inset, panel[3] - inset)
    board = _checkerboard(s, inner, cells=_cells_for(size))
    board_mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(board_mask).rounded_rectangle(
        inner, radius=max(1, panel_radius - inset), fill=255)
    frame.paste(board, (0, 0), board_mask)
    canvas.alpha_composite(frame)

    # ---- 橙色播放徽标（右下角）----
    badge_d = int(s * 0.345)
    bx1, by1 = int(s * 0.945), int(s * 0.945)
    bx0, by0 = bx1 - badge_d, by1 - badge_d

    ring = max(1, int(s * 0.018))
    ImageDraw.Draw(canvas).ellipse(
        [bx0 - ring, by0 - ring, bx1 + ring, by1 + ring], fill=(255, 255, 255, 255))
    ImageDraw.Draw(canvas).ellipse([bx0, by0, bx1, by1], fill=ACCENT + (255,))

    # 播放三角形，光学居中（略向右偏）
    cx, cy = (bx0 + bx1) / 2 + badge_d * 0.05, (by0 + by1) / 2
    r = badge_d * 0.26
    tri = [(cx - r * 0.75, cy - r), (cx - r * 0.75, cy + r), (cx + r * 1.05, cy)]
    ImageDraw.Draw(canvas).polygon(tri, fill=(255, 255, 255, 255))

    return canvas.resize((size, size), Image.LANCZOS)


def save_ico(images: list[Image.Image], path: Path) -> None:
    """手写 ICO 容器：每个尺寸存各自的 PNG，避免 PIL 统一缩放糊掉小图标。"""
    blobs = []
    for im in images:
        import io
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        blobs.append(buf.getvalue())

    n = len(images)
    header = struct.pack("<HHH", 0, 1, n)
    offset = 6 + 16 * n
    entries, data = b"", b""
    for im, blob in zip(images, blobs):
        w = 0 if im.width >= 256 else im.width
        h = 0 if im.height >= 256 else im.height
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
        data += blob
    path.write_bytes(header + entries + data)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    images = [render(s) for s in SIZES]
    ico = OUT_DIR / "matte.ico"
    save_ico(images, ico)

    # 预览图：深/浅两种背景上的实际尺寸，方便肉眼检查
    gap, margin = 18, 24
    total_w = margin * 2 + sum(im.width for im in images) + gap * (len(images) - 1)
    band_h = max(im.height for im in images) + 36
    preview = Image.new("RGB", (total_w, band_h * 2), (245, 246, 250))
    x = margin
    for im in images:
        preview.paste(im, (x, (band_h - im.height) // 2), im)
        x += im.width + gap
    dark = Image.new("RGB", (total_w, band_h), (32, 36, 48))
    x2 = margin
    for im in images:
        dark.paste(im, (x2, (band_h - im.height) // 2), im)
        x2 += im.width + gap
    preview.paste(dark, (0, band_h))
    preview.save(OUT_DIR / "icon_preview.png")

    print(f"已生成 {ico}（{ico.stat().st_size / 1024:.1f} KB，{len(images)} 个尺寸）")
    for s, im in zip(SIZES, images):
        print(f"  {s:>3}×{s:<3} RGBA")


if __name__ == "__main__":
    main()
