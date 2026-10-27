"""名称处理：帧文件名规则、目录名清洗。

这个模块刻意只依赖标准库，方便在没有 torch / OpenCV 的环境里直接单元测试。
"""

from __future__ import annotations

import re

# Windows 文件名非法字符（macOS / Linux 上多限制一些也无害）
_ILLEGAL = r'[\\/:*?"<>|\r\n\t]'


def frame_name(index: int, prefix: str = "frame_", start: int = 0, digits: int = 6) -> str:
    """生成第 index 帧（从 0 起）的文件名。

    >>> frame_name(0)
    'frame_000000.png'
    >>> frame_name(0, prefix="pet_", start=1, digits=4)
    'pet_0001.png'
    """
    return f"{prefix}{start + index:0{digits}d}.png"


def safe_name(text: str, fallback: str = "job") -> str:
    """把用户输入变成合法的文件/目录名。

    非法字符统一替换成下划线，并去掉首尾的点和空格（Windows 不允许
    目录名以点或空格结尾）。空结果返回 fallback。
    """
    text = re.sub(_ILLEGAL, "_", text or "").strip(" .")
    text = re.sub(r"\s+", " ", text)[:60].strip()
    return text or fallback


def clean_prefix(text: str, fallback: str = "frame_") -> str:
    """清洗帧名前缀，与 safe_name 分开是为了保留末尾下划线等合法字符。"""
    text = re.sub(_ILLEGAL, "_", text or "").strip(" .")
    return text[:24] or fallback
