"""抽帧规划：把「速度 + 抽帧模式」换算成「每个输出帧取源视频的第几帧」。

这是整个项目最容易算错、也最值得单测的部分，因此单独抽成纯函数模块，
不导入 OpenCV / torch。

规则说明
--------
调速采用时间重采样，不生成中间视频、不重新编码：

    第 i 个输出帧  ->  源视频第 int(i * speed) 帧

用向下取整而不是 round：Python 的 round 是「银行家舍入」（round(0.5)=0、
round(1.5)=2），会让慢放时的重复节奏变成 0,0,1,2,2,3…，帧间顿挫明显；
向下取整才能得到严格的 0,0,1,1,2,2…
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SPEED = 0.25
MAX_SPEED = 4.0


@dataclass(frozen=True)
class Plan:
    """一次抽帧的完整计划。"""

    source_total: int
    speed: float
    mode: str
    source_indices: tuple[int, ...]   # 长度即输出帧数，元素为源帧号（可能重复）

    @property
    def out_count(self) -> int:
        return len(self.source_indices)

    def duplicates(self) -> dict[int, list[int]]:
        """源帧号 -> 需要写出它的输出序号列表。慢放时一对多。"""
        mapping: dict[int, list[int]] = {}
        for out_index, src_index in enumerate(self.source_indices):
            mapping.setdefault(src_index, []).append(out_index)
        return mapping


def clamp_speed(speed: float) -> float:
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        return 1.0
    if speed <= 0:
        return 1.0
    return min(MAX_SPEED, max(MIN_SPEED, speed))


def build_plan(total: int, speed: float = 1.0, mode: str = "all",
               sample_count: int = 60) -> Plan:
    """根据源视频帧数生成抽帧计划。

    mode="all"    逐帧导出（按速度重采样）
    mode="sample" 在重采样后的序列上再均匀取 sample_count 帧
    """
    if total <= 0:
        return Plan(source_total=0, speed=1.0, mode=mode, source_indices=())

    speed = clamp_speed(speed)
    out_count = max(1, int(total / speed))          # 向下取整，与 source_indices 保持一致

    if mode == "sample":
        k = max(2, min(int(sample_count), out_count))
        positions = ([round(i * (out_count - 1) / (k - 1)) for i in range(k)]
                     if out_count > 1 else [0])
    else:
        positions = list(range(out_count))

    indices = tuple(min(total - 1, int(pos * speed)) for pos in positions)
    return Plan(source_total=total, speed=speed, mode=mode, source_indices=indices)
