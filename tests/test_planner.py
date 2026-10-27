"""抽帧计划：速度、抽帧模式、帧数换算。

这是全项目最容易算错的地方，重点覆盖：
1. 快放按比例丢帧，映射必须是 0,2,4…（不是近似）
2. 慢放必须严格重复 0,0,1,1,2,2…（用 round 会得到 0,0,1,2,2,3 的错误节奏）
3. 输出帧数一律向下取整，和映射公式保持一致
"""

import pytest

from matte_studio.planner import build_plan, clamp_speed

TOTAL = 241        # 10.04 秒 @ 24fps 的实测源片


def test_speed_one_keeps_all_frames():
    plan = build_plan(TOTAL, speed=1.0)
    assert plan.out_count == 241
    assert plan.source_indices[:3] == (0, 1, 2)
    assert plan.source_indices[-1] == 240


@pytest.mark.parametrize("speed,expected", [
    (0.5, 482),
    (1.5, 160),
    (2.0, 120),
    (3.0, 80),
    (4.0, 60),
])
def test_output_count_is_floor_of_total_over_speed(speed, expected):
    assert build_plan(TOTAL, speed=speed).out_count == expected


def test_fast_motion_drops_every_other_frame():
    plan = build_plan(TOTAL, speed=2.0)
    assert plan.source_indices[:4] == (0, 2, 4, 6)


def test_slow_motion_duplicates_each_frame_twice():
    """向下取整才能得到严格的 0,0,1,1,2,2；round 会给出 0,0,1,2,2,3。"""
    plan = build_plan(TOTAL, speed=0.5)
    assert plan.source_indices[:6] == (0, 0, 1, 1, 2, 2)


def test_duplicates_mapping():
    plan = build_plan(10, speed=0.5)
    mapping = plan.duplicates()
    assert mapping[0] == [0, 1]
    assert mapping[1] == [2, 3]


def test_no_index_exceeds_source_range():
    for speed in (0.25, 0.5, 1.0, 1.7, 2.0, 4.0):
        plan = build_plan(TOTAL, speed=speed)
        assert max(plan.source_indices) <= TOTAL - 1
        assert min(plan.source_indices) >= 0


def test_sample_mode_respects_requested_count():
    plan = build_plan(TOTAL, speed=1.0, mode="sample", sample_count=24)
    assert plan.out_count == 24
    assert plan.source_indices[0] == 0
    assert plan.source_indices[-1] == 240


def test_sample_mode_cannot_exceed_available_frames():
    plan = build_plan(10, speed=1.0, mode="sample", sample_count=60)
    assert plan.out_count <= 10


def test_sample_mode_combines_with_speed():
    plan = build_plan(TOTAL, speed=2.0, mode="sample", sample_count=12)
    assert plan.out_count == 12
    # 抽的是重采样后序列里的位置，仍要落在合法源帧号上
    assert max(plan.source_indices) <= TOTAL - 1


def test_speed_is_clamped_to_supported_range():
    assert clamp_speed(10) == 4.0
    assert clamp_speed(0.01) == 0.25
    assert clamp_speed(0) == 1.0
    assert clamp_speed(-2) == 1.0
    assert clamp_speed("bad") == 1.0


def test_empty_video_returns_empty_plan():
    plan = build_plan(0)
    assert plan.out_count == 0
    assert plan.source_indices == ()
