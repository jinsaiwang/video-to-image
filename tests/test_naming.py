"""帧文件名与目录名规则。

只依赖标准库，不需要 torch / OpenCV，CI 里几秒就能跑完。
"""

from matte_studio.naming import clean_prefix, frame_name, safe_name


def test_frame_name_default():
    assert frame_name(0) == "frame_000000.png"
    assert frame_name(240) == "frame_000240.png"


def test_frame_name_custom_prefix_start_digits():
    assert frame_name(0, prefix="pet_", start=1, digits=4) == "pet_0001.png"
    assert frame_name(9, prefix="角色_", start=1, digits=3) == "角色_010.png"


def test_frame_name_digits_expand_when_overflow():
    """位数不够时不能截断，宁可多一位也不能重名。"""
    assert frame_name(1234, prefix="f", start=0, digits=3) == "f1234.png"


def test_safe_name_replaces_illegal_chars():
    assert safe_name("a/b:c*") == "a_b_c_"
    assert safe_name('x<y>z|w') == "x_y_z_w"


def test_safe_name_strips_trailing_dots_and_spaces():
    assert safe_name("  桌宠  ") == "桌宠"
    assert safe_name("name...") == "name"


def test_safe_name_fallback():
    assert safe_name("") == "job"
    assert safe_name(None) == "job"
    assert safe_name("   ", fallback="abc") == "abc"


def test_safe_name_truncates_long_input():
    assert len(safe_name("长" * 200)) <= 60


def test_clean_prefix_keeps_underscore():
    assert clean_prefix("frame_") == "frame_"
    assert clean_prefix("") == "frame_"
    assert clean_prefix("a/b") == "a_b"
