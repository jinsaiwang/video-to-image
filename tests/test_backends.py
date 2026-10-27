"""后端选择、预处理与后处理。

这些测试不加载任何模型、不碰 GPU，所以在 CI 里几十秒就能跑完——
但能挡住最容易出错的几处：目录识别、sigmoid 是否重复施加、蒙版尺寸。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from matte_studio import models
from matte_studio.backends import Backend, attach_alpha, detect, onnx_file
from matte_studio.backends.base import normalize_to_blob


# ---------- 目录识别 ----------
def _touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_detect_onnx_dir(tmp_path):
    _touch(tmp_path / "model.onnx", b"onnx")
    assert detect(tmp_path) == "onnx"


def test_detect_torch_dir(tmp_path):
    _touch(tmp_path / "config.json", b"{}")
    _touch(tmp_path / "model.safetensors", b"w")
    assert detect(tmp_path) == "torch"


def test_detect_torch_without_weight_is_none(tmp_path):
    _touch(tmp_path / "config.json", b"{}")
    assert detect(tmp_path) is None


def test_detect_empty_and_missing(tmp_path):
    assert detect(tmp_path) is None
    assert detect(tmp_path / "nope") is None
    assert detect(None) is None


def test_onnx_preferred_over_torch_when_both_present(tmp_path):
    _touch(tmp_path / "model.onnx", b"onnx")
    _touch(tmp_path / "config.json", b"{}")
    _touch(tmp_path / "model.safetensors", b"w")
    assert detect(tmp_path) == "onnx"


def test_onnx_file_prefers_canonical_name(tmp_path):
    _touch(tmp_path / "aaa.onnx", b"1")
    canonical = _touch(tmp_path / "model.onnx", b"2")
    assert onnx_file(tmp_path) == canonical


def test_onnx_file_accepts_direct_path(tmp_path):
    f = _touch(tmp_path / "custom.onnx", b"1")
    assert onnx_file(f) == f
    assert onnx_file(tmp_path / "cfg.json") is None


# ---------- 后处理：sigmoid 不能加两次 ----------
def test_postprocess_applies_sigmoid_to_logits():
    logits = np.linspace(-8, 8, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
    arr = np.asarray(Backend.postprocess(logits, (16, 16)))
    # sigmoid(-8)≈0 → 0，sigmoid(8)≈1 → 255，中间单调过渡
    assert arr.min() < 5 and arr.max() > 250
    # 输入以 0 为中心对称，sigmoid 后均值应落在 128 附近
    assert abs(arr.mean() - 128) < 3


def test_postprocess_auto_keeps_probabilities_untouched():
    # RMBG-2.0 的 ONNX 导出自带 sigmoid，实测输出就落在 0~1，不能再归一
    probs = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
    arr = np.asarray(Backend.postprocess(probs, (16, 16)))
    assert arr.min() == 0 and arr.max() == 255
    assert arr[0, 0] == 0


def test_postprocess_forced_modes():
    probs = np.zeros((1, 4, 4), dtype=np.float32)
    probs[0, 0, 0] = 1.0
    kept = np.asarray(Backend.postprocess(probs, (4, 4), sigmoid="never"))
    assert kept[0, 0] == 255 and kept[1, 1] == 0
    forced = np.asarray(Backend.postprocess(probs, (4, 4), sigmoid="always"))
    assert forced[1, 1] == 128                          # sigmoid(0) → 0.5


def test_postprocess_negative_logits_become_transparent():
    logits = np.full((1, 8, 8), -10.0, dtype=np.float32)
    mask = np.asarray(Backend.postprocess(logits, (8, 8)))
    assert mask.max() <= 1


def test_postprocess_resizes_to_source():
    logits = np.random.rand(1, 32, 32).astype(np.float32)
    mask = Backend.postprocess(logits, (720, 960))
    assert mask.size == (720, 960)


def test_postprocess_rejects_bad_shape():
    with pytest.raises(ValueError):
        Backend.postprocess(np.zeros((3, 4, 5, 6), np.float32), (10, 10))


# ---------- 预处理：必须与 torch 后端一致 ----------
def test_normalize_shape_and_dtype():
    blob = normalize_to_blob(Image.new("RGB", (300, 200), "white"))
    assert blob.shape == (1, 3, 1024, 1024)
    assert blob.dtype == np.float32


def test_normalize_matches_imagenet_stats():
    blob = normalize_to_blob(Image.new("RGB", (64, 64), "white"))
    expected = (1.0 - 0.485) / 0.229                  # 白色像素的 R 通道
    assert abs(float(blob[0, 0, 0, 0]) - expected) < 1e-5


def test_normalize_accepts_gray_and_rgba():
    for mode in ("L", "RGBA", "P"):
        blob = normalize_to_blob(Image.new(mode, (32, 32)))
        assert blob.shape[0] == 1 and blob.shape[1] == 3


# ---------- 贴 alpha ----------
def test_attach_alpha_produces_rgba():
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    mask = Image.new("L", (8, 8), 128)
    out = attach_alpha(img, mask)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0)) == (255, 0, 0, 128)


# ---------- 下载地址拼接 ----------
def test_resolve_endpoint_aliases():
    assert models.resolve_endpoint("cn") == "https://hf-mirror.com"
    assert models.resolve_endpoint("https://example.com/") == "https://example.com"


def test_file_url_shape():
    url = models.file_url("a/b", "onnx/model.onnx", "https://hf-mirror.com")
    assert url == "https://hf-mirror.com/a/b/resolve/main/onnx/model.onnx"


def test_unknown_variant_rejected(tmp_path):
    with pytest.raises(ValueError):
        models.ensure_onnx_model(tmp_path, variant="q2")


def test_existing_model_is_reused(tmp_path, monkeypatch):
    existing = _touch(tmp_path / "model.onnx", b"already")
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")   # 一旦发起下载就会失败
    assert models.ensure_onnx_model(tmp_path) == existing
