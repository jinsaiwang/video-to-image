"""配置解析：模型目录查找顺序与后端判定。"""

from __future__ import annotations

from pathlib import Path

from matte_studio import config as cfg


def _torch_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"w")
    return root


def _onnx_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.onnx").write_bytes(b"o")
    return root


def test_candidates_put_explicit_first(monkeypatch, tmp_path):
    monkeypatch.setenv("MATTE_MODEL_DIR", str(tmp_path / "from_env"))
    got = cfg.model_candidates(tmp_path / "explicit")
    assert got[0] == tmp_path / "explicit"
    assert got[1] == tmp_path / "from_env"


def test_resolve_prefers_torch_dir_when_both_exist(monkeypatch, tmp_path):
    (monkeypatch.chdir(tmp_path))
    _onnx_dir(tmp_path / "RMBG-2.0-onnx")
    _torch_dir(tmp_path / "RMBG-2.0")
    assert cfg.resolve_model_dir().name == "RMBG-2.0"


def test_resolve_falls_back_to_onnx(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _onnx_dir(tmp_path / "RMBG-2.0-onnx")
    assert cfg.resolve_model_dir().name == "RMBG-2.0-onnx"


def test_resolve_returns_download_target_when_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MATTE_MODEL_DIR", raising=False)
    monkeypatch.setenv("MATTE_HOME", str(tmp_path / "home"))
    assert cfg.resolve_model_dir().name == cfg.DOWNLOAD_DIR_NAME


def test_explicit_dir_wins(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _torch_dir(tmp_path / "RMBG-2.0")
    other = _onnx_dir(tmp_path / "other")
    assert cfg.resolve_model_dir(other) == other


def test_config_load_creates_jobs_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MATTE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MATTE_MODEL_DIR", str(_onnx_dir(tmp_path / "m")))
    c = cfg.Config.load(jobs_dir=tmp_path / "jobs")
    assert c.jobs_dir.is_dir()
    assert c.model_ready is True
    assert c.detected_backend == "onnx"
    assert "onnx" in c.describe()


def test_backend_override(tmp_path):
    _onnx_dir(tmp_path / "m")
    c = cfg.Config.load(model_dir=tmp_path / "m", jobs_dir=tmp_path / "j", backend="torch")
    assert c.active_backend == "torch"


def test_require_model_message_mentions_download(monkeypatch, tmp_path):
    monkeypatch.setenv("MATTE_HOME", str(tmp_path / "home"))
    c = cfg.Config(model_dir=tmp_path / "empty")
    assert c.model_ready is False
    try:
        c.require_model()
    except FileNotFoundError as exc:
        assert "matte download" in str(exc)
    else:
        raise AssertionError("缺模型时应当抛错")
