"""Tests for ui.data.settings round-trip + atomic write."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))

from ui.data import settings as S


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DETECTOR_HOME", str(tmp_path))
    monkeypatch.delenv("DETECTOR_ARTIFACTS", raising=False)
    yield tmp_path


def test_defaults_when_no_file(isolated_home):
    out = S.load_settings()
    assert out == S.DEFAULTS


def test_round_trip(isolated_home):
    settings = dict(S.DEFAULTS)
    settings["model_version"] = "v0.2.0"
    settings["auto_run_on_open"] = False
    S.save_settings(settings)
    out = S.load_settings()
    assert out["model_version"] == "v0.2.0"
    assert out["auto_run_on_open"] is False


def test_unknown_keys_are_preserved_through_load(isolated_home):
    """If a future version adds keys, older load_settings should still
    return the known defaults — unknown keys aren't injected."""
    S.save_settings({**S.DEFAULTS, "future_key": 123})
    out = S.load_settings()
    assert "future_key" not in out
    # Original keys still present
    for k in S.DEFAULTS:
        assert k in out


def test_atomic_no_tmp_leftover(isolated_home):
    S.save_settings(dict(S.DEFAULTS))
    p = isolated_home / "pyqt_settings.json"
    assert p.exists()
    assert not p.with_suffix(".tmp").exists()


def test_update_setting_shortcut(isolated_home):
    S.update_setting("auto_run_on_open", False)
    out = S.load_settings()
    assert out["auto_run_on_open"] is False
