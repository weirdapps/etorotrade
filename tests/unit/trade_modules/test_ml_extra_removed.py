"""finBERT's packages live outside Poetry (issue #312).

torch's nvidia-* markers hung `poetry lock` in a marker-CNF blow-up, so the
`ml` extra was removed from pyproject.toml and its packages moved to the
unlocked requirements-ml.txt. These tests keep it that way, and check that the
sentiment analyzer degrades quietly when those packages are absent.
"""

import builtins
import logging
from pathlib import Path

import pytest
import tomllib

from trade_modules import sentiment_analyzer

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_PACKAGES = ("torch", "transformers", "scikit-learn")


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def test_ml_extra_is_gone_from_pyproject():
    project = _pyproject()["project"]
    extras = project.get("optional-dependencies", {})
    assert "ml" not in extras
    declared = list(project["dependencies"])
    for group in extras.values():
        declared.extend(group)
    for pkg in ML_PACKAGES:
        assert not any(d.lower().startswith(pkg) for d in declared), pkg


def test_poetry_lock_holds_no_ml_packages():
    lock = (REPO_ROOT / "poetry.lock").read_text()
    for pkg in ML_PACKAGES:
        assert f'name = "{pkg}"' not in lock, pkg
    assert "\nml = [" not in lock


def test_requirements_ml_lists_the_finbert_packages():
    lines = (REPO_ROOT / "requirements-ml.txt").read_text().splitlines()
    reqs = [ln.split(">=")[0].strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    assert sorted(reqs) == sorted(ML_PACKAGES)


@pytest.fixture
def fresh_model_state(monkeypatch):
    monkeypatch.setattr(sentiment_analyzer, "_model", None)
    monkeypatch.setattr(sentiment_analyzer, "_tokenizer", None)
    monkeypatch.setattr(sentiment_analyzer, "_model_load_attempted", False)
    monkeypatch.setattr(sentiment_analyzer, "_sentiment_cache", {})
    monkeypatch.setattr(sentiment_analyzer, "_cache_loaded", True)


@pytest.fixture
def no_torch(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "transformers"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_load_model_without_torch_returns_false_and_points_at_requirements_ml(
    fresh_model_state, no_torch, caplog
):
    with caplog.at_level(logging.INFO, logger=sentiment_analyzer.__name__):
        assert sentiment_analyzer._load_model() is False

    assert sentiment_analyzer._model is None
    msgs = [r for r in caplog.records if r.name == sentiment_analyzer.__name__]
    assert any("pip install -r requirements-ml.txt" in r.getMessage() for r in msgs)
    # Absent by default, so no warning noise.
    assert not any(r.levelno >= logging.WARNING for r in msgs)


def test_get_ticker_sentiment_without_torch_returns_none(fresh_model_state, no_torch):
    assert sentiment_analyzer.get_ticker_sentiment("AAPL") is None
