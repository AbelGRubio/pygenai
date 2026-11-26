"""Tests for the agent FastAPI application loader."""

import json

from starlette.applications import Starlette

from pygenai.agent import loader


def test_load_config_missing_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    graphs = loader.load_config()

    assert graphs == {}


def test_load_config_registers_graphs(monkeypatch, tmp_path) -> None:
    config = {"graphs": {"agent": "./graph.py:graph"}}
    monkeypatch.chdir(tmp_path)
    (tmp_path / "langgraph.json").write_text(json.dumps(config), encoding="utf-8")

    graphs = loader.load_config()

    assert graphs == config["graphs"]


def test_build_app_adds_middleware_when_starlette(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_config", lambda: {})

    app = Starlette()
    monkeypatch.setattr(loader, "langgraph_app", app)

    factory = loader.PyGenaiAgent()
    result = factory.build_app()

    assert result is app


def test_build_app_returns_none_for_other_app(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_config", lambda: {})
    monkeypatch.setattr(loader, "langgraph_app", object())

    factory = loader.PyGenaiAgent()

    assert factory.build_app() is None
