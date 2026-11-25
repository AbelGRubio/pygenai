"""Tests for the PyGenAI CLI launcher.

====================================================================================================
Name:         libs/cli/tests/test_launcher.py
Description:  Unit tests for loader discovery, launcher execution, and CLI callbacks.
Project:      PyGenAI
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from types import ModuleType
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from pygenai.cli.cli import app
from pygenai.cli.launcher import PyGenaiLauncher, load_agent, run_cli


runner = CliRunner()


def test_launcher_build_app_delegates_to_loader() -> None:
    """Verify that build_app delegates application construction to the loader."""
    app_instance = object()
    loader = Mock()
    loader.build_app.return_value = app_instance

    launcher = PyGenaiLauncher(loader)

    assert launcher.build_app() is app_instance
    loader.build_app.assert_called_once_with()


def test_launcher_run_passes_options_to_uvicorn() -> None:
    """Verify that launcher.run configures Uvicorn and the built application."""
    built_app = object()
    loader = Mock()
    loader.build_app.return_value = built_app

    with patch("pygenai.cli.launcher.uvicorn.run") as uvicorn_run:
        PyGenaiLauncher(loader).run(host="0.0.0.0", port=9000, log_level="debug", reload=True)

    uvicorn_run.assert_called_once_with(
        app=built_app,
        host="0.0.0.0",
        port=9000,
        log_level="debug",
        log_config=None,
        reload=True,
    )


def test_load_agent_uses_environment_module_and_class() -> None:
    """Verify that load_agent imports the configured module and loader class."""
    module = ModuleType("custom_agent")
    loader_class = Mock()
    module.PyGenaiAgent = loader_class

    with (
        patch.dict("os.environ", {"AGENT_MODULE": "custom_agent"}),
        patch("pygenai.cli.launcher.importlib.import_module", return_value=module),
    ):
        result = load_agent()

    assert result is loader_class


def test_load_agent_defaults_to_missing_project_loader() -> None:
    """Verify the default module lookup returns no class when PyGenaiAgent is absent."""
    module = ModuleType("pygenai.agent.loader")

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pygenai.cli.launcher.importlib.import_module", return_value=module),
    ):
        assert load_agent() is None


def test_run_cli_starts_launcher_with_options() -> None:
    """Verify run_cli instantiates and runs the configured loader."""
    loader_class = Mock()
    launcher_instance = Mock()

    with (
        patch("pygenai.cli.launcher.load_agent", return_value=loader_class),
        patch("pygenai.cli.launcher.PyGenaiLauncher", return_value=launcher_instance),
    ):
        run_cli(host="127.0.0.1", port=8124, log_level="warning", reload=True)

    loader_class.assert_called_once_with()
    launcher_instance.run.assert_called_once_with(
        host="127.0.0.1",
        port=8124,
        log_level="warning",
        reload=True,
    )


def test_run_cli_fails_when_loader_cannot_be_loaded() -> None:
    """Verify the CLI exits cleanly when no loader class is available."""
    with patch("pygenai.cli.launcher.load_agent", return_value=None):
        result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "Failed to load the agent." in result.output


def test_cli_run_forwards_command_options() -> None:
    """Verify the run command forwards all Typer options to the launcher."""
    loader_class = Mock()
    launcher_instance = Mock()

    with (
        patch("pygenai.cli.launcher.load_agent", return_value=loader_class),
        patch("pygenai.cli.launcher.PyGenaiLauncher", return_value=launcher_instance),
    ):
        result = runner.invoke(
            app,
            ["run", "--host", "127.0.0.2", "--port", "9001", "--log-level", "debug", "--reload"],
        )

    assert result.exit_code == 0
    launcher_instance.run.assert_called_once_with(
        host="127.0.0.2",
        port=9001,
        log_level="debug",
        reload=True,
    )


def test_cli_without_subcommand_shows_help() -> None:
    """Verify invoking the base CLI without a subcommand shows the welcome and help."""
    result = runner.invoke(app)

    assert result.exit_code == 0
    assert "Welcome to PyGenAI CLI Launcher!" in result.stdout
    assert "Usage:" in result.stdout
