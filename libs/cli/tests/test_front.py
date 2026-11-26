"""Tests for the PyGenAI Streamlit front-end command.

====================================================================================================
Name:         libs/cli/tests/test_front.py
Description:  Unit tests for front-end port validation and Streamlit process launch.
Project:      PyGenAI
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from pygenai.cli.cli import app


runner = CliRunner()


def test_front_launches_streamlit_with_default_options() -> None:
    """Verify the front command launches Streamlit with default address settings."""
    probe = Mock()
    app_resource = Mock()
    app_resource.__fspath__ = Mock(return_value="")

    with (
        patch("pygenai.cli.commands.front.socket.create_server", return_value=probe),
        patch("pygenai.cli.commands.front.resources.files", return_value=app_resource),
        patch("pygenai.cli.commands.front.resources.as_file") as as_file,
        patch("pygenai.cli.commands.front.subprocess.run") as subprocess_run,
    ):
        as_file.return_value.__enter__.return_value = Path("/tmp/app.py")
        result = runner.invoke(app, ["front"])

    assert result.exit_code == 0
    probe.close.assert_called_once_with()
    subprocess_run.assert_called_once()
    command = subprocess_run.call_args.args[0]
    assert command[:4] == [command[0], "-m", "streamlit", "run"]
    assert "--server.address" in command
    assert "localhost" in command
    assert "--server.port" in command
    assert "8501" in command
    assert "--server.headless" in command
    assert "false" in command
    assert subprocess_run.call_args.kwargs == {"check": True}


def test_front_passes_custom_host_port_and_no_browser() -> None:
    """Verify custom network and browser options are forwarded to Streamlit."""
    probe = Mock()

    with (
        patch("pygenai.cli.commands.front.socket.create_server", return_value=probe),
        patch("pygenai.cli.commands.front.resources.files") as resources_files,
        patch("pygenai.cli.commands.front.resources.as_file") as as_file,
        patch("pygenai.cli.commands.front.subprocess.run") as subprocess_run,
    ):
        resources_files.return_value.joinpath.return_value = Mock()
        as_file.return_value.__enter__.return_value = Path("/tmp/custom-app.py")
        result = runner.invoke(
            app,
            ["front", "--host", "127.0.0.1", "--port", "8600", "--no-browser"],
        )

    assert result.exit_code == 0
    subprocess_run.assert_called_once()
    command = subprocess_run.call_args.args[0]
    assert "127.0.0.1" in command
    assert "8600" in command
    assert "true" in command


def test_front_rejects_port_already_in_use() -> None:
    """Verify a port already in use produces a user-facing Typer parameter error."""
    with patch("pygenai.cli.commands.front.socket.create_server", side_effect=OSError("busy")):
        result = runner.invoke(app, ["front", "--port", "8501"])

    assert result.exit_code == 2
    assert "Port 8501 on localhost is already in use." in result.output


def test_front_validates_port_range() -> None:
    """Verify Typer rejects ports outside the valid range."""
    result = runner.invoke(app, ["front", "--port", "70000"])

    assert result.exit_code == 2
    assert "invalid" in result.output.lower() or "range" in result.output.lower()
