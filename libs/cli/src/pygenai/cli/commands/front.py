"""Launch the PyGenAI Streamlit front end.

====================================================================================================
Name:         src/pygenai/cli/commands/front.py
Description:  CLI command for launching the Streamlit front end
Project:      PyGenAI
Date:         2026-11-19
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

import socket
import subprocess  # noqa: S404
import sys
from importlib import resources

import typer

app = typer.Typer(name="front", help="Launch the Streamlit front end.")


@app.callback(invoke_without_command=True)
def launch(
    host: str = typer.Option("localhost", "--host", help="Network address for the Streamlit server."),
    port: int = typer.Option(8501, "--port", min=1, max=65535, help="Port for the Streamlit server."),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Open the front end in a browser."),
) -> None:
    """Launch the chat front end with Streamlit.

    Args:
        host: Network address for the Streamlit server.
        port: Port for the Streamlit server.
        browser: Whether Streamlit should open a browser automatically.
    """
    try:
        probe = socket.create_server((host, port), reuse_port=False)
    except OSError as exc:
        raise typer.BadParameter(f"Port {port} on {host} is already in use.") from exc
    else:
        probe.close()

    app_resource = resources.files("pygenai.front").joinpath("app.py")

    with resources.as_file(app_resource) as app_path:
        command: list[str] = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true" if not browser else "false",
        ]
        subprocess.run(command, check=True)  # noqa: S603
