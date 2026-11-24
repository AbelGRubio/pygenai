"""Launch PyGenAI applications with Uvicorn.

==================================================
File:    launcher.py
Author:  AbelGRubio
Date:    2026-11-24
Desc:    PyGenAI application launcher using Uvicorn
==================================================
"""
import importlib
import os
import sys
from typing import Any

import typer
import uvicorn

from pygenai.core.loader import PyGenaiLoader


class PyGenaiLauncher:
    """Build and run a PyGenAI application with Uvicorn."""

    def __init__(self, loader: PyGenaiLoader) -> None:
        """Initialize the launcher with an application loader.

        Args:
            loader: Loader responsible for building the application.
        """
        self.loader: PyGenaiLoader = loader

    def build_app(self) -> Any:
        """Build the application using the configured loader.

        Returns:
            ASGI application built by the loader.
        """
        return self.loader.build_app()

    def run(
        self,
        host: str = "localhost",
        port: int = 8123,
        log_level: str = "info",
        reload: bool = False,
    ) -> None:
        """Run the PyGenAI application with Uvicorn.

        Args:
            host: Address where the server is exposed.
            port: Port on which the server listens.
            log_level: Uvicorn logging level.
            reload: Whether to reload the server when files change.
        """
        options: dict[str, Any] = {
            "host": host,
            "port": port,
            "log_level": log_level,
            "log_config": None,
            "reload": reload,
        }

        uvicorn.run(app=self.build_app(), **options)


def load_agent() -> type[PyGenaiLoader] | None:
    """Load the configured PyGenAI agent loader.

    Returns:
        Loader instance for the configured agent module.
    """
    project_root = os.getcwd()

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    agent_module = os.getenv("AGENT_MODULE", "pygenai.agent.loader")
    module = importlib.import_module(agent_module)
    loader_class = getattr(module, "PyGenaiAgent", None)
    return loader_class


def run_cli(
    host: str = typer.Option("localhost", help="Address where the server is exposed."),
    port: int = typer.Option(8123, help="Port on which the server listens."),
    log_level: str = typer.Option("info", help="Uvicorn logging level."),
    reload: bool = typer.Option(False, help="Reload the server when files change."),
) -> None:
    """Run the PyGenAI application from the command line.

    Args:
        host: Address where the server is exposed.
        port: Port on which the server listens.
        log_level: Uvicorn logging level.
        reload: Whether to reload the server when files change.
    """
    loader = load_agent()
    if not loader:
        typer.echo("Failed to load the agent.")
        raise typer.Exit(code=1)

    launcher = PyGenaiLauncher(loader())
    launcher.run(
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )
