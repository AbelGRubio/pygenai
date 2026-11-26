"""This module defines the PyGenaiLoader class, which serves as a base class for loading and configuring the app.

It provides a structure for initializing the application with a configuration dictionary and
outlines the method to build the application, which should be implemented in subclasses.
"""

from typing import Any


class PyGenaiLoader:
    """Base class for loading and configuring the PyGenAI application."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initializes the PyGenaiLoader with an optional configuration dictionary."""
        self.config = config

    def build_app(self) -> Any:
        """Implement the logic to build the PyGenAI application."""
        raise NotImplementedError(
            "PyGenaiLoader.build_app() is not implemented yet. "
            "Please implement the logic to build the PyGenAI application."
        )
