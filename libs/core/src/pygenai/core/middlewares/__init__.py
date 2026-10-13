"""Security package for authentication middleware and identity providers."""

from .auth import AuthMiddleware

__all__ = [
    "AuthMiddleware",
]
