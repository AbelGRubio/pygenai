"""MCP manager: lifecycle and caching utilities for MCP sessions/tools.

This module provides a singleton `MCPManager` that keeps MCP sessions open
across the agent runtime and caches tool lists per MCP configuration hash.
The manager is safe for concurrent access via an `asyncio.Lock`.

========================================================================================================================
Name:         apps/agent/src/pygenai/mcp_manager.py
Description:  Manage MCP ClientSession lifecycle and tool caching.
Project:      Observe me
Date:         2026-06-19 00:00:00
Status:       Development

Copyright ©2026. All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from functools import lru_cache
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession

from pygenai.core.logger import get_logger

logger = get_logger(__name__)


class MCPManager:
    """Manage MCP sessions and cache tools per configuration.

    Attributes:
        sessions: Mapping of server name -> active `ClientSession`.
        tools_cache: Mapping of config-hash -> list of tools loaded for that
          configuration.
        current_config_hash: Last seen configuration hash, used to detect
          configuration changes.
        lock: Asyncio lock to protect concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the MCPManager with empty session and tool caches."""
        self.sessions: dict[str, ClientSession] = {}
        self.tools_cache: dict[str, list[Any]] = {}
        self.current_config_hash: str | None = None
        self.lock = asyncio.Lock()

    def _get_hash(self, mcp_config: dict[str, Any]) -> str:
        """Generate a stable hash for the given MCP configuration mapping."""
        config_str = json.dumps(mcp_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()  # noqa: S324

    async def get_session(self, client: MultiServerMCPClient, name: str) -> ClientSession:
        """Return an active `ClientSession` for `name`, creating it if missing."""
        if name not in self.sessions:
            session = await client.session(name)
            self.sessions[name] = session
        return self.sessions[name]

    async def close_all(self) -> None:
        """Close and clear all active sessions."""
        async with self.lock:
            for session in self.sessions.values():
                try:
                    await session.__aexit__(None, None, None)
                except Exception:
                    try:
                        maybe = session.close()
                        if asyncio.iscoroutine(maybe):
                            await maybe
                    except Exception:
                        logger.debug("Failed to gracefully close session", exc_info=True)

            self.sessions.clear()

    async def get_active_tools(self, mcp_config: dict[str, Any], mcp_client: MultiServerMCPClient) -> list[Any]:
        """Load and return tools for the active MCP configuration with caching."""
        new_hash = self._get_hash(mcp_config)

        if self.current_config_hash != new_hash:
            await self.close_all()
            self.current_config_hash = new_hash

        if new_hash in self.tools_cache:
            return self.tools_cache[new_hash]

        all_tools: list[Any] = []
        for server_name in mcp_config:
            try:
                all_tools = await mcp_client.get_tools()
            except Exception as server_err:
                logger.error(f"Failed to load tools from MCP server '{server_name}': {server_err}")

        self.tools_cache[new_hash] = all_tools
        return all_tools


@lru_cache(maxsize=1)
def get_mcp_manager() -> MCPManager:
    """Return a cached MCPManager singleton.

    The manager is created once per process and reused by callers.
    """
    logger.info("Creating MCPManager singleton")
    return MCPManager()
