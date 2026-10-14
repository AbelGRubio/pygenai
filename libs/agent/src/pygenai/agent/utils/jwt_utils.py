"""JWT helper utilities used by the agent workflow.

========================================================================================================================
Name:         apps/agent/src/pygenai/jwt_utils.py
Description:  Provide helper functions to obtain JWT access tokens for MCP calls.
Project:      Observe me
Date:         2026-06-19 00:00:00
Status:       Development

Copyright ©2026. All rights reserved.
========================================================================================================================
"""

import httpx2

from pygenai.core.logger import get_logger
from pygenai.agent.settings import get_settings

logger = get_logger(__name__)


async def get_jwt_token() -> str:
    """Retrieve a JWT access token for outbound MCP calls.

    This helper uses the client credentials flow to request a token from the
    configured JWT endpoint. It validates that the required client credentials
    are present and raises a clear error if the token is missing from the
    response.

    Returns:
        The JWT access token string.

    Raises:
        ValueError: If required client credentials are missing or the token is
            not present in the JWT response payload.
    """
    settings = get_settings()

    if not settings.client_id or not settings.client_secret:
        raise ValueError("CLIENT_ID and CLIENT_SECRET are required when JWT protection is enabled")

    # Use an async HTTP client to perform the token request. This call is
    # awaited to avoid blocking the async event loop.
    async with httpx2.AsyncClient() as client:
        response = await client.post(
            settings.jwt_url,
            data={"grant_type": "client_credentials"},
            auth=(settings.client_id, settings.client_secret),
            timeout=10.0,
        )

    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ValueError("JWT response does not include access_token")

    return token
