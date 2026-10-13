"""Application factory for the Observe Me FastAPI server.

This module builds and configures the API application instance,
including routing, optional authentication middleware, CORS, and
Prometheus metrics instrumentation.
"""

import json
import os
import sys
from functools import lru_cache

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware

from pygenai.core.logger import get_logger
from pygenai.core.middlewares.auth import AuthMiddleware

sys.path.insert(0, os.path.abspath("src"))
from langgraph_api.server import app as langgraph_app

logger = get_logger(__name__)

LANGGRAPH_JSON_FILENAME = "langgraph.json"
LANGSERVE_GRAPHS = "LANGSERVE_GRAPHS"


def load_config() -> dict:
    """Load the LangGraph configuration file and register graph definitions.

    Returns:
        dict: Parsed graph definitions loaded from `langgraph.json`.
    """
    graphs = {}
    try:
        with open(LANGGRAPH_JSON_FILENAME, encoding="utf-8") as f:
            config_data = json.load(f)

        graphs = config_data.get("graphs", {})
        os.environ[LANGSERVE_GRAPHS] = json.dumps(graphs)

        logger.info(f"Graph file loaded: {graphs}")
    except FileNotFoundError as e:
        logger.error(f"Cannot load langgraph.json in the root: {e}")
    return graphs


@lru_cache(maxsize=1)
def define_app(add_auth: bool = False) -> Starlette:
    """Create and configure the FastAPI application instance.

    Args:
        add_auth: Whether to attach the authentication middleware.

    Returns:
        A configured FastAPI application.

    """
    load_config()

    if add_auth:
        langgraph_app.add_middleware(AuthMiddleware)

    langgraph_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("Define fastapi server.")
    return langgraph_app
