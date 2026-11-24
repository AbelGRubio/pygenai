"""Entry point."""

import uvicorn

from pygenai.agent.loader import PyGenaiAgent
from pygenai.core.logger import get_logger, propague_loggers

logger = get_logger(__name__)

app = PyGenaiAgent(None).build_app()

propague_loggers()


if __name__ == "__main__":
    logger.debug("Starting...")
    uvicorn.run(
        app=app,
        host="localhost",
        port=8123,
        log_config=None,
        reload=False,
    )
    logger.debug("Ending.")
