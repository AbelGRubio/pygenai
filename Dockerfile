# ──────────────────────────────────────────────
# Stage 1: Build — install dependencies
# ──────────────────────────────────────────────
# linux/amd64 avoids ARM64 SIGILL from native-code wheels on Apple Silicon
FROM --platform=linux/amd64 astral/uv:python3.13-bookworm-slim AS build


# ──────────────────────────────────────────────
# Environment variables (build-time only)
# ──────────────────────────────────────────────
ENV UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# ──────────────────────────────────────────────
# Copy (parent project)
# ──────────────────────────────────────────────
COPY --from=observe-core . /observe_core
COPY src/ /app/src/
COPY pyproject.toml /app/

# ──────────────────────────────────────────────
# Authenticate and sync UV dependencies
# ──────────────────────────────────────────────
WORKDIR /app

RUN mkdir -p /app

RUN uv sync --no-dev --no-editable --no-build

# ──────────────────────────────────────────────
# Stage 2: Run — minimal runtime image
# ──────────────────────────────────────────────
FROM --platform=linux/amd64 astral/uv:python3.13-bookworm-slim AS run


# ──────────────────────────────────────────────
# Environment variables
# ──────────────────────────────────────────────
# Ensure uv & venv binaries are in PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DIR=/app \
    AWS_REGION=eu-west-1 \
    AWS_DEFAULT_REGION=eu-west-1 \
    PATH="/app/.venv/bin:/usr/local/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    SERVER__HOST=0.0.0.0

ENV UVICORN_HOST=0.0.0.0

# ──────────────────────────────────────────────
# Create user for non-root execution
# ──────────────────────────────────────────────
RUN useradd -m -u 1000 bedrock_agentcore

# Set up the runtime application directory
WORKDIR /app

RUN mkdir -p /app/.logs && chown -R 1000:1000 /app/.logs
RUN mkdir -p /app/.langgraph_api && chown -R 1000:1000 /app/.langgraph_api

USER bedrock_agentcore

# ──────────────────────────────────────────────
# Copy dependencies from build stage
# ──────────────────────────────────────────────
COPY --from=build --chown=bedrock_agentcore:bedrock_agentcore /app/.venv /app/.venv

# ──────────────────────────────────────────────
# Copy application code as non-root
# ──────────────────────────────────────────────
COPY --chown=bedrock_agentcore:bedrock_agentcore src/ src/

# ──────────────────────────────────────────────
# Entrypoint / CMD
# ──────────────────────────────────────────────
CMD ["opentelemetry-instrument", "uvicorn", "src.__main__:app"]
