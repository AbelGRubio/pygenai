# PyGenAI

> A modular, monorepo-based Python framework and scaffold for building, orchestrating, and deploying scalable AI agents and workflows.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

`pygenai` is designed as a highly modular monorepo to streamline the creation of advanced AI agents, RAG pipelines, and customized reasoning workflows. By splitting key agent responsibilities into reusable internal libraries and leveraging [LiteLLM](https://github.com/BerriAI/litellm) as a provider gateway, `pygenai` allows you to rapidly build, test, and containerize AI-driven applications.

---

## Features

- **Monorepo Architecture**: Clean separation of concerns through isolated libraries (`agent`, `core`, `llm`, `memory`, `rag`).
- **Unified LLM Gateway**: Built-in support for LiteLLM to interact seamlessly with 100+ LLM providers via standard interfaces.
- **Agent Building Blocks**: Generic builders, node executors, and flexible model abstractions for agentic workflows.
- **Extensible Memory Management**: Support for multiple memory strategies to persist and manage conversation context.
- **RAG Subsystems**: Comprehensive tools for indexing, retrieving, and grounding LLM responses with custom data.
- **Production Containerization**: Ready-to-use `Dockerfile` and `docker-compose.yml` setup including an integrated LiteLLM proxy instance.

---

## Architecture & Project Structure

```text
pygenai/
├── docker-compose.yml       # Service orchestration (LiteLLM proxy + App)
├── Dockerfile               # Production image container setup
├── examples/                # Custom reasoning workflows and advanced examples
│   └── reasoning_workflow/  # Example implementation of reasoning agents
├── src/                     # Core application entrypoint
│   ├── pygenai/graph.py     # Graph-based reasoning and agent orchestration
│   └── __main__.py          # Basic agent implementation
└── libs/                    # Core libraries (monorepo packages)
    ├── agent/               # Agent builders, models, and node executors
    ├── core/                # Shared base models, schemas, and utilities
    ├── llm/                 # LiteLLM provider client and integration examples
    ├── memory/              # Conversation and state memory implementations
    └── rag/                 # RAG components, retrievers, and indexers
```

---

## Prerequisites

- **Python**: `3.12+`
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) or `pip`
- **Docker & Docker Compose**: Required for running the containerized stack and local LiteLLM service.

### Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
# General Settings
ENVIRONMENT=development

# LLM & LiteLLM Gateway
LITELLM_MASTER_KEY=sk-1234
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Tracing / Observability (Optional)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
```

---

## Installation & Setup

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/pygenai.git
   cd pygenai
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   uv pip install -e "libs/core" -e "libs/llm" -e "libs/memory" -e "libs/rag" -e "libs/agent"
   ```

### Docker Deployment

To launch the full service along with the local LiteLLM proxy:

```bash
docker-compose up --build
```

---

## Quick Start

### Running the Basic Agent (`src/`)

To execute the default agent implementation found in `src/`:

```bash
python src/main.py
```

### Running Advanced Reasoning Workflows (`examples/`)

To test custom reasoning workflows:

```bash
python examples/reasoning_workflow/main.py
```

---

## Monorepo Libraries (`libs/`)

For detailed guides, examples, and API specifications for each sub-library, consult their respective READMEs:

| Package | Path | Description |
| :--- | :--- | :--- |
| **`pygenai-agent`** | [`libs/agent`](./libs/agent/README.md) | Builders, model wrappers, and node executors for agent creation. |
| **`pygenai-core`** | [`libs/core`](./libs/core/README.md) | Shared data schemas, base models, and common utilities. |
| **`pygenai-llm`** | [`libs/llm`](./libs/llm/README.md) | LiteLLM provider integration client and example calls. |
| **`pygenai-memory`** | [`libs/memory`](./libs/memory/README.md) | Context management and conversation history strategies. |
| **`pygenai-rag`** | [`libs/rag`](./libs/rag/README.md) | Retrieval-Augmented Generation strategies and example usages. |

---

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Ensure code formatting and tests pass across all packages in `libs/`.
4. Open a Pull Request detailing your changes.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
