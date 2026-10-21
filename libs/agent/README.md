# Pygenai Agent

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![Package](https://img.shields.io/badge/package-pygenai--agent-8A2BE2)
![Status](https://img.shields.io/badge/status-experimental-orange)
![License](https://img.shields.io/badge/license-repository%20license-blue)

> Core abstractions, builders, models, and node execution engines for constructing AI agents within the `pygenai` ecosystem.

## Overview

`pygenai-agent` provides the foundational building blocks required to assemble complex agent workflows. It handles the lifecycle of node execution, state transitions, and step-by-step reasoning constructs.

## Key Features

* **Agent Builders**: Declarative tools to compose custom agent pipelines.

* **Node Execution Engine**: Standardized interface for executing isolated execution nodes.

* **Model Abstractions**: Standard definitions for agent states, inputs, and outputs.

## Architecture & Structure

```
libs/agent/
├── pyproject.toml
└── pygenai/
    └── agent/
        ├── builders/       # Pipeline and agent construction helpers
        ├── models/         # State and message schemas
        └── node_executor/  # Execution handlers for graph or sequential nodes

```

## Quick Start

```
from pygenai.agent.builders import AgentBuilder
from pygenai.agent.node_executor import NodeExecutor

# Example initialization
builder = AgentBuilder()
# Add custom nodes and configuration here...
agent = builder.build()

```
