# Pygenai core

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![Package](https://img.shields.io/badge/package-pygenai--core-8A2BE2)
![Status](https://img.shields.io/badge/status-experimental-orange)
![License](https://img.shields.io/badge/license-repository%20license-blue)

> Shared primitives, base domain models, and core utilities for the `pygenai` monorepo.

## Overview

`pygenai-core` provides universal data structures, interface specifications, and shared utilities utilized across all other `libs/` packages and application entry points.

## Key Features

- **Shared Domain Models**: Unified Pydantic base schemas for consistency across modules.
- **Common Utilities**: Helpers for configuration loading, logging, and error handling.

## Architecture & Structure

```text
libs/core/
├── pyproject.toml
└── pygenai/
    └── core/
        ├── models/       # Shared Pydantic data models
        ├── exceptions/   # Common custom exceptions
        └── utils/        # Utility and helper functions
```

## Usage

```python
from pygenai.core.models import BaseMessage, ModelConfig

config = ModelConfig(temperature=0.7, max_tokens=1024)
```
