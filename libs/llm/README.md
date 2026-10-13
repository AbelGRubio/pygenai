# Pygenai LLM - Unified LLM Provider Abstraction

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Development-yellow.svg)](#status)

A modular, production-ready Python library providing unified access to multiple LLM providers with built-in cost tracking, intelligent fallback mechanisms, and comprehensive retry strategies.

---

## 🎯 Features

- **🔗 Unified Provider Interface** — Single abstraction layer for OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, and local Ollama models
- **💰 Automatic Cost Tracking** — Calculate and audit LLM API call costs using LiteLLM's pricing database
- **🔄 Intelligent Fallbacks** — Automatically switch to secondary providers on failure (rate limits, API errors, timeouts)
- **🔁 Robust Retry Logic** — Exponential backoff with configurable retry strategies and per-provider settings
- **📊 Usage Analytics** — Comprehensive tracking of token usage and costs by provider/model
- **⚙️ Type-Safe & Async** — Full Python 3.13 type annotations and async/await support
- **🧩 Modular Architecture** — Clean separation of providers, routing logic, callbacks, and exceptions
- **📝 Configuration-Driven** — YAML-based routing configuration and environment variable support

---

## 🏗️ Architecture

### Directory Structure

```
libs/llm/
├── src/genai/llm/
│   ├── __init__.py                 # Main package exports
│   ├── base.py                     # Base models: LLMProvider, LLMResponse, Message
│   ├── exceptions.py               # Custom exception classes
│   ├── router.py                   # LLMRouter with fallback logic
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── openai_provider.py      # OpenAI (gpt-4o, gpt-4o-mini, etc.)
│   │   ├── anthropic_provider.py   # Anthropic Claude models
│   │   ├── azure_provider.py       # Azure OpenAI deployments
│   │   ├── bedrock_provider.py     # AWS Bedrock with Claude
│   │   └── ollama_provider.py      # Local Ollama inference
│   └── callbacks/
│       ├── __init__.py
│       └── cost_tracker.py         # Cost calculation & audit logging
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_base.py                # Base model tests
│   ├── test_providers.py           # Provider validation tests
│   ├── test_router.py              # Router & fallback tests
│   └── test_cost_tracker.py        # Cost tracking tests
├── examples/
│   ├── 01_basic_openai.py          # Basic OpenAI usage
│   ├── 02_anthropic_provider.py    # Using Claude
│   ├── 03_router_with_fallbacks.py # Fallback strategy demo
│   ├── 04_azure_openai.py          # Azure OpenAI deployment
│   ├── 05_ollama_local.py          # Local inference
│   └── 06_cost_tracking.py         # Cost audit example
├── README.md                       # This file
├── .env.example                    # Configuration template
├── litellm.yaml                    # Router & model configuration
└── pyproject.toml                  # Project metadata & dependencies
```

### Component Overview

| Component | Purpose |
|-----------|---------|
| **Base Models** | `LLMProvider`, `LLMResponse`, `Message`, `ProviderConfig` — core abstractions |
| **Providers** | Implementations for each LLM provider (OpenAI, Anthropic, Azure, Bedrock, Ollama) |
| **Router** | Intelligent request routing with fallback and retry logic |
| **Cost Tracker** | Audit logging and cost calculation per call, provider, or model |
| **Exceptions** | Custom exceptions for provider errors, rate limits, auth failures |

---

## 📋 Prerequisites

- **Python:** 3.13+ (strictly required)
- **Package Manager:** `uv` (not `pip`)
- **Environment Variables:** API keys for providers you plan to use (see `.env.example`)

### Supported Providers & Their Requirements

| Provider | Environment Variable | Installation | Cost |
|----------|----------------------|--------------|------|
| **OpenAI** | `OPENAI_API_KEY` | Included (via `openai` dep) | Paid (token-based) |
| **Anthropic** | `ANTHROPIC_API_KEY` | Included (via `anthropic` dep) | Paid (token-based) |
| **Azure OpenAI** | `AZURE_API_KEY`, `AZURE_API_BASE` | Included (via `azure-identity` dep) | Paid (via Azure) |
| **AWS Bedrock** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` | Included (via `boto3` dep) | Paid (via AWS) |
| **Ollama (Local)** | None (localhost) | Requires local Ollama server | Free (local compute) |

---

## 🚀 Installation

### 1. Clone & Navigate to Module

```bash
cd libs/llm
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Verify installation
uv run pytest tests/ -v
```

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# AZURE_API_KEY=...
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# OLLAMA_API_BASE=http://localhost:11434
```

---

## 💡 Usage Guide

### Basic Usage: Single Provider

```python
import asyncio
from genai.llm import OpenAIProvider, Message, ProviderConfig

async def main():
    # Configure provider
    config = ProviderConfig(
        name="openai",
        api_key="sk-your-key",
        model="gpt-4o-mini",
    )

    # Create provider
    provider = OpenAIProvider(config)

    # Prepare messages
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Explain AI."),
    ]

    # Generate response
    response = await provider.generate(messages)

    print(f"Response: {response.content}")
    print(f"Cost: ${response.cost:.6f}")
    print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

### Using Multiple Providers with Fallbacks

```python
import asyncio
from genai.llm import LLMRouter, Message, ProviderConfig

async def main():
    # Primary provider (OpenAI)
    primary = ProviderConfig(
        name="openai",
        api_key="sk-...",
        model="gpt-4o",
    )

    # Fallback providers
    fallbacks = [
        ProviderConfig(
            name="anthropic",
            api_key="sk-ant-...",
            model="claude-3-5-sonnet-20241022",
        ),
        ProviderConfig(
            name="ollama",
            model="llama3",
            base_url="http://localhost:11434",
        ),
    ]

    # Create router
    router = LLMRouter(primary, fallbacks)

    # Messages
    messages = [
        Message(role="user", content="What is AI?"),
    ]

    # Generate (will fallback if primary fails)
    try:
        response = await router.generate(messages)
        print(f"Provider: {response.provider}")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"All providers failed: {e}")

asyncio.run(main())
```

### Cost Tracking & Auditing

```python
import asyncio
from genai.llm import OpenAIProvider, Message, ProviderConfig, CostTracker

async def main():
    config = ProviderConfig(
        name="openai",
        api_key="sk-...",
        model="gpt-4o-mini",
    )

    provider = OpenAIProvider(config)
    tracker = CostTracker(log_file="/tmp/llm_costs.jsonl")

    messages = [Message(role="user", content="Hello")]

    # Make API calls
    for _ in range(3):
        response = await provider.generate(messages)
        tracker.record_call(response)

    # Display audit report
    summary = tracker.get_summary()
    print(f"Total calls: {summary['total_calls']}")
    print(f"Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"By provider: {summary['cost_by_provider']}")

asyncio.run(main())
```

### Provider-Specific: Anthropic (Claude)

```python
import asyncio
from genai.llm import AnthropicProvider, Message, ProviderConfig

async def main():
    config = ProviderConfig(
        name="anthropic",
        api_key="sk-ant-...",
        model="claude-3-5-sonnet-20241022",
    )

    provider = AnthropicProvider(config)
    messages = [
        Message(role="system", content="Expert Python programmer."),
        Message(role="user", content="Explain decorators."),
    ]

    response = await provider.generate(messages)
    print(response.content)

asyncio.run(main())
```

### Provider-Specific: Azure OpenAI

```python
import asyncio
from genai.llm import AzureOpenAIProvider, Message, ProviderConfig

async def main():
    config = ProviderConfig(
        name="azure",
        api_key="your-azure-key",
        model="gpt-4",  # deployment name
        base_url="https://your-resource.openai.azure.com",
    )

    provider = AzureOpenAIProvider(config)
    messages = [Message(role="user", content="Hello Azure")]

    response = await provider.generate(messages)
    print(response.content)

asyncio.run(main())
```

### Provider-Specific: Local Ollama

```python
import asyncio
from genai.llm import OllamaProvider, Message, ProviderConfig

async def main():
    config = ProviderConfig(
        name="ollama",
        model="llama3",
        base_url="http://localhost:11434",
        timeout=120,  # Local inference may be slower
    )

    provider = OllamaProvider(config)
    messages = [Message(role="user", content="Local inference test")]

    response = await provider.generate(messages)
    print(response.content)
    print(f"Cost: ${response.cost:.2f} (free local model)")

asyncio.run(main())
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Azure OpenAI
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com
AZURE_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT_NAME=gpt-4

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1

# Ollama
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=llama3

# Router & Retry
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY_INITIAL=1.0
LLM_RETRY_DELAY_MAX=60.0

# Cost Tracking
LLM_COST_LOG_FILE=/var/logs/llm_costs.jsonl
```

### Router Configuration (`litellm.yaml`)

The `litellm.yaml` file defines routing strategies, model mappings, and provider-specific settings:

```yaml
models:
  gpt-4o:
    provider: "openai"
    rpm_limit: 100
    tpm_limit: 1000000

  claude-3-sonnet:
    provider: "anthropic"
    rpm_limit: 50
    tpm_limit: 400000

routes:
  high-performance:
    - model: gpt-4o
      weight: 1.0
    - model: claude-3-sonnet
      weight: 0.9
      condition: "openai_failed"

retry_policy:
  max_retries: 3
  initial_delay: 1.0
  exponential_base: 2.0
  retry_on: [429, 500, 502, 503, 504]
```

---

## 🧪 Testing

### Run All Tests

```bash
uv run pytest tests/ -v --cov=src/genai/llm
```

### Run Specific Test Suite

```bash
# Test base models
uv run pytest tests/test_base.py -v

# Test providers
uv run pytest tests/test_providers.py -v

# Test router logic
uv run pytest tests/test_router.py -v

# Test cost tracking
uv run pytest tests/test_cost_tracker.py -v
```

### Test Coverage

```bash
uv run pytest tests/ --cov=src/genai/llm --cov-report=html
# Open htmlcov/index.html
```

---

## 📚 Examples

All examples are in the `examples/` folder:

| Example | Description |
|---------|-------------|
| `01_basic_openai.py` | Simple OpenAI text generation |
| `02_anthropic_provider.py` | Using Claude Anthropic |
| `03_router_with_fallbacks.py` | Automatic fallback to secondary provider |
| `04_azure_openai.py` | Azure OpenAI deployment |
| `05_ollama_local.py` | Local model inference (zero cost) |
| `06_cost_tracking.py` | Comprehensive cost audit example |

### Run an Example

```bash
# Basic OpenAI example
OPENAI_API_KEY=sk-... uv run examples/01_basic_openai.py

# Router with fallbacks (requires multiple API keys)
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \
  uv run examples/03_router_with_fallbacks.py

# Local Ollama (requires running Ollama server)
uv run examples/05_ollama_local.py
```

---

## 🔄 Fallback & Retry Strategies

### How Fallbacks Work

When you configure a router with multiple providers:

1. **Primary Request** — Try primary provider (e.g., OpenAI)
2. **Retry Logic** — If transient error (429, 5xx), retry with exponential backoff
3. **Fallback** — If primary exhausts retries, switch to first fallback
4. **Repeat** — Continue until response received or all fallbacks exhausted
5. **Error** — Raise `NoAvailableProvidersError` if all fail

### Retry Configuration

```python
from genai.llm import RetryConfig, ProviderConfig

retry_config = RetryConfig(
    max_retries=3,           # Number of attempts
    initial_delay=1.0,       # Starting delay in seconds
    max_delay=60.0,          # Max delay (cap exponential backoff)
    exponential_base=2.0,    # Backoff multiplier (1s, 2s, 4s, 8s, ...)
    retry_on=[429, 500, 502, 503, 504],  # HTTP error codes to retry
)

config = ProviderConfig(
    name="openai",
    api_key="sk-...",
    model="gpt-4o",
    retry_config=retry_config,
)
```

---

## 💰 Cost Tracking & Analytics

### Automatic Cost Calculation

Every response includes cost information:

```python
response = await provider.generate(messages)

print(f"Tokens: {response.usage.prompt_tokens} + {response.usage.completion_tokens}")
print(f"Cost: ${response.cost:.6f}")
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
```

### Audit Trail

Track costs across multiple calls:

```python
from genai.llm import CostTracker

tracker = CostTracker(log_file="/var/logs/llm_costs.jsonl")

# Make API calls
for _ in range(100):
    response = await provider.generate(messages)
    tracker.record_call(response, request_id="batch_001")

# Generate audit report
summary = tracker.get_summary()
print(f"Total cost: ${summary['total_cost_usd']:.2f}")
print(f"By provider: {summary['cost_by_provider']}")
print(f"By model: {summary['cost_by_model']}")
print(f"Token usage: {summary['tokens_by_provider']}")
```

### Cost Comparison

```python
# Compare costs across providers
costs = tracker.get_cost_by_provider()
# Output: {'openai': 1.234, 'anthropic': 0.567, 'ollama': 0.0}

models = tracker.get_cost_by_model()
# Output: {'gpt-4o': 0.98, 'claude-3-sonnet': 0.54, 'llama3': 0.0}
```

---

## 🤝 Contributing

1. **Fork & Clone** — Create a feature branch
2. **Follow Conventions** — See [CONVENTIONS.md](../../../CONVENTIONS.md)
3. **Add Tests** — Unit tests required for all new features
4. **Run CI** — `uv run make ci` (lint, type-check, test)
5. **Submit PR** — Include tests and documentation

### Code Quality Standards

- **Type Checking:** `pyrefly` (strict mode)
- **Linting:** `ruff` (line length 120, complexity ≤ 10)
- **Testing:** `pytest` with >80% coverage
- **No `print()`** — Use `logging.getLogger(__name__)`

---

## 📄 License

MIT License — See [LICENSE](LICENSE) file for details.

---

## 🆘 Troubleshooting

### API Key Not Found

```
AuthenticationError: OPENAI_API_KEY not provided in config or environment
```

**Solution:** Set `OPENAI_API_KEY` in `.env` or pass via `ProviderConfig`:

```python
config = ProviderConfig(api_key="sk-...", ...)
```

### Ollama Connection Failed

```
ProviderError: ollama error: Connection refused
```

**Solution:** Ensure Ollama is running:

```bash
ollama serve  # In another terminal
```

### Rate Limit Exceeded (429)

```
RateLimitError: Rate limit exceeded for provider openai
```

**Solution:** Router automatically retries and falls back to secondary provider. No action needed.

### All Providers Failed

```
NoAvailableProvidersError: No available providers. Attempted: openai, anthropic, ollama
```

**Solution:** Verify API keys, internet connection, and provider status.

---

## 📞 Support & Documentation

- **LiteLLM Docs:** https://docs.litellm.ai/
- **OpenAI API:** https://platform.openai.com/docs/
- **Anthropic Claude:** https://claude.ai/
- **AWS Bedrock:** https://aws.amazon.com/bedrock/
- **Ollama:** https://ollama.ai/

---

**Made with ❤️ by the Pygenai team**
