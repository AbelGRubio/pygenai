---
name: pygenai-readme-generator
description: 'Expert technical writer and AI agent skill for generating professional, comprehensive, and well-structured README.md files. Use when documenting repositories, defining project structures, explaining installation steps, configuration guides, usage examples, API references, and contributing guidelines for software projects, monorepos, and Python packages.'
---

# Skill: Expert README.md Generator

## Purpose
This skill equips the agent to act as a software architect and expert technical writer. Its objective is to analyze a repository's structure, source code, and configurations to generate a professional, clear, visually appealing, and highly informative `README.md`.

## Structure Guidelines for README.md
When asked to create or update a `README.md`, you must strictly follow this modular structure:

1. **Title and Short Description (`# Name and Subtitle`)**
   - Clear project name.
   - A concise sentence or paragraph explaining *what* the project does, *what* it is for, and *who* it targets.
   - Relevant badges (Python version, license, build/CI status if applicable).

2. **Main Features (`## Features`)**
   - Clear bullet points highlighting the technical capabilities of the project (e.g., modular support, integration with LiteLLM, LangGraph, etc.).

3. **Architecture and Project Structure (`## Architecture`)**
   - Display the directory structure in a tree format (`tree`) so developers can quickly understand where each component is located (`loaders`, `chunking`, `vectorstore`, `memory`, etc.).

4. **Prerequisites and Environment (`## Prerequisites`)**
   - Required Python versions or other tools (e.g., Python 3.13, package managers like `uv` or `pip`).
   - Required environment variables (e.g., `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`). Provide a `.env.example` template.

5. **Installation and Setup (`## Installation`)**
   - Detailed and reproducible steps to clone the repository, set up the virtual environment, and install dependencies.

6. **Usage Guide / Quick Examples (`## Usage`)**
   - Clear and functional code snippets demonstrating how to initialize and use the main package or agent.

7. **Agent Configuration / Monorepo (if applicable) (`## LangGraph / Monorepo Configuration`)**
   - Explanation of how components (RAG, memory, LLMs) integrate and how it interacts with configuration files (such as `langgraph.json`).

8. **Contribution (`## Contributing`)**
   - Brief guidelines on how to open issues, propose pull requests, or follow repository code conventions.

9. **License (`## License`)**
   - Indication of the license under which the software is distributed.

## Style and Tone
- **Tone:** Professional, technical, direct, and developer-oriented.
- **Format:** Clean Markdown, proper use of syntax-highlighted code blocks (`python`, `bash`, `json`), readable tables, and ordered/unordered lists.
- **Adaptability:** If the project is smaller or specific (e.g., just a RAG or memory module), adapt the sections by omitting non-applicable ones without losing overall quality.
