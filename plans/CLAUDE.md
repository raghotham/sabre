# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python Commands
**Always use `uv run` for running anything related to Python.**

- **Server**: `uv run python -m sabre.server`
- **Client**: `uv run python -m sabre.client`
- **Install dependencies**: `uv sync` or `pip install -e .`
- **Run tests**: `uv run pytest tests/`

## High-Level Architecture

### Core Components

**SABRE** is a simplified, clean reimplementation of LLMVM with focus on:
- Minimal client-server architecture
- Python runtime with bash helper execution
- Conversations API for state management
- WebSocket streaming server
- Clean separation of concerns

### Directory Structure

- **`sabre/server/`**: Server components
  - `orchestrator.py`: Coordinates LLM calls and execution
  - `python_runtime.py`: Python code execution environment
  - `bash_helper.py`: Bash command execution helper
  - `api/server.py`: WebSocket server implementation
  - `prompts/`: System prompts and templates

- **`sabre/client/`**: Minimal terminal client
  - CLI interface for interacting with the server
  - Message rendering and user input handling

- **`sabre/common/`**: Shared components
  - `config/`: Configuration and dependency injection
  - `utils/`: Utilities including performance tracking
  - LLM executors for different providers

- **`tests/`**: Test suite
  - Unit tests for core components
  - Integration tests for client-server communication

### Key Architecture Patterns

**Simplified Execution Model**: The core execution flow:
1. User input → Server orchestrator
2. LLM generates response with optional code blocks
3. Python runtime executes code blocks
4. Results streamed back to client via WebSocket
5. Continue until task complete

**Conversations API**: Manages conversation state, message history, and context. Provides clean separation between conversation management and execution.

**Dependency Injection**: Uses `container.py` for clean dependency management and configuration.

### Configuration

- **Environment Variables**:
  - `ANTHROPIC_API_KEY`: For Claude models
  - `OPENAI_API_KEY`: For OpenAI models
  - Additional provider keys as needed

- **Logging**: Structured logging to `logs/` directory with performance tracking

## Development Notes

- This is a clean, simplified implementation focusing on core LLMVM functionality
- Python runtime executes code in isolated environment with bash helper support
- WebSocket server provides real-time streaming of LLM responses
- Minimal dependencies compared to original LLMVM
- Focus on maintainability and testability
