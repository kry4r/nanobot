# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install (editable/dev mode)
pip install -e ".[dev]"

# Run tests
pytest                        # all tests (asyncio_mode = "auto")
pytest tests/test_foo.py      # single file
pytest tests/test_foo.py::test_bar  # single test

# Lint & format
ruff check                    # lint (rules: E, F, I, N, W; E501 ignored)
ruff check --fix              # auto-fix
ruff format                   # format

# Run the bot
nanobot agent                 # interactive chat
nanobot agent -m "message"    # single message
nanobot gateway               # multi-channel server

# WhatsApp bridge (TypeScript, separate from Python)
cd bridge && npm install && npm run build
```

## Architecture

nanobot is an ultra-lightweight (~3,668 LOC) personal AI assistant framework. A single agent serves multiple chat platforms simultaneously through an async message bus.

```
Channels (10 platforms) → MessageBus → AgentLoop → LLM (via LiteLLM) + Tools
                                          ↕
                                    SessionManager (JSONL, two-layer context)
                                          ↕
                               GraphMemoryStore (SQLite graph.db)

ExtendChannel (WS/HTTP gateway) ──direct──→ AgentLoop.process_streaming()
                                              ↕ (bypasses MessageBus)
                                        streaming chunks
```

### Core flow (`nanobot/agent/`)

- **`loop.py`** — `AgentLoop`: receives messages from bus, builds context, calls LLM, executes tool calls, iterates up to `max_iterations` (default 20). Handles memory consolidation after `memory_window` messages. Also provides `process_streaming()` for streaming responses (used by ExtendChannel) with two-layer session context (summaries + recent window) and `_summarize_old_messages()` for background summary generation.
- **`context.py`** — `ContextBuilder`: assembles system prompt from bootstrap files (`AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md`), memory, and skills. Supports progressive skill loading (summaries in context, full content via `read_file`).
- **`skills.py`** — `SkillsLoader`: discovers skill markdown files from `nanobot/skills/` and `workspace/skills/`. Parses YAML frontmatter for metadata. Supports OpenClaw/ClawHub format.
- **`graph_memory.py`** — `GraphMemoryStore`: SQLite + FTS5 graph-structured associative memory (`workspace/memory/graph.db`). Two-phase recall: anchor + time-chain backtracking, then spreading activation. Auto-promotes keywords to concept hub nodes. Consolidation extracts summaries (event nodes) and long-term facts (entity/concept nodes).
- **`tools/memory_tools.py`** — Six LLM tools wrapping graph memory: `find_memory_cache`, `recall_related`, `get_memory_cache`, `create_memory`, `update_memory`, `delete_memory`.
- **`tools/registry.py`** — `ToolRegistry`: dynamic tool registration. Built-in tools: filesystem, shell, web (Brave API), message, spawn (subagents), cron, MCP.

### Message bus (`nanobot/bus/`)

Async queue-based routing that decouples channels from agent processing. `InboundMessage` flows in, `OutboundMessage` flows out.

### Channels (`nanobot/channels/`)

Ten platform adapters all implementing `BaseChannel`: Telegram, Discord, WhatsApp (via TypeScript bridge on port 3001), Feishu/Lark, Mochat, DingTalk, Slack, Email (IMAP/SMTP), QQ, **Extend** (WebSocket/HTTP gateway). `ChannelManager` starts/stops all enabled channels and routes outbound messages.

### Extend channel (`nanobot/channels/extend/`)

WebSocket/HTTP gateway for external app integration. Unlike other channels, ExtendChannel bypasses MessageBus and holds a direct `AgentLoop` reference for streaming. Registered in `cli/commands.py` (not `ChannelManager._init_channels`) because it needs the agent reference.

- **`__init__.py`** — `ExtendChannel(BaseChannel)`: embeds aiohttp server, lifecycle management.
- **`server.py`** — aiohttp app with routes: `GET /health`, `POST /chat`, `GET /ws/chat`.
- **`ws.py`** — `WebSocketManager`: per-connection handling, per-conversation `asyncio.Lock` for parallel processing, streaming protocol (ack → typing → stream_start → stream_chunk* → stream_end → message).
- **`auth.py`** — Bearer token verification (HMAC-safe comparison).
- **`models.py`** — Pydantic models for the WebSocket protocol.

Session uses two-layer context: old messages are summarized into compact summaries, recent 6-10 turns kept in full. Session data (including summaries) persisted via JSONL. Graph memory is NOT modified by ExtendChannel — the two are separate concerns.

### Providers (`nanobot/providers/`)

`registry.py` is the single source of truth for all 15 LLM providers (Anthropic, OpenAI, DeepSeek, Gemini, etc. plus gateways like OpenRouter). `litellm_provider.py` handles API calls, model prefixing, streaming, and tool calling via LiteLLM. `base.py` defines `LLMProvider` ABC with `chat()` and `chat_stream()` (async generator yielding text chunks).

### Skills format

Markdown files with YAML frontmatter:
```yaml
---
name: skill-name
description: "What this skill does"
metadata: {"nanobot":{"emoji":"🔧","requires":{"bins":["gh"]},"always":false}}
---
```
Skills with `"always": true` are included in every prompt. Others are loaded on demand.

### Configuration

User config lives at `~/.nanobot/config.json`. Schema defined in `nanobot/config/schema.py`. Sections: `agents`, `channels` (including `extend`), `providers`, `gateway`, `tools` (including `mcpServers`).

### Workspace templates (`workspace/`)

Default bootstrap files copied during `nanobot onboard`. These define the agent's personality, instructions, and tool guidelines.

## Code Conventions

- Python 3.11+, async/await throughout
- Line length: 100 (ruff), E501 ignored
- Pydantic v2 for config validation
- loguru for logging
- Type hints expected on function signatures
- Adding a new provider: update both `providers/registry.py` and `config/schema.py`
- Adding a new channel: implement `BaseChannel` in `channels/`, register in `ChannelManager` (or in `cli/commands.py` if it needs AgentLoop reference, like ExtendChannel)
- Adding a new tool: implement tool class in `agent/tools/`, register in `AgentLoop.__init__`

## Gotchas

- `ContextBuilder` and `AgentLoop` share a single `GraphMemoryStore` instance — don't create a second one (causes connection leak)
- `gc_concepts` SQL: avoid LEFT JOIN cross-products for degree counting — use correlated subqueries
- `_batch_get_neighbors`: use UNION ALL (forward + reverse) to ensure bidirectional edges appear for both endpoints
- `loop.py` has pre-existing F821 lint errors (`ExecToolConfig`, `CronService` forward refs) — these are expected
- `ExtendChannel` bypasses `MessageBus` — it calls `AgentLoop.process_streaming()` directly. Don't try to route extend messages through the bus.
- `Session.summaries` and `Session.last_summarized` are persisted in JSONL metadata line — backward compatible (defaults to `[]` and `0` if missing)
- `process_streaming()` is an async generator — callers must iterate it to completion for session state to be saved
