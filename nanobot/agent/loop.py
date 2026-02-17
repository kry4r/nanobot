"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path

import json_repair
from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.graph_memory import GraphMemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.memory_tools import (
    CreateMemoryTool,
    DeleteMemoryTool,
    FindMemoryCacheTool,
    GetMemoryCacheTool,
    RecallRelatedTool,
    UpdateMemoryTool,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.graph_memory = GraphMemoryStore(workspace)
        self._namespace_stores: dict[str, GraphMemoryStore] = {}
        self.context = ContextBuilder(workspace, memory=self.graph_memory)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Memory tools (graph-based associative memory)
        self.tools.register(FindMemoryCacheTool(self.graph_memory))
        self.tools.register(RecallRelatedTool(self.graph_memory))
        self.tools.register(GetMemoryCacheTool(self.graph_memory))
        self.tools.register(CreateMemoryTool(self.graph_memory))
        self.tools.register(UpdateMemoryTool(self.graph_memory))
        self.tools.register(DeleteMemoryTool(self.graph_memory))

    def _get_namespace_store(self, namespace: str) -> GraphMemoryStore:
        """Get or create a GraphMemoryStore for a memory namespace."""
        if namespace not in self._namespace_stores:
            ns_dir = self.workspace / "memory" / namespace
            ns_dir.mkdir(parents=True, exist_ok=True)
            self._namespace_stores[namespace] = GraphMemoryStore(self.workspace, memory_subdir=f"memory/{namespace}")
        return self._namespace_stores[namespace]

    def _swap_memory_store(self, namespace: str | None) -> GraphMemoryStore:
        """Swap memory tools to use a namespace store. Returns the previous store."""
        prev = self.graph_memory
        if not namespace:
            return prev
        store = self._get_namespace_store(namespace)
        self._set_memory_store(store)
        return prev

    def _set_memory_store(self, store: GraphMemoryStore) -> None:
        """Point all memory tools and context builder at the given store."""
        self.graph_memory = store
        self.context.memory = store
        for name in ("find_memory_cache", "recall_related", "get_memory_cache",
                      "create_memory", "update_memory", "delete_memory"):
            tool = self.tools.get(name)
            if tool and hasattr(tool, "_store"):
                tool._store = store

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        self.graph_memory.close()
        for store in self._namespace_stores.values():
            store.close()
        self._namespace_stores.clear()
        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).

        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."

        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into graph memory.

        Args:
            archive_all: If True, consolidate all messages and reset session (for /new command).
                       If False, only consolidate new messages since last consolidation.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly three keys:

1. "summary": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful for future recall.

2. "keywords": A list of 3-8 normalized keywords for the summary. Include WHO (people, teams), WHAT (topics, technologies), WHERE (locations), WHEN (date references if significant). Lowercase, no duplicates.

3. "facts": A list of long-term facts extracted from the conversation. Each fact is an object with "content" (the fact), "keywords" (list of keywords), and "type" ("entity" for people/projects/places, "concept" for ideas/preferences/decisions). Only include facts with lasting value. Return an empty list if no new facts.

Example facts:
{{"content": "User prefers dark mode", "keywords": ["user", "preferences"], "type": "entity"}}
{{"content": "Project uses Python 3.11 with FastAPI", "keywords": ["project", "python", "fastapi"], "type": "concept"}}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            # Store summary as event node
            if summary := result.get("summary"):
                keywords = result.get("keywords", [])
                if isinstance(keywords, list) and keywords:
                    try:
                        self.graph_memory.create_node(
                            content=summary,
                            keywords=keywords,
                            node_type="event",
                        )
                    except Exception as e:
                        logger.warning(f"Graph memory storage (summary) failed: {e}")

            # Store extracted facts as entity/concept nodes
            facts = result.get("facts", [])
            if isinstance(facts, list):
                for fact in facts:
                    if not isinstance(fact, dict) or not fact.get("content") or not fact.get("keywords"):
                        continue
                    try:
                        self.graph_memory.create_node(
                            content=fact["content"],
                            keywords=fact["keywords"],
                            node_type=fact.get("type", "entity"),
                        )
                    except Exception as e:
                        logger.warning(f"Graph memory storage (fact) failed: {e}")

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).

        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).

        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""

    # ── Streaming support (for ExtendChannel) ─────────────────────────────────

    async def _run_agent_loop_streaming(
        self, initial_messages: list[dict]
    ) -> AsyncIterator[str]:
        """Run agent loop: tool calls execute normally, final response streams.

        Yields text chunks from the final LLM response.
        """
        messages = initial_messages
        iteration = 0
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                # Final iteration — stream the response
                # Re-call with streaming (no tools, just generate)
                async for chunk in self.provider.chat_stream(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    yield chunk
                return

        # Hit max iterations — yield whatever we have
        yield "I've reached the maximum number of processing steps."

    async def process_streaming(
        self,
        content: str,
        session_key: str,
        channel: str = "extend",
        chat_id: str = "direct",
        recent_window: int = 8,
        summary_threshold: int = 20,
        extra_system_prompt: str | None = None,
        memory_namespace: str | None = None,
    ) -> AsyncIterator[str]:
        """Process a message with streaming response, using two-layer session context.

        Args:
            content: User message.
            session_key: Session identifier (e.g. "extend:conv_123").
            channel: Channel name for tool context.
            chat_id: Chat ID for tool context.
            recent_window: Number of recent conversation turns to keep in full.
            summary_threshold: Old messages to accumulate before generating summary.
            memory_namespace: If set, use isolated graph memory under this namespace.

        Yields:
            Text chunks of the assistant's response.
        """
        await self._connect_mcp()

        # Swap to namespace memory store if requested
        prev_store = self._swap_memory_store(memory_namespace)

        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(channel, chat_id)

        # Two-layer context: summaries + recent messages
        summary_text, recent_messages = session.get_context_messages(recent_window)

        # Build system prompt
        system_prompt = self.context.build_system_prompt()
        if extra_system_prompt:
            system_prompt += f"\n\n{extra_system_prompt}"
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # Inject summary layer
        if summary_text:
            messages.append({
                "role": "system",
                "content": f"## Conversation Summary\n\n{summary_text}",
            })

        # Recent full messages
        messages.extend(recent_messages)

        # Current user message
        messages.append({"role": "user", "content": content})

        # Stream the response, collecting full text
        full_response = []
        async for chunk in self._run_agent_loop_streaming(messages):
            full_response.append(chunk)
            yield chunk

        final_content = "".join(full_response)
        if not final_content:
            final_content = "I've completed processing but have no response to give."

        # Update session
        session.add_message("user", content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        # Check if summary generation is needed (background task)
        if session.needs_summary(recent_window, summary_threshold):
            asyncio.create_task(
                self._summarize_old_messages(session, recent_window)
            )

        # Restore original memory store
        if memory_namespace:
            self._set_memory_store(prev_store)

    async def _summarize_old_messages(
        self, session: Session, recent_window: int = 8
    ) -> None:
        """Generate a summary of old unsummarized messages and store it in session."""
        old_messages = session.get_unsummarized_old_messages(recent_window)
        if not old_messages:
            return

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            ts = m.get("timestamp", "?")[:16]
            lines.append(f"[{ts}] {m['role'].upper()}: {m['content']}")
        conversation = "\n".join(lines)

        if not conversation.strip():
            return

        prompt = (
            "Summarize this conversation segment in 2-5 concise sentences. "
            "Capture key topics, decisions, and any important context. "
            "Start with a timestamp range. Be factual and brief.\n\n"
            f"{conversation}"
        )

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a conversation summarizer. Be concise."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            summary = (response.content or "").strip()
            if summary:
                session.summaries.append(summary)
                session.mark_summarized(recent_window)
                self.sessions.save(session)
                logger.info(
                    f"Session summary generated for {session.key}: "
                    f"{len(session.summaries)} summaries, "
                    f"last_summarized={session.last_summarized}"
                )
        except Exception as e:
            logger.error(f"Session summary generation failed: {e}")
