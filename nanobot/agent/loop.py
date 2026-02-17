"""Agent loop: the core processing engine."""

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

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.graph_memory = GraphMemoryStore(workspace)
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
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
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

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
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
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
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
                final_content = response.content
                break

        return final_content, tools_used

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
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
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
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
    ) -> AsyncIterator[str]:
        """Process a message with streaming response, using two-layer session context.

        Args:
            content: User message.
            session_key: Session identifier (e.g. "extend:conv_123").
            channel: Channel name for tool context.
            chat_id: Chat ID for tool context.
            recent_window: Number of recent conversation turns to keep in full.
            summary_threshold: Old messages to accumulate before generating summary.

        Yields:
            Text chunks of the assistant's response.
        """
        await self._connect_mcp()

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
