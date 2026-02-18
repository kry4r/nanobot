"""Chat orchestrator: pipeline coordinator for humanlike chat responses."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.channels.extend_chat.aggregator import MessageAggregator
from nanobot.channels.extend_chat.eager import EagerReply
from nanobot.channels.extend_chat.imperfection import ImperfectionInjector
from nanobot.channels.extend_chat.models import (
    ContentPayload,
    MultiReplyEndResponse,
    MultiReplyStartResponse,
    SplitMessageResponse,
    TypingResponse,
)
from nanobot.channels.extend_chat.prompts import HUMANLIKE_PROMPT
from nanobot.channels.extend_chat.splitter import ResponseSplitter

if TYPE_CHECKING:
    from aiohttp import web

    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import ExtendChatConfig, HumanlikeChatConfig


class ChatOrchestrator:
    """Coordinates the humanlike chat pipeline for a single WS connection.

    Pipeline: Aggregator → LLM streaming → Splitter → Imperfection → WS send
    Side-channel: EagerReply monitors typing signals
    """

    def __init__(
        self,
        config: "HumanlikeChatConfig",
        agent: "AgentLoop",
        extend_config: "ExtendChatConfig",
        robot_id: str = "default",
    ):
        self._config = config
        self._agent = agent
        self._extend_config = extend_config
        self._robot_id = robot_id

        # Imperfection injector (shared across conversations)
        self._imperfection: ImperfectionInjector | None = None
        if config.imperfection.enabled and config.imperfection.post_processing:
            self._imperfection = ImperfectionInjector(config.imperfection)

        # Extra system prompt for humanlike style
        self._extra_prompt: str | None = None
        if config.imperfection.enabled and config.imperfection.prompt_injection:
            self._extra_prompt = HUMANLIKE_PROMPT

        # Eager reply (shared, one per orchestrator)
        self._eager: EagerReply | None = None
        if config.eager_reply.enabled:
            self._eager = EagerReply(config.eager_reply, agent)

        # Per-conversation aggregators
        self._aggregators: dict[str, MessageAggregator] = {}
        self._conv_locks: dict[str, asyncio.Lock] = {}

    def get_aggregator(
        self,
        conversation_id: str,
        dispatch_fn: Callable[[str], Awaitable[None]],
    ) -> MessageAggregator:
        """Get or create a per-conversation aggregator."""
        if conversation_id not in self._aggregators:
            self._aggregators[conversation_id] = MessageAggregator(
                config=self._config.aggregator,
                dispatch_callback=dispatch_fn,
            )
        return self._aggregators[conversation_id]

    @property
    def eager(self) -> EagerReply | None:
        return self._eager

    async def dispatch(
        self,
        ws: "web.WebSocketResponse",
        conversation_id: str,
        aggregated_text: str,
        send_fn: Callable[[Any], Awaitable[None]],
        system_context: dict | None = None,
    ) -> None:
        """Run the full pipeline: LLM stream → split → imperfect → send."""
        lock = self._conv_locks.setdefault(conversation_id, asyncio.Lock())

        async with lock:
            await self._dispatch_locked(ws, conversation_id, aggregated_text, send_fn, system_context)

    async def _dispatch_locked(
        self,
        ws: "web.WebSocketResponse",
        conversation_id: str,
        aggregated_text: str,
        send_fn: Callable[[Any], Awaitable[None]],
        system_context: dict | None = None,
    ) -> None:
        """Pipeline execution under conversation lock."""
        reply_group_id = uuid.uuid4().hex[:12]
        session_key = f"extend-chat:{self._robot_id}:{conversation_id}"

        # Notify eager reply
        if self._eager:
            self._eager.on_dispatch_start(conversation_id)

        # Typing on
        await send_fn(TypingResponse(conversation_id=conversation_id))

        # Multi-reply start
        await send_fn(MultiReplyStartResponse(
            conversation_id=conversation_id,
            reply_group_id=reply_group_id,
        ))

        # Create a fresh splitter for this dispatch
        splitter = ResponseSplitter(self._config.splitter)
        sentence_count = 0

        # Consumer task: read sentences from splitter queue → process → send
        async def sentence_consumer():
            nonlocal sentence_count
            while True:
                sentence = await splitter.sentences.get()
                if sentence is None:
                    break

                # Apply imperfection post-processing
                if self._imperfection:
                    sentence = self._imperfection.process(sentence)

                sentence_count += 1

                # Typing indicator before each bubble
                await send_fn(TypingResponse(conversation_id=conversation_id))

                # Simulate typing delay
                delay = splitter.calculate_delay(sentence)
                await asyncio.sleep(delay)

                # Send the split message
                await send_fn(SplitMessageResponse(
                    conversation_id=conversation_id,
                    reply_group_id=reply_group_id,
                    sequence=sentence_count - 1,
                    content=ContentPayload(text=sentence),
                ))

        consumer_task = asyncio.create_task(sentence_consumer())

        # Build extra_system_prompt: merge humanlike prompt + system_context
        extra_prompt = self._extra_prompt or ""
        if system_context:
            ctx_parts = []
            if name := system_context.get("bot_name"):
                ctx_parts.append(f"你的名字是{name}。")
            if persona := system_context.get("persona"):
                ctx_parts.append(f"角色设定：{persona}")
            if style := system_context.get("speaking_style"):
                ctx_parts.append(f"说话风格：{style}")
            if traits := system_context.get("traits"):
                if isinstance(traits, list):
                    ctx_parts.append(f"性格特征：{'、'.join(traits)}")
            if bg := system_context.get("background"):
                ctx_parts.append(f"背景故事：{bg}")
            if gender := system_context.get("gender"):
                ctx_parts.append(f"性别：{gender}")
            if ctx_parts:
                ctx_text = "\n".join(ctx_parts)
                extra_prompt = f"{ctx_text}\n\n{extra_prompt}" if extra_prompt else ctx_text

        # Parse memory namespace from conversation_id (nukara:{userID}:{botID}:{convID})
        memory_namespace = None
        parts = conversation_id.split(":")
        if len(parts) >= 3 and parts[0] == "nukara":
            memory_namespace = f"{parts[1]}_{parts[2]}"

        # Producer: stream from LLM → feed to splitter
        try:
            async for chunk in self._agent.process_streaming(
                content=aggregated_text,
                session_key=session_key,
                channel="extend-chat",
                chat_id=conversation_id,
                recent_window=self._extend_config.recent_window,
                summary_threshold=self._extend_config.summary_threshold,
                extra_system_prompt=extra_prompt or None,
                memory_namespace=memory_namespace,
            ):
                await splitter.feed(chunk)

            await splitter.flush()
            await consumer_task
        except Exception:
            logger.exception(f"[orchestrator:{conversation_id}] pipeline error")
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

        # Multi-reply end
        await send_fn(MultiReplyEndResponse(
            conversation_id=conversation_id,
            reply_group_id=reply_group_id,
            count=sentence_count,
        ))

        # Typing off
        await send_fn(TypingResponse(conversation_id=conversation_id, is_typing=False))

        # Notify eager reply
        if self._eager:
            self._eager.on_dispatch_end(conversation_id)

    async def cleanup(self) -> None:
        """Cancel all aggregators and eager reply timers."""
        for agg in self._aggregators.values():
            agg.cancel()
        self._aggregators.clear()
        self._conv_locks.clear()
        if self._eager:
            await self._eager.cleanup()
