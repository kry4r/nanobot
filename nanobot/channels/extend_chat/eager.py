"""Eager reply: proactive nudges when the user types for too long."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import EagerReplyConfig

# Fallback nudge pool if LLM generation fails
_FALLBACK_NUDGES = [
    "在想什么呢~",
    "慢慢打 不着急哈",
    "嗯嗯 我在听",
    "想好了再说也行~",
    "？？",
    "打字中...我等你",
    "emmm",
    "你是不是在纠结怎么说 hhh",
]


class EagerReply:
    """Monitors client typing signals and sends proactive nudges.

    - On `typing_start`: starts a timer (typing_timeout_ms)
    - If timer expires without `typing_stop` or a message: sends a nudge
    - Max `max_eager_per_turn` nudges per turn, with `cooldown_ms` between them
    - Nudge phrases are LLM-generated per conversation, cached in a pool
    """

    def __init__(
        self,
        config: "EagerReplyConfig",
        agent: "AgentLoop",
        send_callback: Callable[[str, str], Awaitable[None]] | None = None,
    ):
        self._config = config
        self._agent = agent
        self._send = send_callback  # async fn(conversation_id, text)

        self._timer_tasks: dict[str, asyncio.Task] = {}  # conv_id → timer
        self._nudge_pools: dict[str, list[str]] = {}  # conv_id → cached phrases
        self._nudge_counts: dict[str, int] = {}  # conv_id → count this turn
        self._last_nudge_time: dict[str, float] = {}  # conv_id → monotonic time
        self._dispatching: set[str] = set()  # conv_ids currently in dispatch

    def set_send_callback(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Set or update the send callback."""
        self._send = callback

    def on_dispatch_start(self, conversation_id: str) -> None:
        """Called when orchestrator starts dispatching — suppress nudges."""
        self._dispatching.add(conversation_id)
        self._cancel_timer(conversation_id)

    def on_dispatch_end(self, conversation_id: str) -> None:
        """Called when orchestrator finishes dispatching — reset turn counter."""
        self._dispatching.discard(conversation_id)
        self._nudge_counts[conversation_id] = 0

    async def on_typing_start(self, conversation_id: str) -> None:
        """Client started typing — begin countdown."""
        if conversation_id in self._dispatching:
            return
        self._cancel_timer(conversation_id)
        self._timer_tasks[conversation_id] = asyncio.create_task(
            self._typing_timer(conversation_id)
        )

    def on_typing_stop(self, conversation_id: str) -> None:
        """Client stopped typing — cancel countdown."""
        self._cancel_timer(conversation_id)

    def on_message_received(self, conversation_id: str) -> None:
        """User sent a message — cancel any pending nudge timer."""
        self._cancel_timer(conversation_id)

    async def _typing_timer(self, conversation_id: str) -> None:
        """Wait for typing timeout, then attempt to send a nudge."""
        try:
            await asyncio.sleep(self._config.typing_timeout_ms / 1000.0)
            await self._maybe_nudge(conversation_id)
        except asyncio.CancelledError:
            pass

    async def _maybe_nudge(self, conversation_id: str) -> None:
        """Send a nudge if within limits."""
        if not self._send:
            return
        if conversation_id in self._dispatching:
            return

        # Check per-turn limit
        count = self._nudge_counts.get(conversation_id, 0)
        if count >= self._config.max_eager_per_turn:
            logger.debug(f"[eager:{conversation_id}] max nudges reached ({count})")
            return

        # Check cooldown
        now = time.monotonic()
        last = self._last_nudge_time.get(conversation_id, 0)
        if (now - last) * 1000 < self._config.cooldown_ms:
            logger.debug(f"[eager:{conversation_id}] cooldown active")
            return

        # Pick a nudge phrase
        phrase = await self._pick_nudge(conversation_id)
        if not phrase:
            return

        self._nudge_counts[conversation_id] = count + 1
        self._last_nudge_time[conversation_id] = now

        logger.info(f"[eager:{conversation_id}] sending nudge #{count + 1}: {phrase}")
        try:
            await self._send(conversation_id, phrase)
        except Exception:
            logger.exception(f"[eager:{conversation_id}] failed to send nudge")

    async def _pick_nudge(self, conversation_id: str) -> str | None:
        """Pick a nudge phrase from the pool, generating if needed."""
        pool = self._nudge_pools.get(conversation_id)
        if not pool:
            pool = await self._generate_nudge_pool(conversation_id)
            self._nudge_pools[conversation_id] = pool

        if not pool:
            return None

        phrase = random.choice(pool)
        return phrase

    async def _generate_nudge_pool(self, conversation_id: str) -> list[str]:
        """Generate nudge phrases via LLM, falling back to hardcoded pool."""
        try:
            prompt = (
                "生成8条简短的催促语句，用于聊天时对方正在打字但很久没发出来的场景。"
                "要求：口语化、轻松、不要有压迫感、每条10字以内。"
                "直接输出，每行一条，不要编号。"
            )
            full_text = ""
            session_key = f"extend-chat:{conversation_id}"
            async for chunk in self._agent.process_streaming(
                content=prompt,
                session_key=session_key,
                channel="extend-chat",
                chat_id=conversation_id,
            ):
                full_text += chunk

            lines = [line.strip() for line in full_text.strip().split("\n") if line.strip()]
            if len(lines) >= 3:
                logger.debug(f"[eager:{conversation_id}] generated {len(lines)} nudge phrases")
                return lines
        except Exception:
            logger.warning(f"[eager:{conversation_id}] LLM nudge generation failed, using fallback")

        return list(_FALLBACK_NUDGES)

    def _cancel_timer(self, conversation_id: str) -> None:
        """Cancel the typing timer for a conversation."""
        task = self._timer_tasks.pop(conversation_id, None)
        if task and not task.done():
            task.cancel()

    async def cleanup(self, conversation_id: str | None = None) -> None:
        """Cancel timers and clear state. If conv_id is None, clean all."""
        if conversation_id:
            self._cancel_timer(conversation_id)
            self._nudge_pools.pop(conversation_id, None)
            self._nudge_counts.pop(conversation_id, None)
            self._last_nudge_time.pop(conversation_id, None)
            self._dispatching.discard(conversation_id)
        else:
            for cid in list(self._timer_tasks):
                self._cancel_timer(cid)
            self._nudge_pools.clear()
            self._nudge_counts.clear()
            self._last_nudge_time.clear()
            self._dispatching.clear()
