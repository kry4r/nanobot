"""Message aggregator: buffers rapid consecutive user messages before dispatching."""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import AggregatorConfig


class _State(enum.Enum):
    IDLE = "idle"
    AGGREGATING = "aggregating"
    DISPATCHING = "dispatching"


class MessageAggregator:
    """Per-conversation message aggregator with dynamic timeout.

    State machine:
        IDLE → add_message → AGGREGATING (start timer)
        AGGREGATING → add_message → reset timer, adjust timeout
        AGGREGATING → timer expires → DISPATCHING (callback)
        AGGREGATING → max messages reached → DISPATCHING
        DISPATCHING → done → IDLE (process any queued messages)
    """

    def __init__(
        self,
        config: "AggregatorConfig",
        dispatch_callback: Callable[[str], Awaitable[None]],
    ):
        self._config = config
        self._dispatch_callback = dispatch_callback
        self._state = _State.IDLE
        self._buffer: list[str] = []
        self._pending: list[str] = []  # messages arriving during DISPATCHING
        self._timer_task: asyncio.Task | None = None
        self._last_msg_time: float = 0.0
        self._current_timeout_ms: float = config.default_timeout_ms

    @property
    def state(self) -> str:
        return self._state.value

    def add_message(self, text: str) -> None:
        """Add a message to the aggregation buffer (non-blocking).

        During DISPATCHING, messages are queued for the next round.
        """
        now = time.monotonic()

        if self._state == _State.DISPATCHING:
            self._pending.append(text)
            logger.debug(f"Aggregator: queued message for next round ({len(self._pending)} pending)")
            return

        self._buffer.append(text)
        interval_ms = (now - self._last_msg_time) * 1000 if self._last_msg_time else float("inf")
        self._last_msg_time = now

        # Dynamic timeout adjustment
        self._current_timeout_ms = self._config.default_timeout_ms

        if interval_ms < self._config.fast_typing_threshold_ms:
            # User is typing fast — extend the window
            self._current_timeout_ms = self._config.fast_typing_extension_ms

        # If the message ends with sentence-ending punctuation, shorten the wait
        if text.rstrip() and text.rstrip()[-1] in "。！？.!?\n":
            self._current_timeout_ms = max(
                self._config.complete_sentence_reduction_ms,
                self._current_timeout_ms - self._config.complete_sentence_reduction_ms,
            )

        # Check max messages
        if len(self._buffer) >= self._config.max_messages:
            logger.debug("Aggregator: max messages reached, dispatching now")
            self._cancel_timer()
            asyncio.get_event_loop().create_task(self._dispatch())
            return

        # (Re)start timer
        if self._state == _State.IDLE:
            self._state = _State.AGGREGATING
            logger.debug("Aggregator: IDLE → AGGREGATING")

        self._cancel_timer()
        self._timer_task = asyncio.get_event_loop().create_task(self._wait_and_dispatch())

    def _cancel_timer(self) -> None:
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            self._timer_task = None

    async def _wait_and_dispatch(self) -> None:
        """Wait for the dynamic timeout, then dispatch."""
        timeout_s = min(self._current_timeout_ms, self._config.max_window_ms) / 1000.0
        try:
            await asyncio.sleep(timeout_s)
            await self._dispatch()
        except asyncio.CancelledError:
            pass

    async def _dispatch(self) -> None:
        """Aggregate buffer and invoke the dispatch callback."""
        if not self._buffer:
            self._state = _State.IDLE
            return

        self._state = _State.DISPATCHING
        aggregated = "\n".join(self._buffer)
        self._buffer.clear()
        self._last_msg_time = 0.0
        logger.debug(f"Aggregator: DISPATCHING ({len(aggregated)} chars)")

        try:
            await self._dispatch_callback(aggregated)
        except Exception as e:
            logger.error(f"Aggregator dispatch error: {e}")
        finally:
            self._state = _State.IDLE
            logger.debug("Aggregator: DISPATCHING → IDLE")

            # Process any messages that arrived during dispatch
            if self._pending:
                queued = self._pending.copy()
                self._pending.clear()
                for msg in queued:
                    self.add_message(msg)

    def cancel(self) -> None:
        """Cancel any pending timer (e.g. on disconnect)."""
        self._cancel_timer()
        self._buffer.clear()
        self._pending.clear()
        self._state = _State.IDLE
