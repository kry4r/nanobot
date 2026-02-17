"""Response splitter: breaks LLM streaming output into sentence-level messages."""

from __future__ import annotations

import asyncio
import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import SplitterConfig


class ResponseSplitter:
    """Buffers streaming tokens and emits complete sentences via an asyncio.Queue.

    Usage (producer-consumer pattern):
        splitter = ResponseSplitter(config)

        # Consumer task reads sentences:
        async def consumer():
            while True:
                sentence = await splitter.sentences.get()
                if sentence is None:  # sentinel
                    break
                send(sentence)

        # Producer feeds chunks from LLM stream:
        async for chunk in llm_stream:
            await splitter.feed(chunk)
        await splitter.flush()
    """

    def __init__(self, config: "SplitterConfig"):
        self._config = config
        self._buffer = ""
        self.sentences: asyncio.Queue[str | None] = asyncio.Queue()

        # Build regex from configured delimiters
        escaped = [re.escape(d) if d != "\n\n" else r"\n\n" for d in config.delimiters]
        self._pattern = re.compile(f'(?:{"|".join(escaped)})')

    async def feed(self, chunk: str) -> None:
        """Feed a streaming token chunk. Complete sentences are pushed to the queue."""
        self._buffer += chunk
        await self._try_emit()

    async def flush(self) -> None:
        """Flush remaining buffer as the final sentence, then send sentinel."""
        remaining = self._buffer.strip()
        if remaining:
            await self.sentences.put(remaining)
        self._buffer = ""
        await self.sentences.put(None)  # sentinel signals end

    async def _try_emit(self) -> None:
        """Try to extract complete sentences from the buffer."""
        while True:
            match = self._pattern.search(self._buffer)
            if not match:
                break

            # Split at the delimiter (include delimiter in the sentence)
            end = match.end()
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:]

            if not sentence:
                continue

            # Short sentence merging: if too short, keep in buffer for next merge
            if len(sentence) < self._config.min_sentence_length:
                self._buffer = sentence + self._buffer
                break

            await self.sentences.put(sentence)

    def calculate_delay(self, text: str) -> float:
        """Calculate humanlike typing delay in seconds for a sentence.

        Formula: len(text) * speed_per_char ± jitter, clamped to [min_delay, max_delay].
        """
        base_ms = len(text) * self._config.typing_speed_ms_per_char
        jitter_range = base_ms * self._config.jitter_ratio
        jitter_ms = random.uniform(-jitter_range, jitter_range)
        delay_ms = base_ms + jitter_ms
        delay_ms = max(self._config.min_delay_ms, min(delay_ms, self._config.max_delay_ms))
        return delay_ms / 1000.0
