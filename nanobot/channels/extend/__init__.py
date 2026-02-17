"""Extend channel: WebSocket/HTTP gateway for external apps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.extend.server import create_app

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import ExtendConfig


class ExtendChannel(BaseChannel):
    """WebSocket/HTTP gateway channel for external app integration.

    Embeds an aiohttp server that provides:
    - /ws/chat  — WebSocket endpoint with streaming responses
    - /chat     — HTTP POST endpoint (synchronous fallback)
    - /health   — Health check
    """

    name = "extend"

    def __init__(self, config: "ExtendConfig", bus: MessageBus, agent: "AgentLoop"):
        super().__init__(config, bus)
        self._agent = agent
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._ws_manager = None

    async def start(self) -> None:
        """Start the aiohttp server."""
        self._running = True
        self._app, self._ws_manager = create_app(self.config, self._agent)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()
        logger.info(f"Extend channel listening on {self.config.host}:{self.config.port}")

        # Keep running until stopped
        try:
            while self._running:
                import asyncio
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the server and close connections."""
        self._running = False
        if self._ws_manager:
            await self._ws_manager.close_all()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Extend channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Not used — ExtendChannel bypasses MessageBus for streaming."""
        pass
