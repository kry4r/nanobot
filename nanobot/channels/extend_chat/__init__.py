"""Extend-Chat channel: humanlike WebSocket/HTTP gateway for chat apps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.extend_chat.server import create_app

if TYPE_CHECKING:
    from nanobot.channels.extend_chat.robots import RobotManager
    from nanobot.config.schema import ExtendChatConfig


class ExtendChatChannel(BaseChannel):
    """Humanlike WebSocket/HTTP gateway channel.

    Wraps the extend channel with sentence splitting, typing delays,
    imperfection injection, message aggregation, and eager replies.
    """

    name = "extend_chat"

    def __init__(
        self, config: "ExtendChatConfig", bus: MessageBus, robot_manager: "RobotManager"
    ):
        super().__init__(config, bus)
        self._robot_manager = robot_manager
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._ws_manager = None

    async def start(self) -> None:
        """Start the aiohttp server."""
        self._running = True
        self._app, self._ws_manager = create_app(self.config, self._robot_manager)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._ws_manager.start_idle_checker()

        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()
        logger.info(f"Extend-chat channel listening on {self.config.host}:{self.config.port}")

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
        await self._robot_manager.close_all()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Extend-chat channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Not used — ExtendChatChannel bypasses MessageBus for streaming."""
        pass
