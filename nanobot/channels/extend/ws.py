"""WebSocket connection handler for the Extend channel."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web
from loguru import logger

from nanobot.channels.extend.auth import verify_ws_token
from nanobot.channels.extend.models import (
    AckResponse,
    ClientMessage,
    ContentPayload,
    ErrorResponse,
    FullMessageResponse,
    PongResponse,
    StreamChunkResponse,
    StreamEndResponse,
    StreamStartResponse,
    TypingResponse,
)

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import ExtendConfig


class WebSocketManager:
    """Manages WebSocket connections and message processing."""

    def __init__(self, config: "ExtendConfig", agent: "AgentLoop"):
        self.config = config
        self.agent = agent
        self._connections: set[web.WebSocketResponse] = set()
        self._conv_locks: dict[str, asyncio.Lock] = {}

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection at /ws/chat."""
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        # Auth: check query param token
        token = request.query.get("token")
        if not verify_ws_token(token, self.config):
            await ws.send_json(ErrorResponse(message="Unauthorized").model_dump())
            await ws.close(code=aiohttp.WSCloseCode.POLICY_VIOLATION, message=b"Unauthorized")
            return ws

        self._connections.add(ws)
        logger.info(f"Extend WS connected (total: {len(self._connections)})")

        try:
            async for raw_msg in ws:
                if raw_msg.type == aiohttp.WSMsgType.TEXT:
                    asyncio.create_task(self._handle_text_message(ws, raw_msg.data))
                elif raw_msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning(f"Extend WS error: {ws.exception()}")
        finally:
            self._connections.discard(ws)
            logger.info(f"Extend WS disconnected (total: {len(self._connections)})")

        return ws

    async def _handle_text_message(self, ws: web.WebSocketResponse, data: str) -> None:
        """Parse and route a single text message from the client."""
        try:
            import json
            raw = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            await self._send(ws, ErrorResponse(message="Invalid JSON"))
            return

        msg_type = raw.get("type", "")

        if msg_type == "ping":
            await self._send(ws, PongResponse())
            return

        if msg_type != "message":
            await self._send(ws, ErrorResponse(message=f"Unknown message type: {msg_type}"))
            return

        try:
            client_msg = ClientMessage(**raw)
        except Exception as e:
            await self._send(ws, ErrorResponse(message=f"Invalid message format: {e}"))
            return

        if not client_msg.conversation_id:
            await self._send(ws, ErrorResponse(message="conversation_id is required"))
            return

        text = client_msg.content.text if client_msg.content else ""
        if not text.strip():
            await self._send(ws, ErrorResponse(message="Empty message"))
            return

        # Send ack
        msg_id = uuid.uuid4().hex[:12]
        await self._send(ws, AckResponse(client_msg_id=client_msg.client_msg_id, server_msg_id=msg_id))

        # Process with per-conversation lock
        conv_id = client_msg.conversation_id
        lock = self._conv_locks.setdefault(conv_id, asyncio.Lock())

        async with lock:
            await self._process_message(ws, conv_id, msg_id, text)

    async def _process_message(
        self, ws: web.WebSocketResponse, conv_id: str, msg_id: str, text: str
    ) -> None:
        """Stream a response for a single message."""
        session_key = f"extend:{conv_id}"

        # Typing indicator
        await self._send(ws, TypingResponse(conversation_id=conv_id))

        # Stream start
        await self._send(ws, StreamStartResponse(conversation_id=conv_id, msg_id=msg_id))

        full_text: list[str] = []
        try:
            async for chunk in self.agent.process_streaming(
                content=text,
                session_key=session_key,
                channel="extend",
                chat_id=conv_id,
                recent_window=self.config.recent_window,
                summary_threshold=self.config.summary_threshold,
            ):
                full_text.append(chunk)
                await self._send(ws, StreamChunkResponse(msg_id=msg_id, text=chunk))
        except Exception as e:
            logger.error(f"Extend streaming error: {e}")
            await self._send(ws, ErrorResponse(message=f"Processing error: {e}"))
            return

        # Stream end
        await self._send(ws, StreamEndResponse(msg_id=msg_id))

        # Typing off
        await self._send(ws, TypingResponse(conversation_id=conv_id, is_typing=False))

        # Full message
        complete = "".join(full_text)
        await self._send(ws, FullMessageResponse(
            conversation_id=conv_id,
            msg_id=msg_id,
            content=ContentPayload(text=complete),
        ))

    async def _send(self, ws: web.WebSocketResponse, msg: Any) -> None:
        """Send a Pydantic model as JSON to a WebSocket, ignoring closed connections."""
        try:
            if not ws.closed:
                await ws.send_json(msg.model_dump())
        except (ConnectionResetError, RuntimeError):
            pass

    async def close_all(self) -> None:
        """Close all active WebSocket connections."""
        for ws in list(self._connections):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        self._conv_locks.clear()
