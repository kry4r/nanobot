"""WebSocket connection handler for the Extend-Chat humanlike channel."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web
from loguru import logger

from nanobot.channels.extend.auth import verify_ws_token
from nanobot.channels.extend_chat.models import (
    AckResponse,
    BotWaitingResponse,
    ClientMessage,
    ContentPayload,
    ErrorResponse,
    PongResponse,
    ProactiveMessageResponse,
)
from nanobot.channels.extend_chat.orchestrator import ChatOrchestrator

if TYPE_CHECKING:
    from nanobot.channels.extend_chat.robots import RobotManager
    from nanobot.config.schema import ExtendChatConfig


class HumanlikeWebSocketManager:
    """Manages WebSocket connections with multi-robot routing."""

    def __init__(self, config: "ExtendChatConfig", robot_manager: "RobotManager"):
        self.config = config
        self._robot_manager = robot_manager
        self._connections: set[web.WebSocketResponse] = set()
        self._orchestrators: dict[str, ChatOrchestrator] = {}

        # Conversation tracking
        self._conv_ws: dict[str, web.WebSocketResponse] = {}
        self._conv_robot: dict[str, str] = {}
        self._conv_last_activity: dict[str, float] = {}
        self._conv_system_context: dict[str, dict] = {}
        self._idle_check_task: asyncio.Task | None = None

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection at /ws/chat."""
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        # Auth
        token = request.query.get("token")
        if not verify_ws_token(token, self.config):
            await ws.send_json(ErrorResponse(message="Unauthorized").model_dump())
            await ws.close(code=aiohttp.WSCloseCode.POLICY_VIOLATION, message=b"Unauthorized")
            return ws

        self._connections.add(ws)
        logger.info(f"Extend-chat WS connected (total: {len(self._connections)})")

        try:
            async for raw_msg in ws:
                if raw_msg.type == aiohttp.WSMsgType.TEXT:
                    asyncio.create_task(self._handle_text_message(ws, raw_msg.data))
                elif raw_msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning(f"Extend-chat WS error: {ws.exception()}")
        finally:
            self._connections.discard(ws)
            dead_convs = [cid for cid, w in self._conv_ws.items() if w is ws]
            for cid in dead_convs:
                self._conv_ws.pop(cid, None)
                robot_id = self._conv_robot.pop(cid, "default")
                self._conv_last_activity.pop(cid, None)
                self._conv_system_context.pop(cid, None)
                asyncio.create_task(self._archive_conversation(cid, robot_id))
            logger.info(f"Extend-chat WS disconnected (total: {len(self._connections)})")

        return ws

    async def _handle_text_message(self, ws: web.WebSocketResponse, data: str) -> None:
        """Parse and route a single text message from the client."""
        try:
            raw = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            await self._send(ws, ErrorResponse(message="Invalid JSON"))
            return

        msg_type = raw.get("type", "")

        if msg_type == "ping":
            await self._send(ws, PongResponse())
            return

        if msg_type in ("typing_start", "typing_stop"):
            conv_id = raw.get("conversation_id", "")
            robot_id = self._conv_robot.get(conv_id, raw.get("robot_id", "default"))
            orch = self._get_orchestrator(robot_id)
            if conv_id and orch and orch.eager:
                if msg_type == "typing_start":
                    await orch.eager.on_typing_start(conv_id)
                else:
                    orch.eager.on_typing_stop(conv_id)
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

        conv_id = client_msg.conversation_id
        robot_id = client_msg.robot_id or "default"

        # Route to robot orchestrator
        orchestrator = self._get_orchestrator(robot_id)
        if not orchestrator:
            await self._send(ws, ErrorResponse(message=f"Unknown robot: {robot_id}"))
            return

        # Track conversation state
        self._conv_ws[conv_id] = ws
        self._conv_robot[conv_id] = robot_id
        self._conv_last_activity[conv_id] = time.time()

        # Store system_context per conversation (updated on each message)
        sys_ctx = client_msg.system_context
        if sys_ctx:
            self._conv_system_context[conv_id] = sys_ctx

        # Send ack
        msg_id = uuid.uuid4().hex[:12]
        await self._send(ws, AckResponse(
            client_msg_id=client_msg.client_msg_id, server_msg_id=msg_id
        ))

        # Notify eager reply that a message arrived
        if orchestrator.eager:
            orchestrator.eager.on_message_received(conv_id)

        # Feed into aggregator
        if self.config.humanlike.aggregator.enabled:
            aggregator = orchestrator.get_aggregator(
                conv_id,
                dispatch_fn=self._make_dispatch_fn(ws, conv_id, orchestrator),
            )
            await aggregator.add_message(text)
            await self._send(ws, BotWaitingResponse(conversation_id=conv_id))
        else:
            await orchestrator.dispatch(
                ws, conv_id, text,
                send_fn=lambda msg: self._send(ws, msg),
                system_context=sys_ctx,
            )

    def _make_dispatch_fn(
        self, ws: web.WebSocketResponse, conv_id: str, orchestrator: ChatOrchestrator
    ):
        """Create the dispatch callback for the aggregator."""
        async def dispatch(aggregated_text: str) -> None:
            sys_ctx = self._conv_system_context.get(conv_id)
            await orchestrator.dispatch(
                ws, conv_id, aggregated_text,
                send_fn=lambda msg: self._send(ws, msg),
                system_context=sys_ctx,
            )
        return dispatch

    def _get_orchestrator(self, robot_id: str) -> ChatOrchestrator | None:
        """Get or lazily create a per-robot orchestrator."""
        if robot_id in self._orchestrators:
            return self._orchestrators[robot_id]
        agent = self._robot_manager.get_or_default(robot_id)
        if not agent:
            return None
        orch = ChatOrchestrator(
            config=self.config.humanlike,
            agent=agent,
            extend_config=self.config,
            robot_id=robot_id,
        )
        self._orchestrators[robot_id] = orch
        return orch

    async def _send_eager_nudge(self, conversation_id: str, text: str) -> None:
        """Send a proactive nudge message to the right WS connection."""
        ws = self._conv_ws.get(conversation_id)
        if not ws or ws.closed:
            return
        await self._send(ws, ProactiveMessageResponse(
            conversation_id=conversation_id,
            content=ContentPayload(text=text),
        ))

    async def _send(self, ws: web.WebSocketResponse, msg: Any) -> None:
        """Send a Pydantic model as JSON to a WebSocket."""
        try:
            if not ws.closed:
                await ws.send_json(msg.model_dump())
        except (ConnectionResetError, RuntimeError):
            pass

    async def _archive_conversation(self, conv_id: str, robot_id: str) -> None:
        """Archive a conversation's memory to the robot's graph DB."""
        agent = self._robot_manager.get_or_default(robot_id)
        if not agent:
            return
        session_key = f"extend-chat:{robot_id}:{conv_id}"
        session = agent.sessions.get_or_create(session_key)
        if session.messages:
            # Parse memory namespace from conversation_id
            memory_namespace = None
            parts = conv_id.split(":")
            if len(parts) >= 3 and parts[0] == "nukara":
                memory_namespace = f"{parts[1]}_{parts[2]}"

            prev_store = agent._swap_memory_store(memory_namespace)
            try:
                await agent._consolidate_memory(session, archive_all=True)
                session.clear()
                agent.sessions.save(session)
                logger.info(f"Archived conversation {conv_id} for robot {robot_id}")
            except Exception:
                logger.exception(f"Failed to archive {conv_id} for robot {robot_id}")
            finally:
                if memory_namespace:
                    agent._set_memory_store(prev_store)

    def start_idle_checker(self) -> None:
        """Start the background idle-conversation checker."""
        if not self._idle_check_task:
            self._idle_check_task = asyncio.create_task(self._idle_check_loop())

    async def _idle_check_loop(self) -> None:
        """Periodically archive idle conversations."""
        timeout = self.config.idle_timeout_minutes * 60
        try:
            while True:
                await asyncio.sleep(60)
                now = time.time()
                idle = [
                    cid for cid, ts in self._conv_last_activity.items()
                    if now - ts > timeout
                ]
                for cid in idle:
                    robot_id = self._conv_robot.pop(cid, "default")
                    self._conv_ws.pop(cid, None)
                    self._conv_last_activity.pop(cid, None)
                    self._conv_system_context.pop(cid, None)
                    asyncio.create_task(self._archive_conversation(cid, robot_id))
        except asyncio.CancelledError:
            pass

    async def close_all(self) -> None:
        """Close all connections and clean up."""
        if self._idle_check_task:
            self._idle_check_task.cancel()
            try:
                await self._idle_check_task
            except asyncio.CancelledError:
                pass
        for orch in self._orchestrators.values():
            await orch.cleanup()
        for ws in list(self._connections):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        self._conv_ws.clear()
        self._conv_system_context.clear()
        self._orchestrators.clear()
