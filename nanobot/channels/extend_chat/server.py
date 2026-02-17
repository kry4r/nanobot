"""aiohttp application for the Extend-Chat humanlike channel."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from nanobot.channels.extend.auth import verify_token
from nanobot.channels.extend_chat.ws import HumanlikeWebSocketManager

if TYPE_CHECKING:
    from nanobot.channels.extend_chat.robots import RobotManager
    from nanobot.config.schema import ExtendChatConfig


def create_app(
    config: "ExtendChatConfig", robot_manager: "RobotManager"
) -> tuple[web.Application, HumanlikeWebSocketManager]:
    """Create the aiohttp application with routes.

    Returns:
        Tuple of (app, ws_manager) so the channel can manage lifecycle.
    """
    ws_manager = HumanlikeWebSocketManager(config, robot_manager)

    app = web.Application()
    app["config"] = config

    app.router.add_get("/health", _health_handler)
    app.router.add_post("/chat", _make_chat_handler(config, robot_manager))
    app.router.add_get("/ws/chat", ws_manager.handle_ws)

    return app, ws_manager


async def _health_handler(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({"status": "ok", "channel": "extend-chat"})


def _make_chat_handler(config: "ExtendChatConfig", robot_manager: "RobotManager"):
    """Create POST /chat handler (synchronous fallback, no aggregation/splitting)."""

    async def handler(request: web.Request) -> web.Response:
        if not verify_token(request, config):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            body = await request.json()
        except (json.JSONDecodeError, TypeError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        content = (
            body.get("content", {}).get("text", "")
            if isinstance(body.get("content"), dict)
            else ""
        )
        conv_id = body.get("conversation_id", "default")
        robot_id = body.get("robot_id", "default")

        if not content.strip():
            return web.json_response({"error": "Empty message"}, status=400)

        agent = robot_manager.get_or_default(robot_id)
        if not agent:
            return web.json_response({"error": f"Unknown robot: {robot_id}"}, status=400)

        session_key = f"extend-chat:{robot_id}:{conv_id}"

        chunks: list[str] = []
        async for chunk in agent.process_streaming(
            content=content,
            session_key=session_key,
            channel="extend-chat",
            chat_id=conv_id,
            recent_window=config.recent_window,
            summary_threshold=config.summary_threshold,
        ):
            chunks.append(chunk)

        full_text = "".join(chunks)
        return web.json_response({
            "conversation_id": conv_id,
            "content": {"type": "text", "text": full_text},
        })

    return handler
