"""aiohttp application for the Extend channel."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from nanobot.channels.extend.auth import verify_token
from nanobot.channels.extend.ws import WebSocketManager

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import ExtendConfig


def create_app(config: "ExtendConfig", agent: "AgentLoop") -> tuple[web.Application, WebSocketManager]:
    """Create the aiohttp application with routes.

    Returns:
        Tuple of (app, ws_manager) so the channel can manage lifecycle.
    """
    ws_manager = WebSocketManager(config, agent)

    app = web.Application()
    app["config"] = config
    app["agent"] = agent

    # Routes
    app.router.add_get("/health", _health_handler)
    app.router.add_post("/chat", _make_chat_handler(config, agent))
    app.router.add_get("/ws/chat", ws_manager.handle_ws)

    return app, ws_manager


async def _health_handler(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({"status": "ok"})


def _make_chat_handler(config: "ExtendConfig", agent: "AgentLoop"):
    """Create POST /chat handler (synchronous request-response fallback)."""

    async def handler(request: web.Request) -> web.Response:
        if not verify_token(request, config):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            body = await request.json()
        except (json.JSONDecodeError, TypeError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        content = body.get("content", {}).get("text", "") if isinstance(body.get("content"), dict) else ""
        conv_id = body.get("conversation_id", "default")

        if not content.strip():
            return web.json_response({"error": "Empty message"}, status=400)

        session_key = f"extend:{conv_id}"

        # Collect full streaming response
        chunks: list[str] = []
        async for chunk in agent.process_streaming(
            content=content,
            session_key=session_key,
            channel="extend",
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
