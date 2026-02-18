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
        system_context = body.get("system_context")

        if not content.strip():
            return web.json_response({"error": "Empty message"}, status=400)

        agent = robot_manager.get_or_default(robot_id)
        if not agent:
            return web.json_response({"error": f"Unknown robot: {robot_id}"}, status=400)

        session_key = f"extend-chat:{robot_id}:{conv_id}"

        # Build extra_system_prompt from system_context
        extra_prompt = None
        if isinstance(system_context, dict):
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
                extra_prompt = "\n".join(ctx_parts)

        # Parse memory namespace from conversation_id
        memory_namespace = None
        parts = conv_id.split(":")
        if len(parts) >= 3 and parts[0] == "nukara":
            memory_namespace = f"{parts[1]}_{parts[2]}"

        chunks: list[str] = []
        async for chunk in agent.process_streaming(
            content=content,
            session_key=session_key,
            channel="http-chat",
            chat_id=conv_id,
            recent_window=config.recent_window,
            summary_threshold=config.summary_threshold,
            extra_system_prompt=extra_prompt,
            memory_namespace=memory_namespace,
        ):
            chunks.append(chunk)

        full_text = "".join(chunks)

        # Extract clean text from LLM output (strip tool calls)
        import re
        # Try to extract message content from tool call parameters
        match = re.search(r'<parameter name=["\'](?:message|content)["\']>(.*?)</parameter>', full_text, re.DOTALL)
        if match:
            full_text = match.group(1).strip()
        else:
            # Fallback: strip entire tool call blocks
            full_text = re.sub(r'<[^>]+:tool_call>.*?</[^>]+:tool_call>', '', full_text, flags=re.DOTALL)
            full_text = full_text.strip()

        return web.json_response({
            "conversation_id": conv_id,
            "content": {"type": "text", "text": full_text},
        })

    return handler
