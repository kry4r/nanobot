"""Bearer token authentication for the Extend channel."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from nanobot.config.schema import ExtendConfig


def verify_token(request: web.Request, config: "ExtendConfig") -> bool:
    """Verify Bearer token from request headers.

    Returns True if:
    - No auth_tokens configured (open access), or
    - Authorization header contains a valid token.
    """
    if not config.auth_tokens:
        return True

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header[7:]
    return any(
        hmac.compare_digest(token, valid_token)
        for valid_token in config.auth_tokens
    )


def verify_ws_token(token: str | None, config: "ExtendConfig") -> bool:
    """Verify token for WebSocket connections.

    Token can be passed as:
    - Query parameter: /ws/chat?token=xxx
    - First message after connect
    """
    if not config.auth_tokens:
        return True

    if not token:
        return False

    return any(
        hmac.compare_digest(token, valid_token)
        for valid_token in config.auth_tokens
    )
