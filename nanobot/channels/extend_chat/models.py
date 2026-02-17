"""Protocol models for the Extend-Chat humanlike WebSocket channel."""

import time
import uuid

from pydantic import BaseModel, Field

# Re-export base models from extend channel
from nanobot.channels.extend.models import (  # noqa: F401
    AckResponse,
    ClientMessage,
    ContentPayload,
    ErrorResponse,
    PongResponse,
    TypingResponse,
)


class MultiReplyStartResponse(BaseModel):
    """Marks the beginning of a multi-message reply group."""
    type: str = "multi_reply_start"
    conversation_id: str = ""
    reply_group_id: str = ""


class MultiReplyEndResponse(BaseModel):
    """Marks the end of a multi-message reply group."""
    type: str = "multi_reply_end"
    conversation_id: str = ""
    reply_group_id: str = ""
    count: int = 0


class SplitMessageResponse(BaseModel):
    """Individual message within a multi-reply group (each renders as a separate bubble)."""
    type: str = "message"
    conversation_id: str = ""
    msg_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    reply_group_id: str = ""
    sequence: int = 0
    content: ContentPayload = Field(default_factory=ContentPayload)
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class BotWaitingResponse(BaseModel):
    """Sent during message aggregation to indicate bot is listening."""
    type: str = "bot_waiting"
    conversation_id: str = ""


class ProactiveMessageResponse(BaseModel):
    """Unsolicited bot message (eager reply / nudge)."""
    type: str = "proactiveMessage"
    conversation_id: str = ""
    msg_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: ContentPayload = Field(default_factory=ContentPayload)
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class ClientTypingMessage(BaseModel):
    """Client → Server typing indicator."""
    type: str  # "typing_start" | "typing_stop"
    conversation_id: str = ""
