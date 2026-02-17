"""Protocol models for the Extend WebSocket channel."""

import time
import uuid

from pydantic import BaseModel, Field


class ContentPayload(BaseModel):
    """Message content payload."""
    type: str = "text"
    text: str = ""


class ClientMessage(BaseModel):
    """Client → Server message."""
    type: str  # "message" | "ping"
    conversation_id: str = ""
    client_msg_id: str = ""
    content: ContentPayload | None = None
    robot_id: str = "default"
    system_context: dict | None = None


class AckResponse(BaseModel):
    """Acknowledgement sent after receiving a client message."""
    type: str = "ack"
    client_msg_id: str = ""
    server_msg_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class TypingResponse(BaseModel):
    """Typing indicator."""
    type: str = "typing"
    conversation_id: str = ""
    is_typing: bool = True


class StreamStartResponse(BaseModel):
    """Marks the beginning of a streaming response."""
    type: str = "stream_start"
    conversation_id: str = ""
    msg_id: str = ""


class StreamChunkResponse(BaseModel):
    """A single chunk of streamed text."""
    type: str = "stream_chunk"
    msg_id: str = ""
    text: str = ""


class StreamEndResponse(BaseModel):
    """Marks the end of a streaming response."""
    type: str = "stream_end"
    msg_id: str = ""


class FullMessageResponse(BaseModel):
    """Complete message (sent after stream ends)."""
    type: str = "message"
    conversation_id: str = ""
    msg_id: str = ""
    content: ContentPayload = Field(default_factory=ContentPayload)
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class ErrorResponse(BaseModel):
    """Error message."""
    type: str = "error"
    message: str = ""


class PongResponse(BaseModel):
    """Pong reply to client ping."""
    type: str = "pong"
    timestamp: int = Field(default_factory=lambda: int(time.time()))
