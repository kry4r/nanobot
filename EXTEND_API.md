# Extend Channel API

nanobot 的 WebSocket/HTTP 网关通道，允许外部应用通过标准协议接入 nanobot 的 AI 能力，支持流式响应。

## 配置

在 `~/.nanobot/config.json` 中启用：

```json
{
  "channels": {
    "extend": {
      "enabled": true,
      "host": "0.0.0.0",
      "port": 8080,
      "auth_tokens": ["your-secret-token"],
      "summary_threshold": 20,
      "recent_window": 8
    }
  }
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用 |
| `host` | string | `"0.0.0.0"` | 监听地址 |
| `port` | int | `8080` | 监听端口 |
| `auth_tokens` | string[] | `[]` | Bearer token 列表，空则不鉴权 |
| `allow_from` | string[] | `[]` | 允许的来源（预留） |
| `summary_threshold` | int | `20` | 旧消息堆积多少条后生成摘要 |
| `recent_window` | int | `8` | 完整保留最近几轮对话 |

启动：`nanobot gateway`，Extend 通道会在配置端口上监听。

---

## 认证

### HTTP

请求头携带 Bearer token：

```
Authorization: Bearer your-secret-token
```

### WebSocket

连接时通过 query parameter 传递：

```
ws://localhost:8080/ws/chat?token=your-secret-token
```

如果 `auth_tokens` 为空，则不需要认证（开放访问）。Token 验证使用 HMAC 时间安全比较。

---

## HTTP API

### GET /health

健康检查。

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok"}
```

### POST /chat

同步请求-响应模式。收集完整 LLM 响应后一次性返回。

```bash
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_abc123",
    "content": {"type": "text", "text": "你好"}
  }'
```

响应：

```json
{
  "conversation_id": "conv_abc123",
  "content": {
    "type": "text",
    "text": "你好！有什么可以帮你的吗？"
  }
}
```

错误响应：

| 状态码 | 说明 |
|--------|------|
| 401 | `{"error": "Unauthorized"}` — token 无效 |
| 400 | `{"error": "Invalid JSON"}` — 请求体格式错误 |
| 400 | `{"error": "Empty message"}` — 消息内容为空 |

---

## WebSocket API

连接端点：`GET /ws/chat?token=xxx`

连接建立后，服务端自动发送 30 秒心跳（aiohttp heartbeat）。

### 客户端 → 服务端

#### 发送消息

```json
{
  "type": "message",
  "conversation_id": "conv_abc123",
  "client_msg_id": "client_001",
  "content": {
    "type": "text",
    "text": "帮我写一首诗"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 固定 `"message"` |
| `conversation_id` | string | 是 | 会话 ID，建议使用 UUID |
| `client_msg_id` | string | 否 | 客户端消息 ID，用于 ack 关联 |
| `content.type` | string | 否 | 内容类型，目前仅支持 `"text"` |
| `content.text` | string | 是 | 消息文本 |

#### Ping

```json
{"type": "ping"}
```

### 服务端 → 客户端

一条消息的完整响应流程：

```
ack → typing(on) → stream_start → stream_chunk × N → stream_end → typing(off) → message
```

#### 1. ack — 确认收到

```json
{
  "type": "ack",
  "client_msg_id": "client_001",
  "server_msg_id": "a1b2c3d4e5f6",
  "timestamp": 1739808000
}
```

#### 2. typing — 输入指示器

```json
{
  "type": "typing",
  "conversation_id": "conv_abc123",
  "is_typing": true
}
```

响应结束后会发送 `is_typing: false`。

#### 3. stream_start — 流式开始

```json
{
  "type": "stream_start",
  "conversation_id": "conv_abc123",
  "msg_id": "a1b2c3d4e5f6"
}
```

#### 4. stream_chunk — 文本片段

```json
{
  "type": "stream_chunk",
  "msg_id": "a1b2c3d4e5f6",
  "text": "春风"
}
```

多个 chunk 按顺序到达，客户端拼接即可。

#### 5. stream_end — 流式结束

```json
{
  "type": "stream_end",
  "msg_id": "a1b2c3d4e5f6"
}
```

#### 6. message — 完整消息

流式结束后，服务端额外发送一条包含完整文本的消息：

```json
{
  "type": "message",
  "conversation_id": "conv_abc123",
  "msg_id": "a1b2c3d4e5f6",
  "content": {
    "type": "text",
    "text": "春风又绿江南岸，明月何时照我还。"
  },
  "timestamp": 1739808005
}
```

#### 7. pong

```json
{
  "type": "pong",
  "timestamp": 1739808000
}
```

#### 8. error

```json
{
  "type": "error",
  "message": "conversation_id is required"
}
```

---

## 会话管理

### conversation_id

- 客户端提供，建议使用 UUID
- 同一 `conversation_id` 的消息共享对话历史
- 不同 `conversation_id` 完全隔离
- Session key 格式：`extend:{conversation_id}`

### 两层 Context 机制

为避免长对话撑爆 LLM context window，采用滑动窗口 + 摘要压缩：

```
LLM Context = [系统提示] + [摘要层] + [完整层] + [当前消息]
```

- 完整层：最近 `recent_window` 轮对话（默认 8 轮 = 16 条消息），原样传入 LLM
- 摘要层：超出完整层的旧消息，堆积到 `summary_threshold`（默认 20 条）后，由 LLM 压缩为 2-5 句摘要

摘要在后台异步生成，不阻塞当前响应。

### 断线重连

客户端重连同一 `conversation_id` 时，session 从磁盘恢复（JSONL 持久化），包括历史摘要和完整消息，对话无缝继续。进程重启后同样可恢复。

---

## 并发模型

- 不同 `conversation_id` 的请求完全并行处理
- 同一 `conversation_id` 的并发请求通过 `asyncio.Lock` 串行化，避免 session 状态冲突
- 每个 WebSocket 消息通过 `asyncio.create_task()` 独立处理，不阻塞连接上的其他消息

---

## 客户端示例

### Python (websockets)

```python
import asyncio
import json
import websockets

async def chat():
    uri = "ws://localhost:8080/ws/chat?token=your-secret-token"
    async with websockets.connect(uri) as ws:
        # 发送消息
        await ws.send(json.dumps({
            "type": "message",
            "conversation_id": "conv_001",
            "client_msg_id": "msg_001",
            "content": {"type": "text", "text": "你好"}
        }))

        # 接收响应流
        while True:
            data = json.loads(await ws.recv())
            if data["type"] == "stream_chunk":
                print(data["text"], end="", flush=True)
            elif data["type"] == "message":
                print()  # 完整消息到达，结束
                break
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                break

asyncio.run(chat())
```

### JavaScript

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/chat?token=your-secret-token");

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "message",
    conversation_id: "conv_001",
    client_msg_id: "msg_001",
    content: { type: "text", text: "你好" }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case "ack":
      console.log("Message acknowledged:", data.server_msg_id);
      break;
    case "stream_chunk":
      process.stdout.write(data.text);  // 或 DOM 追加
      break;
    case "stream_end":
      console.log("\n--- stream complete ---");
      break;
    case "message":
      console.log("Full response:", data.content.text);
      break;
    case "error":
      console.error("Error:", data.message);
      break;
  }
};
```

### curl (HTTP 模式)

```bash
# 单次请求-响应
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"test","content":{"type":"text","text":"hello"}}'

# 健康检查
curl http://localhost:8080/health
```

### websocat (调试)

```bash
websocat "ws://localhost:8080/ws/chat?token=your-secret-token"
# 输入：
{"type":"message","conversation_id":"test","content":{"type":"text","text":"hi"}}
```
