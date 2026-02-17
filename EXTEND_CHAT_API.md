# Extend-Chat 接入文档

Nukara 后端替换 MCP-over-HTTP 为 extend-chat WebSocket/HTTP 直连。

## 服务地址

| 环境 | 地址 |
|------|------|
| 本地 | `http://localhost:8081` |
| Docker | `http://nanobot:8081` |

## 认证

**HTTP**: `Authorization: Bearer <token>`
**WebSocket**: `ws://host:8081/ws/chat?token=<token>`

token 在 nanobot config 的 `channels.extend_chat.auth_tokens` 中配置。空列表 = 无认证。

## 端点

| 路径 | 方法 | 用途 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat` | POST | 同步聊天（阻塞，返回完整回复） |
| `/ws/chat` | WS | 实时聊天（拟人化分句、打字指示器） |

---

## 一、HTTP POST /chat（简单模式）

适用于测试、proactive 触发等不需要拟人化效果的场景。

### 请求

```json
{
  "conversation_id": "conv_abc123",
  "robot_id": "companion",
  "content": { "type": "text", "text": "你好呀" }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| conversation_id | string | 是 | 会话 ID |
| robot_id | string | 否 | 默认 `"default"`，路由到对应 robot |
| content.text | string | 是 | 用户消息文本 |

### 响应 200

```json
{
  "conversation_id": "conv_abc123",
  "content": { "type": "text", "text": "你好！今天过得怎么样？" }
}
```

### 错误

| 状态码 | 场景 |
|--------|------|
| 401 | token 无效 |
| 400 | JSON 格式错误 / 空消息 / 未知 robot_id |

---

## 二、WebSocket /ws/chat（拟人化模式）

适用于正式聊天，支持消息聚合、分句发送、打字延迟、主动消息。

### 连接

```
ws://host:8081/ws/chat?token=<auth_token>
```

### 客户端 → 服务端

#### 发送消息

```json
{
  "type": "message",
  "conversation_id": "conv_abc123",
  "robot_id": "companion",
  "client_msg_id": "client_001",
  "content": { "type": "text", "text": "你好呀" }
}
```

#### 心跳

```json
{ "type": "ping" }
```

#### 打字状态（可选，触发 eager reply）

```json
{ "type": "typing_start", "conversation_id": "conv_abc123" }
{ "type": "typing_stop", "conversation_id": "conv_abc123" }
```

### 服务端 → 客户端

#### ack — 消息确认

```json
{
  "type": "ack",
  "client_msg_id": "client_001",
  "server_msg_id": "a1b2c3d4e5f6",
  "timestamp": 1708185600
}
```

#### typing — 打字指示器

```json
{ "type": "typing", "conversation_id": "conv_abc123", "is_typing": true }
```

#### bot_waiting — 聚合等待中

```json
{ "type": "bot_waiting", "conversation_id": "conv_abc123" }
```

用户快速连发多条消息时，聚合器缓冲后统一处理。此事件表示"已收到，正在等待更多输入"。

#### multi_reply_start — 分句回复开始

```json
{
  "type": "multi_reply_start",
  "conversation_id": "conv_abc123",
  "reply_group_id": "rg_001"
}
```

#### message — 单条分句消息（气泡）

```json
{
  "type": "message",
  "conversation_id": "conv_abc123",
  "msg_id": "m_001",
  "reply_group_id": "rg_001",
  "sequence": 0,
  "content": { "type": "text", "text": "你好！" },
  "timestamp": 1708185601
}
```

每条 message 前会有 typing 指示器 + 模拟打字延迟，模拟真人节奏。

#### multi_reply_end — 分句回复结束

```json
{
  "type": "multi_reply_end",
  "conversation_id": "conv_abc123",
  "reply_group_id": "rg_001",
  "count": 3
}
```

#### proactiveMessage — 主动消息

```json
{
  "type": "proactiveMessage",
  "conversation_id": "conv_abc123",
  "msg_id": "pm_001",
  "content": { "type": "text", "text": "在忙什么呢？" },
  "timestamp": 1708185700
}
```

用户长时间打字未发送时，eager reply 主动发送关怀消息。

#### pong — 心跳回复

```json
{ "type": "pong", "timestamp": 1708185600 }
```

#### error — 错误

```json
{ "type": "error", "message": "Unknown robot: xxx" }
```

---

## 三、多 Robot 配置

每个 robot 拥有独立的 workspace、记忆数据库、模型配置。

### nanobot config.json 示例

```json
{
  "channels": {
    "extend_chat": {
      "enabled": true,
      "port": 8081,
      "auth_tokens": ["<your-token>"],
      "idle_timeout_minutes": 30,
      "robots": {
        "companion": {
          "workspace": "~/.nanobot/robots/companion",
          "model": "xfyun/spark-max",
          "extra_system_prompt": "你是一个温暖的陪伴者",
          "idle_timeout_minutes": 30
        },
        "proactive": {
          "workspace": "~/.nanobot/robots/proactive",
          "model": "xfyun/spark-max",
          "extra_system_prompt": "你负责主动关怀"
        }
      }
    }
  }
}
```

不配置 `robots` 时，使用全局 AgentLoop 作为 `"default"` robot。

### 记忆隔离

每个 robot 的记忆存储在各自 workspace 下：
```
~/.nanobot/robots/companion/memory/graph.db
~/.nanobot/robots/proactive/memory/graph.db
```

### 记忆归档触发

| 触发条件 | 行为 |
|----------|------|
| WebSocket 断连 | 归档该连接所有活跃会话 |
| 空闲超时 | 超过 `idle_timeout_minutes` 未活动的会话自动归档 |

归档 = `_consolidate_memory(archive_all=True)` → 提取摘要/实体写入 graph.db → 清空 session。

---

## 四、Nukara 后端迁移指南

### 替换前（MCP-over-HTTP）

```
Nukara Go → HTTP POST → nanobot:9090 (MCP JSON-RPC)
```

### 替换后（extend-chat）

```
Nukara Go → WebSocket → nanobot:8081/ws/chat (extend-chat)
Nukara Go → HTTP POST → nanobot:8081/chat (同步回退)
```

### Go 客户端改动要点

1. **删除** `internal/agent/mcp.go`（MCP 客户端）
2. **改写** `internal/agent/agent.go`：
   - WebSocket 连接到 `ws://nanobot:8081/ws/chat?token=xxx`
   - 发送 `{"type":"message", "robot_id":"companion", ...}`
   - 监听 `message` 事件收集分句回复
   - 监听 `typing` 事件驱动 UI 打字指示器
3. **改写** `internal/api/chat_flow.go`：
   - 移除 MCP tool call 逻辑
   - 直接转发 WS 事件到客户端
4. **Proactive 服务**：
   - 使用 `POST /chat` 同步接口（无需拟人化）
   - `robot_id: "proactive"`

### 环境变量

```bash
# 替换
NUKARA_NANOBOT_URL=http://nanobot:8081  # 改端口
# 新增
NUKARA_NANOBOT_WS_URL=ws://nanobot:8081/ws/chat
NUKARA_NANOBOT_TOKEN=<your-token>
```

### Docker Compose

```yaml
nanobot:
  image: nanobot:latest
  command: nanobot gateway
  ports:
    - "8081:8081"
  volumes:
    - nanobot-config:/root/.nanobot
```

---

## 五、消息时序图

```
Client                    Nukara Go                 nanobot:8081
  |                          |                          |
  |-- user msg ------------->|                          |
  |                          |-- WS: type=message ----->|
  |                          |<-- WS: type=ack ---------|
  |                          |<-- WS: bot_waiting ------|  (聚合中)
  |                          |<-- WS: typing(true) -----|
  |                          |<-- WS: multi_reply_start-|
  |                          |<-- WS: typing(true) -----|
  |                          |   (模拟打字延迟)          |
  |                          |<-- WS: message seq=0 ----|  "你好！"
  |                          |<-- WS: typing(true) -----|
  |                          |   (模拟打字延迟)          |
  |                          |<-- WS: message seq=1 ----|  "今天怎么样？"
  |                          |<-- WS: multi_reply_end --|
  |                          |<-- WS: typing(false) ----|
  |<-- push to iOS ----------|                          |
```
