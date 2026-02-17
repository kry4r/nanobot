"""Graph memory tools for associative recall."""

from typing import Any

from nanobot.agent.graph_memory import GraphMemoryStore
from nanobot.agent.tools.base import Tool


class FindMemoryCacheTool(Tool):
    """Search memory graph via recall chain: anchor → time-backtrack → lateral context."""

    def __init__(self, store: GraphMemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "find_memory_cache"

    @property
    def description(self) -> str:
        return (
            "Search the memory graph for related memories. "
            "Finds concept/entity anchors via keyword match, then walks their event "
            "timeline (most recent first) with lateral context for each event. "
            "Returns a structured recall chain, not a flat list."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or phrase to find in memory",
                },
                "max_events": {
                    "type": "integer",
                    "description": "Max events in timeline (default 10)",
                    "minimum": 1,
                    "maximum": 30,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_events: int = 10, **kwargs: Any) -> str:
        try:
            result = self._store.recall_chain(query, max_events=max_events)
            return result.format_for_llm()
        except Exception as e:
            return f"Error searching memory: {e}"


class RecallRelatedTool(Tool):
    """Spreading activation from a specific event to discover distant associations."""

    def __init__(self, store: GraphMemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "recall_related"

    @property
    def description(self) -> str:
        return (
            "From a specific event node, spread activation through the memory graph "
            "to discover distant associations. Use after find_memory_cache to dig "
            "deeper into a particular event's connections. Simulates 'what else is "
            "related to this?' thinking."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "Node ID of the event to spread from",
                },
                "depth": {
                    "type": "integer",
                    "description": "Activation spread depth (default 2, max 3)",
                    "minimum": 1,
                    "maximum": 3,
                },
                "max_nodes": {
                    "type": "integer",
                    "description": "Max associated nodes to return (default 20)",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["event_id"],
        }

    async def execute(
        self, event_id: str, depth: int = 2, max_nodes: int = 20, **kwargs: Any
    ) -> str:
        try:
            node = self._store.get_node(event_id)
            if not node:
                return f"Node [{event_id}] not found"

            associations = self._store.spreading_activation(
                seed_ids=[event_id],
                max_depth=min(depth, 3),
                max_nodes=max_nodes,
            )

            if not associations:
                return f"No distant associations found from [{event_id}]"

            lines = [f"扩散联想 (from: {node['content'][:60]}):\n"]
            for assoc_node, score in associations:
                lines.append(
                    f"  - [{assoc_node['id']}] {assoc_node['content']} "
                    f"(激活度: {score:.2f}, type: {assoc_node['type']})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error in recall_related: {e}"


class GetMemoryCacheTool(Tool):
    """Retrieve specific memory nodes by ID."""

    def __init__(self, store: GraphMemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "get_memory_cache"

    @property
    def description(self) -> str:
        return (
            "Get the full content of specific memory nodes by ID. "
            "Pass a single node_id or a list of node_ids. "
            "Returns node details with their direct neighbors."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Single node ID to retrieve",
                },
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node IDs to retrieve as a chain",
                },
            },
        }

    async def execute(
        self,
        node_id: str | None = None,
        node_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            ids = node_ids or ([node_id] if node_id else [])
            if not ids:
                return "Error: provide node_id or node_ids"

            nodes = self._store.get_chain(ids)
            if not nodes:
                return "No nodes found for the given IDs."

            # Bump weights on access
            self._store._bump_weights([n["id"] for n in nodes])

            parts = []
            for node in nodes:
                neighbors = self._store.get_neighbors(node["id"])
                neighbor_summary = ", ".join(
                    f"{n['content']}({n['relation']})" for n in neighbors[:8]
                )
                parts.append(
                    f"## [{node['id']}] {node['type']} — {node['created_at'][:16]}\n"
                    f"{node['content']}\n"
                    f"Keywords: {node['keywords']}\n"
                    f"Weight: {node['weight']:.1f}\n"
                    f"Connected to: {neighbor_summary or 'none'}"
                )
            return "\n\n".join(parts)
        except Exception as e:
            return f"Error retrieving memory: {e}"


class CreateMemoryTool(Tool):
    """Create a new memory node in the graph."""

    def __init__(self, store: GraphMemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "create_memory"

    @property
    def description(self) -> str:
        return (
            "Create a new memory node. Automatically manages keyword indexing and "
            "concept promotion (keywords referenced by ≥2 events become concept hubs). "
            "Use type 'event' for things that happened, 'concept' for topics/ideas, "
            "'entity' for people/projects/places."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content description",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords/tags for this memory (will be normalized)",
                },
                "type": {
                    "type": "string",
                    "enum": ["event", "concept", "entity"],
                    "description": "Node type (default: event)",
                },
                "linked_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of existing node IDs to link to",
                },
            },
            "required": ["content", "keywords"],
        }

    async def execute(
        self,
        content: str,
        keywords: list[str],
        type: str = "event",
        linked_to: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            node_id = self._store.create_node(
                content=content,
                keywords=keywords,
                node_type=type,
                linked_to=linked_to,
            )
            return f"Created {type} node [{node_id}] with keywords: {', '.join(keywords)}"
        except Exception as e:
            return f"Error creating memory: {e}"


class UpdateMemoryTool(Tool):
    """Update an existing memory node."""

    def __init__(self, store: GraphMemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "update_memory"

    @property
    def description(self) -> str:
        return "Update the content and/or keywords of an existing memory node."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "ID of the node to update",
                },
                "content": {
                    "type": "string",
                    "description": "New content (omit to keep existing)",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New keywords (omit to keep existing)",
                },
            },
            "required": ["node_id"],
        }

    async def execute(
        self,
        node_id: str,
        content: str | None = None,
        keywords: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if self._store.update_node(node_id, content=content, keywords=keywords):
                return f"Updated node [{node_id}]"
            return f"Node [{node_id}] not found"
        except Exception as e:
            return f"Error updating memory: {e}"


class DeleteMemoryTool(Tool):
    """Delete a memory node and its edges."""

    def __init__(self, store: GraphMemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "delete_memory"

    @property
    def description(self) -> str:
        return "Delete a memory node and all its edges from the graph."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "ID of the node to delete",
                },
            },
            "required": ["node_id"],
        }

    async def execute(self, node_id: str, **kwargs: Any) -> str:
        try:
            if self._store.delete_node(node_id):
                return f"Deleted node [{node_id}] and its edges"
            return f"Node [{node_id}] not found"
        except Exception as e:
            return f"Error deleting memory: {e}"
