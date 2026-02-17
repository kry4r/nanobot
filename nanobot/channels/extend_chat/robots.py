"""Robot manager: per-robot AgentLoop instances for extend-chat."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


class RobotManager:
    """Manages per-robot AgentLoop instances."""

    def __init__(self):
        self._agents: dict[str, AgentLoop] = {}

    def register(self, robot_id: str, agent: AgentLoop) -> None:
        self._agents[robot_id] = agent
        logger.info(f"Registered robot '{robot_id}'")

    def get(self, robot_id: str) -> AgentLoop | None:
        return self._agents.get(robot_id)

    def get_or_default(self, robot_id: str) -> AgentLoop | None:
        return self._agents.get(robot_id) or self._agents.get("default")

    @property
    def robot_ids(self) -> list[str]:
        return list(self._agents.keys())

    async def close_all(self) -> None:
        for agent in self._agents.values():
            agent.graph_memory.close()
        self._agents.clear()
