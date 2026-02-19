"""Pre-retrieval: fetch relevant memories before LLM generation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

# Single-character Chinese function words safe to strip
# (almost never part of content words)
_STOP_CHARS = set(
    "的了吗呢吧啊呀嘛哦嗯啦喽么着你我他她它您们还也都就才又再很太最更是有在不没"
)


def _extract_keywords(query: str) -> list[str]:
    """Extract search keywords from conversational Chinese.

    Strips punctuation and function characters, returns overlapping bigrams
    as keyword candidates for FTS5 search.
    """
    text = re.sub(r"[^\u4e00-\u9fff]", "", query)
    cleaned = "".join(c for c in text if c not in _STOP_CHARS)
    if len(cleaned) < 2:
        return [query]
    # Overlapping bigrams, deduplicated, preserving order
    return list(dict.fromkeys(cleaned[i : i + 2] for i in range(len(cleaned) - 1)))


def retrieve_memory_context(
    agent: "AgentLoop", namespace: str, query: str, max_events: int = 5
) -> str | None:
    """Retrieve impression + relevant memories for injection into system prompt.

    Returns formatted text or None if no memories found.
    """
    try:
        store = agent._get_namespace_store(namespace)
    except Exception:
        logger.warning(f"[memory_retriever] failed to get store for {namespace}")
        return None

    parts: list[str] = []

    # 1. Impression node (always inject if exists)
    try:
        impression_hits = store.search("用户印象", limit=1)
        if impression_hits:
            body = impression_hits[0].get("body") or impression_hits[0].get("content", "")
            if body.strip():
                parts.append(f"[用户印象]\n{body.strip()}")
    except Exception:
        logger.debug("[memory_retriever] impression search failed")

    # 2. Extract keywords for better FTS5 matching on Chinese text
    keywords = _extract_keywords(query)
    logger.debug(f"[memory_retriever] keywords from '{query}': {keywords}")

    # 3. Recall chain — try with each keyword until we get results
    recall_found = False
    for kw in keywords:
        if recall_found:
            break
        try:
            recall = store.recall_chain(kw, max_events=max_events)
            if recall and (recall.timeline or recall.associations):
                formatted = recall.format_for_llm()
                if formatted.strip() and "No memories found" not in formatted:
                    parts.append(f"[相关记忆]\n{formatted}")
                    recall_found = True
        except Exception:
            pass

    # 4. Direct search fallback — search each keyword, merge results
    if not recall_found:
        seen: set[str] = set()
        lines: list[str] = []
        for kw in keywords:
            try:
                hits = store.search(kw, limit=max_events)
                for h in hits:
                    content = h.get("content", "")
                    if "用户印象" in content or content in seen:
                        continue
                    seen.add(content)
                    body = h.get("body", "")
                    line = f"- {content}"
                    if body:
                        line += f"（{body[:80]}）"
                    lines.append(line)
            except Exception:
                pass
        if lines:
            parts.append("[相关记忆]\n" + "\n".join(lines[:max_events]))

    if not parts:
        return None

    result = "\n\n".join(parts)
    logger.info(f"[memory_retriever] injecting {len(parts)} sections ({len(result)} chars)")
    return result
