"""Enhanced memory features: consolidation, contradiction detection, community detection."""

import math
import re
from typing import Any

from nanobot.agent.graph_memory import GraphMemoryStore

# ── Importance Scoring ────────────────────────────────────────────────────────

def compute_importance(node: dict[str, Any], half_life_days: float = 30.0) -> float:
    from datetime import datetime
    try:
        age = (datetime.now() - datetime.fromisoformat(node["created_at"])).days
    except (ValueError, TypeError):
        age = 0
    recency = 2.0 ** (-max(age, 0) / half_life_days)
    frequency = math.log2((node.get("access_count", 0) or 0) + 1)
    return node.get("weight", 1.0) * recency * (1 + frequency * 0.3)


# ── Memory Consolidation ─────────────────────────────────────────────────────

def _jaccard(a: set[str], b: set[str]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def consolidate_memories(store: GraphMemoryStore, threshold: float = 0.7) -> list[str]:
    conn = store._get_conn()
    nodes = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE type='event' ORDER BY weight DESC"
    ).fetchall()]
    merged: list[str] = []
    deleted: set[str] = set()

    for i, a in enumerate(nodes):
        if a["id"] in deleted:
            continue
        kws_a = set(a["keywords"].split(",")) - {""}
        if len(kws_a) < 2:
            continue
        for b in nodes[i + 1:]:
            if b["id"] in deleted:
                continue
            kws_b = set(b["keywords"].split(",")) - {""}
            if len(kws_b) < 2 or len(kws_a & kws_b) < 3:
                continue
            if _jaccard(kws_a, kws_b) < threshold:
                continue
            keep, drop = (a, b) if a["weight"] >= b["weight"] else (b, a)
            union_kws = list(kws_a | kws_b)
            merged_content = f"{keep['content']}\n---\n{drop['content']}"
            store.update_node(keep["id"], content=merged_content, keywords=union_kws)
            conn.execute("UPDATE OR IGNORE edges SET source_id=? WHERE source_id=?", (keep["id"], drop["id"]))
            conn.execute("UPDATE OR IGNORE edges SET target_id=? WHERE target_id=?", (keep["id"], drop["id"]))
            store.delete_node(drop["id"])
            deleted.add(drop["id"])
            merged.append(f"{drop['id']} → {keep['id']}")

    return merged


# ── Contradiction Detection ───────────────────────────────────────────────────

_NEG_RE = re.compile(
    r"\b(not|never|no longer|stopped|quit|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|没有|不再|停止|取消)\b",
    re.IGNORECASE,
)


def detect_contradictions(store: GraphMemoryStore, node_id: str) -> list[dict[str, Any]]:
    node = store.get_node(node_id)
    if not node:
        return []
    node_kws = [k for k in node["keywords"].split(",") if k]
    if not node_kws:
        return []

    conn = store._get_conn()
    candidates: dict[str, set[str]] = {}
    for kw in node_kws:
        for r in conn.execute(
            "SELECT node_id FROM keyword_map WHERE keyword=? AND node_id!=?", (kw, node_id)
        ).fetchall():
            candidates.setdefault(r["node_id"], set()).add(kw)

    node_neg = bool(_NEG_RE.search(node["content"]))
    results = []
    for cid, shared in candidates.items():
        if len(shared) < 2:
            continue
        c = store.get_node(cid)
        if not c:
            continue
        if node_neg != bool(_NEG_RE.search(c["content"])):
            results.append({
                "nodeA": node_id, "nodeB": cid,
                "sharedKeywords": list(shared),
                "hint": "Negation mismatch",
            })
    return results


# ── Community Detection (Label Propagation) ───────────────────────────────────

def detect_communities(store: GraphMemoryStore, max_iter: int = 20) -> dict[int, list[str]]:
    conn = store._get_conn()
    nodes = [r["id"] for r in conn.execute("SELECT id FROM nodes").fetchall()]
    if not nodes:
        return {}

    community = {nid: i for i, nid in enumerate(nodes)}

    edges = conn.execute("SELECT source_id, target_id, weight FROM edges").fetchall()
    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in nodes}
    for e in edges:
        adj[e["source_id"]].append((e["target_id"], e["weight"]))
        adj[e["target_id"]].append((e["source_id"], e["weight"]))

    import random
    for _ in range(max_iter):
        changed = False
        order = list(nodes)
        random.shuffle(order)
        for nid in order:
            if not adj[nid]:
                continue
            votes: dict[int, float] = {}
            for nb, w in adj[nid]:
                c = community[nb]
                votes[c] = votes.get(c, 0) + w
            best = max(votes, key=lambda c: votes[c])
            if best != community[nid]:
                community[nid] = best
                changed = True
        if not changed:
            break

    # Write back
    for nid, cid in community.items():
        conn.execute("UPDATE nodes SET community_id=? WHERE id=?", (cid, nid))
    conn.commit()

    result: dict[int, list[str]] = {}
    for nid, cid in community.items():
        result.setdefault(cid, []).append(nid)
    return result
