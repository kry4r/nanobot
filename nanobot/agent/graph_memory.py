"""Graph-based associative memory system using SQLite + FTS5.

Two-phase recall: anchor + time-chain backtracking, then spreading activation.
"""

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id         TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    content    TEXT NOT NULL,
    keywords   TEXT NOT NULL DEFAULT '',
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL DEFAULT 'related',
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS keyword_map (
    keyword    TEXT NOT NULL,
    node_id    TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    PRIMARY KEY (keyword, node_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_keyword_map_kw ON keyword_map(keyword);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED, content,
    content=nodes, content_rowid=rowid,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS nodes_fts_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, content)
    VALUES (new.rowid, new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, content)
    VALUES ('delete', old.rowid, old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS nodes_fts_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, content)
    VALUES ('delete', old.rowid, old.id, old.content);
    INSERT INTO nodes_fts(rowid, id, content)
    VALUES (new.rowid, new.id, new.content);
END;
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RecallFrame:
    """A single event in the recall timeline with its lateral context."""
    event: dict
    context: list[dict] = field(default_factory=list)
    relevance: float = 1.0


@dataclass
class RecallResult:
    """Structured recall output: anchors → timeline → associations."""
    query: str
    anchors: list[dict] = field(default_factory=list)
    timeline: list[RecallFrame] = field(default_factory=list)
    associations: list[tuple[dict, float]] = field(default_factory=list)

    def format_for_llm(self) -> str:
        if not self.timeline and not self.associations:
            return f"No memories found for: {self.query}"

        parts = []
        anchor_names = ", ".join(a["content"] for a in self.anchors) if self.anchors else self.query
        parts.append(f"回忆主题：{anchor_names}\n")

        if self.timeline:
            parts.append("时间线：")
            for frame in self.timeline:
                ts = frame.event.get("created_at", "?")[:16]
                parts.append(f"[{ts}] {frame.event['content']}")
                if frame.context:
                    ctx = ", ".join(n["content"] for n in frame.context[:8])
                    parts.append(f"  ├── 关联：{ctx}")

        if self.associations:
            parts.append("\n扩散联想：")
            for node, score in self.associations[:10]:
                parts.append(f"  - {node['content']} (激活度: {score:.2f})")

        return "\n".join(parts)


# ── Core store ────────────────────────────────────────────────────────────────

class GraphMemoryStore:
    """Graph-structured associative memory backed by SQLite."""

    def __init__(self, workspace: Path, half_life_days: float = 30.0, memory_subdir: str = "memory"):
        self.memory_dir = ensure_dir(workspace / memory_subdir)
        self.db_path = self.memory_dir / "graph.db"
        self.half_life_days = half_life_days
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    # ── Connection ────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        try:
            conn.executescript(_FTS_SQL)
        except sqlite3.OperationalError as e:
            # FTS5 may not be available on all builds
            logger.warning(f"FTS5 setup failed (content fallback disabled): {e}")
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _gen_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _normalize_keyword(kw: str) -> str:
        return kw.strip().lower()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    def _time_decay(self, created_at: str) -> float:
        """Exponential decay: 2^(-age_days / half_life_days)."""
        try:
            age = (datetime.now() - datetime.fromisoformat(created_at)).days
        except (ValueError, TypeError):
            return 0.5
        return 2.0 ** (-max(age, 0) / self.half_life_days)

    # ── CRUD ──────────────────────────────────────────────

    def create_node(
        self,
        content: str,
        keywords: list[str],
        node_type: str = "event",
        linked_to: list[str] | None = None,
    ) -> str:
        """Create a node. Auto-manages keyword_map and concept promotion."""
        conn = self._get_conn()
        node_id = self._gen_id()
        now = self._now()
        norm_kws = [self._normalize_keyword(k) for k in keywords if k.strip()]
        kw_str = ",".join(norm_kws)

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO nodes (id, type, content, keywords, weight, created_at, updated_at) "
                "VALUES (?,?,?,?,1.0,?,?)",
                (node_id, node_type, content, kw_str, now, now),
            )

            # Write keyword_map entries
            for kw in norm_kws:
                conn.execute(
                    "INSERT OR IGNORE INTO keyword_map (keyword, node_id) VALUES (?,?)",
                    (kw, node_id),
                )

            # Check concept promotion for each keyword
            for kw in norm_kws:
                self._maybe_promote_concept(conn, kw, now)

            # Explicit links
            if linked_to:
                for target_id in linked_to:
                    exists = conn.execute(
                        "SELECT 1 FROM nodes WHERE id=?", (target_id,)
                    ).fetchone()
                    if exists:
                        conn.execute(
                            "INSERT OR IGNORE INTO edges "
                            "(source_id, target_id, relation, weight, created_at) "
                            "VALUES (?,?,'related',1.0,?)",
                            (node_id, target_id, now),
                        )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        logger.debug(f"Created {node_type} node [{node_id}] keywords={norm_kws}")
        return node_id

    def _maybe_promote_concept(
        self, conn: sqlite3.Connection, keyword: str, now: str
    ) -> None:
        """If keyword is referenced by ≥2 nodes and no concept exists, create one."""
        # Count references in keyword_map
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM keyword_map WHERE keyword=?", (keyword,)
        ).fetchone()
        if row["cnt"] < 2:
            return

        # Check if concept already exists
        existing = conn.execute(
            "SELECT id FROM nodes WHERE type='concept' AND content=?", (keyword,)
        ).fetchone()
        if existing:
            concept_id = existing["id"]
        else:
            concept_id = self._gen_id()
            conn.execute(
                "INSERT INTO nodes (id, type, content, keywords, weight, created_at, updated_at) "
                "VALUES (?,?,?,?,1.0,?,?)",
                (concept_id, "concept", keyword, keyword, now, now),
            )
            logger.debug(f"Promoted keyword '{keyword}' to concept [{concept_id}]")

        # Link all referencing nodes to the concept
        referencing = conn.execute(
            "SELECT node_id FROM keyword_map WHERE keyword=?", (keyword,)
        ).fetchall()
        for ref in referencing:
            if ref["node_id"] != concept_id:
                conn.execute(
                    "INSERT OR IGNORE INTO edges "
                    "(source_id, target_id, relation, weight, created_at) "
                    "VALUES (?,?,'about',1.0,?)",
                    (ref["node_id"], concept_id, now),
                )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._get_conn().execute(
            "SELECT * FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_node(
        self,
        node_id: str,
        content: str | None = None,
        keywords: list[str] | None = None,
    ) -> bool:
        conn = self._get_conn()
        node = self.get_node(node_id)
        if not node:
            return False

        now = self._now()
        new_content = content if content is not None else node["content"]
        new_kws = (
            [self._normalize_keyword(k) for k in keywords if k.strip()]
            if keywords is not None
            else None
        )
        new_kw_str = ",".join(new_kws) if new_kws is not None else node["keywords"]

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "UPDATE nodes SET content=?, keywords=?, updated_at=? WHERE id=?",
                (new_content, new_kw_str, now, node_id),
            )

            if new_kws is not None:
                # Rebuild keyword_map for this node
                conn.execute("DELETE FROM keyword_map WHERE node_id=?", (node_id,))
                # Remove about edges from this node
                conn.execute(
                    "DELETE FROM edges WHERE source_id=? AND relation='about'",
                    (node_id,),
                )
                for kw in new_kws:
                    conn.execute(
                        "INSERT OR IGNORE INTO keyword_map (keyword, node_id) VALUES (?,?)",
                        (kw, node_id),
                    )
                    self._maybe_promote_concept(conn, kw, now)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return True

    def delete_node(self, node_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ── Search ────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Primary: keyword_map exact match. Fallback: FTS5 content search."""
        conn = self._get_conn()
        norm_q = self._normalize_keyword(query)

        # Primary: keyword_map exact match
        rows = conn.execute(
            """SELECT DISTINCT n.* FROM keyword_map km
               JOIN nodes n ON n.id = km.node_id
               WHERE km.keyword = ?
               ORDER BY n.weight DESC, n.created_at DESC
               LIMIT ?""",
            (norm_q, limit),
        ).fetchall()

        if rows:
            return [dict(r) for r in rows]

        # Fallback: FTS5 content search
        try:
            rows = conn.execute(
                """SELECT n.* FROM nodes_fts f
                   JOIN nodes n ON f.id = n.id
                   WHERE nodes_fts MATCH ?
                   ORDER BY bm25(nodes_fts)
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS5 not available
            return []

    # ── Phase 1: Recall Chain ─────────────────────────────

    def recall_chain(self, query: str, max_events: int = 10) -> RecallResult:
        """
        Phase 1: Anchor on concepts/entities, then walk their event timeline.
        Each event gets lateral context (neighbors excluding anchors).
        """
        conn = self._get_conn()
        result = RecallResult(query=query)

        # Step 1: Find anchor nodes (concepts/entities matching query)
        seeds = self.search(query, limit=5)
        if not seeds:
            return result

        # Separate anchors (concept/entity) from direct event hits
        anchors = [s for s in seeds if s["type"] in ("concept", "entity")]
        direct_events = [s for s in seeds if s["type"] == "event"]

        # If no concept/entity anchors, use events directly
        if not anchors and direct_events:
            result.anchors = []
            events = sorted(direct_events, key=lambda e: e["created_at"], reverse=True)
        else:
            result.anchors = anchors
            anchor_ids = [a["id"] for a in anchors]

            # Step 2: Find events linked to anchors via 'about' edges, time DESC
            placeholders = ",".join("?" for _ in anchor_ids)
            events_rows = conn.execute(
                f"""SELECT DISTINCT n.*
                    FROM edges e
                    JOIN nodes n ON n.id = e.source_id
                    WHERE e.target_id IN ({placeholders})
                      AND e.relation = 'about'
                      AND n.type = 'event'
                    ORDER BY n.created_at DESC
                    LIMIT ?""",
                (*anchor_ids, max_events),
            ).fetchall()
            events = [dict(r) for r in events_rows]

            # Merge any direct event hits not already found
            event_ids = {e["id"] for e in events}
            for de in direct_events:
                if de["id"] not in event_ids:
                    events.append(de)

        # Step 3: For each event, get lateral context (neighbors excluding anchors)
        anchor_id_set = {a["id"] for a in result.anchors}
        for event in events[:max_events]:
            neighbors = self.get_neighbors(event["id"])
            context = [n for n in neighbors if n["id"] not in anchor_id_set]
            result.timeline.append(RecallFrame(
                event=event,
                context=context,
                relevance=event.get("weight", 1.0) * self._time_decay(event["created_at"]),
            ))

        # Bump weight on hit nodes
        hit_ids = [e["id"] for e in events[:max_events]]
        if hit_ids:
            self._bump_weights(hit_ids)

        return result

    # ── Phase 2: Spreading Activation ─────────────────────

    def spreading_activation(
        self,
        seed_ids: list[str],
        seed_weights: dict[str, float] | None = None,
        decay: float = 0.5,
        min_activation: float = 0.1,
        top_k: int = 10,
        max_depth: int = 3,
        max_nodes: int = 30,
    ) -> list[tuple[dict, float]]:
        """
        Weighted spreading activation from seed nodes.
        Returns (node_dict, activation_score) sorted by score DESC.
        """
        if not seed_ids:
            return []

        activation: dict[str, float] = {}
        if seed_weights:
            activation.update(seed_weights)
        else:
            for sid in seed_ids:
                activation[sid] = 1.0

        seed_set = set(seed_ids)
        current_layer = set(seed_ids)

        for _depth in range(max_depth):
            if not current_layer:
                break

            neighbors_map = self._batch_get_neighbors(list(current_layer))
            next_layer: set[str] = set()

            for source_id in current_layer:
                source_act = activation.get(source_id, 0)
                if source_act < min_activation:
                    continue

                edges = neighbors_map.get(source_id, [])
                # Top-K pruning: sort by edge_weight * time_decay, take top_k
                scored_edges = []
                for edge in edges:
                    td = self._time_decay(edge["created_at"])
                    scored_edges.append((edge, edge["weight"] * td))
                scored_edges.sort(key=lambda x: -x[1])

                for edge, edge_score in scored_edges[:top_k]:
                    target_id = edge["neighbor_id"]
                    propagated = source_act * decay * edge_score
                    if propagated < min_activation:
                        continue
                    # Accumulate (multi-path convergence)
                    activation[target_id] = activation.get(target_id, 0) + propagated
                    next_layer.add(target_id)

            current_layer = next_layer - seed_set

        # Fetch node details for non-seed activated nodes
        result_ids = [
            (nid, score)
            for nid, score in activation.items()
            if nid not in seed_set and score >= min_activation
        ]
        result_ids.sort(key=lambda x: -x[1])
        result_ids = result_ids[:max_nodes]

        if not result_ids:
            return []

        nodes_map = {n["id"]: n for n in self.get_chain([r[0] for r in result_ids])}
        return [
            (nodes_map[nid], score)
            for nid, score in result_ids
            if nid in nodes_map
        ]

    # ── Neighbor queries ──────────────────────────────────

    def get_neighbors(self, node_id: str) -> list[dict[str, Any]]:
        """Get all nodes directly connected to a given node."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT n.*, e.relation, e.weight as edge_weight
               FROM edges e
               JOIN nodes n ON n.id = CASE
                   WHEN e.source_id = ? THEN e.target_id
                   ELSE e.source_id
               END
               WHERE e.source_id = ? OR e.target_id = ?
               ORDER BY n.created_at DESC""",
            (node_id, node_id, node_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def _batch_get_neighbors(
        self, node_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch query neighbors for multiple nodes. Returns {node_id: [edge_info]}."""
        if not node_ids:
            return {}

        conn = self._get_conn()
        placeholders = ",".join("?" for _ in node_ids)
        # Use UNION to get edges in both directions, ensuring each node in the
        # input list sees all its neighbors even when both endpoints are queried.
        rows = conn.execute(
            f"""SELECT e.source_id as origin_id, e.target_id as neighbor_id,
                       e.relation, e.weight, e.created_at
                FROM edges e
                WHERE e.source_id IN ({placeholders})
                UNION ALL
                SELECT e.target_id as origin_id, e.source_id as neighbor_id,
                       e.relation, e.weight, e.created_at
                FROM edges e
                WHERE e.target_id IN ({placeholders})""",
            (*node_ids, *node_ids),
        ).fetchall()

        result: dict[str, list[dict[str, Any]]] = {nid: [] for nid in node_ids}
        for r in rows:
            d = dict(r)
            origin = d["origin_id"]
            if origin in result:
                result[origin].append(d)
        return result

    def get_chain(self, node_ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple nodes by ID, preserving requested order."""
        if not node_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in node_ids)
        rows = conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", node_ids
        ).fetchall()
        by_id = {dict(r)["id"]: dict(r) for r in rows}
        return [by_id[nid] for nid in node_ids if nid in by_id]

    # ── Weight management ─────────────────────────────────

    def _bump_weights(self, node_ids: list[str], amount: float = 0.1) -> None:
        """Bump weight for accessed nodes."""
        conn = self._get_conn()
        now = self._now()
        placeholders = ",".join("?" for _ in node_ids)
        conn.execute(
            f"UPDATE nodes SET weight = weight + ?, updated_at = ? WHERE id IN ({placeholders})",
            (amount, now, *node_ids),
        )
        conn.commit()

    # ── Context for LLM injection ─────────────────────────

    def get_memory_context(self) -> str:
        """Minimal stats for system prompt (~50 tokens)."""
        conn = self._get_conn()
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        if node_count == 0:
            return ""

        last = conn.execute(
            "SELECT updated_at FROM nodes ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        last_active = last[0][:16] if last else "unknown"

        return (
            f"## Graph Memory\n"
            f"Graph memory: {node_count} nodes, {edge_count} edges. "
            f"Last active: {last_active}.\n"
            f"Use find_memory_cache to search memories."
        )

    # ── Maintenance ───────────────────────────────────────

    def gc_concepts(self) -> int:
        """Remove concept nodes with degree ≤ 1 (low-value hubs)."""
        conn = self._get_conn()
        # Find concepts with ≤ 1 edge (use subqueries to avoid cross-product)
        rows = conn.execute(
            """SELECT n.id,
                   (SELECT COUNT(*) FROM edges WHERE source_id = n.id)
                 + (SELECT COUNT(*) FROM edges WHERE target_id = n.id) as degree
               FROM nodes n
               WHERE n.type = 'concept'
               GROUP BY n.id
               HAVING degree <= 1"""
        ).fetchall()

        if not rows:
            return 0

        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", ids)
        conn.commit()
        logger.debug(f"GC: removed {len(ids)} cold concept nodes")
        return len(ids)
