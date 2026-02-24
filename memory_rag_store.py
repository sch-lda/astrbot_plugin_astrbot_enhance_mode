from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryRAGStore:
    """SQLite-backed memory store for enhance mode RAG tools."""

    def __init__(self, db_path: Path, default_scan_limit: int = 5000) -> None:
        self.db_path = Path(db_path)
        self.default_scan_limit = max(1, int(default_scan_limit))
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    memory_time REAL NOT NULL,
                    group_scope TEXT DEFAULT '',
                    group_id TEXT DEFAULT '',
                    platform_id TEXT DEFAULT '',
                    extra_metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_memory_roles (
                    memory_id INTEGER NOT NULL,
                    role_id TEXT NOT NULL,
                    PRIMARY KEY (memory_id, role_id),
                    FOREIGN KEY(memory_id) REFERENCES rag_memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_rag_memories_memory_time
                ON rag_memories(memory_time);

                CREATE INDEX IF NOT EXISTS idx_rag_memories_group_scope
                ON rag_memories(group_scope);

                CREATE INDEX IF NOT EXISTS idx_rag_memories_group_id
                ON rag_memories(group_id);

                CREATE INDEX IF NOT EXISTS idx_rag_memories_platform_id
                ON rag_memories(platform_id);

                CREATE INDEX IF NOT EXISTS idx_rag_roles_role_id
                ON rag_memory_roles(role_id);

                CREATE INDEX IF NOT EXISTS idx_rag_roles_memory_id
                ON rag_memory_roles(memory_id);
                """
            )

    @staticmethod
    def _normalize_role_ids(role_ids: list[str]) -> list[str]:
        uniq: set[str] = set()
        normalized: list[str] = []
        for raw in role_ids:
            role_id = str(raw or "").strip()
            if not role_id or role_id in uniq:
                continue
            uniq.add(role_id)
            normalized.append(role_id)
        return normalized

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if len(vec_a) != len(vec_b) or not vec_a:
            return -1.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b

        if norm_a <= 0 or norm_b <= 0:
            return -1.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    @staticmethod
    def _parse_json_dict(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _parse_embedding(raw: str | None) -> list[float]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []

        vector: list[float] = []
        for item in parsed:
            try:
                vector.append(float(item))
            except (TypeError, ValueError):
                return []
        return vector

    def add_memory(
        self,
        *,
        content: str,
        embedding: list[float],
        role_ids: list[str],
        memory_time: float | None = None,
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        clean_content = str(content or "").strip()
        if not clean_content:
            raise ValueError("`content` cannot be empty.")

        clean_roles = self._normalize_role_ids(role_ids)
        if not clean_roles:
            raise ValueError("At least one `role_id` is required.")

        if not embedding:
            raise ValueError("`embedding` cannot be empty.")

        embedding_vector = [float(value) for value in embedding]
        timestamp = float(memory_time) if memory_time is not None else time.time()
        now = time.time()
        metadata = extra_metadata if isinstance(extra_metadata, dict) else {}

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO rag_memories (
                    content, embedding, embedding_dim, memory_time,
                    group_scope, group_id, platform_id, extra_metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_content,
                    json.dumps(embedding_vector, ensure_ascii=False),
                    len(embedding_vector),
                    timestamp,
                    str(group_scope or "").strip(),
                    str(group_id or "").strip(),
                    str(platform_id or "").strip(),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            memory_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT OR IGNORE INTO rag_memory_roles(memory_id, role_id) VALUES (?, ?)",
                [(memory_id, role_id) for role_id in clean_roles],
            )
            return memory_id

    def delete_memory(self, memory_id: int) -> bool:
        target_id = int(memory_id)
        if target_id <= 0:
            return False

        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM rag_memories WHERE id = ?", (target_id,))
            return cursor.rowcount > 0

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        target_id = int(memory_id)
        if target_id <= 0:
            return None

        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    content,
                    embedding_dim,
                    memory_time,
                    group_scope,
                    group_id,
                    platform_id,
                    extra_metadata,
                    created_at,
                    updated_at
                FROM rag_memories
                WHERE id = ?
                """,
                (target_id,),
            ).fetchone()
            if not row:
                return None

            role_rows = conn.execute(
                """
                SELECT role_id
                FROM rag_memory_roles
                WHERE memory_id = ?
                ORDER BY role_id ASC
                """,
                (target_id,),
            ).fetchall()

        return {
            "memory_id": int(row["id"]),
            "content": str(row["content"]),
            "embedding_dim": int(row["embedding_dim"]),
            "memory_time": float(row["memory_time"]),
            "memory_time_iso": datetime.fromtimestamp(
                float(row["memory_time"]), tz=timezone.utc
            ).isoformat(),
            "group_scope": str(row["group_scope"] or ""),
            "group_id": str(row["group_id"] or ""),
            "platform_id": str(row["platform_id"] or ""),
            "related_role_ids": [str(role_row["role_id"]) for role_row in role_rows],
            "extra_metadata": self._parse_json_dict(row["extra_metadata"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def list_memories(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
        group_scope: str = "",
        role_id: str = "",
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = min(200, max(1, int(page_size)))
        offset = (page - 1) * page_size

        where_clauses: list[str] = []
        where_params: list[Any] = []

        clean_keyword = str(keyword or "").strip()
        if clean_keyword:
            where_clauses.append("(m.content LIKE ? OR CAST(m.id AS TEXT) LIKE ?)")
            like_term = f"%{clean_keyword}%"
            where_params.extend([like_term, like_term])

        clean_scope = str(group_scope or "").strip()
        if clean_scope:
            where_clauses.append("m.group_scope = ?")
            where_params.append(clean_scope)

        clean_role_id = str(role_id or "").strip()
        if clean_role_id:
            where_clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM rag_memory_roles r
                    WHERE r.memory_id = m.id AND r.role_id = ?
                )
                """
            )
            where_params.append(clean_role_id)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        with self._lock, self._connect() as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM rag_memories m {where_sql}",
                where_params,
            ).fetchone()
            total = int(count_row["total"]) if count_row else 0

            rows = conn.execute(
                f"""
                SELECT
                    m.id,
                    m.content,
                    m.embedding_dim,
                    m.memory_time,
                    m.group_scope,
                    m.group_id,
                    m.platform_id,
                    m.extra_metadata,
                    m.created_at,
                    m.updated_at
                FROM rag_memories m
                {where_sql}
                ORDER BY m.memory_time DESC, m.id DESC
                LIMIT ? OFFSET ?
                """,
                [*where_params, page_size, offset],
            ).fetchall()

            memory_ids = [int(row["id"]) for row in rows]
            roles_map: dict[int, list[str]] = {}
            if memory_ids:
                placeholders = ",".join("?" for _ in memory_ids)
                role_rows = conn.execute(
                    f"""
                    SELECT memory_id, role_id
                    FROM rag_memory_roles
                    WHERE memory_id IN ({placeholders})
                    ORDER BY memory_id ASC, role_id ASC
                    """,
                    memory_ids,
                ).fetchall()
                for role_row in role_rows:
                    mem_id = int(role_row["memory_id"])
                    roles_map.setdefault(mem_id, []).append(str(role_row["role_id"]))

        items = []
        for row in rows:
            mem_id = int(row["id"])
            memory_time = float(row["memory_time"])
            items.append(
                {
                    "memory_id": mem_id,
                    "content": str(row["content"]),
                    "embedding_dim": int(row["embedding_dim"]),
                    "memory_time": memory_time,
                    "memory_time_iso": datetime.fromtimestamp(
                        memory_time, tz=timezone.utc
                    ).isoformat(),
                    "group_scope": str(row["group_scope"] or ""),
                    "group_id": str(row["group_id"] or ""),
                    "platform_id": str(row["platform_id"] or ""),
                    "related_role_ids": roles_map.get(mem_id, []),
                    "extra_metadata": self._parse_json_dict(row["extra_metadata"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                }
            )

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": (offset + page_size) < total,
        }

    def get_stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS total FROM rag_memories"
            ).fetchone()
            total = int(total_row["total"]) if total_row else 0

            scope_row = conn.execute(
                "SELECT COUNT(DISTINCT group_scope) AS cnt FROM rag_memories WHERE group_scope <> ''"
            ).fetchone()
            scopes = int(scope_row["cnt"]) if scope_row else 0

            group_row = conn.execute(
                "SELECT COUNT(DISTINCT group_id) AS cnt FROM rag_memories WHERE group_id <> ''"
            ).fetchone()
            groups = int(group_row["cnt"]) if group_row else 0

            role_row = conn.execute(
                "SELECT COUNT(DISTINCT role_id) AS cnt FROM rag_memory_roles"
            ).fetchone()
            roles = int(role_row["cnt"]) if role_row else 0

            range_row = conn.execute(
                "SELECT MIN(memory_time) AS oldest, MAX(memory_time) AS latest FROM rag_memories"
            ).fetchone()
            oldest = (
                float(range_row["oldest"])
                if range_row and range_row["oldest"] is not None
                else None
            )
            latest = (
                float(range_row["latest"])
                if range_row and range_row["latest"] is not None
                else None
            )

        return {
            "total_memories": total,
            "group_scope_count": scopes,
            "group_id_count": groups,
            "role_count": roles,
            "oldest_memory_time": oldest,
            "latest_memory_time": latest,
            "oldest_memory_time_iso": datetime.fromtimestamp(
                oldest, tz=timezone.utc
            ).isoformat()
            if oldest is not None
            else None,
            "latest_memory_time_iso": datetime.fromtimestamp(
                latest, tz=timezone.utc
            ).isoformat()
            if latest is not None
            else None,
        }

    def search_memories(
        self,
        *,
        query_embedding: list[float] | None = None,
        embedding_recall_k: int = 20,
        role_ids: list[str] | None = None,
        role_match_mode: str = "any",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        max_results: int = 10,
        scan_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized_roles = self._normalize_role_ids(role_ids or [])
        mode = role_match_mode.lower().strip()
        mode = "all" if mode == "all" else "any"

        where_clauses: list[str] = []
        query_params: list[Any] = []

        if group_scope:
            where_clauses.append("m.group_scope = ?")
            query_params.append(group_scope.strip())
        if group_id:
            where_clauses.append("m.group_id = ?")
            query_params.append(group_id.strip())
        if platform_id:
            where_clauses.append("m.platform_id = ?")
            query_params.append(platform_id.strip())
        if start_time is not None:
            where_clauses.append("m.memory_time >= ?")
            query_params.append(float(start_time))
        if end_time is not None:
            where_clauses.append("m.memory_time <= ?")
            query_params.append(float(end_time))

        if normalized_roles:
            placeholders = ",".join("?" for _ in normalized_roles)
            if mode == "all":
                where_clauses.append(
                    f"""
                    m.id IN (
                        SELECT memory_id
                        FROM rag_memory_roles
                        WHERE role_id IN ({placeholders})
                        GROUP BY memory_id
                        HAVING COUNT(DISTINCT role_id) = ?
                    )
                    """
                )
                query_params.extend(normalized_roles)
                query_params.append(len(normalized_roles))
            else:
                where_clauses.append(
                    f"""
                    m.id IN (
                        SELECT DISTINCT memory_id
                        FROM rag_memory_roles
                        WHERE role_id IN ({placeholders})
                    )
                    """
                )
                query_params.extend(normalized_roles)

        sql = """
            SELECT
                m.id,
                m.content,
                m.embedding,
                m.embedding_dim,
                m.memory_time,
                m.group_scope,
                m.group_id,
                m.platform_id,
                m.extra_metadata
            FROM rag_memories m
        """
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += " ORDER BY m.memory_time DESC"

        final_scan_limit = (
            self.default_scan_limit if scan_limit is None else max(1, int(scan_limit))
        )
        sql += " LIMIT ?"
        query_params.append(final_scan_limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, query_params).fetchall()
            if not rows:
                return []

            memory_ids = [int(row["id"]) for row in rows]
            role_rows = []
            if memory_ids:
                role_placeholders = ",".join("?" for _ in memory_ids)
                role_rows = conn.execute(
                    f"""
                    SELECT memory_id, role_id
                    FROM rag_memory_roles
                    WHERE memory_id IN ({role_placeholders})
                    ORDER BY memory_id ASC, role_id ASC
                    """,
                    memory_ids,
                ).fetchall()

        roles_map: dict[int, list[str]] = {}
        for role_row in role_rows:
            mem_id = int(role_row["memory_id"])
            roles_map.setdefault(mem_id, []).append(str(role_row["role_id"]))

        has_query_embedding = bool(query_embedding)
        clean_query_embedding = [float(x) for x in (query_embedding or [])]

        candidates: list[dict[str, Any]] = []
        for row in rows:
            memory_id = int(row["id"])
            memory_time = float(row["memory_time"])
            embedding = self._parse_embedding(row["embedding"])
            similarity: float | None = None
            if has_query_embedding:
                similarity = self._cosine_similarity(clean_query_embedding, embedding)

            candidates.append(
                {
                    "memory_id": memory_id,
                    "content": str(row["content"]),
                    "memory_time": memory_time,
                    "memory_time_iso": datetime.fromtimestamp(
                        memory_time, tz=timezone.utc
                    ).isoformat(),
                    "group_scope": str(row["group_scope"] or ""),
                    "group_id": str(row["group_id"] or ""),
                    "platform_id": str(row["platform_id"] or ""),
                    "related_role_ids": roles_map.get(memory_id, []),
                    "similarity": similarity,
                    "extra_metadata": self._parse_json_dict(row["extra_metadata"]),
                }
            )

        if has_query_embedding:
            candidates.sort(
                key=lambda item: (
                    item["similarity"]
                    if isinstance(item["similarity"], (int, float))
                    else -1.0
                ),
                reverse=True,
            )
            if embedding_recall_k > 0:
                candidates = candidates[: int(embedding_recall_k)]

        normalized_sort_by = sort_by.lower().strip()
        normalized_sort_order = sort_order.lower().strip()
        reverse = normalized_sort_order != "asc"

        if normalized_sort_by == "time":
            candidates.sort(key=lambda item: item["memory_time"], reverse=reverse)
        elif normalized_sort_by == "relevance" and has_query_embedding:
            candidates.sort(
                key=lambda item: (
                    item["similarity"]
                    if isinstance(item["similarity"], (int, float))
                    else -1.0
                ),
                reverse=reverse,
            )
        else:
            candidates.sort(key=lambda item: item["memory_time"], reverse=reverse)

        if max_results > 0:
            return candidates[: int(max_results)]
        return candidates
