"""Catálogo SQLite de documentos e consultas, compartilhado pela API e CLI."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enterprise_rag.models.document import Document, DocumentStatus


class Catalog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer_found INTEGER,
                    chunks_used INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_document(self, document: Document) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    content_sha256 = excluded.content_sha256,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    ingested_at = excluded.ingested_at,
                    payload = excluded.payload
                """,
                (
                    document.document_id,
                    document.content_sha256,
                    document.status.value,
                    document.chunk_count,
                    document.ingested_at.isoformat(),
                    document.model_dump_json(),
                ),
            )

    def get_document(self, document_id: str) -> Document | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return Document.model_validate_json(row["payload"]) if row else None

    def is_unchanged(self, document_id: str, sha256: str) -> bool:
        document = self.get_document(document_id)
        return (
            document is not None
            and document.status == DocumentStatus.INDEXED
            and document.content_sha256 == sha256
        )

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[Document]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM documents ORDER BY document_id LIMIT ? OFFSET ?",
                (max(0, limit), max(0, offset)),
            ).fetchall()
        return [Document.model_validate_json(row["payload"]) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def stats(self) -> dict[str, int]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS documents,
                    COALESCE(SUM(status = 'indexed'), 0) AS indexed,
                    COALESCE(SUM(status = 'failed'), 0) AS failed,
                    COALESCE(SUM(CASE WHEN status = 'indexed' THEN chunk_count ELSE 0 END), 0)
                        AS chunks
                FROM documents
                """
            ).fetchone()
        return dict(row)

    def log_query(
        self,
        *,
        trace_id: str,
        endpoint: str,
        query: str,
        chunks_used: int,
        latency_ms: float,
        answer_found: bool | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO queries
                    (trace_id, endpoint, query, answer_found, chunks_used,
                     latency_ms, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    endpoint,
                    query,
                    answer_found,
                    chunks_used,
                    latency_ms,
                    json.dumps(detail or {}, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def query_stats(self) -> dict[str, int | float]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS total,
                    COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
                    COALESCE(AVG(CASE WHEN answer_found IS NOT NULL
                        THEN 1.0 - answer_found END), 0.0) AS not_found_rate
                FROM queries
                """
            ).fetchone()
        return dict(row)

    def recent_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (max(0, limit),)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item["detail"])
            if item["answer_found"] is not None:
                item["answer_found"] = bool(item["answer_found"])
            result.append(item)
        return result

    def close(self) -> None:
        with self._lock:
            self._connection.close()
