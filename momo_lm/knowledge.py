from __future__ import annotations

import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

WORD_PATTERN = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]+", re.UNICODE)


@dataclass(slots=True)
class KnowledgeHit:
    source: str
    content: str
    score: float


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
                CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
                """
            )

    def add_document(self, source: str, content: str) -> int:
        cleaned = " ".join(content.split())
        if not cleaned:
            raise ValueError("Document content is empty")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO documents(source, content, created_at) VALUES (?, ?, ?)",
                (source[:500], cleaned, datetime.now(timezone.utc).isoformat()),
            )
            return int(cursor.lastrowid)

    def add_turn(self, role: str, content: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO conversations(role, content, created_at) VALUES (?, ?, ?)",
                (role, content, datetime.now(timezone.utc).isoformat()),
            )
            self._connection.execute(
                "DELETE FROM conversations WHERE id NOT IN (SELECT id FROM conversations ORDER BY id DESC LIMIT 1000)"
            )

    def recent_turns(self, limit: int = 8) -> list[dict[str, str]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT role, content, created_at FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def recent_documents(self, limit: int = 4) -> list[str]:
        bounded = max(0, min(int(limit), 64))
        if not bounded:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT content FROM documents ORDER BY id DESC LIMIT ?", (bounded,)
            ).fetchall()
        return [str(row["content"]) for row in rows]

    def search(self, query: str, limit: int = 4) -> list[KnowledgeHit]:
        terms: set[str] = set()
        for word in WORD_PATTERN.findall(query.lower()):
            if len(word) > 1:
                terms.add(word)
            if any("\u3400" <= character <= "\u9fff" for character in word):
                terms.update(word[index : index + 2] for index in range(len(word) - 1))
        if not terms:
            return []
        with self._lock:
            rows = self._connection.execute(
                "SELECT source, content FROM documents ORDER BY id DESC LIMIT 1000"
            ).fetchall()
        hits: list[KnowledgeHit] = []
        for row in rows:
            haystack = row["content"].lower()
            matches = sum(haystack.count(term) for term in terms)
            if matches:
                score = matches / math.sqrt(max(len(haystack), 1))
                hits.append(KnowledgeHit(row["source"], row["content"], score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def stats(self) -> dict[str, int]:
        with self._lock:
            documents = self._connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            turns = self._connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        return {"documents": int(documents), "conversation_turns": int(turns)}

    def close(self) -> None:
        with self._lock:
            self._connection.close()
