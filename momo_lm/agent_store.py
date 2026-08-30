from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"pending", "running", "waiting_approval"}
SECRET_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}
CANONICAL_SECRET_KEYS = {
    "accesstoken",
    "apikey",
    "authorization",
    "clientsecret",
    "cookie",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "token",
}
SECRET_PATTERN = re.compile(
    r"(?i)(bearer\s+)[^\s,;]+|"
    r"((?:access[_-]?token|api[_-]?key|authorization|client[_-]?secret|cookie|password|"
    r"private[_-]?key|refresh[_-]?token|secret|token)"
    r"\s*[=:]\s*)[^\s,;]+"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def arguments_digest(tool: str, arguments: dict[str, Any]) -> str:
    payload = canonical_json({"tool": tool, "arguments": arguments}).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def redact(value: Any) -> Any:
    """Return a JSON-compatible value with common credentials removed."""

    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in SECRET_KEYS
                or re.sub(r"[^a-z0-9]", "", str(key).lower()) in CANONICAL_SECRET_KEYS
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SECRET_PATTERN.sub(
            lambda match: f"{match.group(1) or match.group(2)}[REDACTED]", value
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


class AgentStore:
    """Thread-safe persistent storage for agent state and append-only events."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        path.chmod(0o600)
        self._connection.row_factory = sqlite3.Row
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=10000")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    budgets_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    next_step INTEGER NOT NULL DEFAULT 0,
                    active_step INTEGER,
                    steps_used INTEGER NOT NULL DEFAULT 0,
                    tool_calls_used INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    step_index INTEGER NOT NULL,
                    tool TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    arguments_sha256 TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_approval_step_pending
                    ON approvals(agent_id, step_index) WHERE status = 'pending';
                CREATE TABLE IF NOT EXISTS agent_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_events_agent_seq
                    ON agent_events(agent_id, seq);
                CREATE INDEX IF NOT EXISTS idx_agents_updated
                    ON agents(updated_at DESC);
                """
            )
        self._secure_files()

    @property
    def journal_mode(self) -> str:
        with self._lock:
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def _secure_files(self) -> None:
        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            with contextlib.suppress(OSError):
                candidate.chmod(0o600)

    def create_agent(
        self,
        *,
        goal: str,
        profile: str,
        capabilities: list[str],
        budgets: dict[str, int],
        plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        agent_id = uuid.uuid4().hex
        timestamp = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO agents(
                    id, goal, profile, status, capabilities_json, budgets_json,
                    plan_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    goal,
                    profile,
                    canonical_json(capabilities),
                    canonical_json(budgets),
                    canonical_json(plan),
                    timestamp,
                    timestamp,
                ),
            )
            self._insert_event(
                agent_id,
                "created",
                "Agent created",
                {"profile": profile, "capabilities": capabilities, "steps": len(plan)},
                timestamp,
            )
        return self.get_agent(agent_id)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown agent: {agent_id}")
            approval = self._connection.execute(
                """
                SELECT * FROM approvals
                WHERE agent_id = ? AND status = 'pending'
                ORDER BY created_at DESC LIMIT 1
                """,
                (agent_id,),
            ).fetchone()
        return self._agent_record(row, approval)

    def list_agents(self, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM agents ORDER BY updated_at DESC, id DESC LIMIT ?", (bounded,)
            ).fetchall()
            approvals = {
                row["agent_id"]: row
                for row in self._connection.execute(
                    "SELECT * FROM approvals WHERE status = 'pending'"
                ).fetchall()
            }
        return [self._agent_record(row, approvals.get(row["id"])) for row in rows]

    def recoverable_agents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM agents WHERE status IN ('pending', 'running') ORDER BY created_at"
            ).fetchall()
        return [self._internal_agent(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM agents GROUP BY status"
            ).fetchall()
        by_status = {str(row["status"]): int(row["count"]) for row in rows}
        return {"total": sum(by_status.values()), "by_status": by_status}

    def internal_agent(self, agent_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown agent: {agent_id}")
        return self._internal_agent(row)

    def set_running(self, agent_id: str, step_index: int) -> None:
        timestamp = utc_now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE agents SET status = 'running', active_step = ?, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'running') AND cancel_requested = 0
                """,
                (step_index, timestamp, agent_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Agent is not runnable")
            self._insert_event(
                agent_id, "step_started", "Agent step started", {"step": step_index}, timestamp
            )

    def reset_pending(self, agent_id: str, *, message: str) -> None:
        timestamp = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE agents SET status = 'pending', active_step = NULL, updated_at = ?
                WHERE id = ? AND status = 'running' AND cancel_requested = 0
                """,
                (timestamp, agent_id),
            )
            self._insert_event(agent_id, "recovered", message, {}, timestamp)

    def complete_step(self, agent_id: str, output: Any) -> dict[str, Any]:
        timestamp = utc_now()
        safe_output = redact(output)
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT next_step, plan_json, result_json, status, cancel_requested
                FROM agents WHERE id = ?
                """,
                (agent_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown agent: {agent_id}")
            if row["status"] == "cancelled" or bool(row["cancel_requested"]):
                return self.get_agent(agent_id)
            if row["status"] != "running":
                raise RuntimeError("Agent step is no longer running")
            step_index = int(row["next_step"])
            plan = json.loads(row["plan_json"])
            results = json.loads(row["result_json"]) if row["result_json"] else {"outputs": []}
            results.setdefault("outputs", []).append(safe_output)
            next_step = step_index + 1
            status = "completed" if next_step >= len(plan) else "pending"
            self._connection.execute(
                """
                UPDATE agents SET status = ?, next_step = ?, active_step = NULL,
                    steps_used = steps_used + 1, tool_calls_used = tool_calls_used + 1,
                    result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, next_step, canonical_json(results), timestamp, agent_id),
            )
            self._insert_event(
                agent_id,
                "completed" if status == "completed" else "step_completed",
                "Agent completed" if status == "completed" else "Agent step completed",
                {"step": step_index, "output": safe_output},
                timestamp,
            )
        return self.get_agent(agent_id)

    def fail(self, agent_id: str, error: str, *, event_type: str = "failed") -> dict[str, Any]:
        timestamp = utc_now()
        safe_error = str(redact(error))[:2000]
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE agents SET status = 'failed', active_step = NULL, error = ?, updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'cancelled')
                """,
                (safe_error, timestamp, agent_id),
            )
            if cursor.rowcount:
                self._connection.execute(
                    """
                    UPDATE approvals SET status = 'revoked'
                    WHERE agent_id = ? AND status = 'pending'
                    """,
                    (agent_id,),
                )
                self._insert_event(agent_id, event_type, safe_error, {}, timestamp)
        return self.get_agent(agent_id)

    def request_cancel(self, agent_id: str) -> dict[str, Any]:
        timestamp = utc_now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT status FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown agent: {agent_id}")
            if row["status"] not in TERMINAL_STATUSES:
                self._connection.execute(
                    """
                    UPDATE agents SET status = 'cancelled', cancel_requested = 1,
                        active_step = NULL, updated_at = ? WHERE id = ?
                    """,
                    (timestamp, agent_id),
                )
                self._connection.execute(
                    "UPDATE approvals SET status = 'revoked' WHERE agent_id = ? AND status = 'pending'",
                    (agent_id,),
                )
                self._insert_event(agent_id, "cancelled", "Agent cancelled", {}, timestamp)
        return self.get_agent(agent_id)

    def is_cancel_requested(self, agent_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT cancel_requested, status FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown agent: {agent_id}")
        return bool(row["cancel_requested"]) or row["status"] == "cancelled"

    def create_approval(
        self,
        agent_id: str,
        *,
        step_index: int,
        tool: str,
        arguments: dict[str, Any],
        reason: str,
        ttl_seconds: int,
    ) -> dict[str, Any]:
        approval_id = uuid.uuid4().hex
        created = datetime.now(timezone.utc)
        expires = created + timedelta(seconds=max(30, min(ttl_seconds, 86_400)))
        with self._lock, self._connection:
            existing = self._connection.execute(
                """
                SELECT id FROM approvals
                WHERE agent_id = ? AND step_index = ? AND status = 'pending'
                """,
                (agent_id, step_index),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO approvals(
                        id, agent_id, step_index, tool, arguments_json, arguments_sha256,
                        reason, status, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        approval_id,
                        agent_id,
                        step_index,
                        tool,
                        canonical_json(arguments),
                        arguments_digest(tool, arguments),
                        reason[:1000],
                        created.isoformat(),
                        expires.isoformat(),
                    ),
                )
            else:
                approval_id = str(existing["id"])
            self._connection.execute(
                """
                UPDATE agents SET status = 'waiting_approval', active_step = NULL, updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (created.isoformat(), agent_id),
            )
            self._insert_event(
                agent_id,
                "approval_required",
                "Explicit approval is required for a mutating tool",
                {
                    "approval_id": approval_id,
                    "tool": tool,
                    "arguments": redact(arguments),
                    "reason": reason,
                },
                created.isoformat(),
            )
        return self.get_agent(agent_id)

    def consume_approval(
        self,
        agent_id: str,
        approval_id: str,
        *,
        tool: str,
        arguments: dict[str, Any],
        step_index: int,
    ) -> None:
        now = datetime.now(timezone.utc)
        digest = arguments_digest(tool, arguments)
        expired = False
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM approvals WHERE id = ? AND agent_id = ?",
                (approval_id, agent_id),
            ).fetchone()
            if row is None:
                raise ValueError("Approval does not exist for this agent")
            if row["status"] != "pending":
                raise ValueError("Approval has already been used or revoked")
            if datetime.fromisoformat(row["expires_at"]) <= now:
                self._connection.execute(
                    "UPDATE approvals SET status = 'expired' WHERE id = ?", (approval_id,)
                )
                self._connection.execute(
                    """
                    UPDATE agents SET status = 'failed', active_step = NULL,
                        error = 'Approval expired', updated_at = ? WHERE id = ?
                    """,
                    (now.isoformat(), agent_id),
                )
                self._insert_event(
                    agent_id,
                    "approval_expired",
                    "Pending approval expired without running the tool",
                    {"approval_id": approval_id},
                    now.isoformat(),
                )
                expired = True
            elif (
                row["tool"] != tool
                or int(row["step_index"]) != int(step_index)
                or row["arguments_sha256"] != digest
            ):
                raise ValueError("Approval does not match the exact pending action")
            if not expired:
                cursor = self._connection.execute(
                    """
                    UPDATE approvals SET status = 'consumed', consumed_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (now.isoformat(), approval_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Approval has already been used")
                self._connection.execute(
                    "UPDATE agents SET status = 'pending', updated_at = ? WHERE id = ?",
                    (now.isoformat(), agent_id),
                )
                self._insert_event(
                    agent_id,
                    "approved",
                    "Exact one-use approval consumed",
                    {"approval_id": approval_id, "tool": tool, "arguments_sha256": digest},
                    now.isoformat(),
                )
        if expired:
            raise ValueError("Approval has expired")

    def has_consumed_approval(
        self, agent_id: str, *, step_index: int, tool: str, arguments: dict[str, Any]
    ) -> bool:
        digest = arguments_digest(tool, arguments)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM approvals
                WHERE agent_id = ? AND step_index = ? AND tool = ?
                    AND arguments_sha256 = ? AND status = 'consumed'
                LIMIT 1
                """,
                (agent_id, step_index, tool, digest),
            ).fetchone()
        return row is not None

    def events(self, agent_id: str, *, after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        self.get_agent(agent_id)
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT seq, agent_id, type, message, data_json, created_at
                FROM agent_events WHERE agent_id = ? AND seq > ? ORDER BY seq LIMIT ?
                """,
                (agent_id, max(0, int(after)), bounded),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "agent_id": row["agent_id"],
                "type": row["type"],
                "message": row["message"],
                "data": json.loads(row["data_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _insert_event(
        self,
        agent_id: str,
        event_type: str,
        message: str,
        data: Any,
        timestamp: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO agent_events(agent_id, type, message, data_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                event_type[:80],
                str(redact(message))[:2000],
                canonical_json(redact(data)),
                timestamp,
            ),
        )

    @staticmethod
    def _internal_agent(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "goal": row["goal"],
            "profile": row["profile"],
            "status": row["status"],
            "capabilities": json.loads(row["capabilities_json"]),
            "budgets": json.loads(row["budgets_json"]),
            "plan": json.loads(row["plan_json"]),
            "next_step": int(row["next_step"]),
            "active_step": int(row["active_step"]) if row["active_step"] is not None else None,
            "steps_used": int(row["steps_used"]),
            "tool_calls_used": int(row["tool_calls_used"]),
            "cancel_requested": bool(row["cancel_requested"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @classmethod
    def _agent_record(
        cls, row: sqlite3.Row, approval: sqlite3.Row | None
    ) -> dict[str, Any]:
        internal = cls._internal_agent(row)
        internal.pop("plan")
        internal.pop("next_step")
        internal.pop("active_step")
        internal.pop("cancel_requested")
        internal["goal"] = redact(internal["goal"])
        internal["pending_approval"] = (
            {
                "id": approval["id"],
                "tool": approval["tool"],
                "arguments": redact(json.loads(approval["arguments_json"])),
                "arguments_sha256": approval["arguments_sha256"],
                "reason": approval["reason"],
                "expires_at": approval["expires_at"],
            }
            if approval is not None
            else None
        )
        return internal
