"""Durable, pluggable storage for Agent sessions.

Storage deliberately has two layers. ``messages`` is the mutable Active
Context used for resume and may contain summaries, while ``events`` is an
append-only Transcript of original user, assistant, tool-call, and tool-result
payloads.  One save commits both layers and session metadata in a transaction.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

SCHEMA_VERSION = 3
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SESSION_ID_LEN = 100
_MAX_SESSION_NAME_LEN = 120


def normalize_session_id(session_id: str | None) -> str:
    """Turn user-controlled IDs into one safe filename/database key."""
    if not session_id:
        return new_session_id()
    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    if len(name) > _MAX_SESSION_ID_LEN:
        name = name[:_MAX_SESSION_ID_LEN].strip(".-_")
    return name or new_session_id()


def new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def normalize_session_name(name: str | None) -> str:
    """Normalize a human-facing label without turning it into an identifier."""
    normalized = " ".join(str(name or "").split())
    return normalized[:_MAX_SESSION_NAME_LEN].rstrip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _preview(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"])[:160]
    return ""


@dataclass
class SessionRecord:
    id: str
    model: str
    messages: list[dict]
    name: str = ""
    status: str = "saved"
    workspace: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float | None = None
    metadata: dict = field(default_factory=dict)
    transcript_complete: bool = False
    transcript_events: list[dict] = field(default_factory=list)
    context_summaries: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_snapshot(cls, session_id: str, snapshot: dict) -> SessionRecord:
        return cls(
            id=normalize_session_id(session_id),
            model=str(snapshot.get("model") or ""),
            messages=list(snapshot.get("messages") or []),
            name=normalize_session_name(snapshot.get("name")),
            status=str(snapshot.get("status") or "saved"),
            workspace=str(snapshot.get("workspace") or ""),
            prompt_tokens=max(0, int(snapshot.get("prompt_tokens") or 0)),
            completion_tokens=max(0, int(snapshot.get("completion_tokens") or 0)),
            estimated_cost=snapshot.get("estimated_cost"),
            metadata=dict(snapshot.get("metadata") or {}),
            transcript_complete=bool(snapshot.get("transcript_complete", False)),
            transcript_events=list(snapshot.get("transcript_events") or []),
            context_summaries=list(snapshot.get("context_summaries") or []),
        )

    def to_snapshot(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "messages": self.messages,
            "status": self.status,
            "workspace": self.workspace,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self.estimated_cost,
            "metadata": self.metadata,
            "transcript_complete": self.transcript_complete,
        }


@dataclass(frozen=True)
class SessionSummary:
    id: str
    name: str
    model: str
    updated_at: str
    status: str
    workspace: str
    preview: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float | None
    transcript_complete: bool = False

    def as_dict(self) -> dict:
        data = asdict(self)
        data["saved_at"] = self.updated_at  # compatibility with the original CLI
        return data


class SessionStore(Protocol):
    """Storage contract used by the CLI and embedders."""

    def save(self, record: SessionRecord) -> str: ...

    def load(self, session_id: str) -> SessionRecord | None: ...

    def list(self, limit: int = 20, offset: int = 0) -> list[SessionSummary]: ...

    def delete(self, session_id: str) -> bool: ...

    def rename(self, session_id: str, name: str) -> bool: ...

    def transcript(self, session_id: str) -> list[dict]: ...

    def summaries(self, session_id: str) -> list[dict]: ...


def _event_type(message: dict) -> str:
    role = str(message.get("role") or "unknown")
    if role == "assistant" and message.get("tool_calls"):
        return "assistant.tool_calls"
    if role == "tool":
        return "tool.result"
    return f"{role}.message"


def _snapshot_events(messages: list[dict], prefix: str) -> list[dict]:
    """Represent an imported snapshot without claiming it is full history."""
    return [
        {
            "event_id": f"{prefix}-{index}",
            "event_type": _event_type(message),
            "run_id": None,
            "round": None,
            "payload": message,
        }
        for index, message in enumerate(messages)
    ]


class JsonSessionStore:
    """Atomic JSON backend and reader for pre-SQLite CoreCoder sessions."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()

    def _path(self, session_id: str) -> Path:
        path = (self.directory / f"{normalize_session_id(session_id)}.json").resolve()
        if path.parent != self.directory:
            raise ValueError("Invalid session id")
        return path

    def save(self, record: SessionRecord) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.directory.chmod(0o700)
        except OSError:
            pass
        session_id = normalize_session_id(record.id)
        now = _now()
        existing = self.load(session_id)
        existing_events = existing.transcript_events if existing else []
        known_event_ids = {event.get("event_id") for event in existing_events}
        transcript_events = existing_events + [
            event for event in record.transcript_events
            if event.get("event_id") not in known_event_ids
        ]
        existing_summaries = existing.context_summaries if existing else []
        known_summary_ids = {item.get("summary_id") for item in existing_summaries}
        context_summaries = existing_summaries + [
            item for item in record.context_summaries
            if item.get("summary_id") not in known_summary_ids
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "id": session_id,
            "name": normalize_session_name(record.name) or (existing.name if existing else ""),
            "model": record.model,
            "status": record.status,
            "workspace": record.workspace,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "estimated_cost": record.estimated_cost,
            "metadata": record.metadata,
            "transcript_complete": (
                existing.transcript_complete if existing else record.transcript_complete
            ),
            "transcript_events": transcript_events,
            "context_summaries": context_summaries,
            "created_at": record.created_at or (existing.created_at if existing else now),
            "updated_at": now,
            "saved_at": now,
            "messages": record.messages,
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        destination = self._path(session_id)
        fd, temporary = tempfile.mkstemp(prefix=f".{session_id}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return session_id

    def load(self, session_id: str) -> SessionRecord | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = list(data["messages"])
            has_transcript = "transcript_events" in data
            return SessionRecord(
                id=normalize_session_id(data.get("id") or path.stem),
                model=str(data["model"]),
                messages=messages,
                name=normalize_session_name(data.get("name")),
                status=str(data.get("status") or "saved"),
                workspace=str(data.get("workspace") or ""),
                prompt_tokens=max(0, int(data.get("prompt_tokens") or 0)),
                completion_tokens=max(0, int(data.get("completion_tokens") or 0)),
                estimated_cost=data.get("estimated_cost"),
                metadata=dict(data.get("metadata") or {}),
                transcript_complete=bool(data.get("transcript_complete", False)),
                transcript_events=(
                    list(data.get("transcript_events") or [])
                    if has_transcript
                    else _snapshot_events(messages, f"legacy-{path.stem}")
                ),
                context_summaries=list(data.get("context_summaries") or []),
                created_at=str(data.get("created_at") or data.get("saved_at") or ""),
                updated_at=str(data.get("updated_at") or data.get("saved_at") or ""),
            )
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            return None

    def list(self, limit: int = 20, offset: int = 0) -> list[SessionSummary]:
        if not self.directory.exists():
            return []
        records = []
        for path in self.directory.glob("*.json"):
            record = self.load(path.stem)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return [
            SessionSummary(
                id=record.id,
                name=record.name,
                model=record.model,
                updated_at=record.updated_at,
                status=record.status,
                workspace=record.workspace,
                preview=_preview(record.messages),
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                estimated_cost=record.estimated_cost,
                transcript_complete=record.transcript_complete,
            )
            for record in records[offset:offset + max(0, limit)]
        ]

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def rename(self, session_id: str, name: str) -> bool:
        record = self.load(session_id)
        if record is None:
            return False
        record.name = normalize_session_name(name)
        self.save(record)
        return True

    def transcript(self, session_id: str) -> list[dict]:
        record = self.load(session_id)
        return list(record.transcript_events) if record is not None else []

    def summaries(self, session_id: str) -> list[dict]:
        record = self.load(session_id)
        return list(record.context_summaries) if record is not None else []


class SQLiteSessionStore:
    """Transactional SQLite store with read-through legacy JSON migration."""

    def __init__(self, path: str | Path, *, legacy_dir: str | Path | None = None):
        self.path = Path(path).expanduser().resolve()
        self.legacy = JsonSessionStore(legacy_dir) if legacy_dir is not None else None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"session database schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version == 0:
                connection.executescript(
                    """
                    CREATE TABLE sessions (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL DEFAULT '',
                        schema_version INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        model TEXT NOT NULL,
                        workspace TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        estimated_cost REAL,
                        preview TEXT NOT NULL DEFAULT '',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        transcript_complete INTEGER NOT NULL DEFAULT 0
                    );
                    -- Mutable Active Context used for the next LLM request and resume.
                    CREATE TABLE messages (
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        position INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (session_id, position)
                    );
                    -- Immutable original conversation payloads, never rewritten by compaction.
                    CREATE TABLE events (
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        event_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        run_id TEXT,
                        round_number INTEGER,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, sequence),
                        UNIQUE (session_id, event_id)
                    );
                    CREATE TABLE summaries (
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        summary_id TEXT NOT NULL,
                        through_event_sequence INTEGER NOT NULL DEFAULT 0,
                        trigger_name TEXT NOT NULL,
                        actions_json TEXT NOT NULL,
                        before_tokens INTEGER NOT NULL,
                        after_tokens INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        messages_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, summary_id)
                    );
                    CREATE INDEX sessions_updated_at_idx ON sessions(updated_at DESC);
                    CREATE INDEX events_session_created_idx
                        ON events(session_id, created_at);
                    PRAGMA user_version = 3;
                    """
                )
            elif version == 1:
                # v1 stored only the mutable Active Context.  Preserve it and
                # seed the new Transcript, but mark it incomplete because any
                # history compacted before migration cannot be reconstructed.
                connection.executescript(
                    """
                    ALTER TABLE sessions
                        ADD COLUMN transcript_complete INTEGER NOT NULL DEFAULT 0;
                    CREATE TABLE events (
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        event_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        run_id TEXT,
                        round_number INTEGER,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, sequence),
                        UNIQUE (session_id, event_id)
                    );
                    CREATE TABLE summaries (
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        summary_id TEXT NOT NULL,
                        through_event_sequence INTEGER NOT NULL DEFAULT 0,
                        trigger_name TEXT NOT NULL,
                        actions_json TEXT NOT NULL,
                        before_tokens INTEGER NOT NULL,
                        after_tokens INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        messages_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, summary_id)
                    );
                    CREATE INDEX events_session_created_idx
                        ON events(session_id, created_at);
                    """
                )
                rows = connection.execute(
                    """
                    SELECT m.session_id, m.position, m.payload_json, s.updated_at
                    FROM messages AS m
                    JOIN sessions AS s ON s.id = m.session_id
                    ORDER BY m.session_id, m.position
                    """
                ).fetchall()
                seeded = []
                for row in rows:
                    try:
                        message = json.loads(row["payload_json"])
                    except (json.JSONDecodeError, TypeError):
                        message = {"role": "unknown", "content": row["payload_json"]}
                    seeded.append((
                        row["session_id"],
                        row["position"] + 1,
                        f"v1-import-{row['position']}",
                        _event_type(message),
                        row["payload_json"],
                        row["updated_at"],
                    ))
                connection.executemany(
                    """
                    INSERT INTO events (
                        session_id, sequence, event_id, event_type,
                        payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    seeded,
                )
                connection.executescript(
                    """
                    ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT '';
                    UPDATE sessions SET schema_version = 3;
                    PRAGMA user_version = 3;
                    """
                )
            elif version == 2:
                connection.executescript(
                    """
                    ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT '';
                    UPDATE sessions SET schema_version = 3;
                    PRAGMA user_version = 3;
                    """
                )
            connection.execute("PRAGMA journal_mode = WAL")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def save(self, record: SessionRecord) -> str:
        session_id = normalize_session_id(record.id)
        messages = [
            json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            for message in record.messages
        ]
        metadata = json.dumps(record.metadata, ensure_ascii=False, separators=(",", ":"))
        now = _now()
        created_at = record.created_at or now
        with self._connect() as connection:
            # Serialize writers before allocating per-session event sequence
            # numbers.  The snapshot, pending Transcript tail, and summaries
            # then commit or roll back as one unit.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO sessions (
                    id, name, schema_version, created_at, updated_at, model, workspace,
                    status, prompt_tokens, completion_tokens, estimated_cost,
                    preview, metadata_json, transcript_complete
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=CASE
                        WHEN excluded.name = '' THEN sessions.name
                        ELSE excluded.name
                    END,
                    schema_version=excluded.schema_version,
                    updated_at=excluded.updated_at,
                    model=excluded.model,
                    workspace=excluded.workspace,
                    status=excluded.status,
                    prompt_tokens=excluded.prompt_tokens,
                    completion_tokens=excluded.completion_tokens,
                    estimated_cost=excluded.estimated_cost,
                    preview=CASE
                        WHEN sessions.preview = '' THEN excluded.preview
                        ELSE sessions.preview
                    END,
                    metadata_json=excluded.metadata_json,
                    transcript_complete=CASE
                        WHEN sessions.transcript_complete = 0 THEN 0
                        ELSE excluded.transcript_complete
                    END
                """,
                (
                    session_id,
                    normalize_session_name(record.name),
                    SCHEMA_VERSION,
                    created_at,
                    now,
                    record.model,
                    record.workspace,
                    record.status,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.estimated_cost,
                    _preview(record.messages),
                    metadata,
                    int(record.transcript_complete),
                ),
            )
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.executemany(
                "INSERT INTO messages (session_id, position, payload_json) VALUES (?, ?, ?)",
                ((session_id, index, payload) for index, payload in enumerate(messages)),
            )

            next_sequence = int(connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0])
            for event in record.transcript_events:
                event_id = str(event.get("event_id") or uuid.uuid4().hex)
                payload = event.get("payload")
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO events (
                        session_id, sequence, event_id, event_type, run_id,
                        round_number, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        next_sequence,
                        event_id,
                        str(event.get("event_type") or "unknown"),
                        event.get("run_id"),
                        event.get("round"),
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        str(event.get("created_at") or now),
                    ),
                )
                if cursor.rowcount:
                    next_sequence += 1

            through_sequence = next_sequence - 1
            for summary in record.context_summaries:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO summaries (
                        session_id, summary_id, through_event_sequence,
                        trigger_name, actions_json, before_tokens, after_tokens,
                        model, prompt_tokens, completion_tokens, messages_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        str(summary.get("summary_id") or uuid.uuid4().hex),
                        through_sequence,
                        str(summary.get("trigger") or "unknown"),
                        json.dumps(
                            summary.get("actions") or [],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        max(0, int(summary.get("before_tokens") or 0)),
                        max(0, int(summary.get("after_tokens") or 0)),
                        str(summary.get("model") or record.model),
                        max(0, int(summary.get("prompt_tokens") or 0)),
                        max(0, int(summary.get("completion_tokens") or 0)),
                        json.dumps(
                            summary.get("messages") or [],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        str(summary.get("created_at") or now),
                    ),
                )
        return session_id

    def _load_sqlite(self, session_id: str) -> SessionRecord | None:
        safe_id = normalize_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (safe_id,)
            ).fetchone()
            if row is None:
                return None
            message_rows = connection.execute(
                "SELECT payload_json FROM messages WHERE session_id = ? ORDER BY position",
                (safe_id,),
            ).fetchall()
        try:
            return SessionRecord(
                id=row["id"],
                name=row["name"],
                model=row["model"],
                messages=[json.loads(item["payload_json"]) for item in message_rows],
                status=row["status"],
                workspace=row["workspace"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                estimated_cost=row["estimated_cost"],
                metadata=json.loads(row["metadata_json"]),
                transcript_complete=bool(row["transcript_complete"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def load(self, session_id: str) -> SessionRecord | None:
        record = self._load_sqlite(session_id)
        if record is not None or self.legacy is None:
            return record
        record = self.legacy.load(session_id)
        if record is not None:
            self.save(record)
        return record

    def _import_legacy(self):
        if self.legacy is None:
            return
        for summary in self.legacy.list(limit=10_000):
            if self._load_sqlite(summary.id) is None:
                record = self.legacy.load(summary.id)
                if record is not None:
                    self.save(record)

    def list(self, limit: int = 20, offset: int = 0) -> list[SessionSummary]:
        self._import_legacy()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, model, updated_at, status, workspace, preview,
                       prompt_tokens, completion_tokens, estimated_cost,
                       transcript_complete
                FROM sessions
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (max(0, limit), max(0, offset)),
            ).fetchall()
        return [
            SessionSummary(**{**dict(row), "transcript_complete": bool(row["transcript_complete"])})
            for row in rows
        ]

    def delete(self, session_id: str) -> bool:
        safe_id = normalize_session_id(session_id)
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (safe_id,))
        deleted = cursor.rowcount > 0
        if self.legacy is not None:
            deleted = self.legacy.delete(safe_id) or deleted
        return deleted

    def rename(self, session_id: str, name: str) -> bool:
        safe_id = normalize_session_id(session_id)
        normalized = normalize_session_name(name)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (normalized, _now(), safe_id),
            )
        renamed = cursor.rowcount > 0
        if self.legacy is not None and self.legacy.load(safe_id) is not None:
            renamed = self.legacy.rename(safe_id, normalized) or renamed
        return renamed

    def transcript(self, session_id: str) -> list[dict]:
        safe_id = normalize_session_id(session_id)
        # Load first so a legacy JSON session is migrated before querying.
        self.load(safe_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_id, event_type, run_id, round_number,
                       payload_json, created_at
                FROM events
                WHERE session_id = ?
                ORDER BY sequence
                """,
                (safe_id,),
            ).fetchall()
        events = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (json.JSONDecodeError, TypeError):
                payload = {"content": row["payload_json"]}
            events.append({
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "run_id": row["run_id"],
                "round": row["round_number"],
                "payload": payload,
                "created_at": row["created_at"],
            })
        return events

    def summaries(self, session_id: str) -> list[dict]:
        safe_id = normalize_session_id(session_id)
        self.load(safe_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT summary_id, through_event_sequence, trigger_name,
                       actions_json, before_tokens, after_tokens, model,
                       prompt_tokens, completion_tokens, messages_json, created_at
                FROM summaries
                WHERE session_id = ?
                ORDER BY created_at, summary_id
                """,
                (safe_id,),
            ).fetchall()
        result = []
        for row in rows:
            try:
                actions = json.loads(row["actions_json"])
                messages = json.loads(row["messages_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            result.append({
                "summary_id": row["summary_id"],
                "through_event_sequence": row["through_event_sequence"],
                "trigger": row["trigger_name"],
                "actions": actions,
                "before_tokens": row["before_tokens"],
                "after_tokens": row["after_tokens"],
                "model": row["model"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "messages": messages,
                "created_at": row["created_at"],
            })
        return result
