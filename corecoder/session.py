"""Compatibility facade over CoreCoder's pluggable session storage.

The original public helpers still return ``(messages, model)`` tuples, while
new callers can use ``SessionRecord`` and ``SessionStore`` for status, usage,
workspace, and crash-recovery metadata.
"""

from __future__ import annotations

from pathlib import Path

from .storage import (
    JsonSessionStore,
    SessionRecord,
    SessionStore,
    SQLiteSessionStore,
    new_session_id,
    normalize_session_id,
)

SESSIONS_DIR = Path.home() / ".corecoder" / "sessions"
DEFAULT_DATABASE_NAME = "sessions.db"


def create_session_store(path: str | Path | None = None) -> SQLiteSessionStore:
    """Create the default SQLite store with legacy JSON read-through."""
    database = Path(path).expanduser() if path is not None else SESSIONS_DIR / DEFAULT_DATABASE_NAME
    return SQLiteSessionStore(database, legacy_dir=SESSIONS_DIR)


def save_session(
    messages: list[dict],
    model: str,
    session_id: str | None = None,
    *,
    name: str = "",
    store: SessionStore | None = None,
    status: str = "saved",
    workspace: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    estimated_cost: float | None = None,
    metadata: dict | None = None,
) -> str:
    """Save one complete, stable conversation snapshot."""
    target = store or create_session_store()
    record = SessionRecord(
        id=normalize_session_id(session_id),
        model=model,
        messages=messages,
        name=name,
        status=status,
        workspace=workspace,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost=estimated_cost,
        metadata=dict(metadata or {}),
    )
    return target.save(record)


def save_snapshot(
    session_id: str,
    snapshot: dict,
    *,
    store: SessionStore | None = None,
) -> str:
    """Persist the storage-neutral snapshot emitted by ``Agent``."""
    target = store or create_session_store()
    return target.save(SessionRecord.from_snapshot(session_id, snapshot))


def load_session_record(
    session_id: str,
    *,
    store: SessionStore | None = None,
) -> SessionRecord | None:
    target = store or create_session_store()
    return target.load(session_id)


def load_session(
    session_id: str,
    *,
    store: SessionStore | None = None,
) -> tuple[list[dict], str] | None:
    """Load a session using the original ``(messages, model)`` API."""
    record = load_session_record(session_id, store=store)
    return (record.messages, record.model) if record is not None else None


def load_transcript(
    session_id: str,
    *,
    store: SessionStore | None = None,
) -> list[dict]:
    """Load the append-only original conversation event stream."""
    target = store or create_session_store()
    return target.transcript(session_id)


def load_context_summaries(
    session_id: str,
    *,
    store: SessionStore | None = None,
) -> list[dict]:
    """Load the recorded context-compaction outputs for one session."""
    target = store or create_session_store()
    return target.summaries(session_id)


def list_sessions(
    *,
    store: SessionStore | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    target = store or create_session_store()
    return [item.as_dict() for item in target.list(limit=limit, offset=offset)]


def delete_session(session_id: str, *, store: SessionStore | None = None) -> bool:
    target = store or create_session_store()
    return target.delete(session_id)


def rename_session(
    session_id: str,
    name: str,
    *,
    store: SessionStore | None = None,
) -> bool:
    """Assign a human-facing name without changing the stable session ID."""
    target = store or create_session_store()
    return target.rename(session_id, name)


# Private aliases retained for forks that imported the old helpers in tests.
_normalize_session_id = normalize_session_id
_new_session_id = new_session_id

__all__ = [
    "DEFAULT_DATABASE_NAME",
    "SESSIONS_DIR",
    "JsonSessionStore",
    "SQLiteSessionStore",
    "SessionRecord",
    "SessionStore",
    "create_session_store",
    "delete_session",
    "list_sessions",
    "load_context_summaries",
    "load_session",
    "load_session_record",
    "load_transcript",
    "new_session_id",
    "normalize_session_id",
    "rename_session",
    "save_session",
    "save_snapshot",
]
