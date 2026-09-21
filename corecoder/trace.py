"""Structured, opt-in execution traces for agents, tools, and providers.

Tracing is disabled by default because prompts and tool payloads may contain
source code or secrets. JSONL traces record metadata only unless the caller
explicitly enables ``capture_content``.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_CONTENT_FIELDS = frozenset({
    "arguments",
    "messages_after",
    "messages_before",
    "result",
    "task",
    "user_input",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value, depth: int = 0):
    """Bound arbitrary event values before they reach a trace sink."""
    if depth > 6:
        return "<max depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 20_000 else value[:20_000] + "... <truncated>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in list(value)[:200]]
    return repr(value)[:2_000]


class TraceSink:
    """Small synchronous event sink interface used throughout CoreCoder."""

    capture_content = False

    def emit(self, event: str, **fields):
        """Record one event. Implementations must be safe across threads."""

    def content(self, **fields) -> dict:
        """Return sensitive payload fields only when explicitly enabled."""
        return fields if self.capture_content else {}


class NullTrace(TraceSink):
    """Default zero-I/O sink."""

    def emit(self, event: str, **fields):
        return None


NULL_TRACE = NullTrace()


class _RecordingTrace(TraceSink):
    def __init__(self, *, capture_content: bool = False, session_id: str | None = None):
        self.capture_content = capture_content
        self.session_id = session_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._sequence = 0

    def _record(self, event: str, fields: dict) -> dict:
        self._sequence += 1
        return {
            "timestamp": _utc_now(),
            "sequence": self._sequence,
            "session_id": self.session_id,
            "event": event,
            **_json_safe(fields),
        }


class MemoryTrace(_RecordingTrace):
    """In-memory sink used by tests, evals, and library integrations."""

    def __init__(self, *, capture_content: bool = False, session_id: str | None = None):
        super().__init__(capture_content=capture_content, session_id=session_id)
        self.events: list[dict] = []

    def emit(self, event: str, **fields):
        with self._lock:
            self.events.append(self._record(event, fields))


class JsonlTrace(_RecordingTrace):
    """Append thread-safe events to one JSON object per line."""

    def __init__(
        self,
        path: str | Path,
        *,
        capture_content: bool = False,
        session_id: str | None = None,
        append: bool = True,
    ):
        super().__init__(capture_content=capture_content, session_id=session_id)
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Validate the destination while CLI errors can still be reported
        # clearly; individual runtime writes remain best-effort.
        self.path.open("a" if append else "w", encoding="utf-8").close()

    def emit(self, event: str, **fields):
        with self._lock:
            record = self._record(event, fields)
            try:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as e:
                log.warning("trace event %s could not be written to %s: %s", event, self.path, e)


class CompositeTrace(TraceSink):
    """Fan out one event to multiple sinks."""

    def __init__(self, *sinks: TraceSink):
        self.sinks = [sink for sink in sinks if sink is not None]
        self.capture_content = any(sink.capture_content for sink in self.sinks)

    def emit(self, event: str, **fields):
        for sink in self.sinks:
            try:
                sink_fields = fields
                if not sink.capture_content:
                    sink_fields = {
                        key: value
                        for key, value in fields.items()
                        if key not in _CONTENT_FIELDS
                    }
                sink.emit(event, **sink_fields)
            except Exception as e:  # noqa: BLE001
                # Observability must never become a new failure mode for the
                # agent, and one broken destination must not starve the rest.
                log.warning(
                    "trace sink %s failed for %s: %s",
                    type(sink).__name__,
                    event,
                    e,
                )
