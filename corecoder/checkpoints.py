"""Agent-scoped, serializable undo checkpoints for file mutations."""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    path: str
    prior: bytes | None

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "existed": self.prior is not None,
            "prior_base64": (
                base64.b64encode(self.prior).decode("ascii")
                if self.prior is not None else ""
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint | None:
        try:
            path = str(data["path"])
            existed = bool(data.get("existed"))
            prior = (
                base64.b64decode(str(data.get("prior_base64") or ""), validate=True)
                if existed else None
            )
        except (KeyError, TypeError, ValueError):
            return None
        return cls(path=path, prior=prior)


class CheckpointManager:
    """Thread-safe undo stack that can be saved inside a session snapshot."""

    def __init__(self, checkpoints: list[dict] | None = None):
        self._lock = threading.RLock()
        self._stack: list[Checkpoint] = []
        if checkpoints:
            self.restore(checkpoints)

    def record(self, path: Path) -> None:
        resolved = path.expanduser().resolve(strict=False)
        checkpoint = Checkpoint(
            path=str(resolved),
            prior=resolved.read_bytes() if resolved.exists() else None,
        )
        with self._lock:
            self._stack.append(checkpoint)

    def undo(self) -> str:
        with self._lock:
            if not self._stack:
                return "Nothing to undo."
            checkpoint = self._stack.pop()
        path = Path(checkpoint.path)
        if checkpoint.prior is None:
            path.unlink(missing_ok=True)
            return f"Removed {checkpoint.path} (created this session)."
        recreated = not path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(checkpoint.prior)
        if recreated:
            return f"Restored {checkpoint.path} (recreated missing parent directories)."
        return f"Restored {checkpoint.path}."

    def pending(self) -> int:
        with self._lock:
            return len(self._stack)

    def clear(self) -> None:
        with self._lock:
            self._stack.clear()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [checkpoint.as_dict() for checkpoint in self._stack]

    def restore(self, checkpoints: list[dict]) -> None:
        restored = []
        for raw in checkpoints:
            if isinstance(raw, dict):
                checkpoint = Checkpoint.from_dict(raw)
                if checkpoint is not None:
                    restored.append(checkpoint)
        with self._lock:
            self._stack = restored


# Backwards-compatible process-global manager for directly constructed Tools.
DEFAULT_MANAGER = CheckpointManager()


def record(path: Path) -> None:
    DEFAULT_MANAGER.record(path)


def undo() -> str:
    return DEFAULT_MANAGER.undo()


def pending() -> int:
    return DEFAULT_MANAGER.pending()


def clear() -> None:
    DEFAULT_MANAGER.clear()


__all__ = [
    "DEFAULT_MANAGER",
    "Checkpoint",
    "CheckpointManager",
    "clear",
    "pending",
    "record",
    "undo",
]
