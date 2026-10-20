"""Cross-agent read/write locks for concrete tool resources."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ResourceClaim:
    """A read or write claim on one stable resource key."""

    key: str
    access: str = "write"

    def __post_init__(self):
        if not self.key:
            raise ValueError("resource key must not be empty")
        if self.access not in {"read", "write"}:
            raise ValueError("resource access must be 'read' or 'write'")


class ResourceLockManager:
    """Acquire multiple resource claims atomically with reader/writer semantics."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._readers: dict[str, int] = {}
        self._writers: set[str] = set()

    @staticmethod
    def normalize(claims: Iterable[ResourceClaim]) -> tuple[ResourceClaim, ...]:
        merged: dict[str, str] = {}
        for claim in claims:
            previous = merged.get(claim.key)
            merged[claim.key] = (
                "write" if claim.access == "write" or previous == "write" else "read"
            )
        return tuple(
            ResourceClaim(key, access)
            for key, access in sorted(merged.items())
        )

    @staticmethod
    def conflict(
        left: Iterable[ResourceClaim],
        right: Iterable[ResourceClaim],
    ) -> bool:
        left_by_key = {claim.key: claim.access for claim in left}
        for claim in right:
            other = left_by_key.get(claim.key)
            if other is not None and (other == "write" or claim.access == "write"):
                return True
        return False

    def _available(self, claims: tuple[ResourceClaim, ...]) -> bool:
        for claim in claims:
            if claim.access == "read":
                if claim.key in self._writers:
                    return False
            elif claim.key in self._writers or self._readers.get(claim.key, 0):
                return False
        return True

    @contextmanager
    def acquire(self, claims: Iterable[ResourceClaim]) -> Iterator[None]:
        normalized = self.normalize(claims)
        if not normalized:
            yield
            return
        with self._condition:
            while not self._available(normalized):
                self._condition.wait()
            for claim in normalized:
                if claim.access == "read":
                    self._readers[claim.key] = self._readers.get(claim.key, 0) + 1
                else:
                    self._writers.add(claim.key)
        try:
            yield
        finally:
            with self._condition:
                for claim in normalized:
                    if claim.access == "read":
                        remaining = self._readers[claim.key] - 1
                        if remaining:
                            self._readers[claim.key] = remaining
                        else:
                            del self._readers[claim.key]
                    else:
                        self._writers.remove(claim.key)
                self._condition.notify_all()


__all__ = ["ResourceClaim", "ResourceLockManager"]
