"""Base class for all tools."""

import threading
from abc import ABC, abstractmethod

from ..capabilities import UNKNOWN
from ..resources import ResourceClaim


class ToolEffect:
    """Coarse side-effect metadata used by the tool scheduler.

    This is deliberately separate from permissions: permissions decide whether
    a call is allowed, while effects decide whether two allowed calls may run
    at the same time.  Unknown tools fail closed and run exclusively.
    """

    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args
    effect: str = ToolEffect.UNKNOWN
    # Stateful tools opt in only when they declare concrete resource claims.
    resource_parallel = False
    # Unknown/custom tools fail closed under a restrictive CapabilityPolicy.
    capabilities: frozenset[str] = frozenset({UNKNOWN})

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def is_concurrency_safe(self) -> bool:
        """Whether calls to this tool may share a parallel read batch."""
        return self.effect in {ToolEffect.PURE, ToolEffect.READ}

    def parallel_group(self, arguments: dict) -> str | None:
        """Scheduler group for this call, or ``None`` for an exclusive barrier.

        Read/pure calls retain their original parallel behavior. A stateful tool
        must explicitly opt in with ``resource_parallel`` and provide resource
        claims, which lets different resources overlap without racing the same
        file or service.
        """
        if self.effect in {ToolEffect.PURE, ToolEffect.READ}:
            return "read"
        if self.resource_parallel and self.resource_claims(arguments):
            return "write"
        return None

    def resource_claims(self, arguments: dict) -> tuple[ResourceClaim, ...]:
        """Concrete resources used by one call for cross-Agent locking."""
        if self.effect == ToolEffect.PURE:
            return ()
        access = "read" if self.effect == ToolEffect.READ else "write"
        return (ResourceClaim(f"tool:{self.name}", access),)

    def required_capabilities(self, arguments: dict) -> frozenset[str]:
        """Authority this call may exercise, checked before Permission."""
        return self.capabilities

    def execution_lock(self):
        """Per-instance lock for unsafe tools shared by multiple Agents."""
        lock = getattr(self, "_corecoder_execution_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._corecoder_execution_lock = lock
        return lock
