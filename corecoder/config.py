"""Configuration - env vars and defaults."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv():
    """Load .env from cwd, walking up to home dir."""
    from dotenv import load_dotenv

    # search cwd first, then parent dirs up to ~
    env_path = Path(".env")
    if not env_path.exists():
        cur = Path.cwd()
        home = Path.home()
        while cur != home and cur != cur.parent:
            candidate = cur / ".env"
            if candidate.exists():
                env_path = candidate
                break
            cur = cur.parent
    load_dotenv(env_path, override=False)


def _route_specs(raw: str) -> list[dict]:
    if not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"CORECODER_FALLBACK_ROUTES is not valid JSON: {error}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("CORECODER_FALLBACK_ROUTES must be a JSON array of objects")
    return value


@dataclass
class Config:
    model: str = "gpt-5.5"
    fallback_models: list[str] = field(default_factory=list)
    fallback_routes: list[dict] = field(default_factory=list)
    api_key: str = ""
    base_url: str | None = None
    max_cost_usd: float | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"
    sandbox: str = "local"
    sandbox_image: str = "corecoder-sandbox:latest"
    sandbox_network: str = "none"
    sandbox_memory: str = "1g"
    sandbox_cpus: float = 1.0
    sandbox_pids: int = 128
    trace_path: str | None = None
    trace_content: bool = False
    storage_path: str | None = None
    autosave: bool = True
    capability_policy_path: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            os.getenv("CORECODER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        fallback_models = [
            model.strip()
            for model in os.getenv("CORECODER_FALLBACK_MODELS", "").split(",")
            if model.strip()
        ]
        max_cost = os.getenv("CORECODER_MAX_COST_USD")
        return cls(
            model=os.getenv("CORECODER_MODEL", "gpt-5.5"),
            fallback_models=fallback_models,
            fallback_routes=_route_specs(os.getenv("CORECODER_FALLBACK_ROUTES", "")),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL"),
            max_cost_usd=float(max_cost) if max_cost else None,
            max_tokens=int(os.getenv("CORECODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("CORECODER_MAX_CONTEXT", "128000")),
            provider=os.getenv("CORECODER_PROVIDER", "openai"),
            sandbox=os.getenv("CORECODER_SANDBOX", "local"),
            sandbox_image=os.getenv("CORECODER_SANDBOX_IMAGE", "corecoder-sandbox:latest"),
            sandbox_network=os.getenv("CORECODER_SANDBOX_NETWORK", "none"),
            sandbox_memory=os.getenv("CORECODER_SANDBOX_MEMORY", "1g"),
            sandbox_cpus=float(os.getenv("CORECODER_SANDBOX_CPUS", "1")),
            sandbox_pids=int(os.getenv("CORECODER_SANDBOX_PIDS", "128")),
            trace_path=os.getenv("CORECODER_TRACE"),
            trace_content=_env_bool("CORECODER_TRACE_CONTENT"),
            storage_path=os.getenv("CORECODER_STORAGE_PATH"),
            autosave=_env_bool("CORECODER_AUTOSAVE", True),
            capability_policy_path=os.getenv("CORECODER_CAPABILITY_POLICY"),
        )
