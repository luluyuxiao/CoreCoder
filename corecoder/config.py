"""Configuration - env vars and defaults."""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class ModelProfile:
    """A named, non-secret LLM connection preset.

    Credentials deliberately remain in environment variables.  A profile may
    select a provider, endpoint, model and policy without putting an API key in
    a checked-in config file or a saved Session.
    """

    name: str
    model: str
    provider: str = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    fallback_models: tuple[str, ...] = ()
    fallback_routes: tuple[dict, ...] = ()
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    description: str = ""

    def resolve_api_key(self, inherited: str) -> str:
        """Read an explicitly named credential, or retain the base config key."""
        if self.api_key_env is None:
            return inherited
        value = os.getenv(self.api_key_env, "")
        if not value:
            raise ValueError(
                f"model profile {self.name!r} requires environment variable "
                f"{self.api_key_env!r}, but it is not set"
            )
        return value


@dataclass(frozen=True)
class ModelProfiles:
    """Validated named model presets loaded from one JSON file."""

    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    default: str | None = None
    source: Path | None = None

    def get(self, name: str) -> ModelProfile | None:
        return self.profiles.get(name)

    def require(self, name: str) -> ModelProfile:
        profile = self.get(name)
        if profile is None:
            available = ", ".join(sorted(self.profiles)) or "(none)"
            raise ValueError(
                f"unknown model profile {name!r}; available profiles: {available}"
            )
        return profile

    def names(self) -> list[str]:
        return sorted(self.profiles)


def _profile_error(source: Path, detail: str) -> ValueError:
    return ValueError(f"invalid model profiles file {source}: {detail}")


def _profile_string(
    source: Path,
    name: str,
    data: dict[str, Any],
    key: str,
) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _profile_error(source, f"model {name!r}.{key} must be a non-empty string")
    return value.strip()


def _model_profile(source: Path, name: str, value: Any) -> ModelProfile:
    if not isinstance(name, str) or not name.strip():
        raise _profile_error(source, "model profile names must be non-empty strings")
    if not isinstance(value, dict):
        raise _profile_error(source, f"model {name!r} must be an object")
    model = _profile_string(source, name, value, "model")
    if model is None:
        raise _profile_error(source, f"model {name!r}.model is required")
    provider = str(value.get("provider", "openai")).lower()
    if provider not in {"openai", "litellm"}:
        raise _profile_error(
            source, f"model {name!r}.provider must be 'openai' or 'litellm'"
        )
    fallback_models = value.get("fallback_models", [])
    if not isinstance(fallback_models, list) or not all(
        isinstance(item, str) and item.strip() for item in fallback_models
    ):
        raise _profile_error(source, f"model {name!r}.fallback_models must be a string list")
    fallback_routes = value.get("fallback_routes", [])
    if not isinstance(fallback_routes, list) or not all(
        isinstance(item, dict) for item in fallback_routes
    ):
        raise _profile_error(source, f"model {name!r}.fallback_routes must be an object list")

    def optional_number(key: str, *, positive: bool = False) -> float | None:
        raw = value.get(key)
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise _profile_error(source, f"model {name!r}.{key} must be a number")
        number = float(raw)
        if positive and number <= 0:
            raise _profile_error(source, f"model {name!r}.{key} must be greater than zero")
        return number

    raw_max_tokens = value.get("max_tokens")
    if raw_max_tokens is not None and (
        isinstance(raw_max_tokens, bool)
        or not isinstance(raw_max_tokens, int)
        or raw_max_tokens <= 0
    ):
        raise _profile_error(source, f"model {name!r}.max_tokens must be a positive integer")

    return ModelProfile(
        name=name.strip(),
        model=model,
        provider=provider,
        base_url=_profile_string(source, name, value, "base_url"),
        api_key_env=_profile_string(source, name, value, "api_key_env"),
        fallback_models=tuple(item.strip() for item in fallback_models),
        fallback_routes=tuple(dict(item) for item in fallback_routes),
        max_cost_usd=optional_number("max_cost_usd", positive=True),
        max_tokens=raw_max_tokens,
        temperature=optional_number("temperature"),
        description=str(value.get("description") or "").strip(),
    )


def _default_model_profiles_path() -> Path | None:
    """Prefer a project preset file, then the user's private preset file."""
    project = Path.cwd() / ".corecoder" / "models.json"
    if project.is_file():
        return project
    user = Path.home() / ".corecoder" / "models.json"
    return user if user.is_file() else None


def load_model_profiles(path: str | Path | None = None) -> ModelProfiles:
    """Load named model presets without ever serializing their credentials."""
    source = Path(path).expanduser() if path is not None else _default_model_profiles_path()
    if source is None:
        return ModelProfiles()
    source = source.resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read model profiles file {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid model profiles JSON {source}: {error}") from error
    if not isinstance(raw, dict):
        raise _profile_error(source, "top level must be an object")
    unknown = set(raw) - {"default", "models"}
    if unknown:
        raise _profile_error(source, f"unknown top-level keys: {', '.join(sorted(unknown))}")
    models = raw.get("models", {})
    if not isinstance(models, dict):
        raise _profile_error(source, "models must be an object keyed by profile name")
    profiles = {
        name.strip(): _model_profile(source, name, value)
        for name, value in models.items()
    }
    default = raw.get("default")
    if default is not None and (
        not isinstance(default, str) or default not in profiles
    ):
        raise _profile_error(source, "default must name one configured model profile")
    return ModelProfiles(profiles=profiles, default=default, source=source)


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
    tool_schema_budget_tokens: int | None = None
    tool_retrieval_limit: int = 5
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
    model_profile: str | None = None
    model_profiles_path: str | None = None

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
        tool_schema_budget = os.getenv("CORECODER_TOOL_SCHEMA_BUDGET")
        tool_retrieval_limit = os.getenv("CORECODER_TOOL_RETRIEVAL_LIMIT")
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
            tool_schema_budget_tokens=(
                int(tool_schema_budget) if tool_schema_budget else None
            ),
            tool_retrieval_limit=(
                int(tool_retrieval_limit) if tool_retrieval_limit else 5
            ),
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
            model_profile=os.getenv("CORECODER_MODEL_PROFILE") or None,
            model_profiles_path=os.getenv("CORECODER_MODEL_PROFILES") or None,
        )


def apply_model_profile(
    config: Config,
    profile: ModelProfile,
    *,
    base: Config | None = None,
) -> Config:
    """Return config with one profile over a stable CLI/environment baseline.

    ``base`` lets an interactive switch avoid carrying a previous profile's
    endpoint or fallback chain into the next one.  Its secret API key is used
    only when the selected profile intentionally omits ``api_key_env``.
    """
    baseline = base or config
    applied = replace(
        baseline,
        model=profile.model,
        provider=profile.provider,
        base_url=profile.base_url,
        api_key=profile.resolve_api_key(baseline.api_key),
        fallback_models=list(profile.fallback_models),
        fallback_routes=[dict(item) for item in profile.fallback_routes],
        max_cost_usd=(
            profile.max_cost_usd
            if profile.max_cost_usd is not None else baseline.max_cost_usd
        ),
        max_tokens=(profile.max_tokens if profile.max_tokens is not None else baseline.max_tokens),
        temperature=(
            profile.temperature if profile.temperature is not None else baseline.temperature
        ),
        model_profile=profile.name,
    )
    return applied
