"""Opt-in PreToolUse hook that protects common secret and production paths.

Install the sample ``examples/hooks.protect-sensitive.json`` as
``~/.corecoder/hooks.json`` to enable it.  The hook deliberately handles only
structured file tools: guessing the effects of arbitrary shell text would
create a false security boundary.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

MUTATING_FILE_TOOLS = frozenset({"write_file", "edit_file"})
DEFAULT_PATTERNS = (
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    ".git/*",
    "*/.git/*",
    "secrets/*",
    "*/secrets/*",
    "production.yaml",
    "production.yml",
)


def _path_candidates(raw_path: str, *, cwd: Path | None = None) -> set[str]:
    """Return lexical and resolved forms so ``..`` and symlinks cannot disguise a path."""
    root = (cwd or Path.cwd()).expanduser().resolve()
    raw = raw_path.replace("\\", "/")
    candidates = {raw, raw.removeprefix("./"), Path(raw_path).name}
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=False)
    candidates.add(resolved.as_posix())
    candidates.add(resolved.name)
    try:
        candidates.add(resolved.relative_to(root).as_posix())
    except ValueError:
        pass
    return {candidate for candidate in candidates if candidate}


def blocked_reason(
    payload: dict,
    *,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
    cwd: Path | None = None,
) -> str | None:
    """Return a model-facing reason when this structured file mutation is protected."""
    if not isinstance(payload, dict) or payload.get("tool_name") not in MUTATING_FILE_TOOLS:
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw_path = tool_input.get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    for pattern in patterns:
        if any(
            fnmatch.fnmatchcase(candidate, pattern)
            for candidate in _path_candidates(raw_path, cwd=cwd)
        ):
            return (
                f"protected path {raw_path!r} matched policy {pattern!r}; "
                "choose a non-sensitive target or ask the user to change the Hook policy"
            )
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Block write_file/edit_file calls targeting protected paths."
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Add a case-sensitive path glob to the built-in protected patterns.",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Use only patterns supplied with --pattern.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    patterns = tuple(args.pattern)
    if not args.no_defaults:
        patterns = DEFAULT_PATTERNS + patterns
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as error:
        print(f"invalid CoreCoder Hook payload: {error}", file=sys.stderr)
        return 1
    reason = blocked_reason(payload, patterns=patterns)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
