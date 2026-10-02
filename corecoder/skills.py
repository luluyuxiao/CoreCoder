"""Discover reusable, lazily loaded Agent instructions.

Skills are deliberately smaller than tools: they teach the model *how* to
approach a task but add no executable capability and never bypass Permission
or Sandbox.  CoreCoder discovers user skills first, then lets a project skill
with the same name override it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

log = logging.getLogger(__name__)

GLOBAL_SKILLS_DIR = Path.home() / ".corecoder" / "skills"
PROJECT_SKILLS_DIR = Path(".corecoder") / "skills"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_SKILL_CHARS = 32_000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path
    scope: str

    @property
    def content_hash(self) -> str:
        """Stable identity for the exact instructions injected into the LLM."""
        return sha256(self.render().encode("utf-8")).hexdigest()

    def render(self) -> str:
        return (
            f"# Loaded skill: {self.name}\n\n"
            f"{self.instructions.strip()}\n\n"
            "---\n"
            "This skill supplies workflow instructions only. It does not grant tools, "
            "permissions, network access, or a sandbox exception."
        )


class SkillRegistry:
    """Immutable-by-convention index of validated ``SKILL.md`` files."""

    def __init__(self, skills: dict[str, Skill] | None = None):
        self._skills = dict(skills or {})

    @classmethod
    def discover(
        cls,
        workspace: str | Path,
        *,
        global_dir: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> SkillRegistry:
        workspace = Path(workspace).expanduser().resolve()
        roots = [
            ("global", Path(global_dir).expanduser() if global_dir else GLOBAL_SKILLS_DIR),
            (
                "project",
                Path(project_dir).expanduser()
                if project_dir
                else workspace / PROJECT_SKILLS_DIR,
            ),
        ]
        found: dict[str, Skill] = {}
        for scope, root in roots:
            root = root.resolve()
            if not root.is_dir():
                continue
            for skill_file in sorted(root.glob("*/SKILL.md")):
                try:
                    skill = _read_skill(skill_file, root, scope)
                except (OSError, UnicodeError, ValueError) as exc:
                    log.warning("ignoring invalid skill %s: %s", skill_file, exc)
                    continue
                found[skill.name] = skill
        return cls(found)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def catalog(self) -> str:
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in sorted(self._skills.values(), key=lambda item: item.name)
        )

    def load(self, name: str) -> str:
        skill = self.get(name)
        if skill is None:
            available = ", ".join(self.names()) or "none"
            return f"Error: unknown skill {name!r}. Available skills: {available}"
        return skill.render()


def _read_skill(path: Path, root: Path, scope: str) -> Skill:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("SKILL.md resolves outside its skill root") from exc

    text = resolved.read_text(encoding="utf-8")
    if len(text) > _MAX_SKILL_CHARS:
        raise ValueError(f"SKILL.md exceeds {_MAX_SKILL_CHARS} characters")
    metadata, instructions = _parse_frontmatter(text)
    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError("name must match [a-z0-9][a-z0-9_-]{0,63}")
    if not description or len(description) > 300:
        raise ValueError("description must contain 1-300 characters")
    if not instructions.strip():
        raise ValueError("skill instructions are empty")
    return Skill(
        name=name,
        description=description,
        instructions=instructions,
        path=resolved,
        scope=scope,
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must begin with frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("SKILL.md frontmatter is not closed") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata, "\n".join(lines[end + 1:]).strip()


__all__ = ["GLOBAL_SKILLS_DIR", "PROJECT_SKILLS_DIR", "Skill", "SkillRegistry"]
