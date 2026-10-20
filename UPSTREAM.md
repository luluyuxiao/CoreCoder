# Upstream and fork provenance

## Origin

This repository is derived from the MIT-licensed
[he-yufeng/CoreCoder](https://github.com/he-yufeng/CoreCoder) project.
The Git history is retained, and the original copyright notice remains in
[LICENSE](LICENSE).

The current repository is maintained at:

```text
https://github.com/luluyuxiao/CoreCoder
```

## Why keep this file

Open-source provenance and personal engineering work are separate facts. This
fork does not remove the upstream author's attribution or rewrite commit
history. Instead, it documents which direction the maintained fork takes and
lets the source and tests demonstrate the implementation.

## Fork direction

The current working tree develops CoreCoder from a minimal teaching agent into
a more complete local Agent Harness. The main areas of continued work are:

- resource-aware Tool scheduling and structured progress events;
- provider routes, fallback, cost budgets, and terminal-selectable profiles;
- whole-request context budgeting and dual-layer Session storage;
- structured task Memory that survives compaction and resume;
- stdio/HTTP MCP, dynamic Tool discovery, recovery, and isolation;
- Permission, Hooks, per-Tool Capability Policy, and Docker Sandbox boundaries;
- foreground/background Sub-Agents with optional Git Worktree isolation;
- privacy-aware Trace, repeatable Eval, and baseline comparison;
- persistent Skills and budgeted Tool Catalog discovery.

The source of truth is the current code and test suite, not this list. Features
that are only future directions are identified as such in the README and
architecture documentation.

## Packaging note

The import package and command remain named `corecoder` for compatibility.
Before publishing a public Python distribution, choose a distinct PyPI project
name or coordinate ownership of the existing name. The fork's release workflow
therefore builds an artifact but does not automatically upload it to PyPI.

## License obligations

The MIT license permits use, modification, distribution, sublicensing, and
commercial use. Copies or substantial portions must retain the copyright and
license notice. See [LICENSE](LICENSE) for the complete terms.
