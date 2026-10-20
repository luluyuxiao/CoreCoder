---
name: corecoder-tool-authoring
description: Add or modify CoreCoder tools while preserving schema, capability, permission, scheduling, sandbox, and Tool Result contracts.
---
# CoreCoder Tool Authoring

Use this workflow when the user asks to add or change a built-in Tool, wrap an external capability, or alter Tool registration in this repository.

## Choose the right extension point

- Use a built-in `Tool` for a small capability implemented and maintained inside CoreCoder.
- Use MCP when the capability already lives in another process or service and should be discovered through `tools/list` and invoked through `tools/call`.
- Use a Skill only for reusable workflow instructions; a Skill must not pretend to add executable authority.

Before editing, inspect `corecoder/tools/base.py`, `build_tools()`, the closest existing Tool, its tests, and the Agent execution path. Treat current source as authoritative.

## Preserve the Tool contract

- Give the Tool a unique, stable `name`, a precise `description`, and a JSON Schema `parameters` object that matches `execute()` exactly.
- Return bounded, useful text. Expected operational failures should become clear `Error:` results so the Agent can feed them back to the model; do not hide partial success.
- Assign `ToolEffect.PURE` or `READ` only when the call has no hidden mutation or external side effect. Use `WRITE`, `EXTERNAL`, or `UNKNOWN` conservatively.
- Define resource claims narrowly enough to permit safe concurrency and broadly enough to serialize conflicting access. Never mark a Tool parallel-safe merely to improve UI throughput.
- Declare every required Capability. Capability metadata, Permission, and Sandbox are separate boundaries; none may be omitted because another layer exists.
- Resolve filesystem paths through the active workspace policy and execute commands through the configured executor. Do not open a second ungoverned path to the host.
- Keep user interaction outside `execute()`. Consent belongs to the Agent permission gate, not inside a Tool implementation.

## Integrate end to end

Register the Tool in the intended instance-level tool set and export it only where public library use requires it. Confirm that its schema reaches the LLM, lookup resolves the same name, its result returns as a matching Tool message, and errors do not break the Agent Loop.

When the Tool can mutate files, integrate Checkpoint or Memory bookkeeping only if the existing abstraction genuinely covers the operation. Do not claim Bash, MCP, or remote side effects are reversible when they are not.

## Verify behavior

Add focused tests for schema/argument agreement, success, expected failure, workspace boundaries, and declared Effect/Capabilities. Add scheduling, Permission, Sandbox, Trace, or progress tests when the new behavior crosses those boundaries. Run focused tests first, then the complete suite when practical.

Report the Tool's capability, side-effect classification, registration path, permission behavior, and verification results. Call out any external effect that remains non-reversible.
