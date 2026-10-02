---
name: corecoder-review
description: Review CoreCoder changes with emphasis on Agent protocol correctness, safety boundaries, context, and durable session state.
---
# CoreCoder Review

Use this workflow when the user asks for a code review, regression audit, or architectural check of this repository.

## Review workflow

1. Treat the current local source as the only authority. Inspect the relevant implementation and tests; do not infer behavior from README claims alone.
2. Stay read-only unless the user explicitly asks for fixes. Prefer `read_file`, `grep`, and `glob`; inspect Git status or diffs only when needed to understand the change.
3. Trace each changed behavior through its complete runtime path instead of reviewing one function in isolation.
4. Check the protocol invariants below before discussing style.
5. Run focused tests when safe and useful, then distinguish verified behavior from inference.
6. Report actionable findings first, ordered by severity, with precise file and line references. If no defect is found, say so and name the remaining test gaps or risks.

## CoreCoder invariants

- Every assistant `tool_calls` message must have one matching Tool Result per call ID before the next provider request or resumable Active Context snapshot.
- Tool effects control scheduling only; Permission controls authorization, and Sandbox limits impact. None of these layers may silently substitute for another.
- Capability Policy must deny undeclared Tool authority before Permission or execution; sandboxed MCP capabilities must reflect its actual container network and workspace mounts rather than optimistic metadata.
- Only explicitly `PURE` or `READ` tools may run concurrently. Writes, external calls, and unknown effects are ordering barriers.
- A fresh Tool Result should be consumed by one successful LLM request before ordinary context snipping; the complete request budget includes system prompt, Tool Schemas, output reserve, and safety margin.
- Transcript Events remain append-only while Active Context may be replaced by compression. Resume loads Active Context, not the full Transcript.
- Loaded Skill activation state must live outside mutable Active Context, survive Session resume, and re-inject instructions only while the saved content hash matches the currently discovered `SKILL.md`; changed or missing Skills fail closed.
- Storage, Trace, hooks, progress callbacks, and child-agent failures are best-effort infrastructure and should not unexpectedly kill the parent Agent Loop.
- Provider fallback and usage accounting must remain attributable to the actual model without double-counting retries.
- MCP and custom tools default to untrusted/unknown effects and must still pass through hooks, Permission, and ordinary Tool Result handling.

## Output format

For each finding provide severity, evidence, impact, and the smallest safe fix direction. Avoid rewriting code during a review-only request.
