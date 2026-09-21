# Parallel execution and sub-agents

When piece one covered the main loop, I left a gap at the "execute tools" step: the model wanting one tool at a time and the model wanting several at once take two different paths. This piece first fills that gap, then covers a tool that lets the agent spawn a "clone" of itself. The two are really two sides of one theme: how to let one agent handle several things at once without scrambling them together.

## Several tool calls come back at once

The model does not always request one tool at a time. Ask it to inspect three files and it may return three `read_file` calls in one response. Those reads are independent and worth overlapping. But if the same response also contains `write_file` or `bash`, throwing the entire batch into a pool creates ordering and race problems.

CoreCoder therefore gives every `Tool` coarse side-effect metadata instead of deciding solely from the number of calls:

```python
class ToolEffect:
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

class Tool(ABC):
    effect = ToolEffect.UNKNOWN

    def is_concurrency_safe(self) -> bool:
        return self.effect in {ToolEffect.PURE, ToolEffect.READ}
```

`read_file`, `glob`, `grep`, and `agent_status` are `READ`; `now` is `PURE`; `write_file`, `edit_file`, and `todo_write` are `WRITE`; and `bash`, `fetch_url`, and `agent` are `EXTERNAL`. Third-party tools and MCP tools remain `UNKNOWN` unless CoreCoder has a trustworthy effect contract. Defaulting to unknown is fail-closed: sacrificing some parallelism is preferable to betting that an external tool has no shared state or side effects.

The main loop still enters `_exec_tools_parallel` when a response contains multiple calls, but the name now identifies the scheduling entry point—not a promise that every call runs concurrently. Condensed into pseudocode, the implementation is:

```python
results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]
pending = [i for i, result in enumerate(results) if result is None]

cursor = 0
while cursor < len(pending):
    if current_tool.is_concurrency_safe():
        batch, cursor = take_consecutive_safe_calls(pending, cursor)
        run_in_thread_pool(batch, max_workers=8)
    else:
        run_exclusively(current_call)
        cursor += 1
```

The scheduler groups only **consecutive** `PURE/READ` calls into a pool of at most eight threads. Every `WRITE/EXTERNAL/UNKNOWN` call is an exclusive barrier: it and its post hook finish before later calls start. Thus `read(A), read(B), write(C), read(D)` runs as “A/B concurrently, then C, then D,” while tool results are still appended to the conversation in the model's original order.

Pre hooks and permission checks remain on the calling thread so several workers never prompt on one terminal at once. Only approved calls reach the scheduler. A call blocked by a hook or permission already has its textual result, does not execute, and does not fire a post hook.

The CLI observes this scheduler through `Agent.chat(..., on_tool_progress=...)`. The callback receives one `batch_started` event after all permission decisions, per-call `tool_started` and `tool_completed` events (possibly from worker threads), then `batch_completed`. That ordering lets Rich render queued/running/done/error/blocked rows and an `n/N` total without competing with an interactive permission prompt. The heading says `Parallel tools` only when the allowed calls contain a batch the scheduler will actually overlap; a merely plural but serial batch says `Tool batch`. The callback is presentation only: it is thread-safe in the CLI, and Agent catches callback failures so a broken display cannot fail a tool call.

Threads fit the safe workload because file reads and searches are largely IO-bound. The policy is intentionally conservative, however: it cannot tell that two writes target unrelated files, or that two MCP tools are genuinely read-only. A later resource-aware scheduler could extend `effect` with read/write sets, path locks, or a dependency DAG.

## Where this simplification falls short of a fuller scheduler

First, CoreCoder waits for the streamed response to finish and for every tool call to be assembled before execution; it does not speculatively launch a complete call while later tokens are still arriving. Second, `ToolEffect` is a coarse category, not proof that two stateful calls cannot conflict, and MCP annotations are not yet promoted into a trusted scheduling policy. In return, the rules stay small and deterministic, and unknown tools are never accidentally parallelized.

## Parallelism isn't free: shared mutable state will bite

Effect metadata governs one Agent's batch scheduler. A tool still needs to tolerate other ways it may be used concurrently—for example, two Agents sharing an instance, or a library caller invoking it directly from a thread pool.

`BashTool` remembers `cd` across commands. Instead of storing cwd in one global string, it keeps a `threading.local()` on each tool instance:

```python
self._local = threading.local()
cwd = getattr(self._local, "cwd", None) or str(default_cwd)
# after a successful cd:
self._local.cwd = running
```

The scheduler currently marks `bash` as `EXTERNAL`, so bash calls in one model batch do not overlap. Per-instance, per-thread cwd isolation is a second line of defense for separate Agents and direct concurrent callers. The core lesson remains: **adding parallelism imposes a concurrency-correctness requirement on every tool with mutable state.** `ToolEffect` controls whether the scheduler dares to overlap calls; internal isolation or locking controls whether the tool stays correct through other concurrent entry points. Neither replaces the other.

## Sub-agents: spawning a clone of yourself

The `agent` tool solves a different problem. Some subtasks are heavy, say "go over this unfamiliar codebase and tell me how authentication is implemented." If the main agent does this itself, it has to read a pile of files and run a bunch of searches, and all that intermediate process piles into the main conversation's window; by the time it's figured things out, the window is nearly stuffed with exploration garbage and the real task has no room left.

The sub-agent's idea is: dispatch a clone with its own independent context to do this heavy work, let it churn in its own window, and hand back only a distilled conclusion when done. The main agent's window stays clean throughout, with just one extra line, "authentication is implemented this way."

The current `agent` splits two concerns into orthogonal parameters:

```python
agent(
    task="analyze authentication and report",
    run_mode="foreground" | "background",
    isolation="shared" | "worktree",
)
```

`run_mode` answers “does the parent wait?” `foreground` preserves blocking behavior; `background` runs in a daemon thread and immediately returns a task ID. `isolation` answers “which checkout does it work in?” `shared` uses the current checkout; `worktree` creates a `corecoder/subagent-<id>` branch and independent worktree from the parent's current `HEAD`. The choices combine freely, including background work inside a worktree.

Every mode constructs a fresh `Agent`, so the child owns new `messages` and a new `ContextManager`. It shares the parent's LLM, permission layer, and hooks, while built-ins are reinstantiated: its own `BashTool` cwd and todo list; in worktree mode file tools are forcibly rooted to the new workspace. Model spend stays on one ledger. A shared lock keeps concurrent parent/child model calls from corrupting token and USD-budget accounting, while tool phases may still overlap.

The child keeps a 20-round cap, truncates final text over 5,000 characters, and converts failure into an ordinary result rather than crashing the parent.

## How a background task comes back

A background `agent` call returns a task ID rather than pretending the work is complete. The new read-only `agent_status` tool lists retained jobs, polls one ID, or waits up to 60 seconds via `wait_seconds`. REPL `/agents` reads the same state.

```python
started = agent(task="run the full suite", run_mode="background")
# => Task ID: 8f31...
agent_status(task_id="8f31...", wait_seconds=30)
```

The table has two bounds: at most four active background children and 32 retained statuses. These are in-process daemon threads, not a durable service; exiting CoreCoder does not guarantee they continue. This is background execution, not a persistent job system.

There is also a terminal race that is easy to miss. A worker inheriting interactive permission could steal stdin while the user is typing. Background children therefore use only `--yes` or existing “always allow” decisions; every other stateful call fails closed as an ordinary tool result. Use foreground when per-call confirmation is required.

## What worktree isolation actually isolates

A worktree starts from `HEAD`, so uncommitted parent changes are deliberately not copied. The result reports the retained path, branch, and `git status --short`, leaving the user to inspect, commit, merge, or cherry-pick. This is not a security sandbox: local bash still sees the host, and hooks plus MCP/custom tools have no universal “rebind to worktree” contract. Docker remains the execution boundary; worktrees prevent parallel code edits from trampling the same files.

If the parent uses the Docker executor, the child copies its image, network, and resource limits while changing only the mounted root. Worktree mode never silently downgrades Docker to a local shell, and a custom command executor without a rebinding contract fails closed.

## Why a sub-agent isn't allowed to spawn a grand-sub-agent

When spawning a child, its tool set drops both `agent` and `agent_status`. It can neither spawn descendants nor receive an orphan status tool pointing at the parent's job table; whatever it cannot do must be handled locally.

Why forbid it? Because a recursive agent is a bomb that can go out of control at any time. Imagine not forbidding it: the main agent dispatches a sub-agent, the sub-agent thinks the task is still too big and dispatches a grand-sub-agent, the grand-sub-agent dispatches again... each layer burns tokens, occupies threads, adds latency, and the model's judgment of "should this subtask be split further" isn't reliable; it can entirely fall into a bottomless pit of ever-finer splitting that never converges. Cutting recursion off cleanly is the most worry-free safety policy: a clone can be only one layer deep, and either this sub-agent handles it itself or it fails and returns, with no third outcome.

Remember piece one stressing that `_tool_by_name` is instance-level? Precisely because which tools an Agent knows is local to that instance, this capability cut is enforceable. Seeing the name `agent` in text does not grant access to it.

## Compared with Claude Code

CoreCoder now covers the two dimensions that best expose the architecture—foreground/background and shared/worktree—but it is still not a production scheduler. There are no durable workers, cross-process recovery, cancellation, event streams, automatic integration policy, or preset agent types. Naming those limits is more honest than calling a daemon thread a complete multi-agent platform.

Independent context is still the first motivation; background adds time concurrency, and worktree adds source-state isolation. Context, scheduling, and checkout are three different kinds of isolation and should not be collapsed into one word.

## What this piece leaves you with

- `Tool.effect` separates permission from concurrency safety: the former answers “may this run?”, the latter “may approved calls overlap?”
- Only consecutive `PURE/READ` calls enter the thread pool. `WRITE/EXTERNAL/UNKNOWN` calls are exclusive barriers, and unknown tools default to serial execution.
- Parallelism is not free: beyond scheduling policy, tools with mutable state still need per-instance isolation, thread-local state, or locks.
- A sub-agent's primary value is context isolation, letting heavy work churn in an independent window and handing back only a distilled conclusion to the main conversation; task decomposition is secondary.
- `run_mode` and `isolation` are orthogonal: foreground/background controls waiting, shared/worktree controls the checkout.
- Background permission never touches stdin, worktree does not pretend to be a sandbox, and failures plus oversized results close inside the parent loop boundary.
- Forbidding recursion trades "cutting it off cleanly" for "never out of control," enforced through instance-level tool sets.

Next piece, we fit these parts into a genuinely usable command-line tool: how sessions get saved, how to resume from a breakpoint, how slash commands hook in, and a security detail hidden inside session saving.
