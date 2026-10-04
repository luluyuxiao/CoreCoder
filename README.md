<div align="center">

# CoreCoder

**The nanoGPT of coding agents. A 3.9k-line engine inside 9,650 readable lines of pure Python: understand how a coding agent actually works, then fork your own.**

*learn from it · fork it · ship something better*

[中文](README_CN.md) | English | [Source-reading series · 8 bilingual essays](article/00-index_EN.md)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-3918_LoC-blue)](article/00-index_EN.md)
[![essays](https://img.shields.io/badge/source--reading-8_bilingual-orange)](article/00-index_EN.md)

</div>

- **Readable end to end.** Read the whole engine in an afternoon, with no magic hidden anywhere you can't follow it.
- **Hackable.** Set a breakpoint on any line, change it, rerun, all on your own machine. It genuinely works, which makes this a living reference rather than a diagram.
- **The gaps are the point.** It deliberately keeps only the minimal core; what's missing isn't half-finished, it's where you branch off and make it your own.

## How it compares

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| Lines of code | ~3,918 engine / 9,650 total | hundreds of thousands (closed) | tens of thousands of Python | ~600 (two files) |
| Time to read it all | one afternoon | can't (closed) | a few days of slogging | one afternoon |
| Breakpoint, change, rerun? | yes, every line | no | yes, but there's a lot | yes |
| What it's for | understand one, then fork your own | production coding assistant | terminal pair-programming | minimal GPT for teaching |

The nanoGPT column is there as a reference point: minimal, readable, but it teaches you to train a GPT. CoreCoder is after the same thing, only the subject is an agent that actually edits code. Sitting it next to Claude Code and aider isn't about competing for their users. CoreCoder is the foundation you stand on while you learn from them and get going; it isn't in the same race.

## What this is

I've always felt coding agents get talked about as if they were arcane. Strip a tool like Claude Code or Cursor all the way down and the core is a `while` loop wrapped around a large model, plus seven or eight tools that let it actually do things. The hard part was never the loop; it's everything the loop has to cope with once it meets the real world. CoreCoder is the minimal version that writes that core out honestly.

The engine (loop, model interface, context, tools, sessions) is 3,918 lines once you drop blank lines and comments. Counting storage, Skills, tracing, evals, the outer CLI, config and packaging too, the whole package is 38 files: 9,650 physical lines, 8,520 net, every one short enough to read in a single sitting. The growth since the original 1,161-line snapshot went into visible features: plan mode, hooks, persistent checkpoints and background jobs, stdio/HTTP MCP, session-persistent Skills, Docker isolation for bash and MCP, per-tool capabilities, resource-aware scheduling with live progress, cross-provider fallback routes, a USD budget, budget-aware context compaction, dual-layer transactional session storage, structured task memory, background/worktree sub-agents, and structured trace/eval, each documented below.

And it really runs: reads and writes files, executes shell, loads project workflows on demand, keeps compactable chat history separate from persistent structured task memory, spawns foreground or background sub-agents, isolates them in Git worktrees when requested, compacts context in three tiers, autosaves resumable sessions, and tells you the tokens and dollars a run burned whenever you ask. Anything that would mutate your disk or run a command stops for your consent first. The test suite now covers 297 cases. But the point of it running isn't to become your daily driver. It runs so the walkthrough can't lie: a reference that shows how an agent works has to actually work.

The code came out of a public teardown: open analyses have already exposed a lot of the load-bearing architecture inside production agents like Claude Code. I took the most essential layer and rewrote it honestly, in as little code as I could. So reading CoreCoder is roughly like reading a runnable, annotated take on how that kind of agent works, except it's only a minimal reimplementation, sitting right there on your machine for you to take apart and change.

<p align="center">
  <img src="https://raw.githubusercontent.com/he-yufeng/CoreCoder/main/assets/demo-plan-hooks.gif" width="760"
       alt="Plan mode in action: the agent reads fib.py, gets its edit refused while plan mode is on, presents a plan, and only after approval edits, tests, and reports — with Pre/PostToolUse hooks firing around every call.">
</p>

<p align="center"><sub><i>These thousand lines really do run a full loop end to end: ask it to fix buggy.py and it reads the file, edits the code, runs it once to confirm, then reports back on its own. Watch it, then come back and read the code.</i></sub></p>

This README follows the same arc: the first half helps you **read it** (the code map, the main loop, eight essays), the second half helps you **fork it** and points at a few directions worth pushing further.

## Run it once first (five minutes before you read)

Before you read the source, get it running on your machine once to build some intuition. It's a foundation meant for forking, so the recommended path is to clone it and install editable, reading and changing as you go:

```bash
git clone https://github.com/he-yufeng/CoreCoder
cd CoreCoder
pip install -e .
```

If you just want to get it running first, `pip install corecoder` works too.

Give it a model and a key and it goes. It speaks the OpenAI-compatible API by default, and switching providers is usually just two environment variables:

| Provider | Example env vars |
|---|---|
| OpenAI (default `gpt-5.5`) | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| OmniRoute | `OPENAI_API_KEY=your-key OPENAI_BASE_URL=http://localhost:20128/v1 CORECODER_MODEL=auto` |
| Local Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

Kimi, Qwen and the like are the same two variables; for providers that don't even offer an OpenAI-compatible endpoint, the optional LiteLLM backend (`pip install "corecoder[litellm]"`) routes to a hundred-plus of them. The third essay goes into this in detail. The key can be `export`ed directly or dropped into a `.env` at the project root, which is loaded on startup. Then:

For graceful degradation and a client-side spend ceiling, configure an ordered fallback chain and USD budget. `CORECODER_FALLBACK_MODELS` remains the short form for models on the primary endpoint. `CORECODER_FALLBACK_ROUTES` can independently select an OpenAI-compatible or LiteLLM backend, endpoint, credential environment variable, model, and optional price override:

```bash
export CORECODER_FALLBACK_MODELS=gpt-5.4-mini,gpt-4o-mini
export QWEN_API_KEY=sk-...
export CORECODER_FALLBACK_ROUTES='[{"name":"qwen","provider":"openai","model":"qwen3-plus","base_url":"https://dashscope.example/v1","api_key_env":"QWEN_API_KEY"}]'
export CORECODER_MAX_COST_USD=1.00
# CLI accepts repeated --fallback-model and --fallback-route JSON values.
```

Smoke-tested end to end (read the file, edit it, run it, report back) against DeepSeek, Qwen3 and Kimi K2 via a single OpenRouter-compatible endpoint; each completed the full loop. One note for one-shot scripts: `-p` refuses mutating tools unless you pass `--yes`, by design.

```bash
corecoder                                             # interactive REPL
corecoder -p "add error handling to parse_config()"   # one-shot mode, exits when done
```

## Read it: the code map

Laid out flat, the whole project is this big. Skim it before you clone and you'll know where everything is. This is the most concrete difference from Claude Code's hundreds of thousands of lines: you can read it like the table of contents of a book. Start from the main loop in `agent.py`; that's the heart of the whole agent.

```
corecoder/
├── agent.py        loop + scheduler + stable snapshots   1220 lines   ← start here
├── capabilities.py per-Tool authority policy              176 lines
├── decisions.py    shared structured gate decisions        97 lines
├── resources.py    cross-Agent read/write resource locks   97 lines
├── llm.py          stream + retry + provider routes       871 lines
├── context.py      request-budgeted context compaction    431 lines
├── session.py      storage-compatible session facade      161 lines
├── storage.py      Transcript + Active Context storage     790 lines
├── memory.py       goal + constraints + plan + decisions   299 lines
├── permissions.py  consent for mutating tools             110 lines
├── hooks.py        Pre/PostToolUse shell hooks             99 lines
├── protect_paths_hook.py  opt-in sensitive-path guard      118 lines
├── mcp.py          stdio/HTTP MCP + recovery/admission    707 lines
├── skills.py       project/user Skill discovery + parser  160 lines
├── sandbox.py      local/Docker execution boundary        342 lines
├── trace.py        metadata-safe memory/JSONL event sinks  160 lines
├── eval.py         repeatable cases, checks, and metrics   586 lines
├── prompt.py       system prompt                           41 lines
├── cli.py          REPL + slash commands + one-shot      1023 lines
├── config.py       env-var config                         109 lines
├── checkpoints.py  durable agent-scoped /undo stack       125 lines
├── demo.py         offline end-to-end demo                 100 lines
└── tools/
    ├── bash.py       shell + execution backend + cd       179 lines
    ├── edit.py       unique-match search/replace + diff   126 lines
    ├── grep.py       content search                       114 lines
    ├── glob_tool.py  filename matching                     67 lines
    ├── read.py       file read                             80 lines
    ├── write.py      file write                            73 lines
    ├── todo.py       agent-maintained task checklist       91 lines
    ├── memory.py     structured task-state updates          83 lines
    ├── agent.py      durable sub-agent/background jobs    645 lines
    ├── fetch.py      bounded HTTP(S) text fetch             46 lines
    ├── now.py        current local timestamp                21 lines
    ├── skill.py      load + persistent activation state    119 lines
    └── base.py       tool base + resource metadata          88 lines
.corecoder/skills/
└── corecoder-review/SKILL.md  repository-aware review workflow
examples/
├── plan_hooks_demo.py  offline plan mode + hooks demo (no API key)
└── eval_cases.json     starter evaluation manifest
```

Thirteen built-in tools: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `todo_write`, `memory_update`, `agent`, `agent_status`, `agent_resume`, `fetch_url`, and `now`. When Skills are discovered, one read-only `load_skill` adapter joins them; it adds workflow instructions, not execution authority. If `~/.corecoder/mcp.json` exists, its MCP servers join as extra `mcp__*` tools.

## A `while` loop is the whole agent

The whole of an agent fits in one sentence: hand the user's words to the model, run whatever tools it asks for, stuff the results back into the context, ask again, and keep going until it stops asking for tools and gives an answer. In code, that's about a dozen lines:

```python
# corecoder/agent.py · the main loop (trimmed skeleton)
def chat(self, user_input):
    self.messages.append(user_input)

    for _ in range(self.max_rounds):                   # bounded, so it can't run away
        reply = self.llm.chat(self.messages, self.tools)   # ask the model what to do next
        if not reply.tool_calls:                       # model wants no more tools
            return reply.text                          #   -> done, hand the answer back
        results = schedule_by_resource(reply.tool_calls) # parallel only when claims do not conflict
        self.messages += results                       # feed results back, loop again

    return "(hit the round limit)"
```

That's the whole thing. The core skeleton is about twenty lines; counting parallel execution and the bookkeeping after a Ctrl+C interrupt, maybe forty. Almost everything else in CoreCoder's thousand-odd lines is there to clean up the mess the loop runs into once it meets the real world. `llm.py` ends up the biggest file in the project, not because calling a model is hard, but because a streamed response splinters each tool call's arguments into fragments you have to restitch in order, a provider will hand you half a JSON object or a null `usage` field, and 429s, timeouts, dropped connections and 5xx all need backoff-and-retry while the other 4xx should just raise. That unglamorous grunt work, not the loop, is where the real engineering of taking an agent from demo to delivery actually lives; the third essay follows it down to the line.

Three decisions are worth a closer look, because they're the kind of call you can only make after you've understood how others did it, and they're judgments you can lift straight into your own fork.

**`edit_file` does search-and-replace on a unique match, not line numbers.** Line numbers are a trap: the model only has to miscount by one and it quietly edits the wrong place. Anchor on a unique snippet of the original instead. If there's no match, it hands the start of the file back so the model can re-anchor; if there are several matches, it makes the model bring more surrounding context rather than gamble on one. On a successful edit it returns a diff. Recoverable on failure, verifiable on success: the whole loop stays inside the tool.

**Context isn't cut all at once when it's full; it is budgeted as a complete request and gives ground in tiers.** CoreCoder subtracts the dynamic system prompt, tool schemas, output-token reserve, and estimator safety margin before applying the 50/70/90% thresholds to the remaining message budget. It mechanically snips stale tool output first, summarizes older turns next, then hard-collapses if needed. A fresh tool result is protected until one successful LLM request has consumed it; a final deterministic fit step guarantees the estimated request is below the configured window, bounding even fresh data only when that batch cannot fit intact.

**You constrain a sub-agent by withholding capabilities, not by writing rules and hoping it obeys.** A spawned sub-agent gets an isolated context and fresh built-in tool instances, but none of `agent`, `agent_status`, or `agent_resume`, so it cannot recursively spawn descendants. It reuses the parent's model connection, spend ledger, and resource-lock manager; results over 5,000 characters are truncated and each child has a 20-round cap. `run_mode` selects blocking foreground or in-process background execution; `isolation` independently selects the current checkout or a retained Git worktree/branch created from `HEAD`. Background records and file checkpoints live in Session snapshots. A saved `queued`/`running` job restores as `interrupted` and requires an explicit, permission-gated `agent_resume`, preventing silent replay of ambiguous side effects.

Every one of these *whys* is traced down to the actual lines of code in the series below.

## The source-reading series · 8 bilingual essays

I also wrote a bilingual source-reading series, one intro plus eight parts, each in Chinese with an English mirror. Against CoreCoder's actual code, it walks through how agents like Claude Code work under the hood. One hard rule I set myself: every line count and every snippet is re-read and re-checked from the repo, never written from memory. The first six get you reading, the seventh gets you forking, and the eighth is about extending it without touching the loop; read them in any order.

- **[Intro · Read Claude Code through CoreCoder, then build your own](article/00-index_EN.md)**
- **[01 · An agent, at its core, is a `while` loop](article/01-the-loop_EN.md)** — the main loop in `agent.py`, interrupts, and the round limit
- **[02 · The tool system: letting the model act, safely](article/02-tools_EN.md)** — the built-in tools in `tools/`, effect metadata, and the bash safety gate
- **[03 · Plug in any LLM, and keep the bill honest](article/03-llm-and-cost_EN.md)** — `llm.py`'s provider wrapper, retries, and cost accounting
- **[04 · Surviving a long task on a finite window](article/04-context_EN.md)** — `context.py`'s three-tier compaction and orphaned tool messages
- **[05 · Parallel execution and sub-agents](article/05-parallel-and-subagents_EN.md)** — resource-aware concurrency and sub-agent isolation
- **[06 · Turning it into a real command-line tool](article/06-session-and-cli_EN.md)** — CLI, transactional session storage, and crash-safe resume
- **[07 · Fork CoreCoder into your own coding agent](article/07-build-your-own_EN.md)** — from fork to custom tools to swapping models
- **[08 · Three ways to extend without touching the loop: MCP, hooks, and plan mode](article/08-extensibility_EN.md)** — the v0.6.0 extensibility trio and the contract that makes them safe

## Fork it, build something better

Once you understand it, the natural next step is to fork. Getting started doesn't take much:

- **Swap in a model you actually use.** It's the two env vars from above; `llm.py` is the entry point for provider adaptation, cross-provider fallback, and spend control.
- **Add a tool of your own.** Write a new file against the tool contract in `tools/base.py`, declare its effect, capabilities, and concrete resource claims, then run tests, fetch a page, call an LSP, whatever. The end of the second essay walks you through your first one by hand.
- **Rewrite the system prompt.** `prompt.py` is all of 41 lines; change one line and you'll watch the agent's temperament shift. It's the cheapest "change one thing, see a result" in the whole project.
- **Import it as a library.** The top level exports `Agent`, `LLM`, and `Config`, ready to embed in your own program:

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("find every TODO comment in this project and list them"))
```

Going deeper, the directions are out in the open too. The Docker sandbox is now a small working baseline; the remaining items are still deliberate extension points:

- **Harden the sandbox further.** `--sandbox docker` supplies a real container boundary for `bash`, and individual MCP servers can opt into the same hardened Docker runtime. Production deployments can still add custom seccomp/AppArmor profiles, per-task images, image signing/scanning, and isolation for hooks.
- **Model resilience is deliberately explicit.** Retryable failures exhaust exponential backoff before advancing through independently configured provider routes; a successful switch stays active. The optional USD cap prices usage per route, reserves the next input, clamps maximum output, and refuses unknown pricing or missing usage. A production fork can extend this into health-weighted routing and account-side billing alerts.
- **Sub-agent modes are explicit but intentionally in-process.** Foreground/background and shared/worktree modes are implemented, and task records survive Session Resume. Threads themselves do not survive process exit: unfinished jobs restore as `interrupted` for explicit `agent_resume`. Production forks can add durable workers, cancellation, event streaming, automatic merge/cherry-pick policy, and process/host isolation.
- **Trace and eval are local, intentionally small building blocks.** JSONL captures one run precisely and the eval runner turns deterministic cases into success/latency/token/tool/cost metrics. Production forks can add OpenTelemetry export, a trace UI, semantic or model-graded checks, datasets, and CI trend storage.
- **No RAG, and the MCP client speaks tools only.** The client supports stdio and Streamable HTTP tools, reconnection, a circuit breaker, refresh, and startup admission, but intentionally omits MCP resources and prompts. Retrieval-based code location and those protocol surfaces remain real extension directions.

The README only points; the seventh essay picks up the code details for each. Pick one and start; that's the whole reason the core is kept this small.

## Commands

Inside the REPL, `/help` lists everything; these are the ones you'll reach for:

```
/model <name>    switch model
/compact         compact the context by hand
/tokens          token usage, active fallback, cost, and remaining budget
/memory          show persistent structured task memory
/goal <text>     set the task goal (`clear` removes it)
/constraint <text>  add a user-owned semantic constraint
/decision <text>    record a durable decision
/diff            files changed this session
/undo            revert the most recent file change
/plan            toggle plan mode (read-only, then a plan to approve)
/save            force-save the active session
/name <name>     name the active session without changing its stable ID
/session         show the active session and storage
/sessions        list saved sessions with names, IDs, and previews
/skills          list project and user workflow skills
/transcript      inspect original history unaffected by context compaction
/delete-session  delete an inactive session by ID
/agents          list background sub-agents
/mcp-refresh     reconnect if needed and refresh MCP tool schemas
quit / exit      exit (Ctrl+C cancels the current round)
```

Interactive and one-shot runs autosave to `~/.corecoder/sessions/sessions.db` by default. `/name Fix login flow` assigns a human-facing name while the generated Session ID remains the stable, unique key used by `corecoder -r <id>`; names need not be unique. `/sessions` shows both, plus the first-message preview, so choosing a session no longer depends on remembering an opaque ID. Storage has two layers: `events` is an append-only Transcript preserving original user/assistant/tool-call/tool-result payloads, while `messages` is the Active Context used by Resume and the next model request and may be summarized or snipped by Context Management. A compressed context snapshot, the pending Transcript tail, and Session metadata commit in one transaction, so the model can run on short context while `/transcript` still shows the original history.

SQLite WAL permits concurrent readers/writers. The Agent emits snapshots only at provider-valid boundaries: after a user message, after every complete tool-result batch, and on completion/failure/interruption. Resume therefore never exposes an Active Context with assistant `tool_calls` but missing observations. Compaction records also enter `summaries`, including actions, before/after token counts, model, and summary messages. `--no-autosave` disables automatic writes while keeping `/save`; `--storage PATH` or `CORECODER_STORAGE_PATH` selects another database. Existing v1/v2/JSON sessions are migrated on read; because already-compacted originals cannot be reconstructed, migrated records are explicitly marked as having an incomplete Transcript.

Stored state includes the full Transcript, compressed Active Context, compaction summaries, structured task memory, model, workspace, status, token/cost counters per model and provider route, plan mode, todo state, active Skill records, file checkpoints, retained background-job records, and unconsumed Tool Result IDs. API keys and permission grants are deliberately excluded. Session IDs are still normalized before becoming database keys or legacy filenames, and SQLite/JSON files are created with user-only permissions where the OS permits it.

The database is local but not encrypted. Messages and Tool Results can themselves contain source code, prompts, command output, or secrets read from the workspace; use `--no-autosave` for sensitive runs or point `--storage` at an appropriately protected location.

## Structured task memory

Conversation history is not the only state the model sees. `MemoryState` keeps a bounded Goal, user-owned Critical Constraints, Plan Steps, Decisions, and system-derived Files Modified outside `messages`; `Agent._full_messages()` re-injects it every round, and Session snapshots restore it after restart. Context summarization can therefore discard old prose without silently deleting the current goal or an explicit user constraint. The internal `memory_update` Tool lets the model maintain goals, plans, and decisions, but deliberately cannot promote Tool output into a Critical Constraint. Successful built-in file writes populate Files Modified from actual Tool arguments rather than trusting a model-written claim.

Constraints in this layer are semantic instructions, not an operating-system boundary. Use `/constraint revoke <id>` to supersede one, and use Hooks, Capability Policy, Permission, or Sandbox when a rule must be mechanically enforced. Foreground/background sub-agents inherit copies of the parent constraints and decisions while keeping their own Plan and modified-file state, so child work cannot mutate parent memory by reference.

## Skills

Skills are reusable workflow instructions, deliberately separate from executable Tools. CoreCoder discovers user Skills under `~/.corecoder/skills/*/SKILL.md` and project Skills under `<workspace>/.corecoder/skills/*/SKILL.md`; a project Skill overrides a same-named user Skill. Startup exposes only each validated `name` and `description`. When a task matches, the model calls the read-only `load_skill` Tool and receives the full instructions as an ordinary Tool Result, so the Context Manager, Transcript, Trace, and sub-agent tool sharing continue to work without a second execution path.

A successful load also creates a small session-persistent `active_skills` record containing the name, scope, load time, status, and SHA-256 content hash. The Agent re-injects hash-matched active instructions into the system message on every LLM request, so context compression and Resume cannot silently forget the workflow. The full instructions remain in `SKILL.md`, not duplicated in session metadata. If the file changes or disappears, CoreCoder fails closed: the old instructions are not injected and `/skills` reports `changed` or `unavailable`; call `load_skill` again to explicitly accept a changed version. `/reset` clears active Skills.

This repository ships one natural example, `corecoder-review`: it reviews changes against the Agent Loop's protocol invariants, scheduling/permission/sandbox boundaries, context protection, and dual-layer Session rules. Type `/skills` to see what was discovered, then ask the model to use `corecoder-review`. A Skill never grants a missing Tool and never bypasses Permission or Sandbox; project Skills are repository-provided instructions, so review them before using an untrusted checkout.

## Tool progress

Tool execution has a dedicated live view instead of a row of fire-and-forget call names. A multi-call response is labelled `Parallel tools` only when the resource-aware scheduler has an actual concurrent batch; otherwise it says `Tool batch`. Pure/read calls overlap as before. Opted-in stateful tools expose read/write `ResourceClaim`s: writes to different files and calls to different MCP servers may overlap, while the same file/server serializes. The shared `ResourceLockManager` also coordinates parent and child Agents. Each call moves through `queued` → `running` → `done`, `error`, or `blocked`, with elapsed time and an overall `n/N` counter. Permission questions are settled before the live display starts, so an interactive consent prompt never fights Rich for the terminal.

Embedders can drive another UI with the same structured stream by passing `on_tool_progress` to `Agent.chat()`. It receives `batch_started`, `tool_started`, `tool_completed`, and `batch_completed`; the older `on_tool(name, arguments)` callback remains supported. Progress callbacks are best-effort presentation code: if one fails, the tool and Agent loop continue.

## Permissions

Read-only and in-memory state tools (`read_file`, `glob`, `grep`, `todo_write`, `memory_update`, `now`, `agent_status`, `load_skill`) run the moment the model asks. The disk-mutating or externally active ones (`edit_file`, `write_file`, `bash`, `fetch_url`, MCP tools, spawning a sub-agent, and `agent_resume`) stop for consent first, and the REPL banner shows which mode you're in. Hooks, Capability Policy, Plan Mode, and Permission now return one structured `ToolDecision`, so traces receive a stable gate/code/reason instead of interpreting mixed strings and tuples:

- In the REPL you get one prompt per call: allow once, always allow this tool, or deny. Foreground sub-agents inherit that interactive layer. A background thread is never allowed to compete with the REPL for input: it can use only `--yes` or tools already marked "always allow"; other stateful calls fail closed and tell the child to route around them.
- In one-shot mode (`-p`) there is nobody to ask, so a mutating call is refused on the spot and the refusal goes back to the model as an ordinary tool result: the loop never hangs on input that can't arrive. Pass `--yes` to approve everything up front (scripts, CI).
- The decision itself is pure logic in `permissions.py`, with the terminal only supplying the prompt callback. You can unit-test consent without a TTY, or reuse the layer in your own embedding.

## Per-tool capabilities

Tools now declare the authority they may exercise: `filesystem_read`, `filesystem_write`, `network`, `process`, `subagent`, `mcp`, or `unknown`. An optional `~/.corecoder/capabilities.json` policy restricts that authority by exact Tool name or glob. It is checked after PreToolUse hooks and before Permission, so a denied capability never reaches a consent prompt or `Tool.execute()`:

```json
{
  "default": "deny",
  "tools": {
    "read_file": {"allow": ["filesystem_read"]},
    "bash": {"allow": ["filesystem_read", "filesystem_write", "process"]},
    "fetch_url": {"allow": ["network"]},
    "mcp__weather__*": {"allow": ["mcp", "process", "network"]}
  }
}
```

Use `--capability-policy PATH` or `CORECODER_CAPABILITY_POLICY`; `/capabilities` shows the active policy and each Tool's effective declaration. With no policy file, CoreCoder keeps the backward-compatible allow policy. Under `default: deny`, tools with no external authority (`now`, `agent_status`, todo, and structured memory state) still run, while unknown custom tools fail closed. MCP applies the server wildcard (for example `mcp__weather__*`) before process launch or the first remote HTTP request, so a denied server never starts and cannot perform startup side effects. The discovered tools are checked again per call. The sample [examples/capabilities.json](examples/capabilities.json) is a fuller starting point.

```bash
corecoder --capability-policy examples/capabilities.json --sandbox docker
# then type /capabilities in the REPL
```

Network policy is deliberately enforceable rather than heuristic. `fetch_url` declares `network`; local `bash` declares it because host processes can reach the network, while Docker `bash` drops it only when `--network none` actually removes that authority. A rule that omits `network` therefore blocks local/online bash entirely instead of pretending to identify which shell commands will connect. Domain allowlists are not claimed: arbitrary shell and MCP implementations can hide or redirect their true destination, so use a network-none container or an external allowlisting proxy for that boundary.

## Docker sandbox

Local execution remains the default for compatibility. To put `bash` behind a container boundary, first build the deliberately small project image, then opt in:

```bash
docker build -f Dockerfile.sandbox -t corecoder-sandbox:latest .
corecoder --sandbox docker
```

Every bash call runs in a fresh container. Only the directory from which CoreCoder was started is mounted, at `/workspace`; the container root filesystem is read-only, network access defaults to `none`, Linux capabilities are dropped, privilege escalation is disabled, and CPU, memory, PID, and `/tmp` limits are applied. The process runs as the host UID/GID and the image is never pulled implicitly. The built-in file tools also reject resolved paths outside that startup directory, including ordinary `..` and symlink escapes. Use project-relative paths or `/workspace/...` inside bash commands.

The defaults can be changed with `CORECODER_SANDBOX*` environment variables or the matching CLI flags (`--sandbox-image`, `--sandbox-network`, `--sandbox-memory`, `--sandbox-cpus`, and `--sandbox-pids`). For example, `--sandbox-network bridge` explicitly grants container networking. A custom image only needs `/bin/sh` plus the runtimes your project requires.

This boundary is deliberately scoped: hooks, the LLM client, and `fetch_url` still run on the host; MCP servers run there too unless their own `sandbox` block selects Docker. The bash workspace mount is writable because this is a coding agent. Host environment variables are not forwarded, but every file already inside the workspace—including `.env` or credentials—is visible to the tools, so use a clean worktree and keep secrets outside it for untrusted tasks. Permission prompts are still required—they decide whether an action is authorized, while capability policy limits which kinds of authority a Tool may request and the sandbox limits the damage after authorization. If the Docker daemon or image is unavailable, the call fails as a tool result; it never falls back to host execution.

## Plan mode

`/plan` toggles plan mode in the REPL. While it's on, the prompt shows `(plan)` and every externally mutating call (writes, edits, bash, MCP tools, sub-agents) is refused on the spot: the refusal goes back to the model as an ordinary tool result, telling it to keep investigating read-only and present a numbered plan instead. The model may maintain structured Plan Steps through `memory_update`; those steps survive context compaction and Session Resume. When the plan looks right, `approve` (or `/plan` again) hands control back and the agent executes. Mechanically the execution restriction remains one flag on `Agent` plus a refusal branch ahead of the consent gate.

## Hooks

Drop a `hooks.json` under `~/.corecoder` and your own shell commands run around every tool call, the same idea as Claude Code's hooks:

```json
{
  "PreToolUse":  [{"matcher": "bash", "command": "cat >> ~/.corecoder/audit.jsonl"}],
  "PostToolUse": [{"matcher": "*",    "command": "cat >> ~/.corecoder/trace.jsonl"}]
}
```

Each hook gets the call as JSON on stdin (`tool_name`, `tool_input`; post hooks also get `tool_response`). The matcher is an exact tool name; empty or `*` fires on every tool. A pre hook can veto the call with exit code 2, and its stderr travels back to the model as the reason so it can route around the block. Post hooks only observe and can never block. A hook that errors or runs past ten seconds is skipped with a warning: hooks assist the loop, they never get to kill it. The whole mechanism is `hooks.py`, and the REPL banner shows how many hooks loaded.

CoreCoder ships one opt-in practical hook. Install the editable package after
pulling this version, then copy the sample only when you do not already have a
hook file (otherwise merge its `PreToolUse` entry):

```bash
python -m pip install -e .
mkdir -p ~/.corecoder
cp examples/hooks.protect-sensitive.json ~/.corecoder/hooks.json
```

The sample uses `CORECODER_PYTHON`, which the Hook runtime sets to the exact
interpreter running CoreCoder, so it does not accidentally invoke another
virtual environment:

```json
{
  "PreToolUse": [{
    "matcher": "*",
    "command": "\"$CORECODER_PYTHON\" -m corecoder.protect_paths_hook"
  }]
}
```

It refuses `write_file` and `edit_file` calls targeting common `.env` files,
private-key extensions, `.git`, `secrets/`, or `production.yaml`/`production.yml`.
It checks lexical, resolved, and workspace-relative path forms, so `..` and an
existing symlink cannot disguise a protected target. Add a project rule with
`corecoder-protect-paths --pattern 'config/prod/*'` (or add those arguments to
the module command in JSON); use `--no-defaults` when
you want only your own patterns. The hook deliberately does not inspect
`bash`: arbitrary shell effects cannot be classified safely from command text,
so use Capability Policy plus the Docker Sandbox for that boundary.

## MCP servers

Drop a `mcp.json` under `~/.corecoder` and tools from any MCP server join the agent over stdio or Streamable HTTP:

```json
{
  "mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    "weather": {"url": "https://mcp.example.test", "headers": {"Authorization": "Bearer ..."}}
  }
}
```

Each admitted server handshakes and lists its tools; every one is registered as `mcp__<server>__<tool>`, so hooks, capability policy, consent, resource scheduling, and traces treat it like a built-in. MCP tools stay out of the read-only set. The handshake gets fifteen seconds and a call gets sixty. A dead stdio process reconnects and re-handshakes before a later call; the failed `tools/call` itself is never replayed because its remote side effect is ambiguous. Repeated transport failures open a configurable circuit (defaults: three failures, thirty seconds). `/mcp-refresh` reconnects if needed, re-runs `tools/list`, and atomically rebuilds the Agent's schema registry and system prompt; a failed refresh retains the previous wrappers. HTTP sessions honor `Mcp-Session-Id`, accept JSON or SSE responses, and send `DELETE` on shutdown. The client still intentionally implements the tools slice only (`initialize`, `ping`, `tools/list`, `tools/call`), not resources or prompts. With no `mcp.json` there is no MCP and nothing changes.

`reconnect`, `circuit_failures`, and `circuit_cooldown` may be set under `defaults` or one server. Stdio servers can use `sandbox`; remote HTTP servers cannot be placed inside a local process container and instead declare only `mcp` + `network` capabilities. Calls to one server serialize because they share state/session, while different servers may run concurrently.

Host mode remains the compatibility default. A server can instead live in one long-running hardened Docker stdio container:

```json
{
  "defaults": {
    "sandbox": {
      "mode": "docker",
      "image": "my-mcp-runtime:latest",
      "network": "none",
      "workspace": "none",
      "memory": "512m",
      "cpus": 0.5,
      "pids": 64
    }
  },
  "mcpServers": {
    "weather": {
      "command": "weather-mcp-server",
      "env": {"WEATHER_API_KEY": "replace-me"},
      "sandbox": {"network": "bridge"}
    },
    "review": {
      "command": "review-mcp-server",
      "sandbox": {"workspace": "ro"}
    }
  }
}
```

The executable must exist inside the selected image. Docker MCP gets the same read-only root, dropped Linux capabilities, no-new-privileges, non-root UID/GID, private `/tmp`, resource limits, `--pull never`, named-container timeout cleanup, and fail-closed startup as Docker bash. It inherits no host environment variables: only the explicit `env` map enters the container. Workspace access defaults to `none`; choose `ro` or `rw` explicitly, which mounts the CoreCoder startup directory at `/workspace`. Network defaults to `none`; `bridge` grants the whole server outbound network, including during startup. Since all tools from one MCP server share that process boundary, place tools with different trust/network requirements in separate servers or images.

For a no-dependency end-to-end smoke test, copy [examples/mcp.sandbox.json](examples/mcp.sandbox.json) to `~/.corecoder/mcp.json`, start CoreCoder from this repository root, and ask it to call `mcp__sandbox_demo__echo`. The example runs [examples/minimal_mcp_server.py](examples/minimal_mcp_server.py) from a read-only Workspace mount with networking disabled.

## Trace and eval

Tracing is opt-in. Pass a path (or set `CORECODER_TRACE`) and CoreCoder appends one JSON object per event:

```bash
corecoder --trace .corecoder/run.jsonl
# Payloads are omitted by default. This explicit switch may record source and secrets:
corecoder --trace .corecoder/debug.jsonl --trace-content
```

The timeline covers agent runs and rounds, LLM latency/usage/retries/fallbacks, tool requests and duration, hook and permission decisions, context compression, and sub-agent lifecycle. Every record has a timestamp, sequence and trace session ID; agent records also carry `run_id`, `agent_id`, `parent_agent_id`, `subagent_task_id`, and the current round where applicable. This is enough to answer “which round went wrong?”, “which tool was slow?”, “what triggered fallback?”, and “how many tokens did this child use?” without storing prompts. With `--trace-content`, the context-compression event also includes bounded before/after messages so information loss can be inspected. JSONL writes are thread-safe and best-effort: a broken trace sink warns but never fails the agent.

Library callers can use the same contract with `MemoryTrace`, `JsonlTrace`, or `CompositeTrace`:

```python
from corecoder import Agent, MemoryTrace

trace = MemoryTrace()
agent = Agent(llm=llm, trace=trace)
agent.chat("inspect this repository")
print(trace.events)
```

`corecoder-eval` runs a JSON manifest through fresh Agent instances. Each case is copied to a temporary workspace by default and can assert final text, regexes, files, file contents, tools used, and maximum rounds/calls/latency/tokens/cost. Repeats produce per-case pass rate and pass@k plus suite totals:

```bash
corecoder-eval examples/eval_cases.json --repeat 3 \
  --output reports/current.json --trace-dir reports/traces
corecoder-eval examples/eval_cases.json --repeat 3 \
  --baseline reports/current.json --output reports/candidate.json
```

The second command adds signed per-run metric deltas against the baseline, so a change can be discussed as a success-rate/latency/token/tool/cost result rather than a nicer demo. The report stores responses, actual models used, non-secret run configuration, and a manifest hash; CLI model/fallback/budget options and environment configuration work as they do in `corecoder`. Mutating tools are denied unless `--yes` is supplied; source workspaces are still copied unless `--in-place` is explicit, and `--keep-workspaces DIR` retains copies for debugging. A copied directory is not a shell security boundary: for untrusted prompts or mutating evals, build the sandbox image and add `--sandbox docker`.

## Related Projects

If working through CoreCoder was useful, here are a few other tools I've built around agents and LLM systems:

- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — dropped into an unfamiliar codebase? It gives you a guided wiki and a where-to-start reading path, a self-hostable DeepWiki alternative.
- **[FindJobs-Agent](https://github.com/he-yufeng/FindJobs-Agent)** — stop sifting job boards by hand: it ranks postings against your resume and runs mock interviews.
- **[ContractGuard](https://github.com/he-yufeng/ContractGuard)** — catch the risky clauses before you sign: it reads contracts and flags the dangerous bits.
- **[GitSense](https://github.com/he-yufeng/GitSense)** — want to contribute to open source? It finds issues worth your time and gauges whether your PR will get merged.
- **[CodeABC](https://github.com/he-yufeng/CodeABC)** — understand any codebase even if you don't code, built for non-programmers.

## Contributing / License

Before you send anything, run `pytest tests/ -q` (297 cases), `ruff check`, and `compileall`, and make sure they're green. MIT licensed: fork it, learn from it, ship something better. A mention of this project is appreciated.

---

By [Yufeng He](https://github.com/he-yufeng), formerly at Moonshot AI (Kimi). I earlier wrote a fairly complete [Claude Code source analysis](https://zhuanlan.zhihu.com/p/1898797658343862272) on Zhihu; this project is its hands-on counterpart: that one walks you through reading it, this one through rebuilding it.

> CoreCoder was formerly named NanoCoder; it was renamed to avoid confusion with [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder), and old links redirect here automatically.
