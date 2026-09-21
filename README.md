<div align="center">

# CoreCoder

**The nanoGPT of coding agents. A 2.6k-line engine inside 5,357 readable lines of pure Python: understand how a coding agent actually works, then fork your own.**

*learn from it · fork it · ship something better*

[中文](README_CN.md) | English | [Source-reading series · 8 bilingual essays](article/00-index_EN.md)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-2597_LoC-blue)](article/00-index_EN.md)
[![essays](https://img.shields.io/badge/source--reading-8_bilingual-orange)](article/00-index_EN.md)

</div>

- **Readable end to end.** Read the whole engine in an afternoon, with no magic hidden anywhere you can't follow it.
- **Hackable.** Set a breakpoint on any line, change it, rerun, all on your own machine. It genuinely works, which makes this a living reference rather than a diagram.
- **The gaps are the point.** It deliberately keeps only the minimal core; what's missing isn't half-finished, it's where you branch off and make it your own.

## How it compares

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| Lines of code | ~2,597 engine / 5,357 total | hundreds of thousands (closed) | tens of thousands of Python | ~600 (two files) |
| Time to read it all | one afternoon | can't (closed) | a few days of slogging | one afternoon |
| Breakpoint, change, rerun? | yes, every line | no | yes, but there's a lot | yes |
| What it's for | understand one, then fork your own | production coding assistant | terminal pair-programming | minimal GPT for teaching |

The nanoGPT column is there as a reference point: minimal, readable, but it teaches you to train a GPT. CoreCoder is after the same thing, only the subject is an agent that actually edits code. Sitting it next to Claude Code and aider isn't about competing for their users. CoreCoder is the foundation you stand on while you learn from them and get going; it isn't in the same race.

## What this is

I've always felt coding agents get talked about as if they were arcane. Strip a tool like Claude Code or Cursor all the way down and the core is a `while` loop wrapped around a large model, plus seven or eight tools that let it actually do things. The hard part was never the loop; it's everything the loop has to cope with once it meets the real world. CoreCoder is the minimal version that writes that core out honestly.

The engine (loop, model interface, context, tools, sessions) is 2,597 lines once you drop blank lines and comments. Counting tracing, evals, the outer CLI, config and packaging too, the whole package is 29 files: 5,357 physical lines, 4,594 net, every one short enough to read in a single sitting. The growth since the original 1,161-line snapshot went into visible features: plan mode, hooks, checkpoints, MCP, an optional Docker sandbox, effect-aware tool scheduling with live progress, model fallback, a USD budget, budget-aware context compaction, background/worktree sub-agents, and structured trace/eval, each documented below.

And it really runs: reads and writes files, executes shell, spawns foreground or background sub-agents, isolates them in Git worktrees when requested, compacts context in three tiers, and tells you the tokens and dollars a run burned whenever you ask. Anything that would mutate your disk or run a command stops for your consent first. The test suite now covers 228 cases. But the point of it running isn't to become your daily driver. It runs so the walkthrough can't lie: a reference that shows how an agent works has to actually work.

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

For graceful degradation and a client-side spend ceiling, configure an ordered fallback chain and USD budget. Every candidate on one `LLM` instance shares its backend, endpoint, and explicit credential; LiteLLM can express provider/model names when the corresponding credentials are available:

```bash
export CORECODER_FALLBACK_MODELS=gpt-5.4-mini,gpt-4o-mini
export CORECODER_MAX_COST_USD=1.00
# CLI equivalents: --fallback-model gpt-5.4-mini --fallback-model gpt-4o-mini --max-cost 1.00
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
├── agent.py        loop + scheduler + trace events        701 lines   ← start here
├── llm.py          stream + retry + fallback + budget     567 lines
├── context.py      request-budgeted context compaction    431 lines
├── session.py      save / resume + path-traversal guard    97 lines
├── permissions.py  consent for mutating tools             108 lines
├── hooks.py        Pre/PostToolUse shell hooks             85 lines
├── mcp.py          MCP stdio client for external tools    210 lines
├── sandbox.py      local/Docker execution boundary        272 lines
├── trace.py        metadata-safe memory/JSONL event sinks  160 lines
├── eval.py         repeatable cases, checks, and metrics   552 lines
├── prompt.py       system prompt                           41 lines
├── cli.py          REPL + slash commands + one-shot       634 lines
├── config.py       env-var config                          88 lines
├── checkpoints.py  /undo snapshot and restore                44 lines
├── demo.py         offline end-to-end demo                 100 lines
└── tools/
    ├── bash.py       shell + execution backend + cd       169 lines
    ├── edit.py       unique-match search/replace + diff   105 lines
    ├── grep.py       content search                       112 lines
    ├── glob_tool.py  filename matching                     65 lines
    ├── read.py       file read                             65 lines
    ├── write.py      file write                            52 lines
    ├── todo.py       agent-maintained task checklist       80 lines
    ├── agent.py      sub-agent modes + background jobs    432 lines
    ├── fetch.py      bounded HTTP(S) text fetch             44 lines
    ├── now.py        current local timestamp                20 lines
    └── base.py       tool base + effect metadata            56 lines
examples/
├── plan_hooks_demo.py  offline plan mode + hooks demo (no API key)
└── eval_cases.json     starter evaluation manifest
```

Eleven built-in tools: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `todo_write`, `agent`, `agent_status`, `fetch_url`, and `now`. Everything else is the CLI shell, config, and packaging wrapped around that engine core. If `~/.corecoder/mcp.json` exists, its MCP servers join them as extra `mcp__*` tools; the MCP section below covers it.

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
        results = schedule_by_effect(reply.tool_calls) # parallel reads; serialize side effects
        self.messages += results                       # feed results back, loop again

    return "(hit the round limit)"
```

That's the whole thing. The core skeleton is about twenty lines; counting parallel execution and the bookkeeping after a Ctrl+C interrupt, maybe forty. Almost everything else in CoreCoder's thousand-odd lines is there to clean up the mess the loop runs into once it meets the real world. `llm.py` ends up the biggest file in the project, not because calling a model is hard, but because a streamed response splinters each tool call's arguments into fragments you have to restitch in order, a provider will hand you half a JSON object or a null `usage` field, and 429s, timeouts, dropped connections and 5xx all need backoff-and-retry while the other 4xx should just raise. That unglamorous grunt work, not the loop, is where the real engineering of taking an agent from demo to delivery actually lives; the third essay follows it down to the line.

Three decisions are worth a closer look, because they're the kind of call you can only make after you've understood how others did it, and they're judgments you can lift straight into your own fork.

**`edit_file` does search-and-replace on a unique match, not line numbers.** Line numbers are a trap: the model only has to miscount by one and it quietly edits the wrong place. Anchor on a unique snippet of the original instead. If there's no match, it hands the start of the file back so the model can re-anchor; if there are several matches, it makes the model bring more surrounding context rather than gamble on one. On a successful edit it returns a diff. Recoverable on failure, verifiable on success: the whole loop stays inside the tool.

**Context isn't cut all at once when it's full; it is budgeted as a complete request and gives ground in tiers.** CoreCoder subtracts the dynamic system prompt, tool schemas, output-token reserve, and estimator safety margin before applying the 50/70/90% thresholds to the remaining message budget. It mechanically snips stale tool output first, summarizes older turns next, then hard-collapses if needed. A fresh tool result is protected until one successful LLM request has consumed it; a final deterministic fit step guarantees the estimated request is below the configured window, bounding even fresh data only when that batch cannot fit intact.

**You constrain a sub-agent by withholding capabilities, not by writing rules and hoping it obeys.** A spawned sub-agent gets an isolated context and fresh built-in tool instances, but neither `agent` nor `agent_status`, so it cannot recursively spawn descendants. It reuses the parent's model connection and spend ledger, truncates results over 5,000 characters, and has a 20-round cap. `run_mode` selects blocking foreground or in-process background execution; `isolation` independently selects the current checkout or a retained Git worktree/branch created from `HEAD`.

Every one of these *whys* is traced down to the actual lines of code in the series below.

## The source-reading series · 8 bilingual essays

I also wrote a bilingual source-reading series, one intro plus eight parts, each in Chinese with an English mirror. Against CoreCoder's actual code, it walks through how agents like Claude Code work under the hood. One hard rule I set myself: every line count and every snippet is re-read and re-checked from the repo, never written from memory. The first six get you reading, the seventh gets you forking, and the eighth is about extending it without touching the loop; read them in any order.

- **[Intro · Read Claude Code through CoreCoder, then build your own](article/00-index_EN.md)**
- **[01 · An agent, at its core, is a `while` loop](article/01-the-loop_EN.md)** — the main loop in `agent.py`, interrupts, and the round limit
- **[02 · The tool system: letting the model act, safely](article/02-tools_EN.md)** — the eleven tools in `tools/`, effect metadata, and the bash safety gate
- **[03 · Plug in any LLM, and keep the bill honest](article/03-llm-and-cost_EN.md)** — `llm.py`'s provider wrapper, retries, and cost accounting
- **[04 · Surviving a long task on a finite window](article/04-context_EN.md)** — `context.py`'s three-tier compaction and orphaned tool messages
- **[05 · Parallel execution and sub-agents](article/05-parallel-and-subagents_EN.md)** — effect-aware read concurrency and sub-agent isolation
- **[06 · Turning it into a real command-line tool](article/06-session-and-cli_EN.md)** — `session.py` and path-traversal defense
- **[07 · Fork CoreCoder into your own coding agent](article/07-build-your-own_EN.md)** — from fork to custom tools to swapping models
- **[08 · Three ways to extend without touching the loop: MCP, hooks, and plan mode](article/08-extensibility_EN.md)** — the v0.6.0 extensibility trio and the contract that makes them safe

## Fork it, build something better

Once you understand it, the natural next step is to fork. Getting started doesn't take much:

- **Swap in a model you actually use.** It's the two env vars from above; `llm.py` (498 lines) is the entry point for provider adaptation, fallback, and spend control.
- **Add a tool of your own.** Write a new file against the tool contract in `tools/base.py` (47 lines), declare its effect, then run tests, fetch a page, call an LSP, whatever. The end of the second essay walks you through your first one by hand.
- **Rewrite the system prompt.** `prompt.py` is all of 41 lines; change one line and you'll watch the agent's temperament shift. It's the cheapest "change one thing, see a result" in the whole project.
- **Import it as a library.** The top level exports `Agent`, `LLM`, and `Config`, ready to embed in your own program:

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("find every TODO comment in this project and list them"))
```

Going deeper, the directions are out in the open too. The Docker sandbox is now a small working baseline; the remaining items are still deliberate extension points:

- **Harden the sandbox further.** `--sandbox docker` now supplies a real container boundary for `bash`, but production deployments can still add a custom seccomp/AppArmor profile, read-only workspace modes, per-task images, and isolation for hooks and MCP server processes.
- **Model resilience is deliberately explicit.** Retryable failures exhaust exponential backoff before advancing through the configured fallback chain; a successful switch stays active. The optional USD cap reserves the next input and clamps maximum output before sending, and refuses unknown pricing or missing usage. A production fork can extend this into health-based routing, per-provider credentials, and account-side billing alerts.
- **Sub-agent modes are explicit but intentionally in-process.** Foreground/background and shared/worktree modes are implemented. Production forks can add durable workers, cancellation, event streaming, automatic merge/cherry-pick policy, and process/host isolation.
- **Trace and eval are local, intentionally small building blocks.** JSONL captures one run precisely and the eval runner turns deterministic cases into success/latency/token/tool/cost metrics. Production forks can add OpenTelemetry export, a trace UI, semantic or model-graded checks, datasets, and CI trend storage.
- **No RAG, and the MCP client speaks tools only.** Retrieval-based code location for big repos is still open, and `mcp.py` leaves resources and prompts unimplemented on purpose. Either one is a real way to grow from a minimal core into your own stronger agent.

The README only points; the seventh essay picks up the code details for each. Pick one and start; that's the whole reason the core is kept this small.

## Commands

Inside the REPL, `/help` lists everything; these are the ones you'll reach for:

```
/model <name>    switch model
/compact         compact the context by hand
/tokens          token usage, active fallback, cost, and remaining budget
/diff            files changed this session
/undo            revert the most recent file change
/plan            toggle plan mode (read-only, then a plan to approve)
/save  /sessions save / list sessions
/agents          list background sub-agents
quit / exit      exit (Ctrl+C cancels the current round)
```

Session IDs are sanitized to safe characters before they become filenames, every archive lands under `~/.corecoder/sessions`, and a malicious session name can't traverse out.

## Tool progress

Tool execution has a dedicated live view instead of a row of fire-and-forget call names. A multi-call response is labelled `Parallel tools` only when the effect-aware scheduler has an actual concurrent read batch; otherwise it says `Tool batch`. Each call moves through `queued` → `running` → `done`, `error`, or `blocked`, with elapsed time and an overall `n/N` counter. Permission questions are settled before the live display starts, so an interactive consent prompt never fights Rich for the terminal.

Embedders can drive another UI with the same structured stream by passing `on_tool_progress` to `Agent.chat()`. It receives `batch_started`, `tool_started`, `tool_completed`, and `batch_completed`; the older `on_tool(name, arguments)` callback remains supported. Progress callbacks are best-effort presentation code: if one fails, the tool and Agent loop continue.

## Permissions

Read-only tools (`read_file`, `glob`, `grep`, `todo_write`, `now`, `agent_status`) run the moment the model asks. The mutating or externally active ones (`edit_file`, `write_file`, `bash`, `fetch_url`, MCP tools, and spawning a sub-agent) stop for consent first, and the REPL banner shows which mode you're in:

- In the REPL you get one prompt per call: allow once, always allow this tool, or deny. Foreground sub-agents inherit that interactive layer. A background thread is never allowed to compete with the REPL for input: it can use only `--yes` or tools already marked "always allow"; other stateful calls fail closed and tell the child to route around them.
- In one-shot mode (`-p`) there is nobody to ask, so a mutating call is refused on the spot and the refusal goes back to the model as an ordinary tool result: the loop never hangs on input that can't arrive. Pass `--yes` to approve everything up front (scripts, CI).
- The decision itself is pure logic in `permissions.py`, with the terminal only supplying the prompt callback. You can unit-test consent without a TTY, or reuse the layer in your own embedding.

## Docker sandbox

Local execution remains the default for compatibility. To put `bash` behind a container boundary, first build the deliberately small project image, then opt in:

```bash
docker build -f Dockerfile.sandbox -t corecoder-sandbox:latest .
corecoder --sandbox docker
```

Every bash call runs in a fresh container. Only the directory from which CoreCoder was started is mounted, at `/workspace`; the container root filesystem is read-only, network access defaults to `none`, Linux capabilities are dropped, privilege escalation is disabled, and CPU, memory, PID, and `/tmp` limits are applied. The process runs as the host UID/GID and the image is never pulled implicitly. The built-in file tools also reject resolved paths outside that startup directory, including ordinary `..` and symlink escapes. Use project-relative paths or `/workspace/...` inside bash commands.

The defaults can be changed with `CORECODER_SANDBOX*` environment variables or the matching CLI flags (`--sandbox-image`, `--sandbox-network`, `--sandbox-memory`, `--sandbox-cpus`, and `--sandbox-pids`). For example, `--sandbox-network bridge` explicitly grants container networking. A custom image only needs `/bin/sh` plus the runtimes your project requires.

This boundary is deliberately scoped: hooks, MCP server processes, the LLM client, and `fetch_url` still run on the host, and the workspace mount is writable because this is a coding agent. Host environment variables are not forwarded, but every file already inside the workspace—including `.env` or credentials—is visible to the tools, so use a clean worktree and keep secrets outside it for untrusted tasks. Permission prompts are still required—they decide whether an action is authorized, while the sandbox limits the damage of an authorized or compromised shell command. If the Docker daemon or image is unavailable, the call fails as a tool result; it never falls back to host execution.

## Plan mode

`/plan` toggles plan mode in the REPL. While it's on, the prompt shows `(plan)` and every mutating call (writes, edits, bash, MCP tools, sub-agents) is refused on the spot: the refusal goes back to the model as an ordinary tool result, telling it to keep investigating read-only and present a numbered plan instead. When the plan looks right, `approve` (or `/plan` again) hands control back and the agent executes. Mechanically it is one flag on the `Agent` plus one refusal branch ahead of the consent gate, which itself stays untouched; there is no plan file and nothing is remembered between sessions.

## Hooks

Drop a `hooks.json` under `~/.corecoder` and your own shell commands run around every tool call, the same idea as Claude Code's hooks:

```json
{
  "PreToolUse":  [{"matcher": "bash", "command": "cat >> ~/.corecoder/audit.jsonl"}],
  "PostToolUse": [{"matcher": "*",    "command": "cat >> ~/.corecoder/trace.jsonl"}]
}
```

Each hook gets the call as JSON on stdin (`tool_name`, `tool_input`; post hooks also get `tool_response`). The matcher is an exact tool name; empty or `*` fires on every tool. A pre hook can veto the call with exit code 2, and its stderr travels back to the model as the reason so it can route around the block. Post hooks only observe and can never block. A hook that errors or runs past ten seconds is skipped with a warning: hooks assist the loop, they never get to kill it. The whole mechanism is `hooks.py`, and the REPL banner shows how many hooks loaded.

Two worth stealing (the commands lean on `jq`, the usual suspect):

```bash
# 1. lint gate: after every edit/write, run the project's fast linter on the
#    touched file. The model sees the output and fixes its own mistakes in
#    the same turn instead of waiting for CI.
{
  "PostToolUse": [{
    "matcher": "edit",
    "command": "f=$(jq -r .tool_input.path); ruff check \"$f\" 2>&1 | head -20"
  }]
}

# 2. write protect: refuse edits under paths you never want an agent to
#    touch. Exit code 2 vetoes the call and the message reaches the model.
{
  "PreToolUse": [{
    "matcher": "edit",
    "command": "case \"$(jq -r .tool_input.path)\" in .env*|*/secrets/*|*.pem) echo 'that path is off-limits' >&2; exit 2;; esac"
  }]
}
```

Both are plain shell; nothing here is CoreCoder-specific syntax beyond the JSON shape and the exit-code-2 veto.

## MCP servers

Drop a `mcp.json` under `~/.corecoder` and tools from any MCP server join the agent over stdio, the same config shape as Claude Code's:

```json
{
  "mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}
  }
}
```

Each configured server starts as a subprocess at launch, handshakes, and lists its tools; every one is registered as `mcp__<server>__<tool>`, so hook matchers and the consent gate treat it exactly like a built-in. MCP tools stay out of the read-only set, meaning the agent asks before running one. The handshake gets fifteen seconds, a call gets sixty, and a server that dies or never answers fails that one call as an ordinary tool result instead of killing the loop. The client speaks the tools slice of the protocol (initialize, tools/list, tools/call) and nothing else, which keeps the whole thing inside `mcp.py` at about 200 lines. With no `mcp.json` there is no MCP and nothing changes.

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

Before you send anything, run `pytest tests/ -q` (228 cases), `ruff check`, and `compileall`, and make sure they're green. MIT licensed: fork it, learn from it, ship something better. A mention of this project is appreciated.

---

By [Yufeng He](https://github.com/he-yufeng), formerly at Moonshot AI (Kimi). I earlier wrote a fairly complete [Claude Code source analysis](https://zhuanlan.zhihu.com/p/1898797658343862272) on Zhihu; this project is its hands-on counterpart: that one walks you through reading it, this one through rebuilding it.

> CoreCoder was formerly named NanoCoder; it was renamed to avoid confusion with [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder), and old links redirect here automatically.
