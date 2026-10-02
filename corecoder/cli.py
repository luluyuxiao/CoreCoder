"""Interactive REPL - the user-facing terminal interface."""

import argparse
import os
import sys
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from . import __version__
from .agent import Agent
from .capabilities import load_capability_policy
from .config import Config
from .hooks import load_hooks
from .llm import LLM, LiteLLM
from .mcp import load_mcp_tools
from .permissions import Permission
from .sandbox import WorkspacePathPolicy, create_command_executor
from .session import (
    SessionStore,
    create_session_store,
    delete_session,
    list_sessions,
    load_session_record,
    load_transcript,
    new_session_id,
    save_snapshot,
)
from .skills import SkillRegistry
from .tools import build_tools
from .trace import JsonlTrace

console = Console()
_ACTIVE_TOOL_PROGRESS = threading.local()


class _ToolProgressDisplay:
    """Thread-safe Rich view of one tool-call batch."""

    def __init__(self, target_console: Console | None = None):
        self.console = target_console or console
        self._lock = threading.RLock()
        self._progress: Progress | None = None
        self._tool_tasks: dict[str, int] = {}
        self._finished: set[str] = set()
        self._overall_task: int | None = None
        self._total = 0

    def __call__(self, event: str, payload: dict):
        with self._lock:
            if event == "batch_started":
                self._start(payload)
            elif event == "tool_started":
                self._tool_started(payload)
            elif event == "tool_completed":
                self._tool_completed(payload)
            elif event == "batch_completed":
                self._batch_completed()

    def _start(self, payload: dict):
        if self._progress is not None:
            self._stop(interrupted=True)
        self._total = payload["total"]
        label = "Parallel tools" if payload.get("parallel") else "Tool batch"
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=18),
            TaskProgressColumn(),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._progress = progress
        self._overall_task = progress.add_task(
            f"[bold cyan]{label}[/bold cyan]",
            total=self._total,
            status=f"0/{self._total}",
        )
        for item in payload["tools"]:
            call_id = item["tool_call_id"]
            details = escape(_brief(item["arguments"], maxlen=58))
            description = f"[cyan]{escape(item['tool_name'])}[/cyan] [dim]{details}[/dim]"
            self._tool_tasks[call_id] = progress.add_task(
                description,
                total=1,
                start=False,
                status="[dim]queued[/dim]",
            )
        progress.start()
        _ACTIVE_TOOL_PROGRESS.display = self

    @contextmanager
    def suspended(self):
        """Temporarily stop Live rendering while the CLI reads nested input."""
        with self._lock:
            progress = self._progress
            if progress is not None:
                progress.stop()
        try:
            yield
        finally:
            with self._lock:
                # The batch may have been closed while input was being read.
                if progress is not None and self._progress is progress:
                    progress.start()

    def _tool_started(self, payload: dict):
        if self._progress is None:
            return
        task_id = self._tool_tasks.get(payload["tool_call_id"])
        if task_id is not None:
            self._progress.start_task(task_id)
            self._progress.update(task_id, status="[yellow]running[/yellow]")

    def _tool_completed(self, payload: dict):
        if self._progress is None:
            return
        call_id = payload["tool_call_id"]
        if call_id in self._finished:
            return
        task_id = self._tool_tasks.get(call_id)
        if task_id is None:
            return
        outcome = payload.get("outcome", "success")
        duration = payload.get("duration_ms", 0.0)
        if outcome == "success":
            status = f"[green]done {duration:.1f}ms[/green]"
        elif outcome == "blocked":
            status = "[yellow]blocked[/yellow]"
        else:
            status = f"[red]error {duration:.1f}ms[/red]"
        self._progress.start_task(task_id)
        self._progress.update(task_id, completed=1, status=status)
        self._progress.stop_task(task_id)
        self._finished.add(call_id)
        if self._overall_task is not None:
            completed = len(self._finished)
            self._progress.update(
                self._overall_task,
                completed=completed,
                status=f"{completed}/{self._total}",
            )

    def _batch_completed(self):
        if self._progress is None:
            return
        if self._overall_task is not None:
            self._progress.update(
                self._overall_task,
                completed=self._total,
                status=f"[green]{self._total}/{self._total} done[/green]",
            )
            self._progress.stop_task(self._overall_task)
        self._stop()

    def close(self, *, interrupted: bool = False):
        """Stop an active display, including after Ctrl+C or an exception."""
        with self._lock:
            self._stop(interrupted=interrupted)

    def _stop(self, *, interrupted: bool = False):
        if self._progress is None:
            return
        if interrupted and self._overall_task is not None:
            self._progress.update(
                self._overall_task,
                status="[yellow]interrupted[/yellow]",
            )
        self._progress.stop()
        if getattr(_ACTIVE_TOOL_PROGRESS, "display", None) is self:
            del _ACTIVE_TOOL_PROGRESS.display
        self._progress = None
        self._tool_tasks = {}
        self._finished = set()
        self._overall_task = None
        self._total = 0


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _parse_args():
    p = argparse.ArgumentParser(
        prog="corecoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $CORECODER_MODEL or gpt-5.5)")
    p.add_argument(
        "--fallback-model",
        action="append",
        dest="fallback_models",
        help="Fallback model after retryable failures; repeat for an ordered chain",
    )
    p.add_argument(
        "--max-cost",
        type=_positive_float,
        dest="max_cost_usd",
        help="Hard estimated USD budget for this process",
    )
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("--yes", action="store_true", help="Auto-approve every tool call (for scripts and CI)")
    p.add_argument(
        "--sandbox", choices=("local", "docker"),
        help="Command execution mode (default: $CORECODER_SANDBOX or local)",
    )
    p.add_argument("--sandbox-image", help="Docker image used by --sandbox docker")
    p.add_argument("--sandbox-network", help="Docker network mode (default: none)")
    p.add_argument("--sandbox-memory", help="Docker memory limit (default: 1g)")
    p.add_argument("--sandbox-cpus", type=float, help="Docker CPU limit (default: 1)")
    p.add_argument("--sandbox-pids", type=int, help="Docker PID limit (default: 128)")
    p.add_argument("--trace", metavar="PATH", help="Append a structured JSONL execution trace")
    p.add_argument(
        "--trace-content",
        action="store_true",
        help="Include prompts and tool payloads in traces (may contain secrets)",
    )
    p.add_argument(
        "--storage",
        metavar="PATH",
        help="SQLite session database (default: ~/.corecoder/sessions/sessions.db)",
    )
    p.add_argument(
        "--no-autosave",
        action="store_true",
        help="Disable stable-state session autosave; /save still works",
    )
    p.add_argument(
        "--capability-policy",
        metavar="PATH",
        help="Per-tool capability policy JSON (default: ~/.corecoder/capabilities.json)",
    )
    p.add_argument("--demo", action="store_true", help="Run the offline scripted demo (no API key needed)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.demo:
        from .demo import run_demo
        raise SystemExit(run_demo())

    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.fallback_models:
        config.fallback_models = args.fallback_models
    if args.max_cost_usd is not None:
        config.max_cost_usd = args.max_cost_usd
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.sandbox:
        config.sandbox = args.sandbox
    if args.sandbox_image:
        config.sandbox_image = args.sandbox_image
    if args.sandbox_network:
        config.sandbox_network = args.sandbox_network
    if args.sandbox_memory:
        config.sandbox_memory = args.sandbox_memory
    if args.sandbox_cpus is not None:
        config.sandbox_cpus = args.sandbox_cpus
    if args.sandbox_pids is not None:
        config.sandbox_pids = args.sandbox_pids
    if args.trace:
        config.trace_path = args.trace
    if args.trace_content:
        config.trace_content = True
    if args.storage:
        config.storage_path = args.storage
    if args.no_autosave:
        config.autosave = False
    if args.capability_policy:
        config.capability_policy_path = args.capability_policy

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or CORECODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    try:
        llm = llm_cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            fallback_models=config.fallback_models,
            max_cost_usd=config.max_cost_usd,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    except ValueError as e:
        console.print(f"[red bold]Invalid model policy:[/] {e}")
        sys.exit(2)
    try:
        executor = create_command_executor(
            config.sandbox,
            Path.cwd(),
            image=config.sandbox_image,
            network=config.sandbox_network,
            memory=config.sandbox_memory,
            cpus=config.sandbox_cpus,
            pids_limit=config.sandbox_pids,
        )
    except ValueError as e:
        console.print(f"[red bold]Invalid sandbox configuration:[/] {e}")
        sys.exit(2)
    try:
        capability_policy = load_capability_policy(config.capability_policy_path)
    except (OSError, TypeError, ValueError) as e:
        console.print(f"[red bold]Invalid capability policy:[/] {e}")
        sys.exit(2)
    path_policy = WorkspacePathPolicy(Path.cwd()) if config.sandbox == "docker" else None
    # consent layer: ask in the REPL, refuse in one-shot mode, --yes skips it
    if args.yes:
        permission = Permission(allow_all=True)
    elif args.prompt:
        permission = Permission()
    else:
        permission = Permission(ask=_ask_permission)
    trace = None
    if config.trace_path:
        try:
            trace = JsonlTrace(
                config.trace_path,
                capture_content=config.trace_content,
            )
        except OSError as e:
            console.print(f"[red bold]Cannot open trace file:[/] {e}")
            sys.exit(2)

    session_store = None
    resumed = None
    if config.autosave or args.resume:
        try:
            session_store = create_session_store(config.storage_path)
            resumed = (
                load_session_record(args.resume, store=session_store)
                if args.resume else None
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red bold]Cannot open session storage:[/] {e}")
            sys.exit(2)
    if args.resume and resumed is None:
        console.print(f"[red]Session '{args.resume}' not found.[/red]")
        sys.exit(1)
    session_id = resumed.id if resumed is not None else new_session_id()
    skill_registry = SkillRegistry.discover(Path.cwd())

    def persist_snapshot(snapshot: dict):
        assert session_store is not None  # autosave always initializes it above
        save_snapshot(session_id, snapshot, store=session_store)

    agent = Agent(
        llm=llm,
        tools=[
            *build_tools(
                executor=executor,
                path_policy=path_policy,
                skill_registry=skill_registry,
            ),
            *load_mcp_tools(workspace=Path.cwd()),
        ],
        max_context_tokens=config.max_context_tokens,
        permission=permission,
        hooks=load_hooks(),
        trace=trace,
        state_callback=persist_snapshot if config.autosave else None,
        capability_policy=capability_policy,
    )

    # resume saved session
    if resumed is not None:
        agent.restore_state(resumed.to_snapshot(), restore_model=not bool(args.model))
        config.model = agent.llm.model
        console.print(
            f"[green]Resumed session: {resumed.id} "
            f"(model: {agent.llm.model}, status: {resumed.status})[/green]"
        )
        if resumed.workspace and Path(resumed.workspace).resolve() != Path.cwd().resolve():
            console.print(
                f"[yellow]Session workspace was {resumed.workspace}; "
                f"current workspace is {Path.cwd()}.[/yellow]"
            )

    # one-shot mode
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config, session_store, session_id, skill_registry)


def _ask_permission(tool_name: str, arguments: dict) -> str:
    """REPL consent prompt. Anything but a clear yes counts as a no."""
    progress = getattr(_ACTIVE_TOOL_PROGRESS, "display", None)
    prompt_context = progress.suspended() if progress is not None else nullcontext()
    with prompt_context:
        console.print(f"\n[bold yellow]permission requested:[/] [cyan]{tool_name}[/cyan]({_brief(arguments)})")
        try:
            answer = pt_prompt("  [y] allow once  [a] always allow this tool  [n] deny: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]denied[/dim]")
            return "deny"
    if answer in ("y", "yes"):
        return "once"
    if answer in ("a", "always"):
        return "always"
    return "deny"


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    perm = agent.permission
    if perm is not None and perm.ask is None and not perm.allow_all:
        console.print("[dim]one-shot mode: mutating tools are refused unless you pass --yes[/dim]")

    def on_token(tok):
        print(tok, end="", flush=True)

    tool_progress = _ToolProgressDisplay()
    try:
        agent.chat(
            prompt,
            on_token=on_token,
            on_tool_progress=tool_progress,
        )
    except KeyboardInterrupt:
        tool_progress.close(interrupted=True)
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        tool_progress.close(interrupted=True)
        # one-shot mode: print whatever went wrong and exit non-zero
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        tool_progress.close()
    print()


def _repl(
    agent: Agent,
    config: Config,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    skill_registry: SkillRegistry | None = None,
):
    """Interactive read-eval-print loop."""
    session_id = session_id or new_session_id()

    def active_store() -> SessionStore:
        nonlocal session_store
        if session_store is None:
            session_store = create_session_store(config.storage_path)
        return session_store

    perm = agent.permission
    mode = "auto-approve every tool call (--yes)" if (perm and perm.allow_all) else "ask before mutating tools"
    mcp_count = sum(1 for t in agent.tools if t.name.startswith("mcp__"))
    mcp_clients = {
        id(t._client): t._client
        for t in agent.tools
        if t.name.startswith("mcp__") and hasattr(t, "_client")
    }
    mcp_modes = {
        sandbox_mode: sum(
            client.sandbox_mode == sandbox_mode for client in mcp_clients.values()
        )
        for sandbox_mode in ("docker", "host")
    }
    mcp_mode_text = ", ".join(
        f"{count} {mode}" for mode, count in mcp_modes.items() if count
    )
    console.print(Panel(
        f"[bold]CoreCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + (
            f"\nFallbacks: [cyan]{' → '.join(config.fallback_models)}[/cyan]"
            if config.fallback_models else ""
        )
        + (
            f"\nUSD budget: [cyan]${config.max_cost_usd:.4f}[/cyan]"
            if config.max_cost_usd is not None else ""
        )
        + f"\nPermissions: [cyan]{mode}[/cyan]"
        + (
            f"\nSandbox: [cyan]docker[/cyan] image={config.sandbox_image} "
            f"network={config.sandbox_network} workspace={Path.cwd()}"
            if config.sandbox == "docker"
            else "\nSandbox: [yellow]local host execution[/yellow]"
        )
        + (f"\nHooks: [cyan]{len(agent.hooks.pre)} pre, {len(agent.hooks.post)} post[/cyan]"
           " from ~/.corecoder/hooks.json" if agent.hooks else "")
        + (
            f"\nMCP: [cyan]{mcp_count} tools[/cyan] from {len(mcp_clients)} servers "
            f"[dim]({mcp_mode_text})[/dim]"
            if mcp_count else ""
        )
        + (
            f"\nSkills: [cyan]{len(skill_registry)} available[/cyan]"
            if skill_registry else ""
        )
        + (
            f"\nCapabilities: [cyan]{agent.capability_policy.source}[/cyan] "
            f"(default {agent.capability_policy.default})"
            if agent.capability_policy.enabled else ""
        )
        + (
            f"\nTrace: [cyan]{config.trace_path}[/cyan]"
            + (" [yellow](content included)[/yellow]" if config.trace_content else " [dim](metadata only)[/dim]")
            if config.trace_path else ""
        )
        + (
            f"\nSession: [cyan]{session_id}[/cyan] "
            f"[dim]({'autosave' if config.autosave else 'manual save'} → "
            f"{getattr(session_store, 'path', 'on-demand store')})[/dim]"
        )
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.corecoder_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You (plan) > " if agent.plan_mode else "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/plan":
            agent.plan_mode = not agent.plan_mode
            agent.persist_state("completed")
            if agent.plan_mode:
                console.print(
                    "[yellow]Plan mode on.[/yellow] The agent can look but not touch: it will "
                    "investigate read-only and present a plan. Type [bold]approve[/bold] to "
                    "accept the plan, or [bold]/plan[/bold] again to exit."
                )
            else:
                console.print("[yellow]Plan mode off.[/yellow]")
            continue
        if agent.plan_mode and user_input.lower() in ("approve", "/approve"):
            agent.plan_mode = False
            console.print("[yellow]Plan mode off.[/yellow]")
            user_input = "approve"  # the approval itself goes to the model, which then executes
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            if agent.llm.max_cost_usd is not None:
                remaining = agent.llm.remaining_budget
                line += (
                    f"  budget ${agent.llm.max_cost_usd:.4f}, "
                    + (f"${remaining:.4f} left" if remaining is not None else "cost unavailable")
                )
            if agent.llm.fallback_history:
                line += f"  active model: {agent.llm.model}"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                agent.persist_state("completed")
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{agent.llm.model}[/cyan]")
            continue
        if user_input == "/memory":
            rendered = agent.memory.render()
            console.print(
                Panel(escape(rendered), title="Structured Memory")
                if rendered else "[dim]Structured memory is empty.[/dim]"
            )
            continue
        if user_input == "/goal" or user_input.startswith("/goal "):
            content = user_input[len("/goal "):].strip() if user_input.startswith("/goal ") else ""
            if not content:
                console.print(
                    f"Goal: [cyan]{escape(agent.memory.goal)}[/cyan]"
                    if agent.memory.goal else "[dim]No goal is set.[/dim]"
                )
            elif content.lower() == "clear":
                agent.memory.clear_goal()
                agent.persist_state("completed")
                console.print("[green]Goal cleared.[/green]")
            else:
                try:
                    agent.memory.set_goal(content)
                except ValueError as error:
                    console.print(f"[yellow]Goal not changed: {escape(str(error))}[/yellow]")
                else:
                    agent.persist_state("completed")
                    console.print(f"[green]Goal set: {escape(agent.memory.goal)}[/green]")
            continue
        if user_input == "/constraint" or user_input.startswith("/constraint "):
            content = (
                user_input[len("/constraint "):].strip()
                if user_input.startswith("/constraint ") else ""
            )
            if not content:
                items = agent.memory.constraints
                if not items:
                    console.print("[dim]No persistent constraints.[/dim]")
                for item in items:
                    console.print(
                        f"  [cyan]{item['id']}[/cyan] [{item['status']}] "
                        f"{escape(item['content'])} [dim](semantic)[/dim]"
                    )
            elif content.startswith("revoke "):
                constraint_id = content[len("revoke "):].strip()
                if agent.memory.revoke_constraint(constraint_id):
                    agent.persist_state("completed")
                    console.print(f"[green]Constraint revoked: {escape(constraint_id)}[/green]")
                else:
                    console.print(f"[yellow]Active constraint not found: {escape(constraint_id)}[/yellow]")
            else:
                try:
                    item = agent.memory.add_constraint(content)
                except ValueError as error:
                    console.print(f"[yellow]Constraint not added: {escape(str(error))}[/yellow]")
                else:
                    agent.persist_state("completed")
                    console.print(
                        f"[green]Constraint added: {item['id']}[/green] "
                        "[dim](semantic; use Hook/Capability/Sandbox for enforcement)[/dim]"
                    )
            continue
        if user_input == "/decision" or user_input.startswith("/decision "):
            content = (
                user_input[len("/decision "):].strip()
                if user_input.startswith("/decision ") else ""
            )
            if not content:
                if not agent.memory.decisions:
                    console.print("[dim]No decisions recorded.[/dim]")
                for item in agent.memory.decisions:
                    console.print(f"  [cyan]{item['id']}[/cyan] {escape(item['content'])}")
            else:
                try:
                    item = agent.memory.add_decision(content, source="user")
                except ValueError as error:
                    console.print(f"[yellow]Decision not added: {escape(str(error))}[/yellow]")
                else:
                    agent.persist_state("completed")
                    console.print(f"[green]Decision recorded: {item['id']}[/green]")
            continue
        if user_input == "/compact":
            compressed, before, after = agent.compress_context()
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
                agent.persist_state("completed")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            snapshot = agent.state_snapshot("saved")
            sid = save_snapshot(
                session_id,
                snapshot,
                store=active_store(),
            )
            agent.acknowledge_persisted(snapshot)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: corecoder -r {sid}")
            continue
        if user_input == "/name" or user_input.startswith("/name "):
            requested_name = (
                user_input[len("/name "):].strip()
                if user_input.startswith("/name ")
                else ""
            )
            if not requested_name:
                record = active_store().load(session_id)
                current_name = record.name if record is not None else ""
                console.print(
                    f"Session name: [cyan]{escape(current_name)}[/cyan]"
                    if current_name else "[dim]This session has no name yet.[/dim]"
                )
                continue
            snapshot = agent.state_snapshot("saved")
            snapshot["name"] = requested_name
            save_snapshot(session_id, snapshot, store=active_store())
            agent.acknowledge_persisted(snapshot)
            record = active_store().load(session_id)
            saved_name = record.name if record is not None else requested_name
            console.print(f"[green]Session named: {escape(saved_name)}[/green]")
            continue
        if user_input == "/session":
            mode_label = "autosave" if config.autosave else "manual save"
            record = active_store().load(session_id)
            name = record.name if record is not None else ""
            console.print(
                f"Session: [cyan]{session_id}[/cyan]  status: [cyan]{mode_label}[/cyan]\n"
                f"Name: [cyan]{escape(name) if name else '(unnamed)'}[/cyan]\n"
                f"Storage: [dim]{getattr(active_store(), 'path', 'custom store')}[/dim]"
            )
            continue
        if user_input == "/diff":
            changed_files = agent.memory.files_modified
            if not changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(changed_files)}):[/bold]")
                for file_path in sorted(changed_files):
                    console.print(f"  [cyan]{escape(file_path)}[/cyan]")
            continue
        if user_input == "/undo":
            from .checkpoints import pending, undo
            console.print(undo())
            left = pending()
            if left:
                console.print(f"[dim]{left} more checkpoint(s) on the stack.[/dim]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions(store=active_store())
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    name = escape(s["name"]) if s["name"] else "(unnamed)"
                    console.print(
                        f"  [bold]{name}[/bold]  [cyan]{s['id']}[/cyan]\n"
                        f"    ({escape(s['model'])}, {s['status']}, {s['saved_at']}) "
                        f"{escape(s['preview'])}"
                    )
            continue
        if user_input == "/skills":
            states = agent.active_skill_states()
            if not skill_registry:
                console.print("[dim]No skills discovered.[/dim]")
            else:
                console.print("[bold]Available skills:[/bold]")
                for name in skill_registry.names():
                    skill = skill_registry.get(name)
                    status = states.get(name, {}).get("status")
                    marker = (
                        f" [yellow][{status}][/]" if status else ""
                    )
                    console.print(
                        f"  [cyan]{name}[/cyan] [dim]({skill.scope})[/dim]{marker} "
                        f"{escape(skill.description)}"
                    )
            unavailable = [
                name for name, state in states.items()
                if state.get("status") == "unavailable"
            ]
            if unavailable:
                console.print(
                    "[yellow]Saved but unavailable:[/] " + ", ".join(unavailable)
                )
            continue
        if user_input == "/capabilities":
            policy = agent.capability_policy
            source = str(policy.source) if policy.source else "built-in allow policy"
            console.print(
                f"[bold]Capability policy:[/] {escape(source)} "
                f"[dim](default {policy.default})[/dim]"
            )
            for tool in agent.tools:
                required = sorted(tool.capabilities)
                console.print(
                    f"  [cyan]{tool.name}[/cyan]: "
                    + (", ".join(required) if required else "[dim]none[/dim]")
                )
            continue
        if user_input == "/transcript" or user_input.startswith("/transcript "):
            target = (
                user_input[len("/transcript "):].strip()
                if user_input.startswith("/transcript ")
                else session_id
            )
            events = load_transcript(target, store=active_store())
            if not events:
                console.print(f"[dim]No transcript events for {target}.[/dim]")
            else:
                console.print(
                    f"[bold]Transcript: {target} ({len(events)} events)[/bold]"
                )
                for event in events:
                    payload = event.get("payload") or {}
                    content = str(payload.get("content") or "").replace("\n", " ")
                    if not content and payload.get("tool_calls"):
                        names = [
                            str((call.get("function") or {}).get("name") or "tool")
                            for call in payload["tool_calls"]
                        ]
                        content = "tool calls: " + ", ".join(names)
                    if len(content) > 180:
                        content = content[:177] + "..."
                    console.print(
                        f"  [cyan]{event['sequence']:>4}[/cyan] "
                        f"[dim]{event['event_type']}[/dim] {escape(content)}"
                    )
            continue
        if user_input.startswith("/delete-session "):
            target = user_input[len("/delete-session "):].strip()
            if target == session_id:
                console.print("[yellow]Cannot delete the active session.[/yellow]")
            elif delete_session(target, store=active_store()):
                console.print(f"[green]Deleted session: {target}[/green]")
            else:
                console.print(f"[yellow]Session not found: {target}[/yellow]")
            continue
        if user_input == "/agents":
            status_tool = next((t for t in agent.tools if t.name == "agent_status"), None)
            console.print(status_tool.execute() if status_tool else "[dim]Sub-agents are disabled.[/dim]")
            continue

        # an unknown /command shouldn't be sent to the model as a prompt
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
            continue

        # call the agent
        streamed: list[str] = []

        def on_token(tok, streamed=streamed):
            streamed.append(tok)
            print(tok, end="", flush=True)

        tool_progress = _ToolProgressDisplay()
        try:
            response = agent.chat(
                user_input,
                on_token=on_token,
                on_tool_progress=tool_progress,
            )
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            tool_progress.close(interrupted=True)
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:  # noqa: BLE001
            tool_progress.close(interrupted=True)
            # keep the REPL alive no matter what chat() throws
            console.print(f"\n[red]Error: {e}[/red]")
        finally:
            tool_progress.close()


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /memory        Show structured goal, constraints, plan, and decisions\n"
        "  /goal <text>   Set the persistent task goal (use 'clear' to remove)\n"
        "  /constraint <text>  Add a persistent semantic constraint\n"
        "  /decision <text>    Record a durable task decision\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /undo          Revert the most recent file change\n"
        "  /plan          Toggle plan mode: read-only, then a plan to approve\n"
        "  /save          Save the active session now\n"
        "  /name <name>   Name the active session for easier resume\n"
        "  /session       Show active session and storage\n"
        "  /sessions      List saved sessions\n"
        "  /skills        List workflow skills and active state\n"
        "  /capabilities  Show Tool capabilities and active policy\n"
        "  /transcript [id]  Show the append-only original event history\n"
        "  /delete-session <id>  Delete an inactive session\n"
        "  /agents        List background sub-agents\n"
        "  quit           Exit CoreCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="CoreCoder Help",
        border_style="dim",
    ))


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
