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
from .config import Config
from .hooks import load_hooks
from .llm import LLM, LiteLLM
from .mcp import load_mcp_tools
from .permissions import Permission
from .sandbox import WorkspacePathPolicy, create_command_executor
from .session import list_sessions, load_session, save_session
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
    agent = Agent(
        llm=llm,
        tools=[*build_tools(executor=executor, path_policy=path_policy), *load_mcp_tools()],
        max_context_tokens=config.max_context_tokens,
        permission=permission,
        hooks=load_hooks(),
        trace=trace,
    )

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config)


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


def _repl(agent: Agent, config: Config):
    """Interactive read-eval-print loop."""
    perm = agent.permission
    mode = "auto-approve every tool call (--yes)" if (perm and perm.allow_all) else "ask before mutating tools"
    mcp_count = sum(1 for t in agent.tools if t.name.startswith("mcp__"))
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
        + (f"\nMCP: [cyan]{mcp_count} tools[/cyan] from ~/.corecoder/mcp.json" if mcp_count else "")
        + (
            f"\nTrace: [cyan]{config.trace_path}[/cyan]"
            + (" [yellow](content included)[/yellow]" if config.trace_content else " [dim](metadata only)[/dim]")
            if config.trace_path else ""
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
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{agent.llm.model}[/cyan]")
            continue
        if user_input == "/compact":
            compressed, before, after = agent.compress_context()
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, agent.llm.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: corecoder -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/undo":
            from .checkpoints import pending, undo
            console.print(undo())
            left = pending()
            if left:
                console.print(f"[dim]{left} more checkpoint(s) on the stack.[/dim]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
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
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /undo          Revert the most recent file change\n"
        "  /plan          Toggle plan mode: read-only, then a plan to approve\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
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
