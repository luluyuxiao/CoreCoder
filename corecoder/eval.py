"""Small repeatable evaluation harness for CoreCoder agents.

An eval manifest describes prompts, source workspaces, and deterministic checks.
Each run gets a fresh copied workspace by default, a fresh Agent conversation,
and an in-memory trace from which latency/tool/token metrics are derived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import Config
from .llm import LLM, LiteLLM
from .permissions import Permission
from .sandbox import WorkspacePathPolicy, create_command_executor
from .tools import build_tools
from .trace import CompositeTrace, JsonlTrace, MemoryTrace, TraceSink


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    workspace: Path
    expect: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    name: str
    repetition: int
    passed: bool
    checks: list[dict]
    metrics: dict
    response: str = ""
    error: str | None = None
    workspace: str = ""


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load and validate a JSON eval manifest."""
    manifest = Path(path).expanduser().resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    raw_cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("eval manifest must contain a non-empty 'cases' list")

    cases = []
    names = set()
    for index, item in enumerate(raw_cases, 1):
        if not isinstance(item, dict):
            raise TypeError(f"eval case {index} must be an object")
        name = str(item.get("name") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name or not prompt:
            raise ValueError(f"eval case {index} requires non-empty name and prompt")
        if name in names:
            raise ValueError(f"duplicate eval case name {name!r}")
        names.add(name)
        workspace_value = item.get("workspace", ".")
        workspace = (manifest.parent / workspace_value).resolve()
        if not workspace.is_dir():
            raise ValueError(f"eval case {name!r} workspace is not a directory: {workspace}")
        expect = item.get("expect") or {}
        if not isinstance(expect, dict):
            raise TypeError(f"eval case {name!r} expect must be an object")
        cases.append(EvalCase(name=name, prompt=prompt, workspace=workspace, expect=expect))
    return cases


AgentFactory = Callable[[EvalCase, Path, TraceSink], Agent]


def run_eval_suite(
    cases: list[EvalCase],
    agent_factory: AgentFactory,
    *,
    repetitions: int = 1,
    in_place: bool = False,
    keep_workspaces: str | Path | None = None,
    trace_dir: str | Path | None = None,
    trace_content: bool = False,
) -> dict:
    """Run cases and return a JSON-serializable report."""
    if not cases:
        raise ValueError("eval suite requires at least one case")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if in_place and keep_workspaces:
        raise ValueError("--in-place and --keep-workspaces cannot be used together")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("eval case names must be unique")
    kept_root = Path(keep_workspaces).expanduser().resolve() if keep_workspaces else None
    traces_root = Path(trace_dir).expanduser().resolve() if trace_dir else None
    if kept_root:
        kept_root.mkdir(parents=True, exist_ok=True)
    if traces_root:
        traces_root.mkdir(parents=True, exist_ok=True)

    results = []
    suite_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="corecoder-eval-") as temp:
        temporary_root = Path(temp)
        for repetition in range(1, repetitions + 1):
            for case in cases:
                slug = _case_slug(case.name, repetition)
                if in_place:
                    workspace = case.workspace
                else:
                    workspace = (kept_root or temporary_root) / slug
                    _copy_workspace(case.workspace, workspace)

                memory_trace = MemoryTrace(capture_content=False)
                trace: TraceSink = memory_trace
                if traces_root:
                    trace = CompositeTrace(
                        memory_trace,
                        JsonlTrace(
                            traces_root / f"{slug}.jsonl",
                            capture_content=trace_content,
                            append=False,
                        ),
                    )
                try:
                    agent = agent_factory(case, workspace, trace)
                except Exception as e:  # noqa: BLE001
                    error = f"{type(e).__name__}: {e}"
                    result = EvalResult(
                        name=case.name,
                        repetition=repetition,
                        passed=False,
                        checks=[{
                            "check": "agent_created",
                            "passed": False,
                            "actual": error,
                        }],
                        metrics=_metrics([], 0.0, 0, 0, None),
                        error=error,
                        workspace=str(workspace),
                    )
                else:
                    result = _run_case(case, repetition, workspace, agent, memory_trace)
                results.append(result)

    serialized = [asdict(result) for result in results]
    passed = sum(result.passed for result in results)
    durations = [result.metrics["duration_seconds"] for result in results]
    run_count = len(results)
    total_prompt = sum(result.metrics["prompt_tokens"] for result in results)
    total_completion = sum(result.metrics["completion_tokens"] for result in results)
    total_tools = sum(result.metrics["tool_calls"] for result in results)
    total_cost = _sum_known_costs(results)
    per_case = []
    for case in cases:
        case_results = [result for result in results if result.name == case.name]
        case_passed = sum(result.passed for result in case_results)
        per_case.append({
            "name": case.name,
            "runs": len(case_results),
            "passed": case_passed,
            "pass_rate": case_passed / len(case_results),
            "pass_at_k": case_passed > 0,
        })
    return {
        "summary": {
            "cases": len(cases),
            "runs": run_count,
            "passed_runs": passed,
            "failed_runs": run_count - passed,
            "success_rate": passed / run_count if results else 0.0,
            "duration_seconds": round(time.perf_counter() - suite_started, 6),
            "mean_case_duration_seconds": round(sum(durations) / len(durations), 6) if durations else 0.0,
            "mean_prompt_tokens": total_prompt / run_count if run_count else 0.0,
            "mean_completion_tokens": total_completion / run_count if run_count else 0.0,
            "mean_tool_calls": total_tools / run_count if run_count else 0.0,
            "mean_estimated_cost_usd": total_cost / run_count if total_cost is not None else None,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tool_calls": total_tools,
            "estimated_cost_usd": total_cost,
        },
        "cases": per_case,
        "results": serialized,
    }


def compare_reports(current: dict, baseline: dict) -> dict:
    """Return signed deltas for the metrics most useful in regressions."""
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        raise TypeError("eval reports must be JSON objects")
    current_summary = current.get("summary", {})
    baseline_summary = baseline.get("summary", {})
    if not isinstance(current_summary, dict) or not isinstance(baseline_summary, dict):
        raise TypeError("eval report summaries must be JSON objects")

    def delta(name: str):
        current_value = current_summary.get(name)
        baseline_value = baseline_summary.get(name)
        if current_value is None or baseline_value is None:
            return None
        return current_value - baseline_value

    return {
        "success_rate_delta": delta("success_rate"),
        "mean_case_duration_seconds_delta": delta("mean_case_duration_seconds"),
        "mean_prompt_tokens_delta": delta("mean_prompt_tokens"),
        "mean_completion_tokens_delta": delta("mean_completion_tokens"),
        "mean_tool_calls_delta": delta("mean_tool_calls"),
        "mean_estimated_cost_usd_delta": delta("mean_estimated_cost_usd"),
    }


def _run_case(
    case: EvalCase,
    repetition: int,
    workspace: Path,
    agent: Agent,
    trace: MemoryTrace,
) -> EvalResult:
    started = time.perf_counter()
    prompt_before = getattr(agent.llm, "total_prompt_tokens", 0)
    completion_before = getattr(agent.llm, "total_completion_tokens", 0)
    cost_before = _cost(agent.llm)
    response = ""
    error = None
    try:
        response = agent.chat(case.prompt)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    duration = time.perf_counter() - started
    metrics = _metrics(
        trace.events,
        duration,
        getattr(agent.llm, "total_prompt_tokens", 0) - prompt_before,
        getattr(agent.llm, "total_completion_tokens", 0) - completion_before,
        _cost_delta(cost_before, _cost(agent.llm)),
    )
    checks = _run_checks(case.expect, response, workspace, metrics, error)
    return EvalResult(
        name=case.name,
        repetition=repetition,
        passed=error is None and all(check["passed"] for check in checks),
        checks=checks,
        metrics=metrics,
        response=response,
        error=error,
        workspace=str(workspace),
    )


def _metrics(events: list[dict], duration: float, prompt: int, completion: int, cost) -> dict:
    requested = [event for event in events if event["event"] == "tool.requested"]
    completed = [event for event in events if event["event"] == "tool.completed"]
    models = [
        event.get("model", "")
        for event in events
        if event["event"] == "llm.request.completed"
    ]
    return {
        "duration_seconds": round(duration, 6),
        "llm_rounds": sum(event["event"] == "llm.request.completed" for event in events),
        "models_used": list(dict.fromkeys(model for model in models if model)),
        "tool_calls": len(requested),
        "tool_names": [event.get("tool_name", "") for event in requested],
        "tool_errors": sum(event.get("outcome") == "error" for event in completed),
        "tool_blocks": sum(event["event"] == "tool.blocked" for event in events)
        + sum(
            event["event"] == "tool.permission_decided" and not event.get("allowed", True)
            for event in events
        ),
        "retries": sum(event["event"] == "llm.retry" for event in events),
        "fallbacks": sum(event["event"] == "llm.fallback" for event in events),
        "context_compressions": sum(event["event"] == "context.compressed" for event in events),
        "subagents": sum(event["event"] == "subagent.started" for event in events),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "estimated_cost_usd": cost,
    }


def _run_checks(expect: dict, response: str, workspace: Path, metrics: dict, error: str | None) -> list[dict]:
    checks = [{"check": "agent_completed", "passed": error is None, "actual": error}]
    for value in _expected_values(expect, "response_contains"):
        checks.append({
            "check": f"response_contains:{value}",
            "passed": str(value) in response,
        })
    for pattern in _expected_values(expect, "response_regex"):
        try:
            matched = re.search(str(pattern), response) is not None
        except re.error:
            matched = False
        checks.append({
            "check": f"response_regex:{pattern}",
            "passed": matched,
        })
    for relative in _expected_values(expect, "files_exist"):
        path = _case_path(workspace, relative)
        checks.append({"check": f"file_exists:{relative}", "passed": bool(path and path.is_file())})
    for relative in _expected_values(expect, "files_not_exist"):
        path = _case_path(workspace, relative)
        checks.append({"check": f"file_not_exists:{relative}", "passed": bool(path and not path.exists())})
    files_contain = expect.get("files_contain", {})
    if not isinstance(files_contain, dict):
        checks.append({"check": "files_contain:is_object", "passed": False})
        files_contain = {}
    for relative, value in files_contain.items():
        path = _case_path(workspace, relative)
        actual = path.read_text(encoding="utf-8", errors="replace") if path and path.is_file() else ""
        checks.append({
            "check": f"file_contains:{relative}:{value}",
            "passed": str(value) in actual,
        })
    for tool_name in _expected_values(expect, "tool_calls_include"):
        checks.append({
            "check": f"tool_called:{tool_name}",
            "passed": tool_name in metrics["tool_names"],
        })
    for key in (
        "max_tool_calls", "max_llm_rounds", "max_duration_seconds",
        "max_prompt_tokens", "max_completion_tokens", "max_cost_usd",
    ):
        if key not in expect:
            continue
        metric = {
            "max_tool_calls": "tool_calls",
            "max_llm_rounds": "llm_rounds",
            "max_duration_seconds": "duration_seconds",
            "max_prompt_tokens": "prompt_tokens",
            "max_completion_tokens": "completion_tokens",
            "max_cost_usd": "estimated_cost_usd",
        }[key]
        actual = metrics[metric]
        try:
            passed = actual is not None and actual <= expect[key]
        except TypeError:
            passed = False
        checks.append({
            "check": key,
            "passed": passed,
            "actual": actual,
            "expected_max": expect[key],
        })
    return checks


def _expected_values(expect: dict, key: str) -> list:
    value = expect.get(key, [])
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _case_path(workspace: Path, relative) -> Path | None:
    try:
        path = (workspace / str(relative)).resolve()
        path.relative_to(workspace.resolve())
        return path
    except (ValueError, OSError):
        return None


def _copy_workspace(source: Path, destination: Path):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
            ".ruff_cache", "node_modules", "dist", "build",
        ),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "case"


def _case_slug(name: str, repetition: int) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(name)}-{digest}-{repetition}"


def _cost(llm):
    try:
        return llm.estimated_cost
    except (AttributeError, TypeError, ValueError):
        return None


def _cost_delta(before, after):
    if before is None or after is None:
        return None
    return max(0.0, after - before)


def _sum_known_costs(results: list[EvalResult]):
    costs = [result.metrics["estimated_cost_usd"] for result in results]
    return round(sum(costs), 10) if all(cost is not None for cost in costs) else None


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _parse_args():
    parser = argparse.ArgumentParser(prog="corecoder-eval", description="Run repeatable CoreCoder eval cases")
    parser.add_argument("manifest", help="JSON manifest containing eval cases")
    parser.add_argument("--output", help="Write the JSON report to this path")
    parser.add_argument("--baseline", help="Compare summary metrics with an earlier JSON report")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case N times")
    parser.add_argument("--yes", action="store_true", help="Allow mutating tools inside isolated eval workspaces")
    parser.add_argument("--in-place", action="store_true", help="Run against source workspaces without copying")
    parser.add_argument("--keep-workspaces", metavar="DIR", help="Keep per-run workspace copies for inspection")
    parser.add_argument("--trace-dir", metavar="DIR", help="Write one JSONL trace per case run")
    parser.add_argument("--trace-content", action="store_true", help="Include sensitive prompt/tool content in traces")
    parser.add_argument("--model")
    parser.add_argument(
        "--fallback-model",
        action="append",
        dest="fallback_models",
        help="Fallback model; repeat for an ordered chain",
    )
    parser.add_argument("--max-cost", type=_positive_float, dest="max_cost_usd")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--sandbox", choices=("local", "docker"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        cases = load_eval_cases(args.manifest)
        config = Config.from_env()
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
        if not config.api_key:
            raise ValueError("no API key configured")

        def factory(case: EvalCase, workspace: Path, trace: TraceSink) -> Agent:
            llm_cls = LiteLLM if config.provider == "litellm" else LLM
            llm = llm_cls(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                fallback_models=config.fallback_models,
                max_cost_usd=config.max_cost_usd,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            executor = create_command_executor(
                config.sandbox,
                workspace,
                image=config.sandbox_image,
                network=config.sandbox_network,
                memory=config.sandbox_memory,
                cpus=config.sandbox_cpus,
                pids_limit=config.sandbox_pids,
            )
            return Agent(
                llm=llm,
                tools=build_tools(
                    executor=executor,
                    path_policy=WorkspacePathPolicy(workspace),
                ),
                max_context_tokens=config.max_context_tokens,
                permission=Permission(allow_all=args.yes),
                workspace=workspace,
                trace=trace,
            )

        report = run_eval_suite(
            cases,
            factory,
            repetitions=args.repeat,
            in_place=args.in_place,
            keep_workspaces=args.keep_workspaces,
            trace_dir=args.trace_dir,
            trace_content=args.trace_content,
        )
        report["config"] = {
            "corecoder_version": __version__,
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "manifest_sha256": hashlib.sha256(
                Path(args.manifest).expanduser().resolve().read_bytes()
            ).hexdigest(),
            "model": config.model,
            "fallback_models": config.fallback_models,
            "provider": config.provider,
            "base_url": config.base_url,
            "sandbox": config.sandbox,
            "repetitions": args.repeat,
            "allow_mutations": args.yes,
            "in_place": args.in_place,
        }
        if args.baseline:
            baseline = json.loads(
                Path(args.baseline).expanduser().resolve().read_text(encoding="utf-8")
            )
            report["comparison"] = compare_reports(report, baseline)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
        print(f"corecoder-eval: {e}")
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"{summary['passed_runs']}/{summary['runs']} runs passed "
        f"({summary['success_rate']:.1%}), {summary['total_tool_calls']} tool calls, "
        f"{summary['duration_seconds']:.3f}s"
    )
    for result in report["results"]:
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['name']} #{result['repetition']}")
    return 0 if summary["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
