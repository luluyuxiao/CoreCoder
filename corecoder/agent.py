"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import copy
import inspect
import logging
import threading
import time
import uuid
from pathlib import Path

from .capabilities import CapabilityPolicy
from .context import ContextManager, estimate_request_tokens, estimate_tokens
from .llm import LLM
from .memory import MemoryState
from .permissions import Permission
from .prompt import PLAN_MODE_PROMPT, system_prompt
from .tools import build_tools
from .tools.agent import AgentStatusTool, AgentTool
from .tools.base import Tool
from .tools.memory import MemoryUpdateTool
from .tools.skill import LoadSkillTool
from .tools.todo import TodoWriteTool
from .trace import NULL_TRACE, TraceSink

log = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        permission=None,
        hooks=None,
        workspace: str | Path | None = None,
        llm_lock=None,
        trace: TraceSink | None = None,
        parent_agent_id: str | None = None,
        agent_id: str | None = None,
        subagent_task_id: str | None = None,
        state_callback=None,
        capability_policy: CapabilityPolicy | None = None,
        memory_state: MemoryState | None = None,
    ):
        self.llm = llm
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self.trace = trace or NULL_TRACE
        self.agent_id = agent_id or uuid.uuid4().hex[:12]
        self.parent_agent_id = parent_agent_id
        self.subagent_task_id = subagent_task_id
        # Optional storage-neutral callback. The Agent emits only complete,
        # provider-valid snapshots; the CLI decides whether SQLite, JSON, or
        # another backend receives them.
        self._state_callback = state_callback
        # Transcript events are independent from ``messages``.  Context
        # management may destructively summarize ``messages`` for the next
        # model request, while these pending events preserve the original
        # user/assistant/tool payloads until durable storage acknowledges them.
        self._pending_transcript_events: list[dict] = []
        self._pending_context_summaries: list[dict] = []
        self._active_run_id: str | None = None
        self._active_round: int | None = None
        self.memory = (
            MemoryState.from_dict(memory_state.to_dict())
            if memory_state is not None else MemoryState()
        )
        # Sub-agents share one LLM and its usage/budget counters. Serializing
        # calls keeps that mutable accounting correct while tool phases can
        # still overlap in background mode.
        self._llm_lock = llm_lock or threading.RLock()
        self.tools = tools if tools is not None else build_tools()
        self.permission = permission
        self.hooks = hooks
        self.capability_policy = capability_policy or CapabilityPolicy()
        self._tool_by_name = {t.name: t for t in self.tools}
        self._skill_loader = next(
            (t for t in self.tools if isinstance(t, LoadSkillTool)), None
        )
        # Skill runtime state is separate from conversation messages so context
        # compression cannot silently deactivate a loaded workflow. Only a
        # name/hash/state record is persisted; instructions are re-read from
        # the validated registry on every model request.
        self.active_skills: dict[str, dict] = {}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        # A tool observation is protected from ordinary history snipping until
        # one successful LLM request has actually consumed it.
        self._unconsumed_tool_call_ids: set[str] = set()
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools, cwd=self.workspace)
        self.plan_mode = False  # toggled by /plan; while on, mutating tools are refused
        try:
            llm_signature = inspect.signature(self.llm.chat)
            self._llm_accepts_events = (
                "on_event" in llm_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in llm_signature.parameters.values()
                )
            )
        except (TypeError, ValueError):
            self._llm_accepts_events = False

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, (AgentTool, AgentStatusTool)):
                t._parent_agent = self
            if isinstance(t, MemoryUpdateTool):
                t.bind(self.memory)
            if not t.is_concurrency_safe():
                t.execution_lock()  # create before any background worker can share it

        self._todo = next((t for t in self.tools if isinstance(t, TodoWriteTool)), None)

    def _full_messages(self) -> list[dict]:
        system = self._system
        # re-injected every round, like the task list below, so a toggle made
        # between turns takes effect on the very next request
        if self.plan_mode:
            system += "\n\n" + PLAN_MODE_PROMPT
        structured_memory = self.memory.render()
        if structured_memory:
            system += "\n\n" + structured_memory
        # the task list is re-injected every round, so the model always sees the
        # current state rather than a stale copy buried in old tool results
        if self._todo is not None:
            rendered = self._todo.render()
            if rendered:
                system += "\n\n# Current task list\n" + rendered
        skill_prompt = self._active_skill_prompt()
        if skill_prompt:
            system += "\n\n" + skill_prompt
        return [{"role": "system", "content": system}] + self.messages

    def _reconcile_active_skills(self):
        """Fail closed when a saved Skill disappeared or changed on disk."""
        if self._skill_loader is None:
            for name, record in self.active_skills.items():
                record.update({"name": name, "status": "unavailable"})
            return
        self.active_skills = {
            name: self._skill_loader.reconcile_activation(name, record)
            for name, record in self.active_skills.items()
        }

    def _active_skill_prompt(self) -> str:
        self._reconcile_active_skills()
        if not self.active_skills:
            return ""
        if self._skill_loader is None:
            names = ", ".join(sorted(self.active_skills))
            return (
                "# Saved Skill state notices\n\n"
                "Saved Skills are unavailable because this Agent has no load_skill "
                f"tool: {names}."
            )
        return self._skill_loader.render_active(self.active_skills)

    def active_skill_states(self) -> dict[str, dict]:
        """Return validated, JSON-safe Skill state for UI and library callers."""
        self._reconcile_active_skills()
        return copy.deepcopy(self.active_skills)

    def _activate_skill_from_result(self, tc, result: str):
        tool = self._tool_by_name.get(tc.name)
        if not isinstance(tool, LoadSkillTool):
            return
        name = tc.arguments.get("name")
        if not isinstance(name, str):
            return
        record = tool.activation_record(name, result)
        if record is None:
            return
        replaced = name in self.active_skills
        self.active_skills[name] = record
        self._trace(
            "skill.activated",
            skill_name=name,
            skill_scope=record["scope"],
            content_hash=record["content_hash"],
            replaced=replaced,
        )

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(
        self,
        user_input: str,
        on_token=None,
        on_tool=None,
        on_tool_progress=None,
    ) -> str:
        """Process one user message and trace its complete lifecycle.

        ``on_tool`` is the original lightweight notification fired when the
        model requests a call. ``on_tool_progress`` receives structured batch
        and lifecycle events suitable for a progress UI. Callback failures are
        deliberately isolated from the agent loop.
        """
        run_id = uuid.uuid4().hex
        previous_run_id = self._active_run_id
        previous_round = self._active_round
        self._active_run_id = run_id
        self._active_round = None
        started = time.perf_counter()
        self._trace(
            "agent.run.started",
            model=getattr(self.llm, "model", ""),
            workspace=self.workspace,
            plan_mode=self.plan_mode,
            user_input_chars=len(user_input),
            **self._trace_content(user_input=user_input),
        )
        try:
            result = self._chat_impl(user_input, on_token, on_tool, on_tool_progress)
        except BaseException as e:
            self._trace(
                "agent.run.failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                error_type=type(e).__name__,
            )
            self.persist_state(
                "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
            )
            self._active_run_id = previous_run_id
            self._active_round = previous_round
            raise
        self._trace(
            "agent.run.completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            result_chars=len(result),
            outcome="max_rounds" if result == "(reached maximum tool-call rounds)" else "success",
            **self._trace_content(result=result),
        )
        self.persist_state(
            "max_rounds" if result == "(reached maximum tool-call rounds)" else "completed"
        )
        self._active_run_id = previous_run_id
        self._active_round = previous_round
        return result

    def _chat_impl(
        self,
        user_input: str,
        on_token=None,
        on_tool=None,
        on_tool_progress=None,
    ) -> str:
        self._append_message({"role": "user", "content": user_input})
        self.persist_state("running")

        for round_index in range(self.max_rounds):
            round_number = round_index + 1
            self._active_round = round_number
            # Recompute the complete request budget every round: plan mode,
            # todos, schemas, and fallback output limits can all be dynamic.
            self.compress_context(trigger="before_llm")
            llm_started = time.perf_counter()
            full_messages = self._full_messages()
            schemas = self._tool_schemas()
            estimated_input = estimate_request_tokens(full_messages, schemas)
            output_reserve = self._output_token_reserve()
            self._trace(
                "llm.request.started",
                round=round_number,
                model=getattr(self.llm, "model", ""),
                message_count=len(full_messages),
                estimated_context_tokens=estimated_input,
                output_token_reserve=output_reserve,
                safety_margin_tokens=self.context.safety_margin_tokens,
                estimated_reserved_tokens=(
                    estimated_input
                    + output_reserve
                    + self.context.safety_margin_tokens
                ),
                tool_schema_count=len(schemas),
            )
            try:
                with self._llm_lock:
                    kwargs = {
                        "messages": full_messages,
                        "tools": schemas,
                        "on_token": on_token,
                    }
                    if self._llm_accepts_events:
                        kwargs["on_event"] = lambda event, fields, rn=round_number: self._trace(
                            event, round=rn, **fields
                        )
                    resp = self.llm.chat(**kwargs)
            except Exception as e:
                self._trace(
                    "llm.request.failed",
                    round=round_number,
                    duration_ms=round((time.perf_counter() - llm_started) * 1000, 3),
                    error_type=type(e).__name__,
                    status_code=getattr(e, "status_code", None),
                )
                raise
            # Every previously fresh observation was present in the successful
            # request that produced this response, so it is now history.
            self._unconsumed_tool_call_ids.clear()
            self._trace(
                "llm.request.completed",
                round=round_number,
                duration_ms=round((time.perf_counter() - llm_started) * 1000, 3),
                model=getattr(resp, "model", "") or getattr(self.llm, "model", ""),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                response_chars=len(resp.content),
                tool_call_count=len(resp.tool_calls),
                tool_names=[tc.name for tc in resp.tool_calls],
                estimated_cost=self._estimated_cost(),
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self._append_message(resp.message)
                return resp.content

            # tool calls -> execute. Multiple calls are scheduled by effect:
            # reads may overlap, while writes/unknown effects are barriers.
            self._append_message(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    self._trace_tool_requested(tc)
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._gate_tool(tc)
                    self._start_tool_progress(
                        [tc], on_tool_progress, parallel=False
                    )
                    if result is None:
                        result = self._exec_tool(tc, on_tool_progress)
                        self._post_hooks(tc, result)
                    else:
                        self._finish_blocked_tool_progress(tc, result, on_tool_progress)
                    self._finish_tool_batch_progress(
                        [tc], on_tool_progress, parallel=False
                    )
                    self._activate_skill_from_result(tc, result)
                    self._append_tool_result(tc.id, result)
                else:
                    # effect-aware execution for multiple tool calls
                    results = self._exec_tools_parallel(
                        resp.tool_calls, on_tool, on_tool_progress
                    )
                    for tc, result in zip(resp.tool_calls, results):
                        self._activate_skill_from_result(tc, result)
                        self._append_tool_result(tc.id, result)
                # The assistant tool_calls and every corresponding observation
                # are committed as one stable snapshot. Never persist halfway
                # through a batch: resuming an orphan tool call is invalid.
                self.persist_state("running")
            except BaseException:
                # An interrupt or an unexpected gate/callback failure during
                # execution would otherwise leave assistant tool_calls without
                # replies. Repair first so the failed autosave is resumable.
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

        return "(reached maximum tool-call rounds)"

    def compress_context(self, trigger: str = "manual") -> tuple[bool, int, int]:
        """Compress against the complete request budget and trace the delta."""
        schemas = self._tool_schemas()
        system_message = self._full_messages()[0]
        fixed_tokens = estimate_request_tokens([system_message], schemas)
        output_reserve = self._output_token_reserve()
        safety_margin = self.context.safety_margin_tokens
        before = fixed_tokens + estimate_tokens(self.messages) + output_reserve + safety_margin
        before_messages = copy.deepcopy(self.messages) if self.trace.capture_content else None
        prompt_before = getattr(self.llm, "total_prompt_tokens", 0)
        completion_before = getattr(self.llm, "total_completion_tokens", 0)
        started = time.perf_counter()
        with self._llm_lock:
            compressed = self.context.maybe_compress(
                self.messages,
                self.llm,
                fixed_tokens=fixed_tokens,
                output_reserve=output_reserve,
                protected_tool_call_ids=self._unconsumed_tool_call_ids,
            )
        after = fixed_tokens + estimate_tokens(self.messages) + output_reserve + safety_margin
        if compressed:
            budget = self.context.last_budget
            summary_messages = [
                copy.deepcopy(message)
                for message in self.messages
                if str(message.get("content") or "").startswith((
                    "[Context compressed - conversation summary]",
                    "[Hard context reset]",
                    "[Emergency context reset",
                    "[Context reset]",
                ))
            ]
            self._pending_context_summaries.append({
                "summary_id": uuid.uuid4().hex,
                "trigger": trigger,
                "actions": list(self.context.last_actions),
                "before_tokens": before,
                "after_tokens": after,
                "model": getattr(self.llm, "model", ""),
                "prompt_tokens": (
                    getattr(self.llm, "total_prompt_tokens", 0) - prompt_before
                ),
                "completion_tokens": (
                    getattr(self.llm, "total_completion_tokens", 0) - completion_before
                ),
                "messages": summary_messages,
            })
            self._trace(
                "context.compressed",
                trigger=trigger,
                actions=list(self.context.last_actions),
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                before_tokens=before,
                after_tokens=after,
                reclaimed_tokens=max(0, before - after),
                message_count=len(self.messages),
                fixed_tokens=fixed_tokens,
                output_token_reserve=output_reserve,
                safety_margin_tokens=safety_margin,
                message_budget=budget.get("message_budget"),
                protected_tool_results=len(self._unconsumed_tool_call_ids),
                summary_prompt_tokens=(
                    getattr(self.llm, "total_prompt_tokens", 0) - prompt_before
                ),
                summary_completion_tokens=(
                    getattr(self.llm, "total_completion_tokens", 0) - completion_before
                ),
                **self._trace_content(
                    messages_before=before_messages,
                    messages_after=self.messages,
                ),
            )
        return compressed, before, after

    def _output_token_reserve(self) -> int:
        """Read the configured completion ceiling without provider coupling."""
        extra = getattr(self.llm, "extra", None)
        if not isinstance(extra, dict):
            return 0
        raw = extra.get("max_completion_tokens", extra.get("max_tokens", 0))
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            return 0

    def _append_tool_result(self, tool_call_id: str, result: str):
        self._append_message({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        })
        self._unconsumed_tool_call_ids.add(tool_call_id)

    def _append_message(self, message: dict):
        """Append to active context and preserve the original in Transcript."""
        active = copy.deepcopy(message)
        self.messages.append(active)
        role = str(active.get("role") or "unknown")
        if role == "assistant" and active.get("tool_calls"):
            event_type = "assistant.tool_calls"
        elif role == "tool":
            event_type = "tool.result"
        else:
            event_type = f"{role}.message"
        self._pending_transcript_events.append({
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "run_id": self._active_run_id,
            "round": self._active_round,
            "payload": copy.deepcopy(active),
        })

    def state_snapshot(self, status: str = "saved") -> dict:
        """Return a storage-neutral, JSON-serializable session snapshot."""
        metadata = {
            "agent_id": self.agent_id,
            "plan_mode": self.plan_mode,
            "unconsumed_tool_call_ids": sorted(self._unconsumed_tool_call_ids),
            "usage_by_model": copy.deepcopy(
                getattr(self.llm, "usage_by_model", {})
            ),
            "fallback_history": copy.deepcopy(
                getattr(self.llm, "fallback_history", [])
            ),
            "active_skills": self.active_skill_states(),
            "memory_state": self.memory.to_dict(),
        }
        if self._todo is not None:
            metadata["todo_tasks"] = self._todo.snapshot()
        return {
            "model": getattr(self.llm, "model", ""),
            "messages": copy.deepcopy(self.messages),
            "status": status,
            "workspace": str(self.workspace),
            "prompt_tokens": max(
                0, int(getattr(self.llm, "total_prompt_tokens", 0) or 0)
            ),
            "completion_tokens": max(
                0, int(getattr(self.llm, "total_completion_tokens", 0) or 0)
            ),
            "estimated_cost": self._estimated_cost(),
            "metadata": metadata,
            "transcript_complete": True,
            "transcript_events": copy.deepcopy(self._pending_transcript_events),
            "context_summaries": copy.deepcopy(self._pending_context_summaries),
        }

    def restore_state(self, snapshot: dict, *, restore_model: bool = True):
        """Restore one previously stable snapshot into this Agent instance."""
        self.messages = copy.deepcopy(list(snapshot.get("messages") or []))
        if restore_model and snapshot.get("model"):
            self.llm.model = str(snapshot["model"])
        self.llm.total_prompt_tokens = max(
            0, int(snapshot.get("prompt_tokens") or 0)
        )
        self.llm.total_completion_tokens = max(
            0, int(snapshot.get("completion_tokens") or 0)
        )
        metadata = dict(snapshot.get("metadata") or {})
        if hasattr(self.llm, "usage_by_model"):
            self.llm.usage_by_model = copy.deepcopy(metadata.get("usage_by_model") or {})
        if hasattr(self.llm, "fallback_history"):
            self.llm.fallback_history = copy.deepcopy(
                metadata.get("fallback_history") or []
            )
        self.plan_mode = bool(metadata.get("plan_mode", False))
        self.memory = MemoryState.from_dict(metadata.get("memory_state"))
        for tool in self.tools:
            if isinstance(tool, MemoryUpdateTool):
                tool.bind(self.memory)
        raw_skills = metadata.get("active_skills") or {}
        self.active_skills = (
            {
                str(name): copy.deepcopy(record)
                for name, record in raw_skills.items()
                if isinstance(name, str) and isinstance(record, dict)
            }
            if isinstance(raw_skills, dict)
            else {}
        )
        self._reconcile_active_skills()
        self._unconsumed_tool_call_ids = set(
            metadata.get("unconsumed_tool_call_ids") or []
        )
        if self._todo is not None:
            tasks = metadata.get("todo_tasks")
            if isinstance(tasks, list):
                self._todo.restore(tasks)

    def persist_state(self, status: str = "saved") -> bool:
        """Best-effort persistence hook; storage failure never kills the run."""
        if self._state_callback is None:
            return False
        snapshot = self.state_snapshot(status)
        try:
            self._state_callback(snapshot)
            self.acknowledge_persisted(snapshot)
            self._trace(
                "session.state.saved",
                status=status,
                message_count=len(self.messages),
            )
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("session state callback failed: %s", e)
            self._trace(
                "session.state.failed",
                status=status,
                error_type=type(e).__name__,
            )
            return False

    def acknowledge_persisted(self, snapshot: dict):
        """Forget only the pending records included in a successful save.

        Event IDs make storage writes idempotent; filtering by ID also avoids
        discarding records that may have arrived after the snapshot was built.
        """
        event_ids = {
            event.get("event_id")
            for event in snapshot.get("transcript_events") or []
        }
        summary_ids = {
            item.get("summary_id")
            for item in snapshot.get("context_summaries") or []
        }
        self._pending_transcript_events = [
            event for event in self._pending_transcript_events
            if event.get("event_id") not in event_ids
        ]
        self._pending_context_summaries = [
            item for item in self._pending_context_summaries
            if item.get("summary_id") not in summary_ids
        ]

    def _trace(self, event: str, **fields):
        try:
            if self._active_round is not None and "round" not in fields:
                fields["round"] = self._active_round
            self.trace.emit(
                event,
                agent_id=self.agent_id,
                parent_agent_id=self.parent_agent_id,
                subagent_task_id=self.subagent_task_id,
                run_id=self._active_run_id,
                **fields,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("trace sink failed for %s: %s", event, e)

    def _trace_content(self, **fields) -> dict:
        """Read the opt-in content policy without letting a custom sink fail work."""
        try:
            return self.trace.content(**fields)
        except Exception as e:  # noqa: BLE001
            log.warning("trace content policy failed: %s", e)
            return {}

    def _estimated_cost(self):
        try:
            return self.llm.estimated_cost
        except (AttributeError, TypeError, ValueError):
            return None

    def _trace_tool_requested(self, tc):
        tool = self._tool_by_name.get(tc.name)
        self._trace(
            "tool.requested",
            tool_call_id=tc.id,
            tool_name=tc.name,
            argument_keys=sorted(tc.arguments),
            capabilities=(
                sorted(tool.required_capabilities(tc.arguments)) if tool else []
            ),
            **self._trace_content(arguments=tc.arguments),
        )

    def _emit_tool_progress(self, callback, event: str, **fields):
        """Deliver one best-effort progress event without affecting execution."""
        if callback is None:
            return
        payload = {"round": self._active_round, **fields}
        try:
            callback(event, payload)
        except Exception as e:  # noqa: BLE001
            log.warning("tool progress callback failed for %s: %s", event, e)

    def _tool_progress_item(self, tc) -> dict:
        tool = self._tool_by_name.get(tc.name)
        return {
            "tool_call_id": tc.id,
            "tool_name": tc.name,
            "arguments": tc.arguments,
            "effect": getattr(tool, "effect", "unknown"),
            "capabilities": (
                sorted(tool.required_capabilities(tc.arguments)) if tool else []
            ),
        }

    def _has_parallel_work(self, tool_calls, gated_results) -> bool:
        """Return whether scheduling will contain a concurrent safe batch."""
        pending = [
            i for i, result in enumerate(gated_results) if result is None
        ]
        cursor = 0
        while cursor < len(pending):
            index = pending[cursor]
            tool = self._tool_by_name.get(tool_calls[index].name)
            if tool is None or not tool.is_concurrency_safe():
                cursor += 1
                continue
            batch_size = 1
            cursor += 1
            while cursor < len(pending):
                candidate = pending[cursor]
                candidate_tool = self._tool_by_name.get(tool_calls[candidate].name)
                if candidate_tool is None or not candidate_tool.is_concurrency_safe():
                    break
                batch_size += 1
                cursor += 1
            if batch_size > 1:
                return True
        return False

    def _start_tool_progress(
        self,
        tool_calls,
        callback,
        *,
        parallel: bool,
    ):
        """Start UI progress only after hooks and permission prompts finish."""
        self._emit_tool_progress(
            callback,
            "batch_started",
            total=len(tool_calls),
            parallel=parallel,
            tools=[self._tool_progress_item(tc) for tc in tool_calls],
        )

    def _finish_blocked_tool_progress(self, tc, result: str, callback):
        self._emit_tool_progress(
            callback,
            "tool_completed",
            tool_call_id=tc.id,
            tool_name=tc.name,
            outcome="blocked",
            duration_ms=0.0,
            result_chars=len(result),
        )

    def _finish_tool_batch_progress(self, tool_calls, callback, *, parallel: bool):
        self._emit_tool_progress(
            callback,
            "batch_completed",
            total=len(tool_calls),
            parallel=parallel,
        )

    def _gate_tool(self, tc) -> str | None:
        hook_started = time.perf_counter()
        result = self._pre_hooks(tc)
        self._trace(
            "hook.pre.completed",
            tool_call_id=tc.id,
            tool_name=tc.name,
            configured=self.hooks is not None,
            blocked=result is not None,
            duration_ms=round((time.perf_counter() - hook_started) * 1000, 3),
        )
        if result is not None:
            self._trace(
                "tool.blocked",
                tool_call_id=tc.id,
                tool_name=tc.name,
                gate="hook",
                **self._trace_content(result=result),
            )
            return result
        capability_started = time.perf_counter()
        decision, result = self._capability_decision(tc)
        self._trace(
            "tool.capability_decided",
            tool_call_id=tc.id,
            tool_name=tc.name,
            decision=decision,
            allowed=result is None,
            duration_ms=round((time.perf_counter() - capability_started) * 1000, 3),
            **(self._trace_content(result=result) if result is not None else {}),
        )
        if result is not None:
            self._trace(
                "tool.blocked",
                tool_call_id=tc.id,
                tool_name=tc.name,
                gate="capability",
                **self._trace_content(result=result),
            )
            return result
        permission_started = time.perf_counter()
        decision, result = self._permission_decision(tc)
        content_fields = self._trace_content(result=result) if result is not None else {}
        self._trace(
            "tool.permission_decided",
            tool_call_id=tc.id,
            tool_name=tc.name,
            decision=decision,
            allowed=result is None,
            duration_ms=round((time.perf_counter() - permission_started) * 1000, 3),
            **content_fields,
        )
        return result

    def _capability_decision(self, tc) -> tuple[str, str | None]:
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return "unknown_tool", None
        return self.capability_policy.decide(tool, tc.arguments)

    def _permission_decision(self, tc) -> tuple[str, str | None]:
        if self.plan_mode and tc.name not in Permission.READ_ONLY:
            return "plan_deny", self._permit(tc)
        if self.permission is None:
            return "no_permission_layer", None
        decide = getattr(self.permission, "decide", None)
        if callable(decide):
            return decide(tc.name, tc.arguments)
        result = self.permission.check(tc.name, tc.arguments)
        return ("custom_deny" if result is not None else "custom_allow"), result

    def _pre_hooks(self, tc) -> str | None:
        """PreToolUse hooks, fired before consent. A string return blocks the
        call and becomes the tool result the model sees; None lets it through."""
        if self.hooks is None:
            return None
        return self.hooks.run_pre(tc.name, tc.arguments)

    def _post_hooks(self, tc, result: str):
        """PostToolUse hooks observe a finished call; they can never block."""
        started = time.perf_counter()
        if self.hooks is not None:
            self.hooks.run_post(tc.name, tc.arguments, result)
        self._trace(
            "hook.post.completed",
            tool_call_id=tc.id,
            tool_name=tc.name,
            configured=self.hooks is not None,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _permit(self, tc) -> str | None:
        """Consent check for one call. None means go ahead; a string is the
        refusal, returned as the tool result instead of executing."""
        # plan mode outranks consent, even --yes: while it's on nothing mutates
        if self.plan_mode and tc.name not in Permission.READ_ONLY:
            return (
                "Plan mode is on, so this call was refused: plan mode is "
                "read-only. Do not retry it. Keep investigating with the "
                "read-only tools, then present the plan and stop. The user can "
                'approve it by typing "approve", or exit plan mode with /plan.'
            )
        if self.permission is None:
            return None
        return self.permission.check(tc.name, tc.arguments)

    def _exec_tool(self, tc, on_tool_progress=None) -> str:
        """Execute a single tool call, returning the result string."""
        started = time.perf_counter()
        self._trace(
            "tool.started",
            tool_call_id=tc.id,
            tool_name=tc.name,
        )
        self._emit_tool_progress(
            on_tool_progress,
            "tool_started",
            tool_call_id=tc.id,
            tool_name=tc.name,
        )

        def finish(result: str) -> str:
            lowered = result.lstrip().lower()
            outcome = "error" if lowered.startswith((
                "error", "permission denied", "sandbox violation", "⚠ blocked",
            )) else "success"
            if outcome == "success":
                self._record_memory_effect(tc)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self._trace(
                "tool.completed",
                tool_call_id=tc.id,
                tool_name=tc.name,
                duration_ms=duration_ms,
                outcome=outcome,
                result_chars=len(result),
                **self._trace_content(result=result),
            )
            self._emit_tool_progress(
                on_tool_progress,
                "tool_completed",
                tool_call_id=tc.id,
                tool_name=tc.name,
                duration_ms=duration_ms,
                outcome=outcome,
                result_chars=len(result),
            )
            return result

        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return finish(f"Error: unknown tool '{tc.name}'")
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return finish(f"Error: bad arguments for {tc.name}: {e}")
        # a tool that blows up gets reported back as text, never kills the loop
        try:
            # Separate Agents may share MCP/custom Tool instances. Unsafe
            # effects use an instance lock so background sub-agents cannot race
            # the parent through that second concurrency entrance.
            if tool.is_concurrency_safe():
                result = tool.execute(**tc.arguments)
            else:
                with tool.execution_lock():
                    result = tool.execute(**tc.arguments)
            return finish(result)
        except Exception as e:  # noqa: BLE001
            return finish(f"Error executing {tc.name}: {e}")

    def _record_memory_effect(self, tc):
        """Derive modified files from successful built-in write calls."""
        if tc.name not in {"write_file", "edit_file"}:
            return
        raw_path = tc.arguments.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        resolved = path.resolve(strict=False)
        try:
            rendered = str(resolved.relative_to(self.workspace))
        except ValueError:
            rendered = str(resolved)
        self.memory.record_file(rendered)

    def _exec_tools_parallel(
        self,
        tool_calls,
        on_tool=None,
        on_tool_progress=None,
    ) -> list[str]:
        """Run safe read batches concurrently; serialize all other effects.

        Order is a correctness boundary. Consecutive ``pure``/``read`` calls
        may share a thread pool, but a ``write``, ``external`` or ``unknown``
        call completes (including its post hooks) before later calls begin.
        """
        for tc in tool_calls:
            self._trace_tool_requested(tc)
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # hooks and consent are settled up front on this thread: prompting
        # from pool workers would interleave several prompts on one terminal
        results = [self._gate_tool(tc) for tc in tool_calls]
        parallel = self._has_parallel_work(tool_calls, results)
        self._start_tool_progress(
            tool_calls, on_tool_progress, parallel=parallel
        )
        for tc, result in zip(tool_calls, results):
            if result is not None:
                self._finish_blocked_tool_progress(tc, result, on_tool_progress)
        pending = [i for i, result in enumerate(results) if result is None]
        cursor = 0
        while cursor < len(pending):
            index = pending[cursor]
            tool = self._tool_by_name.get(tool_calls[index].name)

            # Unknown tools and any tool not explicitly marked safe are an
            # exclusive barrier. _exec_tool still owns the unknown-tool error.
            if tool is None or not tool.is_concurrency_safe():
                results[index] = self._exec_tool(
                    tool_calls[index], on_tool_progress
                )
                self._post_hooks(tool_calls[index], results[index])
                cursor += 1
                continue

            # Form one maximal, consecutive batch of explicitly safe calls.
            batch = [index]
            cursor += 1
            while cursor < len(pending):
                candidate = pending[cursor]
                candidate_tool = self._tool_by_name.get(tool_calls[candidate].name)
                if candidate_tool is None or not candidate_tool.is_concurrency_safe():
                    break
                batch.append(candidate)
                cursor += 1

            if len(batch) == 1:
                results[index] = self._exec_tool(
                    tool_calls[index], on_tool_progress
                )
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(8, len(batch))
                ) as pool:
                    futures = {
                        i: pool.submit(
                            self._exec_tool, tool_calls[i], on_tool_progress
                        )
                        for i in batch
                    }
                    # Assign by original index so conversation order stays stable
                    # even when workers finish in a different order.
                    for i in batch:
                        results[i] = futures[i].result()
            for i in batch:
                self._post_hooks(tool_calls[i], results[i])
        self._finish_tool_batch_progress(
            tool_calls, on_tool_progress, parallel=parallel
        )
        return results

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self._append_tool_result(tc.id, "[interrupted]")

    def reset(self):
        """Clear conversation history."""
        self._pending_transcript_events.append({
            "event_id": uuid.uuid4().hex,
            "event_type": "session.reset",
            "run_id": self._active_run_id,
            "round": self._active_round,
            "payload": {},
        })
        self.messages.clear()
        self._unconsumed_tool_call_ids.clear()
        self.active_skills.clear()
        self.memory.clear()
        self.persist_state("reset")
