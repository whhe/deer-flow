"""Middleware for task-level tool call progress tracking with a state machine.

Implements RFC #3177: structured tool result signals drive a per-(thread, tool)
state machine that detects stagnation and repetition, injects hints early
(WARNED), and hard-blocks the tool when it has stopped producing value (BLOCKED).

Architecture:
  ToolProgressMiddleware (outer)
    └── handler → ToolErrorHandlingMiddleware (inner) → actual tool
                                                              ↓
  ToolProgressMiddleware reads deerflow_tool_meta from the normalized result

State machine transitions per (thread_id, tool_name):
  ACTIVE → WARNED (at stagnation_threshold problems) → BLOCKED (after warn_escalation_count more)
  Any problem-free call resets consecutive_problems=0 and reverts to ACTIVE.
  Auth/config errors are immediately BLOCKED (not recoverable by model).
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from deerflow.agents.middlewares.tool_result_meta import TOOL_META_KEY, ToolResultMeta

if TYPE_CHECKING:
    from deerflow.config.tool_progress_config import ToolProgressConfig

logger = logging.getLogger(__name__)

_MAX_PENDING_PER_RUN = 3


# ---------------------------------------------------------------------------
# State data structures


@dataclass(slots=True)
class ToolPhaseState:
    """Per (thread_id, tool_name) tracking state."""

    phase: Literal["active", "warned", "blocked"] = "active"
    consecutive_problems: int = 0
    block_reason: str | None = None
    recent_word_sets: list[frozenset[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Content helpers


def word_set(content: str) -> frozenset[str]:
    """Extract lowercase words of length >= 3 for Jaccard similarity."""
    return frozenset(re.findall(r"\b\w{3,}\b", content.lower()))


def is_near_duplicate(
    current: frozenset[str],
    recent: list[frozenset[str]],
    threshold: float,
    min_words: int,
) -> bool:
    """Return True if current is similar to any of the last 3 recent word sets."""
    if len(current) < min_words:
        return False
    for prev in recent[-3:]:
        if len(prev) < min_words:
            continue
        union = len(current | prev)
        if union == 0:
            continue
        if len(current & prev) / union >= threshold:
            return True
    return False


def _message_content_str(msg: ToolMessage) -> str:
    return msg.content if isinstance(msg.content, str) else ""


def _parse_tool_meta(meta_dict: object) -> ToolResultMeta | None:
    """Safely deserialize a ToolResultMeta from a raw dict; returns None on schema mismatch."""
    if not isinstance(meta_dict, dict):
        return None
    try:
        return ToolResultMeta(**meta_dict)
    except TypeError:
        logger.warning("Unexpected tool meta schema, skipping progress tracking: %s", meta_dict)
        return None


# ---------------------------------------------------------------------------
# Hint / block reason formatting


def _format_hint(meta: ToolResultMeta) -> str:
    action_map = {
        "rewrite_query": "Try rephrasing your search query with different keywords or approach.",
        "try_alternative": "Consider using a different tool or strategy.",
        "summarize": "Consider summarizing your current findings and moving forward.",
        "stop": "Do not retry this operation — it is not recoverable.",
    }
    base = {
        "no_results": "[PROGRESS HINT] Your search returned no results.",
        "not_found": "[PROGRESS HINT] The resource was not found repeatedly.",
        "rate_limited": "[PROGRESS HINT] The tool is being rate-limited.",
        "transient": "[PROGRESS HINT] The tool encountered repeated transient failures.",
        "partial_success": "[PROGRESS HINT] The tool has returned incomplete results multiple times.",
    }.get(
        meta.error_type or meta.status,
        "[PROGRESS HINT] The tool is not producing new information.",
    )
    suffix = action_map.get(meta.recommended_next_action, "")
    return f"{base} {suffix}".strip()


def _block_reason(meta: ToolResultMeta) -> str:
    return {
        "no_results": "Repeated no-results — rewrite your query or try a different tool.",
        "not_found": "Repeated not-found — rewrite your query or try a different resource.",
        "rate_limited": "Repeated rate-limiting — summarize current findings and proceed.",
        "transient": "Repeated transient failures — try a different approach.",
        "auth": "Authentication failure — this tool cannot be used.",
        "config": "Tool is not configured — this tool cannot be used.",
        "internal": "Repeated internal errors — this tool is unavailable.",
    }.get(
        meta.error_type or "",
        "Tool has not produced new information after multiple attempts — summarize and move on.",
    )


# ---------------------------------------------------------------------------
# Middleware


class ToolProgressMiddleware(AgentMiddleware[AgentState]):
    """State-machine-based tool stagnation guard (RFC #3177)."""

    def __init__(
        self,
        *,
        stagnation_threshold: int = 3,
        warn_escalation_count: int = 2,
        inject_assessment: bool = True,
        jaccard_threshold: float = 0.8,
        min_words: int = 10,
        exempt_tools: set[str] | None = None,
        max_tracked_threads: int = 100,
    ) -> None:
        self._stagnation_threshold = stagnation_threshold
        self._warn_escalation = warn_escalation_count
        self._inject_assessment = inject_assessment
        self._jaccard_threshold = jaccard_threshold
        self._min_words = min_words
        self._exempt_tools: set[str] = exempt_tools or {"ask_clarification", "write_todos", "present_files"}
        self._max_tracked_threads = max_tracked_threads

        self._lock = threading.Lock()
        # LRU-evicting store: thread_id → {tool_name → ToolPhaseState}
        self._phase_states: OrderedDict[str, dict[str, ToolPhaseState]] = OrderedDict()
        # Pending hint queue: (thread_id, run_id) → [hint texts]
        self._pending: dict[tuple[str, str], list[str]] = defaultdict(list)

    @classmethod
    def from_config(cls, config: ToolProgressConfig) -> ToolProgressMiddleware:
        return cls(
            stagnation_threshold=config.stagnation_threshold,
            warn_escalation_count=config.warn_escalation_count,
            inject_assessment=config.inject_assessment,
            jaccard_threshold=config.jaccard_similarity_threshold,
            min_words=config.min_word_count_for_similarity,
            exempt_tools=set(config.exempt_tools),
            max_tracked_threads=config.max_tracked_threads,
        )

    # ------------------------------------------------------------------
    # Runtime helpers

    @staticmethod
    def _thread_id(runtime: Runtime) -> str:
        tid = runtime.context.get("thread_id") if runtime.context else None
        return str(tid) if tid else "default"

    @staticmethod
    def _run_id(runtime: Runtime) -> str:
        rid = runtime.context.get("run_id") if runtime.context else None
        return str(rid) if rid else "default"

    def _pending_key(self, runtime: Runtime) -> tuple[str, str]:
        return self._thread_id(runtime), self._run_id(runtime)

    # ------------------------------------------------------------------
    # State store (caller holds lock)

    def _get_state(self, thread_id: str, tool_name: str) -> ToolPhaseState:
        if thread_id not in self._phase_states:
            self._phase_states[thread_id] = {}
            while len(self._phase_states) > self._max_tracked_threads:
                self._phase_states.popitem(last=False)
        self._phase_states.move_to_end(thread_id)
        return self._phase_states[thread_id].get(tool_name, ToolPhaseState())

    def _set_state(self, thread_id: str, tool_name: str, state: ToolPhaseState) -> None:
        self._phase_states[thread_id][tool_name] = state

    def _get_block_reason(self, runtime: Runtime, tool_name: str) -> str | None:
        thread_id = self._thread_id(runtime)
        with self._lock:
            state = self._get_state(thread_id, tool_name)
        return state.block_reason if state.phase == "blocked" else None

    def _make_blocked_message(self, request: ToolCallRequest, tool_name: str, block_reason: str) -> ToolMessage:
        return ToolMessage(
            content=f"[TOOL_BLOCKED] {block_reason}",
            tool_call_id=str(request.tool_call.get("id", "")),
            name=tool_name,
            status="error",
            additional_kwargs={
                TOOL_META_KEY: {
                    "status": "error",
                    "error_type": "blocked_by_progress_guard",
                    "retryable": False,
                    "recoverable_by_model": True,
                    "recommended_next_action": "summarize",
                    "source": "progress_middleware",
                }
            },
        )

    def _update_state_from_result(
        self,
        result: ToolMessage | Command,
        tool_name: str,
        runtime: Runtime,
    ) -> ToolMessage | Command:
        """Update the state machine from a tool result; queue hints if warranted."""
        if not isinstance(result, ToolMessage):
            return result
        meta = _parse_tool_meta((result.additional_kwargs or {}).get(TOOL_META_KEY))
        if meta is None:
            return result
        content = _message_content_str(result)
        thread_id = self._thread_id(runtime)
        with self._lock:
            state = self._get_state(thread_id, tool_name)
            new_state, hint = self._assess_and_transition(state, meta, content)
            self._set_state(thread_id, tool_name, new_state)
        if hint and self._inject_assessment:
            self._queue_assessment(runtime, hint)
        return result

    # ------------------------------------------------------------------
    # State machine

    def _assess_and_transition(
        self,
        state: ToolPhaseState,
        meta: ToolResultMeta,
        content: str,
    ) -> tuple[ToolPhaseState, str | None]:
        """Return (new_state, hint_text_or_None)."""
        # Immediately block on unrecoverable stop signals (auth, config, internal).
        if not meta.recoverable_by_model and meta.recommended_next_action == "stop":
            return replace(
                state,
                phase="blocked",
                block_reason=_block_reason(meta),
            ), None

        ws = word_set(content)
        is_problem = meta.status in ("error", "partial_success") or (meta.status == "success" and is_near_duplicate(ws, state.recent_word_sets, self._jaccard_threshold, self._min_words))

        if not is_problem:
            # Good result: reset consecutive count, return to active.
            new_recent = (state.recent_word_sets + [ws])[-5:]
            return replace(state, consecutive_problems=0, phase="active", recent_word_sets=new_recent), None

        new_count = state.consecutive_problems + 1
        hint: str | None = None

        if new_count >= self._stagnation_threshold + self._warn_escalation:
            reason = _block_reason(meta)
            new_state = replace(state, consecutive_problems=new_count, phase="blocked", block_reason=reason)
        elif new_count >= self._stagnation_threshold:
            hint = _format_hint(meta)
            new_state = replace(state, consecutive_problems=new_count, phase="warned")
        else:
            new_state = replace(state, consecutive_problems=new_count)

        return new_state, hint

    # ------------------------------------------------------------------
    # Pending queue helpers

    def _queue_assessment(self, runtime: Runtime, text: str) -> None:
        key = self._pending_key(runtime)
        with self._lock:
            queue = self._pending[key]
            if len(queue) < _MAX_PENDING_PER_RUN:
                queue.append(text)

    def _drain_pending(self, runtime: Runtime) -> list[str]:
        key = self._pending_key(runtime)
        with self._lock:
            return self._pending.pop(key, [])

    def _clear_stale_pending(self, runtime: Runtime) -> None:
        thread_id, current_run = self._pending_key(runtime)
        with self._lock:
            for key in list(self._pending):
                if key[0] == thread_id and key[1] != current_run:
                    del self._pending[key]

    # ------------------------------------------------------------------
    # wrap_tool_call

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = str(request.tool_call.get("name", ""))
        if not tool_name or tool_name in self._exempt_tools:
            return handler(request)
        runtime = getattr(request, "runtime", None)
        if runtime is None:
            return handler(request)
        block_reason = self._get_block_reason(runtime, tool_name)
        if block_reason:
            return self._make_blocked_message(request, tool_name, block_reason)
        return self._update_state_from_result(handler(request), tool_name, runtime)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = str(request.tool_call.get("name", ""))
        if not tool_name or tool_name in self._exempt_tools:
            return await handler(request)
        runtime = getattr(request, "runtime", None)
        if runtime is None:
            return await handler(request)
        block_reason = self._get_block_reason(runtime, tool_name)
        if block_reason:
            return self._make_blocked_message(request, tool_name, block_reason)
        return self._update_state_from_result(await handler(request), tool_name, runtime)

    # ------------------------------------------------------------------
    # wrap_model_call: drain pending hints and inject before model sees messages

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        hints = self._drain_pending(request.runtime)
        if not hints:
            return request
        deduped = list(dict.fromkeys(hints))
        new_messages = [
            *request.messages,
            HumanMessage(content="\n\n".join(deduped), name="progress_hint"),
        ]
        return request.override(messages=new_messages)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._augment_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._augment_request(request))

    # ------------------------------------------------------------------
    # before_agent: clean up stale pending hints from previous runs

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_stale_pending(runtime)
        return None

    @override
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._clear_stale_pending(runtime)
        return None
