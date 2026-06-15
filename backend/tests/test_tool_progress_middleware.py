"""Tests for ToolProgressMiddleware state machine (RFC #3177)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from deerflow.agents.middlewares.tool_progress_middleware import (
    ToolProgressMiddleware,
    is_near_duplicate,
    word_set,
)
from deerflow.agents.middlewares.tool_result_meta import TOOL_META_KEY

# ---------------------------------------------------------------------------
# Helpers


def _make_runtime(thread_id: str = "t1", run_id: str = "r1") -> MagicMock:
    rt = MagicMock()
    rt.context = {"thread_id": thread_id, "run_id": run_id}
    return rt


def _make_tool_request(tool_name: str = "web_search", *, runtime: MagicMock | None = None) -> SimpleNamespace:
    rt = runtime or _make_runtime()
    return SimpleNamespace(
        tool_call={"name": tool_name, "id": f"tc-{tool_name}"},
        runtime=rt,
    )


def _meta_kwargs(
    *,
    status: str = "success",
    error_type: str | None = None,
    retryable: bool = False,
    recoverable_by_model: bool = True,
    recommended_next_action: str = "continue",
    source: str = "content_analysis",
) -> dict[str, object]:
    return {
        TOOL_META_KEY: {
            "status": status,
            "error_type": error_type,
            "retryable": retryable,
            "recoverable_by_model": recoverable_by_model,
            "recommended_next_action": recommended_next_action,
            "source": source,
        }
    }


def _make_tool_message(
    content: str = "A" * 200,
    *,
    tool_name: str = "web_search",
    meta_kwargs: dict[str, object] | None = None,
) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=f"tc-{tool_name}",
        name=tool_name,
        status="success",
        additional_kwargs=meta_kwargs or _meta_kwargs(),
    )


def _make_error_message(
    content: str = "Error: no results found",
    *,
    tool_name: str = "web_search",
    error_type: str = "no_results",
    recoverable_by_model: bool = True,
    recommended_next_action: str = "rewrite_query",
) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=f"tc-{tool_name}",
        name=tool_name,
        status="error",
        additional_kwargs=_meta_kwargs(
            status="error",
            error_type=error_type,
            recoverable_by_model=recoverable_by_model,
            recommended_next_action=recommended_next_action,
        ),
    )


def _make_model_request(messages: list, runtime: MagicMock) -> MagicMock:
    req = MagicMock()
    req.messages = list(messages)
    req.runtime = runtime

    def _override(**kw) -> MagicMock:
        updated = MagicMock()
        updated.messages = kw.get("messages", req.messages)
        updated.runtime = runtime
        updated.override = req.override
        return updated

    req.override = _override
    return req


def _make_mw(**kwargs) -> ToolProgressMiddleware:
    defaults = {
        "stagnation_threshold": 3,
        "warn_escalation_count": 2,
        "inject_assessment": True,
        "jaccard_threshold": 0.8,
        "min_words": 5,
    }
    defaults.update(kwargs)
    return ToolProgressMiddleware(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: word_set and is_near_duplicate


def test_word_set_extracts_words_ge_3():
    ws = word_set("go quick brown fox")
    assert "go" not in ws
    assert "quick" in ws
    assert "brown" in ws
    assert "fox" in ws


def test_is_near_duplicate_above_threshold():
    ws1 = frozenset("quick brown fox jumps over lazy dog".split())
    ws2 = frozenset("quick brown fox jumps over lazy dog".split())
    assert is_near_duplicate(ws2, [ws1], threshold=0.8, min_words=5)


def test_is_near_duplicate_below_threshold():
    ws1 = frozenset("apple banana cherry delta echo".split())
    ws2 = frozenset("xray yankee zulu alpha bravo".split())
    assert not is_near_duplicate(ws2, [ws1], threshold=0.8, min_words=5)


def test_is_near_duplicate_too_short_skips_check():
    ws1 = frozenset("apple".split())
    ws2 = frozenset("apple".split())
    # min_words=5 but len==1, so not a duplicate
    assert not is_near_duplicate(ws2, [ws1], threshold=0.8, min_words=5)


# ---------------------------------------------------------------------------
# Scenario 1: Normal call → no hint, phase stays active


def test_normal_call_no_hint_phase_active():
    mw = _make_mw()
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    msg = _make_tool_message("A" * 300)

    def handler(_r):
        return msg

    result = mw.wrap_tool_call(req, handler)

    assert result is msg
    assert mw._phase_states["t1"]["web_search"].phase == "active"
    assert mw._phase_states["t1"]["web_search"].consecutive_problems == 0


# ---------------------------------------------------------------------------
# Scenario 2: consecutive no_results → hint injected, phase=warned


def test_repeated_no_results_reaches_warned():
    mw = _make_mw(stagnation_threshold=2)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()

    def handler(_r):
        return error_msg

    # stagnation_threshold=2, so the second problem call tips into warned
    mw.wrap_tool_call(req, handler)
    mw.wrap_tool_call(req, handler)

    state = mw._phase_states["t1"]["web_search"]
    assert state.phase == "warned"
    assert state.consecutive_problems == 2

    # Hint should be queued
    hints = mw._drain_pending(rt)
    assert len(hints) == 1
    assert "PROGRESS HINT" in hints[0]


# ---------------------------------------------------------------------------
# Scenario 3: Additional calls after warned → blocked


def test_warned_to_blocked_after_escalation():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=2)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()

    def handler(_r):
        return error_msg

    for _ in range(4):
        mw.wrap_tool_call(req, handler)

    state = mw._phase_states["t1"]["web_search"]
    assert state.phase == "blocked"
    assert state.block_reason is not None


# ---------------------------------------------------------------------------
# Scenario 4: Blocked tool is front-gate intercepted (handler NOT called)


def test_blocked_tool_is_intercepted_without_calling_handler():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=1)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()
    call_count = [0]

    def handler(r):
        call_count[0] += 1
        return error_msg

    # 2 calls → warned + 1 more = blocked
    for _ in range(3):
        mw.wrap_tool_call(req, handler)

    assert mw._phase_states["t1"]["web_search"].phase == "blocked"
    call_count_before = call_count[0]

    # Next call should be intercepted
    result = mw.wrap_tool_call(req, handler)

    assert call_count[0] == call_count_before
    assert isinstance(result, ToolMessage)
    assert "[TOOL_BLOCKED]" in result.content


# ---------------------------------------------------------------------------
# Scenario 5: Auth error → immediately blocked (no warned phase)


def test_auth_error_immediately_blocked():
    mw = _make_mw(stagnation_threshold=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    auth_msg = _make_error_message(
        error_type="auth",
        recoverable_by_model=False,
        recommended_next_action="stop",
    )

    def handler(_r):
        return auth_msg

    mw.wrap_tool_call(req, handler)

    state = mw._phase_states["t1"]["web_search"]
    assert state.phase == "blocked"
    assert "auth" in state.block_reason.lower() or "Authentication" in state.block_reason


# ---------------------------------------------------------------------------
# Scenario 6: Valid result after problems resets to active


def test_valid_result_after_problems_resets_to_active():
    mw = _make_mw(stagnation_threshold=3, warn_escalation_count=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()
    good_msg = _make_tool_message("A" * 300)

    def handler_error(_r):
        return error_msg

    def handler_good(_r):
        return good_msg

    mw.wrap_tool_call(req, handler_error)
    mw.wrap_tool_call(req, handler_error)
    mw.wrap_tool_call(req, handler_error)

    state = mw._phase_states["t1"]["web_search"]
    assert state.phase == "warned"

    # Good result resets
    mw.wrap_tool_call(req, handler_good)

    state = mw._phase_states["t1"]["web_search"]
    assert state.phase == "active"
    assert state.consecutive_problems == 0


# ---------------------------------------------------------------------------
# Scenario 7: Two different tools have independent states


def test_two_tools_have_independent_states():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=1)
    rt = _make_runtime()
    req_search = _make_tool_request("web_search", runtime=rt)
    req_read = _make_tool_request("read_file", runtime=rt)

    error_search = _make_error_message(tool_name="web_search")
    error_read = _make_error_message(tool_name="read_file")

    # Block web_search (2 → warned, 1 more → blocked)
    for _ in range(3):
        mw.wrap_tool_call(req_search, lambda r: error_search)

    assert mw._phase_states["t1"]["web_search"].phase == "blocked"

    # read_file should still be active
    mw.wrap_tool_call(req_read, lambda r: error_read)
    assert mw._phase_states["t1"]["read_file"].phase == "active"


# ---------------------------------------------------------------------------
# Scenario 8: Jaccard near-duplicate result counts as problem


def test_jaccard_near_duplicate_counts_as_problem():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=5, jaccard_threshold=0.8, min_words=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)

    # First call: good unique content (establishes baseline)
    words = "apple banana cherry delta echo foxtrot golf hotel india juliet"
    msg1 = _make_tool_message(words)
    mw.wrap_tool_call(req, lambda r: msg1)

    # Second call: exact same content (Jaccard = 1.0) → near-duplicate → problem count goes up
    msg2 = _make_tool_message(words)
    mw.wrap_tool_call(req, lambda r: msg2)

    state = mw._phase_states["t1"]["web_search"]
    assert state.consecutive_problems >= 1


# ---------------------------------------------------------------------------
# Scenario 9: Different Jaccard content does NOT count as problem


def test_jaccard_different_content_not_a_problem():
    mw = _make_mw(stagnation_threshold=3, warn_escalation_count=5, jaccard_threshold=0.8, min_words=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)

    words1 = "apple banana cherry delta echo foxtrot golf hotel india juliet"
    words2 = "xray yankee zulu alpha bravo charlie sierra tango uniform victor"
    msg1 = _make_tool_message(words1)
    msg2 = _make_tool_message(words2)

    mw.wrap_tool_call(req, lambda r: msg1)
    mw.wrap_tool_call(req, lambda r: msg2)

    state = mw._phase_states["t1"]["web_search"]
    assert state.consecutive_problems == 0
    assert state.phase == "active"


# ---------------------------------------------------------------------------
# Scenario 10: exempt_tools are not tracked


def test_exempt_tools_not_tracked():
    mw = _make_mw(stagnation_threshold=1, warn_escalation_count=1)
    rt = _make_runtime()
    req = _make_tool_request("ask_clarification", runtime=rt)
    error_msg = _make_error_message(tool_name="ask_clarification")

    def handler(_r):
        return error_msg

    for _ in range(5):
        mw.wrap_tool_call(req, handler)

    assert "ask_clarification" not in mw._phase_states.get("t1", {})


# ---------------------------------------------------------------------------
# Scenario 11: before_agent clears stale pending hints from previous runs


def test_before_agent_clears_stale_pending():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=5)
    rt_run1 = _make_runtime(thread_id="t1", run_id="old-run")
    rt_run2 = _make_runtime(thread_id="t1", run_id="new-run")
    req = _make_tool_request(runtime=rt_run1)
    error_msg = _make_error_message()

    # Produce a hint for old-run
    mw.wrap_tool_call(req, lambda r: error_msg)
    mw.wrap_tool_call(req, lambda r: error_msg)

    mw._drain_pending(rt_run1)
    # Re-queue manually to simulate pending state
    mw._queue_assessment(rt_run1, "old hint")

    # before_agent with new-run should clear the old-run's pending hints
    state_mock = MagicMock()
    mw.before_agent(state_mock, rt_run2)

    # Old pending should be gone
    leftovers = mw._pending.get(("t1", "old-run"), [])
    assert leftovers == []


# ---------------------------------------------------------------------------
# Scenario 12: LRU eviction when max_tracked_threads exceeded


def test_lru_eviction_of_oldest_thread():
    mw = _make_mw(max_tracked_threads=2)
    error_msg = _make_error_message()

    for i in range(3):
        rt = _make_runtime(thread_id=f"thread-{i}")
        req = _make_tool_request(runtime=rt)
        mw.wrap_tool_call(req, lambda r: error_msg)

    assert len(mw._phase_states) == 2
    # thread-0 should have been evicted (oldest); thread-1 and thread-2 remain
    assert "thread-0" not in mw._phase_states
    assert "thread-1" in mw._phase_states
    assert "thread-2" in mw._phase_states


# ---------------------------------------------------------------------------
# Hint injection via wrap_model_call


def test_hint_injected_into_model_call():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()

    # Trigger hint
    mw.wrap_tool_call(req, lambda r: error_msg)
    mw.wrap_tool_call(req, lambda r: error_msg)

    model_req = _make_model_request([], rt)
    captured_messages = []

    def model_handler(r):
        captured_messages.extend(r.messages)
        return MagicMock()

    mw.wrap_model_call(model_req, model_handler)

    assert any(isinstance(m, HumanMessage) for m in captured_messages)
    hint_msgs = [m for m in captured_messages if isinstance(m, HumanMessage)]
    assert any("PROGRESS HINT" in m.content for m in hint_msgs)


def test_partial_success_hint_is_specific_not_generic():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    partial_msg = ToolMessage(
        content="Here are some partial results from the search.",
        tool_call_id="tc-web_search",
        name="web_search",
        status="success",
        additional_kwargs=_meta_kwargs(
            status="partial_success",
            recommended_next_action="rewrite_query",
        ),
    )

    def handler(_r):
        return partial_msg

    mw.wrap_tool_call(req, handler)
    mw.wrap_tool_call(req, handler)

    hints = mw._drain_pending(rt)
    assert len(hints) == 1
    assert "incomplete results" in hints[0].lower()
    assert "not producing new information" not in hints[0]


def test_no_hint_when_inject_assessment_disabled():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=5, inject_assessment=False)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()

    mw.wrap_tool_call(req, lambda r: error_msg)
    mw.wrap_tool_call(req, lambda r: error_msg)

    hints = mw._drain_pending(rt)
    assert hints == []


# ---------------------------------------------------------------------------
# Tool without runtime attribute is passed through


def test_no_runtime_passthrough():
    mw = _make_mw()
    req = SimpleNamespace(tool_call={"name": "web_search", "id": "tc-1"})
    # No runtime attribute
    msg = _make_tool_message()

    def handler(_r):
        return msg

    result = mw.wrap_tool_call(req, handler)
    assert result is msg


# ---------------------------------------------------------------------------
# Command results are passed through unchanged


def test_command_result_passthrough():
    mw = _make_mw()
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    cmd = Command(goto="some_node")

    def handler(_r):
        return cmd

    result = mw.wrap_tool_call(req, handler)
    assert result is cmd


# ---------------------------------------------------------------------------
# from_config round-trip


def test_from_config():
    from deerflow.config.tool_progress_config import ToolProgressConfig

    cfg = ToolProgressConfig(
        enabled=True,
        stagnation_threshold=4,
        warn_escalation_count=3,
        jaccard_similarity_threshold=0.7,
        min_word_count_for_similarity=8,
    )
    mw = ToolProgressMiddleware.from_config(cfg)
    assert mw._stagnation_threshold == 4
    assert mw._warn_escalation == 3
    assert mw._jaccard_threshold == pytest.approx(0.7)
    assert mw._min_words == 8


# ---------------------------------------------------------------------------
# Defensive meta parsing: malformed dicts must not crash the middleware


def test_wrap_tool_call_malformed_meta_passthrough():
    """Malformed deerflow_tool_meta dict must not crash the middleware."""
    mw = _make_mw()
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    bad_msg = ToolMessage(
        content="some content",
        tool_call_id="tc-web_search",
        name="web_search",
        status="success",
        additional_kwargs={TOOL_META_KEY: {"unexpected_field": True}},
    )

    def handler(_r):
        return bad_msg

    result = mw.wrap_tool_call(req, handler)

    assert result is bad_msg
    assert mw._phase_states.get("t1", {}).get("web_search") is None


# ---------------------------------------------------------------------------
# Async path: awrap_tool_call mirrors sync path


@pytest.mark.anyio
async def test_awrap_tool_call_normal_passthrough():
    mw = _make_mw()
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    msg = _make_tool_message("A" * 300)

    result = await mw.awrap_tool_call(req, AsyncMock(return_value=msg))

    assert result is msg
    assert mw._phase_states["t1"]["web_search"].phase == "active"


@pytest.mark.anyio
async def test_awrap_tool_call_blocked_intercepted_without_calling_handler():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=1)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()
    call_count = [0]

    async def handler(r):
        call_count[0] += 1
        return error_msg

    # 3 calls: 2 → warned, 1 more → blocked
    for _ in range(3):
        await mw.awrap_tool_call(req, handler)

    assert mw._phase_states["t1"]["web_search"].phase == "blocked"
    before = call_count[0]

    result = await mw.awrap_tool_call(req, handler)

    assert call_count[0] == before
    assert isinstance(result, ToolMessage)
    assert "[TOOL_BLOCKED]" in result.content


@pytest.mark.anyio
async def test_awrap_tool_call_auth_error_immediately_blocked():
    mw = _make_mw(stagnation_threshold=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    auth_msg = _make_error_message(
        error_type="auth",
        recoverable_by_model=False,
        recommended_next_action="stop",
    )

    await mw.awrap_tool_call(req, AsyncMock(return_value=auth_msg))

    state = mw._phase_states["t1"]["web_search"]
    assert state.phase == "blocked"
    assert state.block_reason is not None


@pytest.mark.anyio
async def test_awrap_tool_call_no_runtime_passthrough():
    mw = _make_mw()
    req = SimpleNamespace(tool_call={"name": "web_search", "id": "tc-1"})
    msg = _make_tool_message()

    result = await mw.awrap_tool_call(req, AsyncMock(return_value=msg))

    assert result is msg
    assert "t1" not in mw._phase_states


@pytest.mark.anyio
async def test_awrap_tool_call_command_result_passthrough():
    mw = _make_mw()
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    cmd = Command(goto="some_node")

    result = await mw.awrap_tool_call(req, AsyncMock(return_value=cmd))

    assert result is cmd


@pytest.mark.anyio
async def test_awrap_tool_call_malformed_meta_passthrough():
    """Malformed deerflow_tool_meta dict must not crash the middleware."""
    mw = _make_mw()
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    bad_msg = ToolMessage(
        content="some content",
        tool_call_id="tc-web_search",
        name="web_search",
        status="success",
        additional_kwargs={TOOL_META_KEY: {"unexpected_field": True}},
    )

    result = await mw.awrap_tool_call(req, AsyncMock(return_value=bad_msg))

    assert result is bad_msg
    # No state was tracked — malformed meta is silently skipped
    assert mw._phase_states.get("t1", {}).get("web_search") is None


@pytest.mark.anyio
async def test_awrap_model_call_drains_and_injects_hints():
    mw = _make_mw(stagnation_threshold=2, warn_escalation_count=5)
    rt = _make_runtime()
    req = _make_tool_request(runtime=rt)
    error_msg = _make_error_message()

    # Trigger hint via sync path (state machine is shared)
    mw.wrap_tool_call(req, lambda r: error_msg)
    mw.wrap_tool_call(req, lambda r: error_msg)

    model_req = _make_model_request([], rt)
    captured: list = []

    async def model_handler(r):
        captured.extend(r.messages)
        return MagicMock()

    await mw.awrap_model_call(model_req, model_handler)

    hint_msgs = [m for m in captured if isinstance(m, HumanMessage)]
    assert any("PROGRESS HINT" in m.content for m in hint_msgs)
