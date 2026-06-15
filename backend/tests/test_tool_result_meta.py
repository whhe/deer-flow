"""Tests for tool_result_meta normalization logic."""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deerflow.agents.middlewares.tool_result_meta import (
    TOOL_META_KEY,
    ToolResultMeta,
    normalize_tool_message,
    normalize_tool_result,
    stamp_exception_meta,
)


def _make_msg(content: str, *, status: str = "success", kwargs: dict[str, object] | None = None) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id="tc-1",
        name="test_tool",
        status=status,
        additional_kwargs=kwargs or {},
    )


def _meta(msg: ToolMessage) -> dict[str, object]:
    return msg.additional_kwargs[TOOL_META_KEY]


# ---------------------------------------------------------------------------
# Already-stamped messages are not overwritten


def test_existing_meta_is_preserved():
    existing = {"status": "success", "source": "custom"}
    msg = _make_msg("hello", kwargs={TOOL_META_KEY: existing})
    result = normalize_tool_message(msg)
    assert result.additional_kwargs[TOOL_META_KEY] is existing


# ---------------------------------------------------------------------------
# Error prefix (tool_return path)


@pytest.mark.parametrize(
    "snippet,expected_type",
    [
        ("Error: 401 unauthorized", "auth"),
        ("Error: permission denied for path", "permission"),
        ("Error: 429 rate limit exceeded", "rate_limited"),
        ("Error: connection timeout", "transient"),
        ("Error: tool not configured", "config"),
        ("Error: no results found for query", "no_results"),
        ("Error: file not found", "not_found"),
        ("Error: internal error 500", "internal"),
        ("Error: something completely unexpected happened", "unknown"),
    ],
)
def test_error_prefix_classification(snippet: str, expected_type: str):
    msg = _make_msg(snippet, status="error")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["status"] == "error"
    assert m["error_type"] == expected_type
    assert m["source"] == "tool_return"


def test_auth_error_is_not_retryable_and_stop():
    msg = _make_msg("Error: invalid api key", status="error")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["retryable"] is False
    assert m["recoverable_by_model"] is False
    assert m["recommended_next_action"] == "stop"


def test_rate_limited_error_is_retryable():
    msg = _make_msg("Error: rate limited", status="error")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["retryable"] is True
    assert m["recommended_next_action"] == "summarize"


def test_no_results_suggests_rewrite_query():
    msg = _make_msg("Error: no results found", status="error")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["recoverable_by_model"] is True
    assert m["recommended_next_action"] == "rewrite_query"


# ---------------------------------------------------------------------------
# Exception path (status="error", no "Error:" prefix)


def test_exception_path_classifies_from_content():
    msg = _make_msg("ConnectionError: connection refused", status="error")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["status"] == "error"
    assert m["source"] == "exception"
    assert m["error_type"] == "transient"


def test_exception_path_timeout_content():
    msg = _make_msg("timeout occurred", status="error")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["source"] == "exception"
    assert m["error_type"] == "transient"


# ---------------------------------------------------------------------------
# Partial success detection


def test_partial_markers_detected():
    for marker in ("partial results available", "limited results returned", "truncated output", "results may be incomplete"):
        msg = _make_msg(f"Here are some {marker} from the search.", status="success")
        result = normalize_tool_message(msg)
        m = _meta(result)
        assert m["status"] == "partial_success", f"expected partial_success for: {marker}"
        assert m["recommended_next_action"] == "rewrite_query"


def test_short_content_is_partial():
    msg = _make_msg("Ok.", status="success")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["status"] == "partial_success"
    assert m["source"] == "content_analysis"


def test_empty_content_is_not_partial():
    # Empty content: len == 0, so the `0 < len(content) < 80` condition is False
    msg = _make_msg("", status="success")
    result = normalize_tool_message(msg)
    m = _meta(result)
    # Empty content falls through to success (no partial markers, len == 0)
    assert m["status"] == "success"


# ---------------------------------------------------------------------------
# Success path


def test_substantial_content_is_success():
    content = "A" * 200
    msg = _make_msg(content, status="success")
    result = normalize_tool_message(msg)
    m = _meta(result)
    assert m["status"] == "success"
    assert m["source"] == "content_analysis"
    assert m["recommended_next_action"] == "continue"
    assert m["error_type"] is None


# ---------------------------------------------------------------------------
# ToolResultMeta dataclass round-trip


def test_tool_result_meta_from_dict():
    msg = _make_msg("A" * 200)
    result = normalize_tool_message(msg)
    meta_dict = _meta(result)
    meta = ToolResultMeta(**meta_dict)
    assert meta.status == "success"
    assert meta.error_type is None
    assert meta.retryable is False
    assert meta.recommended_next_action == "continue"


# ---------------------------------------------------------------------------
# stamp_exception_meta


def test_stamp_exception_meta_classifies_from_exc_info_not_content():
    # Content says "no results" but exc_info says "connection refused" —
    # stamp_exception_meta must use exc_info, producing transient, not no_results.
    msg = _make_msg("Error: no results found", status="error")
    result = stamp_exception_meta(msg, "ConnectionError: connection refused")
    m = _meta(result)
    assert m["source"] == "exception"
    assert m["error_type"] == "transient"


def test_stamp_exception_meta_overwrites_existing_meta():
    pre_existing = {TOOL_META_KEY: {"source": "tool_return", "error_type": "unknown"}}
    msg = _make_msg("Error: no results found", status="error", kwargs=pre_existing)
    result = stamp_exception_meta(msg, "PermissionError: access denied")
    m = _meta(result)
    assert m["source"] == "exception"
    assert m["error_type"] == "permission"


def test_stamp_exception_meta_preserves_other_additional_kwargs():
    msg = _make_msg("irrelevant", status="error", kwargs={"subagent_status": "running"})
    result = stamp_exception_meta(msg, "TimeoutError: timed out")
    assert result.additional_kwargs["subagent_status"] == "running"
    assert TOOL_META_KEY in result.additional_kwargs


# ---------------------------------------------------------------------------
# normalize_tool_result handles Command wrappers


def test_normalize_tool_result_passthrough_command():
    cmd = Command(goto="next_node")
    result = normalize_tool_result(cmd)
    assert result is cmd


def test_normalize_tool_result_stamps_tool_message():
    msg = _make_msg("A" * 200)
    result = normalize_tool_result(msg)
    assert isinstance(result, ToolMessage)
    assert TOOL_META_KEY in result.additional_kwargs
