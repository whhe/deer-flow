"""Unified tool result semantics for structured signal production.

Every tool result that passes through ToolErrorHandlingMiddleware gets a
``deerflow_tool_meta`` entry in additional_kwargs. Downstream consumers
(ToolProgressMiddleware, etc.) read this key instead of parsing text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.types import Command

TOOL_META_KEY = "deerflow_tool_meta"

_ERROR_PREFIX = "Error:"
_PARTIAL_MARKERS = ("partial results", "limited results", "truncated", "results may be incomplete")
_MIN_SUBSTANTIAL_CONTENT = 80


@dataclass(frozen=True, slots=True)
class ToolResultMeta:
    status: Literal["success", "error", "partial_success"]
    error_type: str | None
    retryable: bool
    recoverable_by_model: bool
    recommended_next_action: Literal["continue", "rewrite_query", "try_alternative", "summarize", "stop"]
    source: Literal["exception", "tool_return", "content_analysis", "progress_middleware"]


_ERROR_RULES: list[tuple[list[str], dict[str, object]]] = [
    (
        ["401", "403", "unauthorized", "authentication", "invalid api key"],
        {"error_type": "auth", "retryable": False, "recoverable_by_model": False, "recommended_next_action": "stop"},
    ),
    (
        ["rate limit", "rate limited", "rate_limit"],
        {"error_type": "rate_limited", "retryable": True, "recoverable_by_model": False, "recommended_next_action": "summarize"},
    ),
    (
        ["timeout", "timed out", "connection", "network error", "temporarily unavailable"],
        {"error_type": "transient", "retryable": True, "recoverable_by_model": False, "recommended_next_action": "try_alternative"},
    ),
    (
        ["not configured", "not installed", "missing required", "disabled", "no api key"],
        {"error_type": "config", "retryable": False, "recoverable_by_model": False, "recommended_next_action": "stop"},
    ),
    (
        ["permission denied", "access denied", "path traversal", "forbidden"],
        {"error_type": "permission", "retryable": False, "recoverable_by_model": True, "recommended_next_action": "try_alternative"},
    ),
    (
        ["no results found", "no content found", "no images found", "no results"],
        {"error_type": "no_results", "retryable": False, "recoverable_by_model": True, "recommended_next_action": "rewrite_query"},
    ),
    (
        ["not found", "no such file", "does not exist", "404"],
        {"error_type": "not_found", "retryable": False, "recoverable_by_model": True, "recommended_next_action": "rewrite_query"},
    ),
    (
        ["unexpected error", "internal error", "500"],
        {"error_type": "internal", "retryable": False, "recoverable_by_model": False, "recommended_next_action": "stop"},
    ),
]

_UNKNOWN_ERROR: dict[str, object] = {
    "error_type": "unknown",
    "retryable": False,
    "recoverable_by_model": True,
    "recommended_next_action": "try_alternative",
}


def _classify_error_text(text: str) -> dict[str, object]:
    lower = text.lower()
    for keywords, attrs in _ERROR_RULES:
        if any(kw in lower for kw in keywords):
            return {**attrs}
    return {**_UNKNOWN_ERROR}


def _make_meta(*, status: str, source: str, error_type: str | None = None, retryable: bool = False, recoverable_by_model: bool = True, recommended_next_action: str = "continue") -> dict[str, object]:
    return {
        "status": status,
        "error_type": error_type,
        "retryable": retryable,
        "recoverable_by_model": recoverable_by_model,
        "recommended_next_action": recommended_next_action,
        "source": source,
    }


def stamp_exception_meta(msg: ToolMessage, exc_info: str) -> ToolMessage:
    """Stamp deerflow_tool_meta with source='exception' onto an exception-derived ToolMessage."""
    attrs = _classify_error_text(exc_info)
    updated_kwargs = dict(msg.additional_kwargs or {})
    updated_kwargs[TOOL_META_KEY] = _make_meta(status="error", source="exception", **attrs)
    msg.additional_kwargs = updated_kwargs
    return msg


def normalize_tool_message(msg: ToolMessage) -> ToolMessage:
    """Attach deerflow_tool_meta to a ToolMessage if not already present."""
    existing = (msg.additional_kwargs or {}).get(TOOL_META_KEY)
    if existing is not None:
        return msg

    content = msg.content if isinstance(msg.content, str) else ""

    # Non-standard error: tool returned status="error" without the "Error:" prefix convention.
    # (Actual exceptions from ToolErrorHandlingMiddleware are pre-stamped by stamp_exception_meta
    # and exit early above — they never reach this branch.)
    if msg.status == "error" and not content.startswith(_ERROR_PREFIX):
        attrs = _classify_error_text(content)
        meta = _make_meta(status="error", source="exception", **attrs)
    elif content.startswith(_ERROR_PREFIX):
        attrs = _classify_error_text(content[len(_ERROR_PREFIX) :])
        meta = _make_meta(status="error", source="tool_return", **attrs)
    elif any(m in content.lower() for m in _PARTIAL_MARKERS) or (0 < len(content) < _MIN_SUBSTANTIAL_CONTENT):
        meta = _make_meta(
            status="partial_success",
            source="content_analysis",
            recommended_next_action="rewrite_query",
        )
    else:
        meta = _make_meta(status="success", source="content_analysis")

    updated_kwargs = dict(msg.additional_kwargs or {})
    updated_kwargs[TOOL_META_KEY] = meta
    msg.additional_kwargs = updated_kwargs
    return msg


def normalize_tool_result(result: ToolMessage | Command) -> ToolMessage | Command:
    """Normalize a tool result, handling Command wrappers transparently."""
    if isinstance(result, ToolMessage):
        return normalize_tool_message(result)
    return result
