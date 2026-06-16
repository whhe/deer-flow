"""Unified tool result semantics for structured signal production.

Every tool result that passes through ToolErrorHandlingMiddleware gets a
``deerflow_tool_meta`` entry in additional_kwargs. Downstream consumers
(ToolProgressMiddleware, etc.) read this key instead of parsing text.
"""

from __future__ import annotations

import json
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


_SEMANTIC_ZERO_ERROR_STRINGS: frozenset[str] = frozenset({"none", "null", "false", "no", "ok", "success", "n/a", ""})


def _extract_json_error_text(content: str) -> str | None:
    """Return the error string from a JSON-wrapped error like {"error": "...", "query": "..."}.

    Returns None when the ``error`` field is falsy (JSON null / 0 / false / empty
    string) or is a sentinel string that conventionally means "no error" (e.g.
    ``"none"``, ``"null"``, ``"false"``).  This prevents tools that return
    ``{"error": "none", "results": [...]}`` on success from being misclassified
    as errors.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    error = data.get("error") if isinstance(data, dict) else None
    if not error:
        return None
    if isinstance(error, str) and error.lower().strip() in _SEMANTIC_ZERO_ERROR_STRINGS:
        return None
    return str(error)


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
    """Stamp deerflow_tool_meta with source='exception' onto an exception-derived ToolMessage.

    Unlike normalize_tool_message (which preserves existing stamps), this function always
    overwrites any pre-existing TOOL_META_KEY entry.  Exception-derived classification is
    more authoritative than a tool's own return-time stamp.
    """
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
    # Try JSON extraction first so classification uses only the "error" field value, not
    # keywords that appear incidentally in other JSON fields (e.g. "query").
    if msg.status == "error" and not content.startswith(_ERROR_PREFIX):
        json_error = _extract_json_error_text(content)
        attrs = _classify_error_text(json_error if json_error is not None else content)
        meta = _make_meta(status="error", source="tool_return", **attrs)
    elif content.startswith(_ERROR_PREFIX):
        attrs = _classify_error_text(content[len(_ERROR_PREFIX) :])
        meta = _make_meta(status="error", source="tool_return", **attrs)
    elif (json_error := _extract_json_error_text(content)) is not None:
        attrs = _classify_error_text(json_error)
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
