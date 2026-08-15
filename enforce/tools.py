"""Tool calling, treated as structured output under a different name.

A tool call is a schema-constrained generation that arrives in a different
envelope, and it fails in the same ways plus a few of its own. Arguments come
back as a JSON *string* rather than an object, so a malformed one is an ordinary
parse failure wearing different clothes.

Nothing here trusts the provider's framing. Tool arguments are model output, and
the fact that they arrived in a `tool_calls` field says nothing about whether
they match the schema that was sent — the capability probe measured a model that
accepts a tool definition and emits no call at all.

Failure modes handled, each observed rather than imagined:

* arguments that are not valid JSON,
* arguments that parse but do not match the tool's schema,
* a call to a tool that was never offered,
* prose returned where a call was required,
* several calls when one was expected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from contracts.translate import SchemaTranslator


class ToolFailure(str, Enum):
    """Why a tool call could not be turned into validated arguments."""

    NONE = "none"
    NO_CALL = "no_call"
    UNKNOWN_TOOL = "unknown_tool"
    ARGS_NOT_JSON = "args_not_json"
    ARGS_MISMATCH = "args_mismatch"


@dataclass
class ToolResult:
    """One tool call, validated or explained."""

    ok: bool
    tool_name: str = ""
    arguments: Any = None
    failure: ToolFailure = ToolFailure.NONE
    detail: str = ""
    raw_arguments: str = ""
    field_errors: list[dict[str, Any]] = field(default_factory=list)


def tool_definition(
    name: str,
    description: str,
    parameters: type[BaseModel],
    translator: SchemaTranslator | None = None,
) -> dict[str, Any]:
    """Build a tool definition from a Pydantic model.

    The parameter schema goes through the same translator as everything else.
    A tool schema is a schema, and the strict-mode rules that reject a Pydantic
    optional field in a `response_format` reject it here too.
    """
    translation = (translator or SchemaTranslator()).translate(parameters)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": translation.schema,
        },
    }


def validate_tool_call(
    tool_calls: list[Any],
    expected: dict[str, type[BaseModel]],
    require_one: bool = True,
) -> ToolResult:
    """Validate the first tool call against the schema it claims to satisfy.

    Args:
        tool_calls: Calls as the provider returned them.
        expected: Pydantic model per tool name.
        require_one: Fail when several calls arrive and only one was expected.
            Reported rather than silently taking the first, since the extra
            calls may have been the ones that mattered.

    Returns:
        The validated arguments, or the reason they could not be validated.
    """
    if not tool_calls:
        return ToolResult(
            ok=False,
            failure=ToolFailure.NO_CALL,
            detail="the model returned prose where a tool call was required",
        )

    if require_one and len(tool_calls) > 1:
        names = ", ".join(_name_of(c) for c in tool_calls)
        return ToolResult(
            ok=False,
            failure=ToolFailure.NO_CALL,
            detail=f"expected one tool call, got {len(tool_calls)}: {names}",
        )

    call = tool_calls[0]
    name = _name_of(call)
    raw = _arguments_of(call)

    contract = expected.get(name)
    if contract is None:
        known = ", ".join(sorted(expected))
        return ToolResult(
            ok=False,
            tool_name=name,
            failure=ToolFailure.UNKNOWN_TOOL,
            detail=f"called {name!r}, which was not offered. Offered: {known}",
            raw_arguments=raw,
        )

    try:
        data = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError) as exc:
        # Arguments arrive as a JSON string, so this is an ordinary parse
        # failure that happens to have come through the tool envelope.
        return ToolResult(
            ok=False,
            tool_name=name,
            failure=ToolFailure.ARGS_NOT_JSON,
            detail=f"arguments were not valid JSON: {exc}",
            raw_arguments=raw,
        )

    try:
        validated = contract.model_validate(data)
    except ValidationError as exc:
        return ToolResult(
            ok=False,
            tool_name=name,
            failure=ToolFailure.ARGS_MISMATCH,
            detail=str(exc).splitlines()[0],
            raw_arguments=raw,
            field_errors=exc.errors(),
        )

    return ToolResult(ok=True, tool_name=name, arguments=validated, raw_arguments=raw)


def _name_of(call: Any) -> str:
    """Read a tool name from either an SDK object or a plain dict."""
    function = getattr(call, "function", None)
    if function is not None:
        return getattr(function, "name", "") or ""
    if isinstance(call, dict):
        return call.get("function", {}).get("name", "")
    return ""


def _arguments_of(call: Any) -> str:
    """Read raw argument text from either an SDK object or a plain dict."""
    function = getattr(call, "function", None)
    if function is not None:
        return getattr(function, "arguments", "") or ""
    if isinstance(call, dict):
        return call.get("function", {}).get("arguments", "") or ""
    return ""
