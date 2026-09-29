"""Explicit plugin registration, strict schema checks, and scoped dispatch.

The intentionally small JSON Schema subset is validated on registration. Unknown
keywords fail closed rather than silently weakening a plugin's contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from importlib import metadata
import math
import re
import signal
import threading
from types import MappingProxyType
from typing import Any

from .artifacts import json_text
from .contracts import ResearchRequest, Tool, ToolCall, ToolContext, ToolObservation, ToolSpec


class ToolInputError(ValueError):
    """A repairable policy action error."""


class ToolScopeError(ToolInputError):
    """A policy attempted to use an unavailable tool."""


class ToolTimeoutError(TimeoutError):
    pass


_SCHEMA_KEYS = {"type", "properties", "required", "additionalProperties", "items", "enum", "const",
                "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
                "description", "title", "default", "pattern"}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict) or set(schema) - _SCHEMA_KEYS:
        raise ValueError("unsupported JSON schema keyword or invalid schema")
    if schema.get("type") not in _TYPES:
        raise ValueError("schema requires a supported explicit type")
    json_text(schema)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not all(isinstance(k, str) for k in properties):
        raise ValueError("properties must be a mapping")
    for child in properties.values():
        validate_schema(child)
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(k, str) and k in properties for k in required):
        raise ValueError("required must list declared properties")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError("additionalProperties must be boolean")
    if schema["type"] == "array":
        validate_schema(schema.get("items"))
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("enum must be a nonempty list")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise ValueError("pattern must be a string")
        re.compile(schema["pattern"])
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("minimum", "maximum"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            raise ValueError(f"{key} must be finite numeric")


def validate_value(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    kind = schema["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "integer": type(value) is int,
             "number": type(value) in (int, float) and math.isfinite(value),
             "boolean": type(value) is bool, "null": value is None}[kind]
    if not valid:
        raise ToolInputError(f"{path} must have type {kind}")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise ToolInputError(f"{path} is outside enum")
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise ToolInputError(f"{path} differs from const")
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ToolInputError(f"{path}.{key} is required")
        for key, child in value.items():
            if not isinstance(key, str):
                raise ToolInputError(f"{path} keys must be strings")
            if key not in properties:
                if schema.get("additionalProperties", True) is False:
                    raise ToolInputError(f"{path}.{key} is not allowed")
            else:
                validate_value(child, properties[key], f"{path}.{key}")
    elif kind == "array":
        for index, child in enumerate(value):
            validate_value(child, schema["items"], f"{path}[{index}]")
    if kind in {"string", "array"}:
        low, high = ("minLength", "maxLength") if kind == "string" else ("minItems", "maxItems")
        if len(value) < schema.get(low, 0) or len(value) > schema.get(high, float("inf")):
            raise ToolInputError(f"{path} has invalid length")
    if kind == "string" and "pattern" in schema and re.search(schema["pattern"], value) is None:
        raise ToolInputError(f"{path} does not match pattern")
    if kind in {"integer", "number"}:
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            raise ToolInputError(f"{path} is out of range")
    try:
        json_text(value)
    except (ValueError, TypeError) as exc:
        raise ToolInputError(f"{path} must be JSON serializable") from exc


@contextmanager
def call_deadline(seconds: float):
    """Bound trusted synchronous calls on POSIX main threads.

    Plugins remain trusted host code and must clean up children on interruption.
    This is a deadline guard, not a security sandbox for arbitrary Python plugins.
    """
    if seconds <= 0:
        raise ToolTimeoutError("call deadline exhausted")
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("synchronous harness requires a POSIX main thread for deadline enforcement")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        raise RuntimeError("harness cannot replace an existing process alarm")
    def expired(signum, frame):
        raise ToolTimeoutError("call exceeded deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec
        if not isinstance(spec, ToolSpec) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", spec.name):
            raise ValueError("invalid tool spec/name")
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        if not spec.type or not spec.version or not callable(getattr(tool, "execute", None)):
            raise ValueError("tools require type, version and execute")
        if not math.isfinite(spec.timeout_seconds) or spec.timeout_seconds <= 0 or spec.max_output_bytes < 512:
            raise ValueError("tool limits require a positive timeout and at least 512 output bytes")
        if any(not isinstance(v, str) or not v for v in (*spec.capabilities, *spec.required_resources)):
            raise ValueError("capabilities/resources must be nonempty strings")
        validate_schema(spec.input_schema)
        if spec.input_schema["type"] != "object":
            raise ValueError("tool input schema must be an object")
        if spec.output_schema is not None:
            validate_schema(spec.output_schema)
        self._tools[spec.name] = tool

    def load_plugins(self, names: list[str], group: str = "agent_harness.tools") -> None:
        """Load only explicit entry-point names; factories return a Tool or list of Tools."""
        entries = list(metadata.entry_points(group=group))
        for name in names:
            matches = [entry for entry in entries if entry.name == name]
            if len(matches) != 1:
                raise ValueError(f"plugin must resolve uniquely: {name}")
            loaded = matches[0].load()()
            for tool in loaded if isinstance(loaded, (list, tuple)) else [loaded]:
                self.register(tool)

    @staticmethod
    def _allowed(spec: ToolSpec, request: ResearchRequest) -> bool:
        return ((request.allowed_types is None or spec.type in request.allowed_types)
                and (request.allowed_capabilities is None or set(spec.capabilities) <= request.allowed_capabilities)
                and (request.permitted_tools is None or spec.name in request.permitted_tools)
                and set(spec.required_resources) <= set(request.resources))

    def available(self, request: ResearchRequest) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values() if self._allowed(tool.spec, request)]

    def dispatch(self, call: ToolCall, context: ToolContext, request: ResearchRequest,
                 timeout_seconds: float | None = None) -> ToolObservation:
        tool = self._tools.get(call.name)
        if tool is None or not self._allowed(tool.spec, request):
            raise ToolScopeError(f"tool is unavailable: {call.name}")
        # A plugin receives only handles it declares, never all host resources.
        scoped = ToolContext(MappingProxyType({key: context.resources[key] for key in tool.spec.required_resources}),
                             context.artifacts, context.episode_id)
        timeout = min(tool.spec.timeout_seconds, timeout_seconds) if timeout_seconds is not None else tool.spec.timeout_seconds
        with call_deadline(timeout):
            validate_value(call.arguments, tool.spec.input_schema)
            observation = tool.execute(call.arguments, scoped)
            if not isinstance(observation, ToolObservation) or observation.status not in {"ok", "error"}:
                raise TypeError("tool returned an invalid observation")
            if tool.spec.output_schema is not None and observation.status == "ok":
                try:
                    validate_value(observation.content, tool.spec.output_schema, "tool output")
                except ToolInputError as exc:
                    raise TypeError(str(exc)) from exc
            raw = asdict(observation)
            encoded = json_text(raw).encode("utf-8")
        if len(encoded) > tool.spec.max_output_bytes:
            artifact_id = context.artifacts.put(context.episode_id, raw)
            return ToolObservation(status=observation.status,
                                   content={"artifact_id": artifact_id, "bytes": len(encoded),
                                            "message": "Output exceeded tool limit; retrieve bounded artifact ranges."},
                                   truncated=True, error="Error details archived." if observation.error else None)
        return observation
