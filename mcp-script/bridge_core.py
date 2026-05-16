from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return
    load_dotenv(override=False)


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return value


def _import_mcp_client():
    try:
        from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore

        try:
            from mcp import ClientSession  # type: ignore
        except Exception:
            from mcp.client.session import ClientSession  # type: ignore

        return ClientSession, StdioServerParameters, stdio_client
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "MCP SDK is not installed. Run: pip install -r mcp-script/requirements.txt"
        ) from e


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


def _openai_tools() -> list[dict[str, Any]]:
    tools: list[ToolSpec] = [
        ToolSpec(
            name="list_tables",
            description="List available user tables (schema + name) in PostgreSQL.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="query_database",
            description=(
                "Run a read-only SQL SELECT query against PostgreSQL. "
                "Only SELECT/CTE queries are allowed; no modification statements."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT statement (optionally WITH/CTE).",
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional named parameters for psycopg (%(name)s style).",
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Optional max number of rows to return (capped).",
                    },
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        ),
    ]

    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _responses_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
        }
        for tool in _openai_tools()
    ]


def _system_prompt() -> str:
    return (
        "You are a helpful assistant running fully locally. "
        "If you need facts from the local PostgreSQL database, you may call tools. "
        "The database tools are strictly read-only (SELECT only). "
        "When you call query_database, always write safe, narrow SELECT queries "
        "and limit the result size."
    )


def _extract_response_tool_calls(response: Any) -> list[Any]:
    return [
        item
        for item in (response.output or [])
        if item.type == "function_call"
    ]


def _tool_call_name_and_args(tool_call: Any) -> tuple[str, dict[str, Any]]:
    raw_args = tool_call.arguments or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return str(tool_call.name), args


def _response_text(response: Any) -> str:
    return response.output_text or ""


def _mcp_result_to_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if not content:
        return json.dumps({"result": getattr(result, "result", None)}, ensure_ascii=False, default=str)

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(json.dumps(item, ensure_ascii=False, default=str))
    return "\n".join(parts)
