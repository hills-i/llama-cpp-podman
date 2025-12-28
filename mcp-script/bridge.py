from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


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
    """Import MCP client pieces from the MCP SDK."""

    try:
        # Common MCP SDK layout.
        from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore

        try:
            from mcp import ClientSession  # type: ignore
        except Exception:
            from mcp.client.session import ClientSession  # type: ignore

        return ClientSession, StdioServerParameters, stdio_client
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "MCP SDK is not installed. Run: pip install -r requirements.txt"
        ) from e


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


def _openai_tools() -> list[dict[str, Any]]:
    # Keep tool schema stable and simple.
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


def _system_prompt() -> str:
    return (
        "You are a helpful assistant running fully locally. "
        "If you need facts from the local PostgreSQL database, you may call tools. "
        "The database tools are strictly read-only (SELECT only). "
        "When you call query_database, always write safe, narrow SELECT queries "
        "and limit the result size."
    )


def _extract_tool_calls(message: Any) -> list[Any]:
    # openai>=1.x returns message.tool_calls; llama.cpp may vary slightly.
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return list(tool_calls)
    return []


def _tool_call_name_and_args(tool_call: Any) -> tuple[str, dict[str, Any]]:
    fn = getattr(tool_call, "function", None)
    if fn is None:
        raise RuntimeError("tool_call.function is missing")
    name = getattr(fn, "name", None)
    if not name:
        raise RuntimeError("tool_call.function.name is missing")

    raw_args = getattr(fn, "arguments", "{}") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return str(name), args


def _mcp_result_to_text(result: Any) -> str:
    # MCP CallToolResult typically has `.content` as a list of content blocks.
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


async def _chat_once(llm: AsyncOpenAI, session, user_text: str) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_text},
    ]

    tools = _openai_tools()
    model = _get_env("LLM_MODEL")

    while True:
        resp = await llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = resp.choices[0].message
        tool_calls = _extract_tool_calls(msg)
        if not tool_calls:
            return msg.content or ""

        # Append assistant message including tool calls so the LLM can continue.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name, args = _tool_call_name_and_args(tc)
            try:
                result = await session.call_tool(name, args)
                tool_text = _mcp_result_to_text(result)
            except Exception as e:
                tool_text = json.dumps({"error": f"Tool call failed: {name}", "detail": str(e)})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_text,
                }
            )


async def main() -> None:
    _load_env()

    base_url = _get_env("LLM_BASE_URL", "http://localhost:8080/v1")
    api_key = _get_env("OPENAI_API_KEY", "local")

    llm = AsyncOpenAI(base_url=base_url, api_key=api_key)

    ClientSession, StdioServerParameters, stdio_client = _import_mcp_client()

    pg_server_path = Path(__file__).with_name("pg_server.py")
    if not pg_server_path.exists():
        raise RuntimeError(f"pg_server.py not found at: {pg_server_path}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(pg_server_path)],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("Bridge ready. Type your question (or 'exit').")
            while True:
                try:
                    user_text = input("> ").strip()
                except EOFError:
                    break

                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    break

                answer = await _chat_once(llm, session, user_text)
                print(answer)

if __name__ == "__main__":
    asyncio.run(main())
