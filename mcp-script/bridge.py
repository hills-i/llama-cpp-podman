from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from bridge_core import (
    _extract_response_tool_calls,
    _get_env,
    _import_mcp_client,
    _load_env,
    _mcp_result_to_text,
    _response_text,
    _responses_tools,
    _system_prompt,
    _tool_call_name_and_args,
)


async def _chat_once(llm: AsyncOpenAI, session, user_text: str) -> str:
    input_items: list[Any] = [
        {"role": "user", "content": user_text},
    ]

    tools = _responses_tools()
    model = _get_env("LLM_MODEL")

    while True:
        resp = await llm.responses.create(
            model=model,
            instructions=_system_prompt(),
            input=input_items,
            tools=tools,
        )

        tool_calls = _extract_response_tool_calls(resp)
        if not tool_calls:
            return _response_text(resp)

        input_items.extend(resp.output or [])

        for tc in tool_calls:
            name, args = _tool_call_name_and_args(tc)
            try:
                result = await session.call_tool(name, args)
                tool_text = _mcp_result_to_text(result)
            except Exception as e:
                tool_text = json.dumps({"error": f"Tool call failed: {name}", "detail": str(e)})

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": tool_text,
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
