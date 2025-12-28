from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

# ============================================================================
# Configuration Constants
# ============================================================================

# Maximum iterations for tool call loop to prevent infinite loops
MAX_TOOL_ITERATIONS = 10

# Rate limiting: requests per minute per client
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60


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


def _system_prompt() -> str:
    return (
        "You are a helpful assistant running fully locally. "
        "If you need facts from the local PostgreSQL database, you may call tools. "
        "The database tools are strictly read-only (SELECT only). "
        "When you call query_database, always write safe, narrow SELECT queries "
        "and limit the result size."
    )


def _extract_tool_calls(message: Any) -> list[Any]:
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


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model: str | None = None
    max_tokens: int | None = Field(default=2000, ge=1, le=8192)
    stream: bool | None = False


class ToolTraceItem(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool
    result: str | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_trace: list[ToolTraceItem] = []


# ============================================================================
# Rate Limiter (simple in-memory implementation)
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter using sliding window.
    
    For production, consider using Redis-backed rate limiting.
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed and record it if so."""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old entries
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > window_start
            ]
            
            if len(self._requests[client_id]) >= self.max_requests:
                return False
            
            self._requests[client_id].append(now)
            return True
    
    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until next request is allowed."""
        if not self._requests[client_id]:
            return 0
        oldest = min(self._requests[client_id])
        retry_after = int(oldest + self.window_seconds - time.time()) + 1
        return max(0, retry_after)


# ============================================================================
# MCP State Management
# ============================================================================

class _MCPState:
    def __init__(self) -> None:
        self.exit_stack: AsyncExitStack | None = None
        self.session: Any | None = None
        self.lock: asyncio.Lock = asyncio.Lock()
        self.llm: AsyncOpenAI | None = None
        self.client_session_cls: Any | None = None
        self.stdio_server_params_cls: Any | None = None
        self.stdio_client_fn: Any | None = None
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


async def _start_mcp_session(mcp_state: _MCPState) -> None:
    """Start the MCP stdio server subprocess once and keep a persistent session."""
    if mcp_state.exit_stack is not None:
        return

    base_url = _get_env("LLM_BASE_URL", "http://localhost:8080/v1")
    api_key = _get_env("OPENAI_API_KEY", "local")
    mcp_state.llm = AsyncOpenAI(base_url=base_url, api_key=api_key)

    ClientSession, StdioServerParameters, stdio_client = _import_mcp_client()
    mcp_state.client_session_cls = ClientSession
    mcp_state.stdio_server_params_cls = StdioServerParameters
    mcp_state.stdio_client_fn = stdio_client

    pg_server_path = Path(__file__).with_name("pg_server.py")
    if not pg_server_path.exists():
        raise RuntimeError(f"pg_server.py not found at: {pg_server_path}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(pg_server_path)],
        env=os.environ.copy(),
    )

    exit_stack = AsyncExitStack()
    try:
        read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server_params))
        session = ClientSession(read_stream, write_stream)
        session = await exit_stack.enter_async_context(session)
        await session.initialize()
    except Exception:
        await exit_stack.aclose()
        raise

    mcp_state.exit_stack = exit_stack
    mcp_state.session = session


async def _stop_mcp_session(mcp_state: _MCPState) -> None:
    """Stop the MCP session and clean up resources."""
    if mcp_state.exit_stack is None:
        return

    try:
        await mcp_state.exit_stack.aclose()
    finally:
        mcp_state.exit_stack = None
        mcp_state.session = None


async def _restart_mcp_session(mcp_state: _MCPState) -> None:
    """Restart the MCP session (stop then start)."""
    await _stop_mcp_session(mcp_state)
    await _start_mcp_session(mcp_state)


# ============================================================================
# FastAPI App with Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP session lifecycle with FastAPI lifespan.
    
    This replaces the deprecated @app.on_event("startup") and
    @app.on_event("shutdown") handlers.
    """
    _load_env()
    mcp_state = _MCPState()
    app.state.mcp = mcp_state
    
    try:
        await _start_mcp_session(mcp_state)
        yield
    finally:
        await _stop_mcp_session(mcp_state)


app = FastAPI(title="mcp-bridge", version="0.1", lifespan=lifespan)


@app.get("/mcp/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "llm_base_url": os.getenv("LLM_BASE_URL", ""),
        "llm_model": os.getenv("LLM_MODEL", ""),
    }


@app.get("/mcp/tools")
async def tools() -> dict[str, Any]:
    return {"tools": _openai_tools()}


async def _chat_with_trace(
    llm: AsyncOpenAI,
    session,
    req: ChatRequest,
    *,
    session_lock: asyncio.Lock,
) -> ChatResponse:
    """Execute chat with tool calling and return trace.
    
    Includes iteration limit to prevent infinite tool call loops.
    """
    model = req.model or _get_env("LLM_MODEL")
    tools = _openai_tools()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": req.message},
    ]

    trace: list[ToolTraceItem] = []
    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_TOOL_ITERATIONS:
            # Return partial response with warning instead of infinite loop
            return ChatResponse(
                answer="[Error: Maximum tool call iterations exceeded. Partial response may be incomplete.]",
                tool_trace=trace,
            )
        
        resp = await llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=req.max_tokens,
        )

        msg = resp.choices[0].message
        tool_calls = _extract_tool_calls(msg)
        if not tool_calls:
            return ChatResponse(answer=msg.content or "", tool_trace=trace)

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name, args = _tool_call_name_and_args(tc)
            try:
                async with session_lock:
                    result = await session.call_tool(name, args)
                tool_text = _mcp_result_to_text(result)
                trace.append(ToolTraceItem(name=name, args=args, ok=True, result=tool_text))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                tool_text = json.dumps({"error": "Tool call failed", "tool": name, "detail": err})
                trace.append(ToolTraceItem(name=name, args=args, ok=False, error=err))

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_text})


def _get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting.
    
    Trusts X-Real-IP header if set by the reverse proxy (Apache in this deployment).
    X-Real-IP is safer than X-Forwarded-For because:
    1. It contains a single IP (not a chain)
    2. Apache is configured to overwrite it with REMOTE_ADDR
    3. Clients cannot spoof it if Apache is properly configured
    
    Falls back to direct client address if header is not present.
    """
    # Trust X-Real-IP set by Apache (configure Apache to overwrite this header)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    # Fallback to direct connection address
    return request.client.host if request.client else "unknown"


@app.post("/mcp/chat", response_model=ChatResponse)
async def chat(request: Request, req: ChatRequest) -> ChatResponse:
    """Execute chat with MCP tool support.
    
    This endpoint intentionally executes MCP tools server-side 
    (no direct browser MCP). Includes rate limiting and proper
    session management.
    """
    mcp_state: _MCPState = app.state.mcp
    
    # Rate limiting check
    client_id = _get_client_id(request)
    if not await mcp_state.rate_limiter.is_allowed(client_id):
        retry_after = mcp_state.rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Ensure session is started with proper locking to avoid race conditions
    async with mcp_state.lock:
        if mcp_state.exit_stack is None or mcp_state.session is None or mcp_state.llm is None:
            try:
                await _start_mcp_session(mcp_state)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail="MCP session initialization failed. Service temporarily unavailable."
                ) from e

    try:
        assert mcp_state.session is not None
        assert mcp_state.llm is not None

        return await _chat_with_trace(
            mcp_state.llm,
            mcp_state.session,
            req,
            session_lock=mcp_state.lock,
        )
    except HTTPException:
        raise
    except Exception as e:
        # If the stdio subprocess/session died, restart once and retry.
        try:
            async with mcp_state.lock:
                await _restart_mcp_session(mcp_state)
            
            assert mcp_state.session is not None
            assert mcp_state.llm is not None
            
            return await _chat_with_trace(
                mcp_state.llm,
                mcp_state.session,
                req,
                session_lock=mcp_state.lock,
            )
        except Exception:
            # Don't leak internal error details
            raise HTTPException(
                status_code=500, 
                detail="Bridge encountered an error. Please try again."
            ) from e


def main() -> None:
    # Convenience entrypoint for running via `python web_bridge.py`.
    import uvicorn

    host = os.getenv("BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("BRIDGE_PORT", "8090"))

    uvicorn.run("web_bridge:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    # Running this module directly should work (uvicorn dependency required).
    asyncio.run(asyncio.sleep(0))
    main()
