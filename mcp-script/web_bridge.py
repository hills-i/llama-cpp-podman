from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

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

# ============================================================================
# Configuration Constants
# ============================================================================

# Maximum iterations for tool call loop to prevent infinite loops
MAX_TOOL_ITERATIONS = 10

# Rate limiting: requests per minute per client
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# MCP session startup can race while the stdio child process comes up.
# Do not retry once chat/tool execution has started.
MCP_SESSION_INIT_ATTEMPTS = 2


def _parse_tool_payload(tool_text: str) -> dict[str, Any] | None:
    """Best-effort parse of JSON tool responses produced by the MCP server."""
    try:
        payload = json.loads(tool_text)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _tool_error_from_text(tool_text: str) -> str | None:
    """Extract a tool-level error from a JSON payload, if present."""
    payload = _parse_tool_payload(tool_text)
    if not payload:
        return None

    error = payload.get("error")
    if not error:
        return None

    detail = payload.get("detail")
    if detail:
        return f"{error}: {detail}"
    return str(error)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model: str | None = None
    max_tokens: int | None = Field(default=2000, ge=1, le=8192)


class ToolTraceItem(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool
    result: str | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_trace: list[ToolTraceItem] = Field(default_factory=list)


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
        self.lock: asyncio.Lock = asyncio.Lock()
        self.llm: AsyncOpenAI | None = None
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


@asynccontextmanager
async def _mcp_client_session():
    """Create a per-request MCP stdio session in a single task context."""
    ClientSession, StdioServerParameters, stdio_client = _import_mcp_client()

    pg_server_path = Path(__file__).with_name("pg_server.py")
    if not pg_server_path.exists():
        raise RuntimeError(f"pg_server.py not found at: {pg_server_path}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(pg_server_path)],
        env=os.environ.copy(),
    )

    for attempt in range(MCP_SESSION_INIT_ATTEMPTS):
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except Exception:
            await stack.aclose()
            if attempt == MCP_SESSION_INIT_ATTEMPTS - 1:
                raise
            continue

        try:
            yield session
        finally:
            await stack.aclose()
        return


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
    base_url = _get_env("LLM_BASE_URL", "http://localhost:8080/v1")
    api_key = _get_env("OPENAI_API_KEY", "local")
    mcp_state.llm = AsyncOpenAI(base_url=base_url, api_key=api_key)
    yield


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
    return {"tools": _responses_tools()}


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
    tools = _responses_tools()

    input_items: list[Any] = [
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
        
        resp = await llm.responses.create(
            model=model,
            instructions=_system_prompt(),
            input=input_items,
            tools=tools,
            max_output_tokens=req.max_tokens,
        )

        tool_calls = _extract_response_tool_calls(resp)
        if not tool_calls:
            return ChatResponse(answer=_response_text(resp), tool_trace=trace)

        input_items.extend(resp.output or [])

        for tc in tool_calls:
            name, args = _tool_call_name_and_args(tc)
            try:
                async with session_lock:
                    result = await session.call_tool(name, args)
                tool_text = _mcp_result_to_text(result)
                tool_error = _tool_error_from_text(tool_text)
                trace.append(
                    ToolTraceItem(
                        name=name,
                        args=args,
                        ok=tool_error is None,
                        result=tool_text,
                        error=tool_error,
                    )
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                tool_text = json.dumps({"error": "Tool call failed", "tool": name, "detail": err})
                trace.append(ToolTraceItem(name=name, args=args, ok=False, error=err))

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": tool_text,
                }
            )


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

    try:
        assert mcp_state.llm is not None

        async with _mcp_client_session() as session:
            return await _chat_with_trace(
                mcp_state.llm,
                session,
                req,
                session_lock=mcp_state.lock,
            )
    except HTTPException:
        raise
    except Exception as e:
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
