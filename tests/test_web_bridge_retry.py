from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from conftest import load_module


def _request():
    return SimpleNamespace(
        headers={},
        client=SimpleNamespace(host="127.0.0.1"),
    )


def _chat_request(module):
    return module.ChatRequest(
        message="hello",
        model=None,
        max_tokens=2000,
        stream=False,
    )


def test_chat_does_not_retry_after_chat_execution_starts(monkeypatch):
    module = load_module("web_bridge_retry_under_test", "mcp-script/web_bridge.py")
    module.app.state.mcp = module._MCPState()
    module.app.state.mcp.llm = object()
    calls = {"chat": 0}

    @asynccontextmanager
    async def fake_mcp_client_session():
        yield object()

    async def failing_chat_with_trace(llm, session, req, *, session_lock):
        calls["chat"] += 1
        raise RuntimeError("llm failed after chat started")

    monkeypatch.setattr(module, "_mcp_client_session", fake_mcp_client_session)
    monkeypatch.setattr(module, "_chat_with_trace", failing_chat_with_trace)

    async def run_chat():
        await module.chat(_request(), _chat_request(module))

    with pytest.raises(module.HTTPException) as exc_info:
        asyncio.run(run_chat())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Bridge encountered an error. Please try again."
    assert calls["chat"] == 1


def test_mcp_client_session_retries_initialization_only_once(monkeypatch):
    module = load_module("web_bridge_session_retry_under_test", "mcp-script/web_bridge.py")
    calls = {
        "initialize": 0,
        "stdio_exit": 0,
        "session_exit": 0,
    }

    class FakeServerParameters:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeStdioClient:
        async def __aenter__(self):
            return "read-stream", "write-stream"

        async def __aexit__(self, exc_type, exc, tb):
            calls["stdio_exit"] += 1

    class FakeClientSession:
        def __init__(self, read_stream, write_stream):
            self.read_stream = read_stream
            self.write_stream = write_stream

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            calls["session_exit"] += 1

        async def initialize(self):
            calls["initialize"] += 1
            if calls["initialize"] == 1:
                raise RuntimeError("transient init failure")

    def fake_stdio_client(server_params):
        return FakeStdioClient()

    monkeypatch.setattr(
        module,
        "_import_mcp_client",
        lambda: (FakeClientSession, FakeServerParameters, fake_stdio_client),
    )

    async def run_session():
        async with module._mcp_client_session() as session:
            assert isinstance(session, FakeClientSession)
            assert calls["initialize"] == 2

    asyncio.run(run_session())

    assert calls["initialize"] == 2
    assert calls["session_exit"] == 2
    assert calls["stdio_exit"] == 2
