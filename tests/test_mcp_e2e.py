"""E2E-тест MCP-сервера: реальный stdio-транспорт, handshake, вызовы тулов.
Именно эта схема используется при подключении к ZCode."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SRC = Path(__file__).resolve().parents[1] / "src"


def _server_params(path):
    # PYTHONPATH со src: подпроцесс работает и без `pip install -e .`
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "realmemory.api.mcp_server",
            "--path", str(path),
            "--embedder", "hashing",  # без скачивания модели в тесте
        ],
        env=env,
    )


def _text_of(result) -> dict:
    assert result.content, "пустой ответ сервера"
    return json.loads(result.content[0].text)


def test_full_tool_cycle(tmp_path):
    async def scenario() -> None:
        async with stdio_client(_server_params(tmp_path / "rm")) as (read, write), \
                ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {"recall", "memorize", "reflect", "revise",
                    "introspect", "dream_log"} <= names

            written = _text_of(await session.call_tool(
                "memorize",
                {"text": "Проект realMemory использует PostgreSQL 16"},
            ))
            assert written["created"] and written["action"] == "create"

            recalled = _text_of(await session.call_tool(
                "recall",
                {"query": "PostgreSQL 16 realMemory", "k": 3},
            ))
            assert not recalled["abstained"]
            assert any(it["id"] == written["memory_id"] for it in recalled["items"])

            fb = _text_of(await session.call_tool(
                "reflect",
                {"memory_ids": [written["memory_id"]], "reward": 1.0},
            ))
            assert fb["touched"] == 1

            upd = _text_of(await session.call_tool(
                "revise",
                {"old_id": written["memory_id"],
                 "new_text": "Проект realMemory перешёл на PostgreSQL 17"},
            ))
            assert upd["new_id"] != upd["old_id"]

            recalled2 = _text_of(await session.call_tool(
                "recall",
                {"query": "PostgreSQL 17 realMemory", "k": 3},
            ))
            by_id = {it["id"]: it["text"] for it in recalled2["items"]}
            assert upd["new_id"] in by_id
            assert "PostgreSQL 17" in by_id[upd["new_id"]]
            assert upd["old_id"] not in by_id

            log_raw = await session.call_tool("dream_log", {})
            log = json.loads(log_raw.content[0].text)
            assert log["memories_active"] >= 1

    asyncio.run(scenario())
