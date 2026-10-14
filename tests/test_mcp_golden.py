"""Golden-тест wire-контракта MCP-тулов: точный словарь recall-ответа.

_packet_to_dict — единственное место, где RecallPacket превращается в ответ
клиенту (recall → json.dumps). Потребители MCP-протокола завязаны на имена
ключей ("id", не "memory_id"), округление latency_ms и отсутствие служебного
updated_at. Golden-тест держит весь словарь дословно: любое изменение
контракта становится сознательным решением, а не случайностью.
Пакет собирается прямыми dataclass-конструкторами — store не нужен.
"""
from realmemory.api.mcp_server import _packet_to_dict
from realmemory.types import RecalledMemory, RecallPacket


def _packet() -> RecallPacket:
    item = RecalledMemory(
        memory_id=42,
        text="Проект realMemory использует SQLite WAL",
        kind="semantic",
        cosine=0.518484,
        confidence=0.406806,
        retention=0.692303,
        source="direct",
        created_at=1787826847.2837832,
        updated_at=1787900000.5,
        meta={"supersedes": 1},
        scope="realMemory",
        author="re-obscura",
    )
    return RecallPacket(
        query="wire contract",
        items=(item,),
        abstained=False,
        latency_ms=115.2364891,
    )


def test_recall_packet_wire_dict_exact():
    assert _packet_to_dict(_packet()) == {
        "query": "wire contract",
        "abstained": False,
        "latency_ms": 115.24,  # округление до сотых — единственное в выдаче
        "items": [
            {
                "id": 42,  # наружу id, не memory_id
                "text": "Проект realMemory использует SQLite WAL",
                "kind": "semantic",
                "cosine": 0.518484,
                "confidence": 0.406806,
                "retention": 0.692303,
                "source": "direct",
                "scope": "realMemory",
                "author": "re-obscura",
                "created_at": 1787826847.2837832,
                "meta": {"supersedes": 1},
            }
        ],
    }


def test_updated_at_absent_from_wire():
    out = _packet_to_dict(_packet())
    assert out["items"] and all("updated_at" not in it for it in out["items"])


def test_item_keys_exact_set():
    out = _packet_to_dict(_packet())
    assert set(out) == {"query", "abstained", "latency_ms", "items"}
    assert set(out["items"][0]) == {
        "id", "text", "kind", "cosine", "confidence", "retention",
        "source", "scope", "author", "created_at", "meta",
    }


def test_meta_dict_copied_not_aliased():
    meta = {"k": "v"}
    item = RecalledMemory(
        memory_id=1, text="t", kind="episodic", cosine=0.1, confidence=0.1,
        retention=0.5, source="direct", created_at=1.0, updated_at=2.0,
        meta=meta,
    )
    out = _packet_to_dict(RecallPacket("q", (item,), False, 0.0))
    out["items"][0]["meta"]["k"] = "mutated"  # type: ignore[index]
    assert meta == {"k": "v"}  # dict(it.meta): мутация выдачи не трогает след
