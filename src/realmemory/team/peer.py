"""Живой peer-endpoint: сетевой доступ ТОЛЬКО к опубликованным следам.

Каждый участник может поднять демон (`python -m realmemory.team serve`),
который делает две вещи: периодически шлёт presence-хартбит координатору со
своим адресом и отвечает на /recall — поиск строго по следам, на которые
ссылается АКТИВНАЯ публикация из локального registry.

Гарантия приватности конструктивная, а не фильтрацией: множество кандидатов
задаётся JOIN'ом memories × publications, личные следы вне публикаций не
попадают в рассмотрение ни на одном шаге. Auth тот же shared-token; несовпадение
эмбеддера — честный 409 по той же конвенции, что у координатора.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from ._http import (
    BadRequest,
    BoundedThreadingHTTPServer,
    PayloadTooLarge,
    TeamHandler,
    cosine_scores,
    decode_vector_b64,
)
from .policy import require_bind_token

if TYPE_CHECKING:
    from ..types import MemoryRecord

HEARTBEAT_INTERVAL_S = 30.0


def lan_ip() -> str:
    """Свой адрес в LAN без внешних запросов: UDP-connect только смотрит
    таблицу маршрутизации, пакет не отправляется. Пусто — не удалось."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.255.255", 1))
        return str(s.getsockname()[0])
    except OSError:
        return ""
    finally:
        s.close()


class PeerState:
    """Читающий доступ к мозгу владельца: только опубликованное подмножество."""

    def __init__(self, root: Path) -> None:
        from ..config import MemoryConfig
        from ..hook_cli import _infer_dim, _load_cfg
        from ..store.sqlite_store import MemoryStore

        cfg = _load_cfg(root) or MemoryConfig(dim=_infer_dim(root))
        self.store = MemoryStore(root / "memory.db", dim=cfg.dim)
        self.embedder_name = self.store.get_meta("embedder") or ""

    def published_traces(self) -> dict[int, float]:
        """{trace_id: published_at последней активной публикации}."""
        out: dict[int, float] = {}
        for pub in self.store.publications_active():
            tid, pub_at = int(pub[1]), float(pub[4])
            out[tid] = max(out.get(tid, 0.0), pub_at)
        return out

    def recall_live(self, query_vec, k: int, embedder: str,
                    project: str | None) -> tuple[list[dict], str]:
        """Топ-k по косинусу среди активных публикаций владельца.

        Возвращает (хиты, имя эмбеддера владельца); сравнение паритетности —
        на вызывавшей стороне/хендлере через исключение Mismatch. Строки с
        размерностью, отличной от запроса, несравнимы — пропускаются."""
        import numpy as np

        if embedder and self.embedder_name and embedder != self.embedder_name:
            raise EmbedderMismatchPeer([self.embedder_name], embedder)
        allowed = self.published_traces()
        if not allowed:
            return [], self.embedder_name
        qvec = np.asarray(query_vec, dtype=np.float32)
        rows: list[tuple[MemoryRecord, np.ndarray]] = []
        for rec in self.store.get_many(sorted(allowed)):
            if project is not None and rec.scope != project:
                continue
            vec = np.asarray(rec.embedding, dtype=np.float32)
            if vec.size != qvec.size:
                continue  # битая/чужая строка: косинус имел бы нулевой смысл
            rows.append((rec, vec))
        scores = cosine_scores(qvec, [vec for _rec, vec in rows])
        hits: list[dict] = []
        for score, (rec, _vec) in zip(scores.tolist(), rows):
            tid = int(rec.id or 0)
            hits.append({
                "trace_id": tid, "text": rec.text,
                "author": rec.author or "", "project": rec.scope,
                "score": round(score, 6),
                "published_at": allowed.get(tid, 0.0),
                "embedder": self.embedder_name,
            })
        hits.sort(key=lambda h: (-h["score"], h["trace_id"]))
        return hits[:max(1, int(k))], self.embedder_name


class EmbedderMismatchPeer(Exception):
    def __init__(self, known: list[str], requested: str) -> None:
        super().__init__(
            f"peer работает на {known}, запрос закодирован {requested!r}")
        self.known = known
        self.requested = requested


class PeerHandler(TeamHandler):
    server_version = "realmemory-peer/0.8"

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            return self._send(200, {"ok": True})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        if path != "/recall":
            return self._send(404, {"error": "not found"})
        try:
            payload = self._read_json()
        except PayloadTooLarge as exc:
            return self._send(413, {"error": str(exc)})
        except BadRequest as exc:
            return self._send(400, {"error": str(exc)})
        try:
            embedder = str(payload.get("embedder", ""))
            if not embedder:
                # без имени эмбеддера сравнение бессмысленно: честный отказ
                return self._send(400, {"error": "embedder required"})
            qvec = decode_vector_b64(payload.get("query_embedding_b64", ""))
            k = int(payload.get("k") or 5)
        except (BadRequest, TypeError, ValueError) as exc:
            return self._send(400, {"error": f"bad request: {exc}"})
        try:
            hits, _ = self.state.recall_live(qvec, k, embedder,
                                             payload.get("project"))
            return self._send(200, {"hits": hits, "live": True})
        except EmbedderMismatchPeer as exc:
            return self._send(409, {"error": "embedder_mismatch",
                                    "known_embedders": exc.known,
                                    "requested": exc.requested})
        except Exception:  # noqa: BLE001 - соединению нужен ответ всегда
            import traceback

            traceback.print_exc(file=sys.stderr)
            return self._send(500, {"error": "internal error"})


def make_peer_server(root, host="127.0.0.1", port=8410, token=None):
    require_bind_token(host, token, what="peer-endpoint")
    srv = BoundedThreadingHTTPServer((host, int(port)), PeerHandler)
    srv.state = PeerState(Path(root))  # type: ignore[attr-defined]
    srv.token = token  # type: ignore[attr-defined]
    return srv


def start_heartbeat(policy, address: str, stop_event: threading.Event,
                    interval_s: float = HEARTBEAT_INTERVAL_S) -> threading.Thread:
    """Фоновый цикл presence: «я online, мой peer-endpoint по адресу»."""

    def loop() -> None:  # pragma: no cover - фоновый поток интерактивного демона
        from .sync import make_client

        while not stop_event.is_set():
            try:
                client = make_client(policy)
                client.heartbeat(policy.identity, address=address,
                                 projects=[p.name for p in policy.projects])
            except Exception as exc:  # noqa: BLE001 - демон не должен падать
                print(f"[realmemory-peer] heartbeat skipped: {exc}",
                      file=sys.stderr)
            stop_event.wait(interval_s)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th
