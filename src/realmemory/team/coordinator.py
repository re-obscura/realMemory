"""Координатор командного слоя: presence + read-cache опубликованного.

Пассивный сервис (stdlib http.server, ноль зависимостей): никогда не пишет в
локальные базы участников и не инициирует запросов. Хранит только то, что
участники явно опубликовали через registry-sync, и эфемерные хартбиты с TTL.

Запуск:
    python -m realmemory.team.coordinator --data ./coord_data \
        [--host 127.0.0.1] [--port 8400] [--token-env REALMEMORY_TEAM_TOKEN]

Маршруты:
    GET  /health                              {"ok": true}
    POST /heartbeat   {identity, address?, projects?}
    GET  /presence                    → [{identity, last_seen, online}]
    POST /publish     {items:[{publication_id, project, author, text,
                               embedding_b64, published_at,
                               content_hash, embedder}]}
    POST /retract     {tombstones:[{publication_id, revoked_at}]}
    POST /search      {query_embedding_b64, k, embedder, author?, project?}
    GET  /cache/dump                  → активные публикации + tombstones

Аутентификация: если задан токен, все маршруты кроме /health требуют заголовок
`Authorization: Bearer <token>`. Поиск честно отказывает при несовпадении
эмбеддера: косинусы разных моделей несравнимы.
"""
from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PRESENCE_TTL_S = 90.0


def _b64decode_vec(s: str):
    import numpy as np

    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=np.float32)


def _open_state(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(data_dir / "coordinator.db"), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS pubs (
            publication_id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT '',
            author TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL DEFAULT '',
            embedding BLOB NOT NULL,
            published_at REAL NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            embedder TEXT NOT NULL DEFAULT '',
            revoked_at REAL
        );
        CREATE TABLE IF NOT EXISTS hearts (
            identity TEXT PRIMARY KEY,
            last_seen REAL NOT NULL,
            address TEXT NOT NULL DEFAULT '',
            projects TEXT NOT NULL DEFAULT '[]'
        );
        """
    )
    con.commit()
    return con


class CoordinatorState:
    """Состояние под одним замком: sqlite + presence-словарь."""

    def __init__(self, data_dir: Path) -> None:
        self.lock = threading.Lock()
        self.con = _open_state(data_dir)

    def heartbeat(self, identity: str, address: str = "",
                  projects: list[str] | None = None) -> None:
        now = time.time()
        with self.lock, self.con:
            self.con.execute(
                "INSERT INTO hearts(identity,last_seen,address,projects)"
                " VALUES(?,?,?,?) ON CONFLICT(identity) DO UPDATE SET"
                " last_seen=excluded.last_seen, address=excluded.address,"
                " projects=excluded.projects",
                (identity[:64], now, address[:120],
                 json.dumps(projects or [])),
            )

    def presence(self) -> list[dict]:
        cutoff = time.time() - PRESENCE_TTL_S
        with self.lock:
            rows = self.con.execute(
                "SELECT identity, last_seen, address FROM hearts"
                " WHERE last_seen > ? ORDER BY identity", (cutoff,)
            ).fetchall()
        return [{"identity": r[0], "last_seen": float(r[1]),
                 "online": True, "address": r[2]} for r in rows]

    def publish(self, items: list[dict]) -> int:
        accepted = 0
        with self.lock, self.con:
            for it in items:
                cur = self.con.execute(
                    "INSERT OR IGNORE INTO pubs(publication_id, project,"
                    " author, text, embedding, published_at, content_hash,"
                    " embedder, revoked_at) VALUES(?,?,?,?,?,?,?,?,NULL)",
                    (str(it["publication_id"]), str(it.get("project", "")),
                     str(it.get("author", "")), str(it.get("text", "")),
                     base64.b64decode(it["embedding_b64"]),
                     float(it["published_at"]),
                     str(it.get("content_hash", "")),
                     str(it.get("embedder", ""))),
                )
                accepted += int(cur.rowcount)
        return accepted

    def retract(self, tombstones: list[dict]) -> int:
        touched = 0
        with self.lock, self.con:
            for tb in tombstones:
                cur = self.con.execute(
                    "UPDATE pubs SET revoked_at=? WHERE publication_id=?"
                    " AND revoked_at IS NULL",
                    (float(tb["revoked_at"]), str(tb["publication_id"])),
                )
                touched += int(cur.rowcount)
        return touched

    def search(self, query_b64: str, k: int, embedder: str,
               author: str | None, project: str | None) -> tuple[list[dict], dict]:
        """Топ-k по косинусу среди активных публикаций. Возвращает (хиты,
        мета); мета.nonempty_mismatch=True если кэш содержит записи других
        эмбеддеров и сравнение было бы некорректным."""
        q = _b64decode_vec(query_b64)
        with self.lock:
            rows = self.con.execute(
                "SELECT publication_id, project, author, text, embedding,"
                " published_at, embedder FROM pubs WHERE revoked_at IS NULL"
            ).fetchall()
        rows = [r for r in rows
                if (author is None or r[2] == author)
                and (project is None or r[1] == project)]
        known = sorted({r[6] for r in rows})
        mismatch_meta = {"known_embedders": known}
        comparable = [r for r in rows if not embedder or r[6] == embedder]
        if embedder and known and embedder not in known:
            # всё содержимое — другие модели: косинус имел бы нулевой смысл
            return [], {"error": "embedder_mismatch",
                        "known_embedders": known,
                        "requested": embedder}
        scored = []
        for pid, proj, auth, text, emb_blob, pub_at, emb_name in comparable:
            import numpy as np

            vec = np.frombuffer(emb_blob, dtype=np.float32)
            denom = float(np.linalg.norm(vec) * (np.linalg.norm(q) or 1.0))
            score = float(np.dot(vec, q) / denom) if denom else 0.0
            scored.append({"publication_id": pid, "project": proj,
                           "author": auth, "text": text,
                           "published_at": pub_at, "score": round(score, 6)})
        scored.sort(key=lambda h: (-h["score"], h["publication_id"]))
        return scored[:max(1, int(k))], mismatch_meta

    def dump(self) -> dict:
        with self.lock:
            active = self.con.execute(
                "SELECT publication_id, project, author, text, published_at,"
                " content_hash, embedder FROM pubs WHERE revoked_at IS NULL"
            ).fetchall()
            tombs = self.con.execute(
                "SELECT publication_id, revoked_at FROM pubs"
                " WHERE revoked_at IS NOT NULL"
            ).fetchall()
        return {
            "active": [{"publication_id": r[0], "project": r[1],
                        "author": r[2], "text": r[3], "published_at": r[4],
                        "content_hash": r[5], "embedder": r[6]} for r in active],
            "tombstones": [{"publication_id": r[0], "revoked_at": float(r[1])}
                           for r in tombs],
        }


class CoordinatorHandler(BaseHTTPRequestHandler):
    server_version = "realmemory-coordinator/0.7"

    @property
    def state(self) -> CoordinatorState:
        return self.server.state  # type: ignore[attr-defined]

    @property
    def expected_token(self) -> str | None:
        return getattr(self.server, "token", None)  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # тихо: шум координатора не нужен
        return

    # -- служебное -------------------------------------------------------------

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        want = self.expected_token
        if not want:
            return True
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {want}"

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 32 * 1024 * 1024:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- маршруты --------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._send(200, {"ok": True})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        if path == "/presence":
            return self._send(200, {"presence": self.state.presence()})
        if path == "/cache/dump":
            return self._send(200, self.state.dump())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._send(200, {"ok": True})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        payload = self._read_json()
        try:
            if path == "/heartbeat":
                self.state.heartbeat(str(payload.get("identity", "")),
                                     str(payload.get("address", "")),
                                     payload.get("projects"))
                return self._send(200, {"ok": True,
                                        "ttl_s": PRESENCE_TTL_S})
            if path == "/publish":
                n = self.state.publish(payload.get("items") or [])
                return self._send(200, {"accepted": n,
                                        "received": len(payload.get("items") or [])})
            if path == "/retract":
                n = self.state.retract(payload.get("tombstones") or [])
                return self._send(200, {"retracted": n})
            if path == "/search":
                hits, meta = self.state.search(
                    str(payload.get("query_embedding_b64", "")),
                    int(payload.get("k") or 5),
                    str(payload.get("embedder", "")),
                    payload.get("author"), payload.get("project"))
                code = 409 if meta.get("error") == "embedder_mismatch" else 200
                return self._send(code, {"hits": hits, **meta})
        except (KeyError, ValueError, TypeError) as exc:
            return self._send(400, {"error": f"bad request: {exc}"})
        except Exception as exc:  # noqa: BLE001 - соединению нужен ответ ВСЕГДА
            return self._send(500, {"error": f"internal: {exc!r}"})
        return self._send(404, {"error": "not found"})


def make_server(data_dir, host="127.0.0.1", port=8400, token=None):
    """Собрать сервер (для тестов: порт 0 → фактический из server_address)."""
    srv = ThreadingHTTPServer((host, int(port)), CoordinatorHandler)
    srv.state = CoordinatorState(Path(data_dir))  # type: ignore[attr-defined]
    srv.token = token  # type: ignore[attr-defined]
    return srv


def main(argv=None) -> None:  # pragma: no cover - долгоживущий процесс
    parser = argparse.ArgumentParser(prog="realmemory-coordinator")
    parser.add_argument("--data", required=True, help="каталог состояния")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--token-env", default="REALMEMORY_TEAM_TOKEN",
                        help="имя переменной окружения с общим токеном команды")
    args = parser.parse_args(argv)
    token = __import__("os").environ.get(args.token_env, "").strip() or None
    srv = make_server(args.data, args.host, args.port, token)
    actual = srv.server_address[1]
    print(f"[realmemory-coordinator] listening on {args.host}:{actual}"
          f" ({'токен включён' if token else 'БЕЗ токена'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
